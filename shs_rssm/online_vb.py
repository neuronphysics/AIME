"""Online global-update bookkeeping: full-batch, streaming, memoized, and legacy-EMA.

This module makes the online update semantics of the switching head explicit and
selectable, instead of hard-wiring the EMA forgetting factor:

  * ``full_batch``  — totals are the sum over the batches added since the last
    ``reset()``; the caller sweeps the whole corpus, then runs one global step.
    Exact batch VB.
  * ``streaming``   — Broderick et al.'s Streaming VB sufficient-statistics-update
    (SSU) primitive: natural parameters are ADDITIVE, each batch is absorbed once,
    the prior is counted once (it lives in the M-step, not in the totals). Adding
    the same ``batch_id`` twice raises, because streaming semantics see each datum
    exactly once.
  * ``memoized``    — Hughes/Sudderth memoized VI: per-batch summaries are STORED;
    revisiting a batch replaces its stale summary,
        S_total <- S_total - S_old(batch) + S_new(batch),
    so the totals are always an exact sum of one summary per batch (each computed
    at the parameters current at its last visit). ``is_complete`` certifies that
    every expected batch has been visited at least once, which is the precondition
    for whole-corpus ELBO statements.
  * ``legacy_ema``  — the previous behaviour, S <- (1-tau) S + tau S_batch: a
    forgetting-factor estimator appropriate for Dreamer's non-stationary replay
    (the representation drifts, so old summaries SHOULD decay), but not a batch-VB
    fixed point. Kept, explicitly labelled.

The store is agnostic to what a "summary" is: it handles a dict of tensors (regime
sufficient statistics), a transition-count matrix, a start-count vector, and an
optional dict of Polya-Gamma natural statistics, all with the same arithmetic.
"""
from __future__ import annotations

import torch

MODES = ("full_batch", "streaming", "memoized", "legacy_ema")


def _clone(x):
    if x is None:
        return None
    if isinstance(x, dict):
        return {k: _clone(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        t = [_clone(v) for v in x]
        return tuple(t) if isinstance(x, tuple) else t
    return x.detach().clone()


def _zeros_like(x):
    if x is None:
        return None
    if isinstance(x, dict):
        return {k: _zeros_like(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        t = [_zeros_like(v) for v in x]
        return tuple(t) if isinstance(x, tuple) else t
    return torch.zeros_like(x)


def _iadd(tot, x, sign=1.0):
    if x is None:
        return tot
    if tot is None:
        tot = _zeros_like(x)
    if isinstance(x, dict):
        for k, v in x.items():
            if k not in tot:
                tot[k] = _zeros_like(v)
            tot[k] = _iadd(tot[k], v, sign=sign)
    elif isinstance(x, (tuple, list)):
        merged = [_iadd(t, v, sign=sign) for t, v in zip(tot, x)]
        tot = tuple(merged) if isinstance(tot, tuple) else merged
    else:
        tot.add_(x, alpha=sign)
    return tot


def _ema(tot, x, tau):
    if x is None:
        return tot
    if tot is None:
        return _clone(x)
    if isinstance(x, dict):
        for k, v in x.items():
            if k not in tot:
                tot[k] = v.detach().clone()
            else:
                tot[k].mul_(1.0 - tau).add_(v, alpha=tau)
    else:
        tot.mul_(1.0 - tau).add_(x, alpha=tau)
    return tot


def _all_finite(x) -> bool:
    """Recursive finiteness over tensors, dicts, lists and tuples (round-9)."""
    if x is None:
        return True
    if torch.is_tensor(x):
        return bool(torch.isfinite(x).all())
    if isinstance(x, dict):
        return all(_all_finite(v) for v in x.values())
    if isinstance(x, (tuple, list)):
        return all(_all_finite(v) for v in x)
    return True


def _stream_offset(bid):
    """Strict non-negative integer stream offset, or None to use the id-set fallback.
    Rejects non-integral floats (1.5) and non-numeric strings ('auto0') so they cannot
    silently truncate or collide onto the monotonic cursor (round-6 review, issue 1)."""
    if isinstance(bid, bool):
        return None
    if isinstance(bid, int):
        return bid if bid >= 0 else None
    if isinstance(bid, float):
        return int(bid) if (float(bid).is_integer() and bid >= 0.0) else None
    if isinstance(bid, str) and bid.isdigit():
        return int(bid)
    return None


# ROUND-12 review item 14: unambiguous mode names. The aliases make it explicit that
# "online Dreamer training" (live_ema) is NOT streaming VB, while episode_stream /
# memoized_corpus / full_batch_corpus name the actual VB contracts.
_MODE_ALIASES = dict(live_ema="legacy_ema", episode_stream="streaming",
                     memoized_corpus="memoized", full_batch_corpus="full_batch")


class SuffStatStore:
    """Totals + (in memoized mode) per-batch summaries for the switching-head globals.

    Each summary is the 4-tuple ``(regime_stats: dict, trans_counts, start_counts,
    pg_stats: dict|None)``. All tensors are cloned/detached on entry.
    """

    def __init__(self, mode: str = "memoized", ema_tau: float = 0.02,
                 expected_batches: int | None = None,
                 expected_ids: set | None = None, strict_stream: bool = False):
        mode = _MODE_ALIASES.get(mode, mode)
        if mode not in MODES:
            raise ValueError(f"online mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.ema_tau = float(ema_tau)
        self.expected_batches = expected_batches
        self.expected_ids = set(expected_ids) if expected_ids is not None else None
        self.strict_stream = bool(strict_stream)   # streaming: no id fallback, contiguous offsets
        self.pass_id = 0                 # full_batch: explicit pass counter
        self.n_stale_invalidated = 0     # memoized: batches dropped on repr change
        self.reset()

    # ------------------------------------------------------------- bookkeeping
    def reset(self):
        self._tot_stats = None
        self._tot_C = None
        self._tot_s = None
        self._tot_pg = None
        self._per_batch: dict = {}
        self._batch_repr: dict = {}      # memoized: repr_version per stored batch
        self._pass_repr = None           # streaming/full_batch: repr of the open pass
        self.n_updates = 0
        self._stream_hi = None            # streaming: high-water integer offset (O(1))
        self._stream_count = 0            # streaming: count of absorbed minibatches
        self._fb_finalized = False        # full_batch: single-step-per-pass guard
        self._async_lo = -1               # SDA-Bayes async: contiguous commit watermark
        self._async_ahead = set()         # SDA-Bayes async: bounded out-of-order set
        self._stream_api = None           # streaming: 'serial' | 'async' (no mixing)

    def begin_full_batch_pass(self):
        """Explicit pass boundary for full_batch mode (auto ids never repeat, so
        without this call -- or stable repeated ids -- accumulation would run across
        epochs forever, which is NOT full-batch semantics)."""
        if self.mode != "full_batch":
            raise ValueError(f"begin_full_batch_pass() is for mode='full_batch', "
                             f"store is '{self.mode}'")
        self.reset()
        self.pass_id += 1

    def has_batch(self, batch_id) -> bool:
        return batch_id in self._per_batch

    @property
    def n_batches(self) -> int:
        # streaming's monotonic cursor keeps a count instead of an id set (O(1) memory)
        return len(self._per_batch) + getattr(self, "_stream_count", 0)

    def is_complete(self, expected: int | None = None) -> bool:
        """True iff every expected batch has a stored summary (memoized mode)."""
        if self.mode != "memoized":
            return False
        if self.expected_ids is not None and expected is None:
            return set(self._per_batch) == self.expected_ids
        exp = self.expected_batches if expected is None else expected
        if exp is None:
            return False
        return self.n_batches == exp

    # ------------------------------------------------------------------ update
    def add_batch(self, batch_id, regime_stats, trans_counts, start_counts,
                  pg_stats=None, repr_version=None):
        """Absorb one batch summary under the store's semantics; returns totals."""
        s = _clone(regime_stats)
        C = _clone(trans_counts)
        v = _clone(start_counts)
        p = _clone(pg_stats)
        if (self.expected_ids is not None and self.mode in ("memoized", "full_batch")
                and batch_id not in self.expected_ids):
            raise ValueError(
                f"batch_id {batch_id!r} is not in the declared expected_ids; a fixed "
                "corpus rejects foreign partitions AT INGESTION, before touching the "
                "totals (round-7 review, issue 5)")
        if self.mode == "full_batch" and self._fb_finalized:
            raise ValueError(
                "full_batch pass is FINALIZED; call begin_full_batch_pass() to open a "
                "new pass before adding data (round-7 review, issue 6)")
        if self.mode == "legacy_ema":
            self._tot_stats = _ema(self._tot_stats, s, self.ema_tau)
            self._tot_C = _ema(self._tot_C, C, self.ema_tau)
            self._tot_s = _ema(self._tot_s, v, self.ema_tau)
            self._tot_pg = _ema(self._tot_pg, p, self.ema_tau)
        elif self.mode in ("full_batch", "streaming"):
            if repr_version is not None and self._pass_repr is not None \
                    and repr_version != self._pass_repr:
                if self.mode == "streaming":
                    raise ValueError(
                        f"streaming pass saw repr_version {repr_version} after absorbing "
                        f"data at version {self._pass_repr}; streamed data cannot be "
                        "recomputed, so the representation must be frozen for the "
                        "stream (or use memoized mode, which invalidates stale batches)")
                # full_batch: a representation change invalidates the open pass
                self.reset()
                self.pass_id += 1
            if self.mode == "streaming":
                if self._stream_api == "async":
                    raise ValueError(
                        "this streaming store already absorbed data via async_commit(); "
                        "mixing async commits and the serial cursor would double-count "
                        "(round-8 review, issue 2)")
                # O(1) MEMORY (round-5 review, issue 6): with integer stream offsets keep
                # only a high-water mark + count, not every id forever. Absorb-once is
                # enforced by strict monotonicity. Non-integer ids fall back to the set
                # UNLESS strict_stream, which forbids the fallback and requires contiguity.
                off = _stream_offset(batch_id)
                if self.strict_stream:
                    if off is None:
                        raise ValueError(
                            f"strict streaming requires a non-negative INTEGER offset with "
                            f"no id-set fallback; got {batch_id!r} (round-7 review, issue 4)")
                    exp = 0 if self._stream_hi is None else self._stream_hi + 1
                    if off != exp:
                        raise ValueError(
                            f"strict streaming requires CONTIGUOUS offsets to detect skipped "
                            f"data: expected {exp}, got {off} (round-7 review, issue 4)")
                if off is not None:
                    hi = getattr(self, "_stream_hi", None)
                    if hi is not None and off <= hi:
                        raise ValueError(
                            f"streaming saw stream offset {off} <= high-water mark {hi}: "
                            "each datum must be absorbed exactly once and in increasing "
                            "order (use a monotonic stream cursor).")
                    self._stream_hi = off
                    self._stream_count = getattr(self, "_stream_count", 0) + 1
                    self._stream_api = "serial"   # marker AFTER success (round-9)
                else:
                    if batch_id in self._per_batch:
                        raise ValueError(
                            f"streaming mode saw batch_id {batch_id!r} twice; each datum "
                            "must be absorbed exactly once (use memoized mode to revisit)")
                    self._per_batch[batch_id] = True
                    self._stream_api = "serial"   # marker AFTER success (round-9)
            else:
                if self.mode == "full_batch" and batch_id in self._per_batch:
                    # a repeated id means a new sweep over the same corpus: full-batch
                    # semantics accumulate within ONE pass, so start the pass fresh
                    self.reset()
                # OVERFLOW CHECKED BEFORE MUTATING (round-6 review, issue 6): a foreign
                # batch beyond the declared size must not touch the ledger or the totals,
                # so a caught exception leaves the pass exactly as it was.
                if (self.mode == "full_batch" and self.expected_batches is not None
                        and batch_id not in self._per_batch
                        and len(self._per_batch) >= self.expected_batches):
                    raise ValueError(
                        f"full_batch pass already holds {len(self._per_batch)} summaries "
                        f"(expected_batches={self.expected_batches}); refusing foreign "
                        f"batch {batch_id!r}. Call begin_full_batch_pass() at each epoch "
                        "boundary or reuse stable batch ids.")
                self._per_batch[batch_id] = True          # id set only (no summaries kept)
            self._pass_repr = repr_version if repr_version is not None else self._pass_repr
            self._tot_stats = _iadd(self._tot_stats, s)
            self._tot_C = _iadd(self._tot_C, C)
            self._tot_s = _iadd(self._tot_s, v)
            self._tot_pg = _iadd(self._tot_pg, p)
        else:                                          # memoized: replace stale summary
            if repr_version is not None:
                stale = [b for b, rv in self._batch_repr.items()
                         if rv is not None and rv != repr_version]
                for b in stale:
                    ss = self._per_batch.pop(b)
                    self._batch_repr.pop(b, None)
                    self._tot_stats = _iadd(self._tot_stats, ss[0], sign=-1.0)
                    self._tot_C = _iadd(self._tot_C, ss[1], sign=-1.0)
                    self._tot_s = _iadd(self._tot_s, ss[2], sign=-1.0)
                    if ss[3] is not None:
                        self._tot_pg = _iadd(self._tot_pg, ss[3], sign=-1.0)
                    self.n_stale_invalidated += 1
            old = self._per_batch.get(batch_id)
            if old is not None:
                self._tot_stats = _iadd(self._tot_stats, old[0], sign=-1.0)
                self._tot_C = _iadd(self._tot_C, old[1], sign=-1.0)
                self._tot_s = _iadd(self._tot_s, old[2], sign=-1.0)
                if old[3] is not None:
                    self._tot_pg = _iadd(self._tot_pg, old[3], sign=-1.0)
            self._per_batch[batch_id] = (s, C, v, p)
            self._batch_repr[batch_id] = repr_version
            self._tot_stats = _iadd(self._tot_stats, s)
            self._tot_C = _iadd(self._tot_C, C)
            self._tot_s = _iadd(self._tot_s, v)
            self._tot_pg = _iadd(self._tot_pg, p)
        self.n_updates += 1
        return self.totals()

    def remap(self, row_map):
        """Row-remap every cached summary and total after a structural move.

        `row_map` is a (K_new, K_old) mapping matrix: new row n receives the SUM of the
        old rows it selects (a merge row carries two 1s; fresh birth/split rows are
        all-zero and start with zero cached mass -- exactly bnpy's sufficient-statistic
        memory expansion for HMM merges/expansions). Regime stats, start counts and PG
        naturals remap the leading axis; transition counts remap both axes (M C M^T).
        Batches not yet revisited keep remapped summaries and are replaced wholesale on
        their next visit under memoized semantics, so totals stay exactly the sum of
        the per-batch ledger at all times.
        """
        if row_map is None:
            return
        M = row_map.detach()
        Ko = int(M.shape[1])

        def lead(t):
            if not torch.is_tensor(t) or t.dim() < 1 or t.shape[0] != Ko:
                return t
            return torch.einsum("nk,k...->n...", M.to(t.dtype).to(t.device), t)

        def both(C):
            if not torch.is_tensor(C):
                return C
            Md = M.to(C.dtype).to(C.device)
            return Md @ C @ Md.T

        def rec(o, kind):
            if o is None:
                return None
            if isinstance(o, dict):
                return {k: rec(v, kind) for k, v in o.items()}
            if isinstance(o, (tuple, list)):
                t = [rec(v, kind) for v in o]
                return tuple(t) if isinstance(o, tuple) else t
            return both(o) if kind == "C" else lead(o)

        self._tot_stats = rec(self._tot_stats, "lead")
        self._tot_C = rec(self._tot_C, "C")
        self._tot_s = rec(self._tot_s, "lead")
        self._tot_pg = rec(self._tot_pg, "lead")
        for bid, entry in list(self._per_batch.items()):
            if isinstance(entry, tuple) and len(entry) == 4:
                s, C, v, p = entry
                self._per_batch[bid] = (rec(s, "lead"), rec(C, "C"),
                                        rec(v, "lead"), rec(p, "lead"))

    def drop_batch(self, batch_id):
        """Remove a batch's contribution entirely (memoized mode)."""
        if self.mode != "memoized":
            raise ValueError("drop_batch is only meaningful in memoized mode")
        old = self._per_batch.pop(batch_id, None)
        self._batch_repr.pop(batch_id, None)
        if old is not None:
            self._tot_stats = _iadd(self._tot_stats, old[0], sign=-1.0)
            self._tot_C = _iadd(self._tot_C, old[1], sign=-1.0)
            self._tot_s = _iadd(self._tot_s, old[2], sign=-1.0)
            if old[3] is not None:
                self._tot_pg = _iadd(self._tot_pg, old[3], sign=-1.0)

    def totals(self):
        return self._tot_stats, self._tot_C, self._tot_s, self._tot_pg

    # ------------------------------------------------------------ serialization
    def async_commit(self, offset, delta, max_reorder: int = 4096):
        """SDA-Bayes MASTER accumulate primitive: xi_post <- xi_post + d_xi, applied
        ATOMICALLY and in ANY order. Every new total (including DICT-valued regime
        statistics and the PG naturals) is built into a temporary and recursively
        finiteness-checked BEFORE any is installed; the streaming-API marker is set only
        AFTER a successful install, so a failed first commit leaves the store exactly as
        it was (round-9 review, issue 3)."""
        if self.mode != "streaming":
            raise ValueError("async_commit requires online_mode='streaming'")
        if self._stream_api == "serial":
            raise ValueError(
                "this streaming store already absorbed data via add_batch(); mixing the "
                "serial cursor and async commits would double-count (round-8 review, issue 2)")
        for key in ("s", "C", "v"):
            if key not in delta or delta[key] is None:
                raise ValueError(f"async delta is missing required field {key!r}")
        off = _stream_offset(offset)
        if off is None:
            raise ValueError(f"async offset must be a non-negative integer, got {offset!r}")
        if off <= self._async_lo or off in self._async_ahead:
            raise ValueError(
                f"async offset {off} already committed (watermark {self._async_lo}); each "
                "datum is absorbed exactly once")
        if off - self._async_lo > max_reorder:
            raise ValueError(
                f"async offset {off} is {off - self._async_lo} beyond the contiguous "
                f"watermark {self._async_lo}; more than max_reorder={max_reorder} items "
                "appear skipped (supply the missing offsets or widen the contract)")
        try:
            new_stats = _iadd(_clone(self._tot_stats), delta["s"])
            new_C = _iadd(_clone(self._tot_C), delta["C"])
            new_s = _iadd(_clone(self._tot_s), delta["v"])
            new_pg = self._tot_pg
            if delta.get("p") is not None:
                new_pg = _iadd(_clone(self._tot_pg), delta["p"])
        except Exception as e:
            raise ValueError(f"async delta is malformed; master totals unchanged ({e})")
        # RECURSIVE finiteness over tensors, DICTS and tuples -- covers the dict-valued
        # regime statistics and the PG naturals, which the round-8 top-level
        # torch.is_tensor() check silently skipped (round-9 review, issue 3).
        for nm, t in (("s", new_stats), ("C", new_C), ("v", new_s), ("p", new_pg)):
            if not _all_finite(t):
                raise ValueError(
                    f"async delta made totals non-finite in field {nm!r}; master unchanged")
        self._tot_stats, self._tot_C, self._tot_s, self._tot_pg = new_stats, new_C, new_s, new_pg
        self._async_ahead.add(off)
        while (self._async_lo + 1) in self._async_ahead:
            self._async_lo += 1
            self._async_ahead.discard(self._async_lo)
        self._stream_count = getattr(self, "_stream_count", 0) + 1
        self.n_updates += 1
        self._stream_api = "async"   # marker set only AFTER a successful install

    def async_watermark(self):
        """Highest contiguous offset absorbed (all <= this are committed); -1 if none."""
        return self._async_lo

    def state_dict(self):
        """Checkpoint payload: mode, certificates, totals AND the per-batch ledger,
        so memoized replacement semantics survive a save/load round trip."""
        def cpu(o):
            if o is None:
                return None
            if isinstance(o, dict):
                return {k: cpu(v) for k, v in o.items()}
            if isinstance(o, (tuple, list)):
                t = [cpu(v) for v in o]
                return tuple(t) if isinstance(o, tuple) else t
            return o.detach().cpu() if torch.is_tensor(o) else o
        return dict(mode=self.mode, ema_tau=self.ema_tau,
                    expected_batches=self.expected_batches,
                    expected_ids=(sorted(self.expected_ids)
                                  if self.expected_ids is not None else None),
                    n_updates=self.n_updates,
                    pass_id=self.pass_id,
                    n_stale_invalidated=self.n_stale_invalidated,
                    batch_repr=dict(self._batch_repr),
                    pass_repr=self._pass_repr,
                    totals=cpu((self._tot_stats, self._tot_C, self._tot_s, self._tot_pg)),
                    per_batch={k: cpu(v) for k, v in self._per_batch.items()},
                    stream_hi=self._stream_hi, stream_count=self._stream_count,
                    fb_finalized=self._fb_finalized, strict_stream=self.strict_stream,
                    async_lo=self._async_lo, async_ahead=sorted(self._async_ahead),
                    stream_api=self._stream_api)

    def load_state_dict(self, sd, device=None):
        self.mode = sd["mode"]
        self.ema_tau = sd["ema_tau"]
        self.expected_batches = sd["expected_batches"]
        self.expected_ids = (set(sd["expected_ids"])
                             if sd.get("expected_ids") is not None else None)
        self.n_updates = int(sd["n_updates"])
        self.pass_id = int(sd.get("pass_id", 0))
        self.n_stale_invalidated = int(sd.get("n_stale_invalidated", 0))
        self._batch_repr = dict(sd.get("batch_repr", {}))
        self._pass_repr = sd.get("pass_repr")
        self._tot_stats, self._tot_C, self._tot_s, self._tot_pg = sd["totals"]
        self._per_batch = dict(sd["per_batch"])
        self._stream_hi = sd.get("stream_hi")
        self._stream_count = int(sd.get("stream_count", 0))
        self._fb_finalized = bool(sd.get("fb_finalized", False))
        self.strict_stream = bool(sd.get("strict_stream", False))
        self._async_lo = int(sd.get("async_lo", -1))
        self._async_ahead = set(sd.get("async_ahead", []))
        self._stream_api = sd.get("stream_api")
        if device is not None:
            def mv(o):
                if o is None:
                    return None
                if isinstance(o, dict):
                    return {k: mv(v) for k, v in o.items()}
                if isinstance(o, (tuple, list)):
                    t = [mv(v) for v in o]
                    return tuple(t) if isinstance(o, tuple) else t
                return o.to(device) if torch.is_tensor(o) else o
            self._tot_stats = mv(self._tot_stats)
            self._tot_C = mv(self._tot_C)
            self._tot_s = mv(self._tot_s)
            self._tot_pg = mv(self._tot_pg)
            self._per_batch = {k: mv(v) for k, v in self._per_batch.items()}
