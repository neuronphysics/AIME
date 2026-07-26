"""Offline fixed-corpus VB trainer (round-4 review, issue 6).

The runnable end-to-end loop for the guarantees this package certifies: a FIXED corpus
with STABLE batch ids, memoized laps under frozen globals-per-visit semantics, optional
per-lap re-encoding of a learning representation (repr-version bump + automatic
stale-batch invalidation), and complete-corpus consolidation sweeps whose acceptance is
the exact whole-corpus bound. This is exactly the loop of the Lorenz study, packaged.

Usage sketch::

    head = RegimeHead(..., online_mode="memoized", expected_batches=len(corpus))
    out = fit_offline_corpus(head, corpus, laps=20, sweep_every=3)

where ``corpus`` is a list of ``(batch_id, stoch, deter, is_first)`` tuples (optionally
``(..., z_var)`` for the analytic E-step), or pass ``encode_fn`` returning such a list
to re-encode a *learning* representation each lap: the trainer bumps the head's
representation version so the memoized ledger invalidates stale summaries instead of
silently mixing encoder states.
"""
import torch

from .moves import MoveBuffer, sweep_moves, aggregate_bound


@torch.no_grad()
def fit_offline_corpus(head, corpus=None, laps: int = 10, encode_fn=None,
                       sweep_every: int = 0, sweep_kwargs: dict | None = None,
                       verbose: bool = False, merge_topm=None):
    """Run memoized VB laps over a fixed, stable-id corpus; return per-lap diagnostics.

    Args:
        head: a RegimeHead constructed with ``online_mode="memoized"`` and
            ``expected_batches`` equal to the corpus size.
        corpus: list of ``(batch_id, stoch, deter, is_first[, z_var])``. Ignored when
            ``encode_fn`` is given.
        laps: number of full passes over the corpus.
        encode_fn: optional callable returning a fresh corpus list each lap (a learning
            encoder). Each call after the first bumps the representation version.
        sweep_every: if > 0, run a complete-corpus move sweep after every N-th lap.
        sweep_kwargs: forwarded to :func:`sweep_moves` (defaults: exact acceptance at
            threshold 0, no creation bonus, exhaustive confirmation).

    Returns:
        dict with ``bounds`` (per-lap whole-corpus profile bound), ``K_trace``,
        ``ledger_ids`` and ``move_log`` (per-sweep accepted flags and exact gains).
    """
    if getattr(head, "stat_store", None) is None or head.stat_store.mode != "memoized":
        raise ValueError(
            "fit_offline_corpus requires online_mode='memoized' with a declared "
            "expected_batches: the fixed-corpus contract IS the memoized contract "
            "(stable ids, replace-on-revisit, complete-corpus certificates).")
    if corpus is None and encode_fn is None:
        raise ValueError("provide a corpus or an encode_fn")
    if head.stat_store.expected_batches is None:
        raise ValueError("fit_offline_corpus requires a DECLARED expected_batches "
                         "(the fixed-corpus size); got None (round-6 review, issue 7)")

    exp = head.stat_store.expected_batches
    if corpus is not None and exp is not None and exp != len(corpus):
        raise ValueError(
            f"expected_batches ({exp}) must equal len(corpus) ({len(corpus)}): the "
            "memoized completeness certificate IS the corpus size (round-5 review, issue 4).")

    def _get_corpus(first):
        if encode_fn is None:
            return corpus
        fresh = encode_fn()
        if not first:
            head.bump_repr_version()   # a re-encode is a NEW representation version
        return fresh

    first_ids, bounds, k_trace, move_log = None, [], [], []
    for lap in range(int(laps)):
        batches = _get_corpus(first=(lap == 0))
        ids = [b[0] for b in batches]
        if len(batches) != head.stat_store.expected_batches:
            raise ValueError(
                f"lap {lap}: encode_fn/corpus produced {len(batches)} batches but "
                f"expected_batches={head.stat_store.expected_batches} (round-6 issue 7)")
        if len(set(ids)) != len(ids):
            raise ValueError(f"corpus batch ids must be unique, got {ids}")
        if first_ids is None:
            first_ids = set(ids)
        elif set(ids) != first_ids:
            raise ValueError(
                f"corpus id set changed across laps ({sorted(first_ids)} -> "
                f"{sorted(set(ids))}): a fixed corpus must present the SAME stable ids "
                "every lap, else stale summaries linger in the ledger (round-5 review, issue 4).")
        # Clean full pass (round-5 review, issue 4): RESET the ledger, accumulate every
        # batch with globals FROZEN (stats_only => ledger-only writes), then ONE global
        # step from the complete totals -- correct for both a fixed and a re-encoding
        # representation, and a re-encode never fires corpus-PREFIX updates.
        head.stat_store.reset()
        buf = MoveBuffer(max_batches=len(batches), complete=True, expected_ids=ids)
        for entry in batches:
            bid, z, h, isf = entry[0], entry[1], entry[2], entry[3]
            zv = entry[4] if len(entry) > 4 else None
            gamma, counts, sc, _ = head.regime_inference(
                z, h, is_first=isf, z_var=zv, cache_estep=True)
            head.update_globals(z, h, gamma, counts, sc, is_first=isf,
                                z_var=zv, batch_id=bid, stats_only=True)
            buf.add(z, h, isf, zv, batch_id=bid,
                    repr_version=int(head.repr_version))
        head.global_step_from_totals()             # one global update from complete totals
        if sweep_every and (lap + 1) % int(sweep_every) == 0:
            kw = dict(threshold=0.0, create_bonus=0.0, refine_iters=2,
                      confirm_top=None)
            kw.update(sweep_kwargs or {})
            log = sweep_moves(head, buffer=buf, **kw,
                          merge_topm=merge_topm)
            move_log.append({m: (bool(a), float(g)) for m, (a, g) in log.items()})
        # bound and K recorded TOGETHER, both reflecting the SAME post-move state
        bounds.append(float(aggregate_bound(head, buf)))
        k_trace.append(int(head.K))
        if verbose:
            print(f"[offline] lap {lap}: bound {bounds[-1]:.2f} K={head.K}")
    return dict(bounds=bounds, K_trace=k_trace,
                ledger_ids=sorted(map(str, head.stat_store._per_batch)),
                move_log=move_log)
