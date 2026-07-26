"""Exactly-once completed-episode ingestion for persistent streaming SHS-VB.

This is the queue DATA STRUCTURE and its contract, fully unit-tested. Wiring `simulate()`
to call `push` on real episode-completion events is Dreamer-loop integration; the trigger
is provided (see dreamer.py) but not executed in the test environment.
"""
from __future__ import annotations


class CompletedEpisodeQueue:
    def __init__(self):
        self._next_id = 0          # id to assign to the next pushed episode
        self._consumed = -1        # highest id drained (checkpoint watermark)
        self._pending = []         # list of (id, payload, replay_key) pushed, not drained
        self._reserved = []        # open transaction (review P0 #5), empty when none

    @property
    def next_id(self):
        return self._next_id

    @property
    def consumed_watermark(self):
        return self._consumed

    def __len__(self):
        return len(self._pending)

    def push(self, payload, is_complete: bool = True, replay_key=None):
        """Enqueue one COMPLETED episode; returns its monotonic id. `replay_key` is a
        DURABLE identifier (e.g. the replay filename/index) by which the payload can be
        re-fetched after a checkpoint resume (review P0 #2). A non-complete episode is
        rejected."""
        if not is_complete:
            raise ValueError("CompletedEpisodeQueue only accepts COMPLETED episodes "
                             "(an active/incomplete episode must not be ingested)")
        eid = self._next_id
        self._pending.append((eid, payload, replay_key))
        self._next_id += 1
        return eid

    def drain(self, max_n: int | None = None):
        """Non-transactional drain (kept for back-compat): remove and return pending
        episodes in id order, advancing the watermark. Prefer reserve/commit for the
        transactional path."""
        n = len(self._pending) if max_n is None else max(0, min(int(max_n), len(self._pending)))
        out = self._pending[:n]
        self._pending = self._pending[n:]
        if out:
            self._consumed = out[-1][0]
        return [(e[0], e[1]) for e in out]

    # ---- transactional protocol (review P0 #5): reserve -> compute -> commit | abort ----
    def reserve(self, max_n: int | None = None):
        """PEEK the next pending episodes WITHOUT advancing the watermark or removing
        them. Returns [(id, payload, replay_key)]. A crash/failure before commit leaves
        the queue exactly as it was, so nothing is ever lost or double-counted."""
        if self._reserved:
            raise RuntimeError("a reservation is already open; commit or abort it first")
        n = len(self._pending) if max_n is None else max(0, min(int(max_n), len(self._pending)))
        self._reserved = list(self._pending[:n])
        return [(e[0], e[1], e[2] if len(e) > 2 else None) for e in self._reserved]

    def commit(self):
        """ATOMICALLY consume the reserved episodes: remove them and advance the
        watermark. Call ONLY after the ingestion + global update succeeded."""
        if not self._reserved:
            return 0
        n = len(self._reserved)
        self._pending = self._pending[n:]
        self._consumed = self._reserved[-1][0]
        self._reserved = []
        return n

    def abort(self):
        """Roll back an open reservation: the episodes stay pending, the watermark does
        NOT move. Call on any failure during compute/validate/commit."""
        self._reserved = []

    def state_dict(self):
        # payloads are NOT serialised (they live in replay); pending ids + DURABLE replay
        # keys let a resume re-fetch exactly the not-yet-ingested episodes (review P0 #2).
        # An open reservation is intentionally dropped (rolled back) across a checkpoint.
        return dict(next_id=int(self._next_id), consumed=int(self._consumed),
                    pending_ids=[int(e[0]) for e in self._pending],
                    pending_keys=[(e[2] if len(e) > 2 else None) for e in self._pending])

    def load_state_dict(self, sd):
        self._next_id = int(sd.get("next_id", 0))
        self._consumed = int(sd.get("consumed", -1))
        # pending payloads are re-supplied by the caller (from replay, keyed by id); we keep
        # only the ids so the caller knows which episodes still need ingestion.
        _ids = sd.get("pending_ids", [])
        _keys = sd.get("pending_keys", [None] * len(_ids))
        self._pending = [(int(i), None, k) for i, k in zip(_ids, _keys)]
        self._reserved = []
        return self
