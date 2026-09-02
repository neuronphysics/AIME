from __future__ import annotations


class CompletedEpisodeQueue:
    def __init__(self):
        self._next_id = 0
        self._consumed = -1
        self._pending = []
        self._reserved = []

    @property
    def next_id(self):
        return self._next_id

    @property
    def consumed_watermark(self):
        return self._consumed

    def __len__(self):
        return len(self._pending)

    def push(self, payload, is_complete: bool = True, replay_key=None):
        if not is_complete:
            raise ValueError("CompletedEpisodeQueue only accepts COMPLETED episodes "
                             "(an active/incomplete episode must not be ingested)")
        eid = self._next_id
        self._pending.append((eid, payload, replay_key))
        self._next_id += 1
        return eid

    def drain(self, max_n: int | None = None):
        n = len(self._pending) if max_n is None else max(0, min(int(max_n), len(self._pending)))
        out = self._pending[:n]
        self._pending = self._pending[n:]
        if out:
            self._consumed = out[-1][0]
        return [(e[0], e[1]) for e in out]

    def reserve(self, max_n: int | None = None):
        if self._reserved:
            raise RuntimeError("a reservation is already open; commit or abort it first")
        n = len(self._pending) if max_n is None else max(0, min(int(max_n), len(self._pending)))
        self._reserved = list(self._pending[:n])
        return [(e[0], e[1], e[2] if len(e) > 2 else None) for e in self._reserved]

    def commit(self):
        if not self._reserved:
            return 0
        n = len(self._reserved)
        self._pending = self._pending[n:]
        self._consumed = self._reserved[-1][0]
        self._reserved = []
        return n

    def abort(self):
        self._reserved = []

    def state_dict(self):
        return dict(next_id=int(self._next_id), consumed=int(self._consumed),
                    pending_ids=[int(e[0]) for e in self._pending],
                    pending_keys=[(e[2] if len(e) > 2 else None) for e in self._pending])

    def load_state_dict(self, sd):
        self._next_id = int(sd.get("next_id", 0))
        self._consumed = int(sd.get("consumed", -1))
        _ids = sd.get("pending_ids", [])
        _keys = sd.get("pending_keys", [None] * len(_ids))
        self._pending = [(int(i), None, k) for i, k in zip(_ids, _keys)]
        self._reserved = []
        return self
