from __future__ import annotations
import threading
import time

import torch


def sda_train_threaded(head, batches, *, n_workers: int = 4, refresh_every: int = 5,
                       snap_every: int = 4, local_iters: int = 1, is_first=None,
                       tolerate_stale: bool = True, max_lag: int = 8,
                       window: int | None = None, start_offset: int = 0,
                       commit_max_stale: int | None = None):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    n = len(batches)
    tl = threading.local()
    task_lock = threading.Lock()
    per_worker_tasks = {}

    def _replica():
        if not hasattr(tl, "worker"):
            tl.worker = head.make_worker()
            tl.done = 0
            with task_lock:
                per_worker_tasks[threading.get_ident()] = 0
        return tl.worker

    def compute(off):
        w = _replica()
        lag = int(getattr(head, "_async_version", 0)) - int(w._snap_meta["version"])
        if lag > max_lag or (tl.done > 0 and tl.done % snap_every == 0):
            w.load_snapshot(head.master_snapshot())
        tl.done += 1
        with task_lock:
            per_worker_tasks[threading.get_ident()] += 1
        b = batches[off - start_offset]
        if isinstance(b, dict):
            d = head.async_worker_delta(
                b["stoch"], b["deter"],
                is_first=b.get("is_first", is_first), z_var=b.get("z_var"),
                action=b.get("action"), valid=b.get("valid"),
                worker=w, local_iters=local_iters,
                data_repr_version=int(head.repr_version))
        else:
            z, h = b
            d = head.async_worker_delta(z, h, is_first=is_first, worker=w,
                                        local_iters=local_iters,
                                        data_repr_version=int(head.repr_version))
        return off, d

    if window is None:
        window = 2 * n_workers
    from concurrent.futures import FIRST_COMPLETED, wait
    staleness, committed = [], 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        pending, next_off = set(), start_offset
        end_off = start_offset + n
        while next_off < end_off or pending:
            while next_off < end_off and len(pending) < window:
                pending.add(ex.submit(compute, next_off))
                next_off += 1
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                off, d = fut.result()
                staleness.append(int(getattr(head, "_async_version", 0))
                                 - d["base_version"])
                committed += 1
                head.async_commit(off, d,
                                  refresh=(committed % refresh_every == 0
                                           or committed == n),
                                  tolerate_stale=tolerate_stale,
                                  max_stale=commit_max_stale)
    wall = time.time() - t0
    return dict(wall=wall, commits=committed, max_lag=max_lag,
                version=int(getattr(head, "_async_version", 0)),
                watermark=int(head.stat_store.async_watermark()),
                staleness_mean=(sum(staleness) / max(1, len(staleness))),
                staleness_max=(max(staleness) if staleness else 0),
                worker_tasks=sorted(per_worker_tasks.values()))
