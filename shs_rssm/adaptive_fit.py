from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch

from .init_data import init_from_random_blocks, init_contig_blocks
from .offline_trainer import fit_offline_corpus


@dataclass
class AdaptiveResult:
    bound: float
    K: int
    head: Any
    K0: int
    seed: int
    arm: str
    out: dict
    all_runs: list = field(default_factory=list)


def _arm_kwargs(arm: str, base: dict) -> dict:
    kw = dict(base)
    if arm == "grow":
        kw.update(do_birth=True, do_split=True)
    else:
        kw.update(do_birth=False, do_split=True)
    return kw


def fit_adaptive(
    head_factory: Callable[[int], Any],
    corpus: Sequence,
    K0_grid: Sequence[int] = (8, 16, 30),
    seeds: Sequence[int] = (0,),
    laps: int = 12,
    sweep_every: int = 3,
    sweep_kwargs: dict | None = None,
    merge_topm: int | None = 5,
    init: str = "randcontigblocks",
    verbose: bool = True,
) -> AdaptiveResult:
    sweep_kwargs = dict(sweep_kwargs or {})
    median_K0 = sorted(K0_grid)[len(K0_grid) // 2]
    runs, best = [], None

    for K0 in K0_grid:
        arm = "grow" if K0 < median_K0 else "prune"
        for seed in seeds:
            torch.manual_seed(int(seed))
            head = head_factory(int(K0))
            if init in ("randcontigblocks", "contig_blocks"):
                try:
                    init_contig_blocks(head, corpus, int(K0), seed=int(seed))
                except ValueError as _e:
                    import warnings
                    warnings.warn(
                        f"init_contig_blocks infeasible for K0={K0} ({_e}); falling "
                        "back to init_from_random_blocks for this arm")
                    init_from_random_blocks(head, corpus, int(K0), seed=int(seed))
            elif init in ("random_blocks", "blocks"):
                init_from_random_blocks(head, corpus, int(K0), seed=int(seed))
            out = fit_offline_corpus(
                head, corpus=corpus, laps=laps, sweep_every=sweep_every,
                sweep_kwargs=_arm_kwargs(arm, sweep_kwargs),
                merge_topm=merge_topm, verbose=False,
            )
            b = float(out["bounds"][-1]) if out.get("bounds") else -math.inf
            rec = AdaptiveResult(bound=b, K=int(head.K), head=head, K0=int(K0),
                                 seed=int(seed), arm=arm, out=out)
            runs.append(rec)
            if verbose:
                tr = out.get("K_trace", [])
                still = (len(tr) > 3 and tr[-1] < tr[-4])
                print(f"  [{arm:>5}] K0={K0:>3} seed={seed}  K={head.K:>3}  "
                      f"bound={b:>10.1f}" + ("   (K still descending)" if still else ""),
                      flush=True)
            if best is None or b > best.bound:
                best = rec

    best.all_runs = runs
    if verbose:
        print(f"  -> selected K={best.K} from the {best.arm} arm at K0={best.K0} "
              f"(bound {best.bound:.1f})")
        tr = best.out.get("K_trace", [])
        if len(tr) > 3 and tr[-1] < tr[-4]:
            print("  !  K was STILL DESCENDING at the last lap: this K is an upper "
                  "bound, not a converged estimate. Increase `laps`.")
    return best


def k_trace_converged(out: dict, tail: int = 4) -> bool:
    tr = out.get("K_trace", [])
    return len(tr) >= tail and len(set(tr[-tail:])) == 1
