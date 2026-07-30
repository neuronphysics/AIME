"""Multi-start structure search for the SHS-RSSM regime head.

Because `aggregate_bound` returns the exact variational bound of the SAME data at each
model's own K -- including the parameter-complexity terms, and with the proper Beta q(u)
that makes it a genuine bound rather than a pseudo-bound (Hughes NIPS'15 Sec. 3.2) --
bounds from runs at different K are directly comparable.  Selecting the best-bound run is
therefore principled Bayesian model selection, not a heuristic tie-break.

Usage
-----
    from shs_rssm.adaptive_fit import fit_adaptive

    def make_head(K):
        return RegimeHead(stoch=L, deter=H, K=K, ...)

    best = fit_adaptive(make_head, corpus, K0_grid=(8, 16, 30), laps=12)
    head = best.head
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch

from .init_data import init_from_random_blocks
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
    """Grow arms lead with births; prune arms lead with merge/delete."""
    kw = dict(base)
    if arm == "grow":
        kw.update(do_birth=True, do_split=True)
    else:                                   # "prune"
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
    """Run several structure searches and keep the one with the best bound.

    Small `K0` values are run as GROW arms (births enabled) and large ones as PRUNE arms
    (merge + delete), because the two reach different basins and the bound arbitrates.

    Parameters
    ----------
    head_factory : callable K -> RegimeHead
        Must build a *fresh* head at truncation K.  Do not reuse a fitted head.
    K0_grid : truncations to try.  A spread that brackets your expectation works best;
        (8, 16, 30) is a reasonable default when you expect O(10) regimes.
    seeds : repeats per K0.  More seeds is the cheapest variance reduction available.
    init : "randcontigblocks" applies the data-driven symmetry-breaking initialiser.
        With `identity_init=True` every regime starts identical and the first E-step is
        exactly symmetric in k, so without this the MAP path collapses to one state and
        results are bit-identical across seeds.  Pass "none" only if the head has already
        been initialised from a trained encoder.
    """
    sweep_kwargs = dict(sweep_kwargs or {})
    median_K0 = sorted(K0_grid)[len(K0_grid) // 2]
    runs, best = [], None

    for K0 in K0_grid:
        arm = "grow" if K0 < median_K0 else "prune"
        for seed in seeds:
            torch.manual_seed(int(seed))
            head = head_factory(int(K0))
            if init == "randcontigblocks":
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
    """True when the K trace has been flat for the last `tail` laps.

    Use this before quoting a state count.  In our MoCap6 runs K was still falling at
    lap 12 in every prune arm, so any number read off at that point overstates K.
    """
    tr = out.get("K_trace", [])
    return len(tr) >= tail and len(set(tr[-tail:])) == 1
