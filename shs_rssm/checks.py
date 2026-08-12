import os
import numpy as np
import torch

_MODE = os.environ.get("SHS_CHECKS", "raise")


class CheckFailure(AssertionError):
    pass


def _fail(msg):
    if _MODE == "off":
        return False
    if _MODE == "warn":
        print(f"[shs_rssm.checks] WARNING: {msg}")
        return False
    raise CheckFailure(msg)


def verify_bound_tracking(head, buffer, accepted_bound, tag="move",
                          rtol=1e-6, aggregate_fn=None):
    if _MODE == "off":
        return True
    if aggregate_fn is None:
        from .moves import aggregate_bound as aggregate_fn
    got = float(aggregate_fn(head, buffer))
    exp = float(accepted_bound)
    tol = rtol * max(1.0, abs(exp))
    if not np.isfinite(got):
        return _fail(f"[{tag}] recomputed bound is not finite: {got}")
    if abs(got - exp) > tol:
        return _fail(
            f"[{tag}] bound tracking mismatch: accepted={exp:.6f} "
            f"recomputed={got:.6f} diff={got - exp:+.3e} tol={tol:.3e}. "
            f"The install did not reproduce the scored candidate -- suspect "
            f"_apply's row_map / statistic remap, not the acceptance rule.")
    return True


def check_occupancy_vs_beta(head, gamma, max_ratio=1.25, min_occ_frac=0.005,
                            tag="lap", hard=False):
    if _MODE == "off":
        return True
    from .sticky_hdp import rho2beta
    g = gamma.detach() if torch.is_tensor(gamma) else torch.as_tensor(gamma)
    occ = g.reshape(-1, g.shape[-1]).sum(0)
    frac = (occ / occ.sum().clamp_min(1e-12)).cpu().numpy()
    n_occ = int((frac > min_occ_frac).sum())

    beta = rho2beta(head.hdp.rho).detach().cpu().numpy()
    beta = beta[:head.K] if len(beta) > head.K else beta
    beta = beta / max(beta.sum(), 1e-12)
    eff = 1.0 / max(float((beta ** 2).sum()), 1e-12)

    ratio = eff / max(n_occ, 1)
    ok = ratio <= max_ratio
    if not ok:
        msg = (f"[{tag}] allocation/E-step disagree: beta supports {eff:.1f} states "
               f"but only {n_occ} are occupied (ratio {ratio:.2f} > {max_ratio}). "
               f"The HDP is holding room the responsibilities are not using.")
        if hard:
            return _fail(msg)
        print(f"[shs_rssm.checks] WARNING: {msg}")
    return ok


def check_merge_stats(C, S, i, j, C_merged=None, S_merged=None, atol=1e-8):
    C = np.asarray(C, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    K = C.shape[0]
    if not (0 <= i < K and 0 <= j < K and i != j):
        raise ValueError(f"bad merge pair ({i}, {j}) for K={K}")

    keep = [k for k in range(K) if k != j]
    Cm = C.copy()
    Cm[:, i] += Cm[:, j]
    Cm[i, :] += Cm[j, :]
    Cm[i, i] = C[i, i] + C[j, j] + C[j, i] + C[i, j]
    Cm = Cm[np.ix_(keep, keep)]

    Sm = S.copy()
    Sm[i] = S[i] + S[j]
    Sm = Sm[keep]

    total_before, total_after = C.sum(), Cm.sum()
    if abs(total_before - total_after) > atol * max(1.0, abs(total_before)):
        _fail(f"merge lost transition mass: {total_before:.6f} -> {total_after:.6f}")

    if C_merged is not None:
        d = np.abs(np.asarray(C_merged, dtype=np.float64) - Cm).max()
        if d > atol:
            diag = abs(float(np.asarray(C_merged)[min(i, K - 2), min(i, K - 2)])
                       - Cm[min(i, K - 2), min(i, K - 2)])
            _fail(f"merged transition matrix mismatch: max|diff|={d:.3e} "
                  f"(self-count diff={diag:.3e} -- check all FOUR terms)")
    if S_merged is not None:
        d = np.abs(np.asarray(S_merged, dtype=np.float64) - Sm).max()
        if d > atol:
            _fail(f"merged data statistics mismatch: max|diff|={d:.3e}")
    return Cm, Sm


def check_beta_merge(beta, i, j, beta_merged, atol=1e-8):
    beta = np.asarray(beta, dtype=np.float64)
    bm = np.asarray(beta_merged, dtype=np.float64)
    if abs(beta.sum() - bm.sum()) > atol * max(1.0, abs(beta.sum())):
        _fail(f"beta mass not conserved under merge: {beta.sum():.8f} -> {bm.sum():.8f}")
    keep = [k for k in range(len(beta)) if k != j]
    ref = beta.copy(); ref[i] = beta[i] + beta[j]; ref = ref[keep]
    if len(bm) == len(ref) and np.abs(bm - ref).max() > atol:
        _fail(f"beta merge mismatch: max|diff|={np.abs(bm - ref).max():.3e}")
    return ref


class MoveLog:

    def __init__(self):
        self.touched = {}

    def reset(self):
        self.touched = {}

    def claim(self, move, uids):
        uids = set(int(u) for u in np.atleast_1d(uids))
        clash = {u: self.touched[u] for u in uids if u in self.touched}
        if clash:
            _fail(f"[{move}] would touch states already changed this lap by "
                  f"{clash}; skip them or defer to the next lap")
            return False
        for u in uids:
            self.touched[u] = move
        return True

    def available(self, uids):
        return [int(u) for u in np.atleast_1d(uids) if int(u) not in self.touched]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    K = 6
    C = rng.random((K, K)) * 10
    S = rng.random((K, 3)) * 10
    Cm, Sm = check_merge_stats(C, S, 1, 4)
    print("merge surgery: reference computed, mass conserved  OK")
    assert Cm.shape == (K - 1, K - 1) and Sm.shape == (K - 1, 3)
    assert abs(Cm[1, 1] - (C[1, 1] + C[4, 4] + C[4, 1] + C[1, 4])) < 1e-12
    print("  four-term self-count verified")

    bad = Cm.copy(); bad[1, 1] = C[1, 1] + C[4, 4]
    try:
        check_merge_stats(C, S, 1, 4, C_merged=bad)
        print("  FAILED to catch two-term self-count")
    except CheckFailure as e:
        print(f"  caught two-term self-count error  OK")

    b = rng.random(K); b /= b.sum()
    check_beta_merge(b, 1, 4, np.delete(np.where(np.arange(K) == 1, b[1] + b[4], b), 4))
    print("beta merge: conservation verified  OK")

    log = MoveLog()
    assert log.claim("merge", [1, 2])
    try:
        log.claim("delete", [2, 3]); print("  FAILED to catch interlock")
    except CheckFailure:
        print("interlock: caught overlapping claim  OK")
    print(f"  available after merge claim: {log.available([1, 2, 3, 4])}")
