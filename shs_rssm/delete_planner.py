from collections import defaultdict
import numpy as np

__all__ = ["per_sequence_occupancy", "target_set", "FailureRecords",
           "plan_delete", "DeletePlan"]


def per_sequence_occupancy(gamma, valid=None):
    g = np.asarray(gamma, dtype=np.float64)
    if g.ndim != 3:
        raise ValueError(f"gamma must be (N,T,K), got {g.shape}")
    if valid is not None:
        g = g * np.asarray(valid, dtype=np.float64)[..., None]
    return g.sum(axis=1)


def target_set(u, j, min_count=0.01):
    return set(np.flatnonzero(np.asarray(u)[:, j] > float(min_count)).tolist())


class FailureRecords:

    def __init__(self, min_perc_change=0.01, n_lap_to_reactivate=5, fail_limit=5):
        self.rec = defaultdict(lambda: dict(n_fail=0, n_fail_recent=0,
                                            latest_count=0.0, latest_lap=-np.inf))
        self.min_perc_change = float(min_perc_change)
        self.n_lap_to_reactivate = float(n_lap_to_reactivate)
        self.fail_limit = int(fail_limit)

    def record_attempt(self, uid, count, lap):
        r = self.rec[uid]
        r["latest_count"] = float(count)
        r["latest_lap"] = float(lap)

    def record_fail(self, uid, count, lap):
        r = self.rec[uid]
        r["n_fail"] += 1
        r["n_fail_recent"] += 1
        r["latest_count"] = float(count)
        r["latest_lap"] = float(lap)

    def record_success(self, uid):
        self.rec.pop(uid, None)

    def is_blocked(self, uid, size, lap):
        if uid not in self.rec:
            return False
        r = self.rec[uid]
        if r["n_fail_recent"] <= 0 or r["latest_count"] <= 0:
            return False
        if r["n_fail"] >= self.fail_limit:
            old = r["latest_count"]
            return abs(size - old) / (1e-100 + abs(old)) <= self.min_perc_change
        old = r["latest_count"]
        if abs(size - old) / (1e-100 + abs(old)) > self.min_perc_change:
            return False
        if self.n_lap_to_reactivate > 0 and (lap - r["latest_lap"]) > self.n_lap_to_reactivate:
            return False
        return True


class DeletePlan:
    def __init__(self, target_uids, absorbing_uids, target_seqs, reason=""):
        self.target_uids = list(target_uids)
        self.absorbing_uids = set(absorbing_uids)
        self.target_seqs = set(target_seqs)
        self.reason = reason

    def __repr__(self):
        return (f"DeletePlan(targets={self.target_uids}, "
                f"n_absorbing={len(self.absorbing_uids)}, "
                f"n_target_seqs={len(self.target_seqs)})")


def plan_delete(u, uids=None, lap=0, records=None, era="2015",
                min_count=0.01, max_target_seqs=10,
                max_atoms=50000.0, min_active=2, busy_uids=()):
    u = np.asarray(u, dtype=np.float64)
    N, K = u.shape
    uids = list(range(K)) if uids is None else list(uids)
    if len(uids) != K:
        raise ValueError(f"uids has {len(uids)} entries, u has {K} columns")
    records = records or FailureRecords()
    busy = set(busy_uids)
    counts = u.sum(axis=0)

    if K < min_active + 1:
        return None

    eligible, too_big, blocked = [], [], []
    for k, uid in enumerate(uids):
        if uid in busy:
            blocked.append(uid); continue
        if counts[k] <= float(min_count):
            too_big.append(uid); continue
        if era == "2015":
            if len(target_set(u, k, min_count)) > max_target_seqs:
                too_big.append(uid); continue
        else:
            if counts[k] > max_atoms:
                too_big.append(uid); continue
        if records.is_blocked(uid, counts[k], lap):
            blocked.append(uid); continue
        eligible.append((uid, k))

    if not eligible:
        return None
    if K - 1 < min_active:
        return None

    if era == "modern":
        uid, k = max(eligible, key=lambda t: counts[t[1]])
        absorbing = set(uids) - {uid}
        return DeletePlan([uid], absorbing, target_set(u, k, min_count),
                          reason="modern: single largest eligible")

    eligible.sort(key=lambda t: (len(target_set(u, t[1], min_count)), -counts[t[1]]))
    chosen, union = [], set()
    for uid, k in eligible:
        ts = target_set(u, k, min_count)
        if len(union | ts) > max_target_seqs:
            continue
        if len(chosen) + min_active > K:
            break
        chosen.append(uid); union |= ts
    if not chosen:
        return None
    plan = DeletePlan(chosen, set(uids) - set(chosen), union,
                      reason="2015: union-budget group")
    plan.empty_uids = [uids[k] for k in range(K) if counts[k] <= float(min_count)]
    return plan


if __name__ == "__main__":
    ok = 0

    def check(name, cond):
        global ok
        print(("  PASS  " if cond else "  FAIL  ") + name)
        ok += bool(cond)

    print("per-sequence occupancy")
    g = np.zeros((3, 10, 4)); g[0, :5, 0] = 1; g[0, 5:, 1] = 1
    g[1, :, 2] = 1; g[2, :3, 0] = 1; g[2, 3:, 3] = 1
    u = per_sequence_occupancy(g)
    check("u[0,0]==5, u[1,2]==10", u[0, 0] == 5 and u[1, 2] == 10)
    check("state 0 used by seqs {0,2}", target_set(u, 0) == {0, 2})
    check("state 2 used by seq {1}", target_set(u, 2) == {1})

    print("threshold boundary (paper uses >, snapshot uses >=)")
    ub = np.array([[0.009], [0.010], [0.011]])
    check("0.009 excluded", 0 not in target_set(ub, 0, 0.01))
    check("0.010 excluded under strict >", 1 not in target_set(ub, 0, 0.01))
    check("0.011 included", 2 in target_set(ub, 0, 0.01))

    print("budget: state used weakly in 11 sequences is ineligible at budget 10")
    u11 = np.zeros((12, 3)); u11[:11, 0] = 0.5; u11[:, 1] = 5.0; u11[:2, 2] = 5.0
    p = plan_delete(u11, max_target_seqs=10)
    check("state 0 not targeted", p is None or 0 not in p.target_uids)

    print("union budget: two states each fit, union does not")
    uu = np.zeros((12, 3))
    uu[0:6, 0] = 1.0; uu[6:12, 1] = 1.0; uu[0:1, 2] = 1.0
    p = plan_delete(uu, max_target_seqs=8)
    check("not both 0 and 1 chosen", p is not None and not ({0, 1} <= set(p.target_uids)))
    check("union within budget", p is not None and len(p.target_seqs) <= 8)

    print("no candidate fits -> None, no fallback")
    ubig = np.ones((30, 3)) * 5.0
    check("returns None", plan_delete(ubig, max_target_seqs=10) is None)

    print("never delete the last state")
    check("K=1 returns None", plan_delete(np.ones((3, 1))) is None)

    print("failure record: blocked then reactivated by mass change")
    r = FailureRecords(min_perc_change=0.01, n_lap_to_reactivate=5, fail_limit=5)
    r.record_fail("a", count=100.0, lap=1.0)
    check("blocked at same mass", r.is_blocked("a", 100.0, lap=2.0))
    check("reactivated by +20% mass", not r.is_blocked("a", 120.0, lap=2.0))
    check("still blocked at lap 4", r.is_blocked("a", 100.0, lap=4.0))
    check("reactivated at lap 7 (>5 laps)", not r.is_blocked("a", 100.0, lap=7.0))

    print("failure limit: laps no longer reactivate after the limit")
    r2 = FailureRecords(fail_limit=2)
    for lap in (1.0, 2.0):
        r2.record_fail("b", count=100.0, lap=lap)
    check("blocked at lap 99 despite elapsed laps", r2.is_blocked("b", 100.0, lap=99.0))
    check("mass change still reactivates", not r2.is_blocked("b", 150.0, lap=99.0))

    print("busy states excluded (cross-move interlock)")
    u2 = np.zeros((4, 3)); u2[:2, 0] = 1; u2[:2, 1] = 1; u2[:2, 2] = 1
    p = plan_delete(u2, busy_uids={0, 1})
    check("busy uids not targeted", p is None or not ({0, 1} & set(p.target_uids)))

    print("empty states are not delete targets")
    ue = np.zeros((4, 5)); ue[:, 0] = 2.0; ue[:, 1] = 3.0
    p = plan_delete(ue, max_target_seqs=10)
    check("empties excluded from targets", p is not None and set(p.target_uids) <= {0, 1})
    check("empties reported separately", p is not None and set(p.empty_uids) == {2, 3, 4})
    check("at least 2 states survive", p is not None and len(p.target_uids) <= 5 - 2)

    print("modern era: single largest eligible target")
    um = np.zeros((3, 4)); um[:, 0] = 1.0; um[:, 1] = 9.0; um[:, 2] = 4.0; um[:, 3] = 2.0
    p = plan_delete(um, era="modern")
    check("one target", p is not None and len(p.target_uids) == 1)
    check("largest count chosen", p is not None and p.target_uids == [1])
    check("all others absorb", p is not None and p.absorbing_uids == {0, 2, 3})

    print(f"\n{ok} checks passed")
