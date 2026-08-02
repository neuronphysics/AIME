"""Meta-World task tiers and the suite recommended for this repo's comparison.

Tier source: Seo et al., "Masked World Models for Visual Control" (2022), which
partitions the 50 tasks into easy / medium / hard / very hard and trains them
for 500K / 1M / 2M / 3M env steps respectively at action repeat 2. Most later
Meta-World papers report against this partition, so using it keeps the numbers
legible to reviewers.

Provenance caveat, stated because it matters if you report per-tier means:
EASY (28) and MEDIUM (11) below are the published lists verbatim. The remaining
11 tasks split 6 hard / 5 very hard, and only 7 of those 11 assignments are
confirmed from secondary sources (MoDem's task table, which selects from
medium+hard+very-hard). The 4 in ``HARD_TIER_UNCONFIRMED`` are definitely in
the top 11 but their exact hard-vs-very-hard bucket is not verified here. Do
not publish a hard-vs-very-hard breakdown without checking Seo et al. Appendix
F directly. Aggregating them as one "hard tier (11)" is safe and is what
``TIERS`` does by default.
"""

EASY = [
    "button-press", "button-press-topdown", "button-press-topdown-wall",
    "button-press-wall", "coffee-button", "dial-turn", "door-close",
    "door-lock", "door-open", "door-unlock", "drawer-close", "drawer-open",
    "faucet-close", "faucet-open", "handle-press", "handle-press-side",
    "handle-pull", "handle-pull-side", "lever-pull", "peg-unplug-side",
    "plate-slide", "plate-slide-back", "plate-slide-back-side",
    "plate-slide-side", "reach", "reach-wall", "window-close", "window-open",
]

MEDIUM = [
    "basketball", "bin-picking", "box-close", "coffee-pull", "coffee-push",
    "hammer", "peg-insert-side", "push-wall", "soccer", "sweep", "sweep-into",
]

# Confirmed hard.
HARD = ["assembly", "hand-insert", "pick-place", "push"]
# Confirmed very hard.
VERY_HARD = ["pick-place-wall", "stick-pull", "stick-push"]
# In the top-11 tier, exact bucket unverified (see module docstring).
HARD_TIER_UNCONFIRMED = [
    "disassemble", "pick-out-of-hole", "push-back", "shelf-place",
]

HARD_TIER = HARD + VERY_HARD + HARD_TIER_UNCONFIRMED  # 11 tasks

TIERS = {"easy": EASY, "medium": MEDIUM, "hard": HARD_TIER}

# Per-tier step budgets from Seo et al. (env steps, action repeat 2).
# The hard tier merges their 2M and 3M buckets; 2M is the conservative choice
# and is what SUITE_15 below assumes.
TIER_STEPS = {"easy": 500_000, "medium": 1_000_000, "hard": 2_000_000}

# ---------------------------------------------------------------------------
# Recommended suite: 15 tasks, ~5 per tier, chosen to span difficulty rather
# than to flatter the method. Deliberately NOT cherry-picked -- fixing the task
# list before you see any results is the single cheapest thing you can do to
# make the comparison credible, and the first thing a reviewer will ask about.
#
# The easy five are near-ceiling for most agents and exist to catch regressions,
# not to generate a win. The hard five are where a switching dynamics prior has
# room to actually help: each has a clear phase structure (approach / contact /
# grasp / transport / release) that a single amortised Gaussian prior smears
# together.
# ---------------------------------------------------------------------------
SUITE_15 = {
    "easy": ["button-press-topdown", "door-open", "drawer-open",
             "reach", "window-open"],
    "medium": ["basketball", "hammer", "peg-insert-side", "soccer",
               "sweep-into"],
    "hard": ["assembly", "pick-place", "pick-place-wall", "push",
             "stick-push"],
}

# Smallest defensible suite if compute is tight: 6 tasks, skewed hard, since a
# result on easy tasks is uninformative (everything solves them).
SUITE_6 = ["reach", "door-open", "hammer", "peg-insert-side", "assembly",
           "pick-place"]


def flat(suite):
    """Flatten a tiered suite dict into (task, tier, steps) triples."""
    out = []
    for tier, tasks in suite.items():
        for task in tasks:
            out.append((task, tier, TIER_STEPS[tier]))
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="suite15",
                    choices=["suite15", "suite6", "easy", "medium", "hard",
                             "all"])
    ap.add_argument("--with-steps", action="store_true")
    args = ap.parse_args()

    if args.suite == "suite15":
        rows = flat(SUITE_15)
    elif args.suite == "suite6":
        rows = [(t, "hard", 2_000_000) for t in SUITE_6]
    elif args.suite == "all":
        rows = flat(TIERS)
    else:
        rows = [(t, args.suite, TIER_STEPS[args.suite])
                for t in TIERS[args.suite]]

    for task, tier, steps in rows:
        print(f"{task}\t{tier}\t{steps}" if args.with_steps else task)
