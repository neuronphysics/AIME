"""Meta-World task tiers and the suite recommended for this repo's comparison.

Tier source: Seo et al., "Masked World Models for Visual Control" (2022), which
partitions the 50 tasks into easy / medium / hard / very hard and trains them
for 500K / 1M / 2M / 3M env steps respectively at action repeat 2.

Provenance caveat: EASY (28) and MEDIUM (11) are the published lists verbatim
and both validate against the installed registry. The remaining 11 split
6 hard / 5 very hard, and only 7 of those assignments are confirmed from
secondary sources. The 4 in HARD_TIER_UNCONFIRMED are definitely in the top 11
but their exact bucket is not verified. Do not publish a hard-vs-very-hard
breakdown without checking Seo et al. Appendix F. Aggregating them as one
"hard tier (11)" is safe and is what TIERS does.
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
HARD = ["assembly", "hand-insert", "pick-place", "push"]           # confirmed
VERY_HARD = ["pick-place-wall", "stick-pull", "stick-push"]        # confirmed
HARD_TIER_UNCONFIRMED = ["disassemble", "pick-out-of-hole", "push-back",
                         "shelf-place"]
HARD_TIER = HARD + VERY_HARD + HARD_TIER_UNCONFIRMED               # 11 tasks

TIERS = {"easy": EASY, "medium": MEDIUM, "hard": HARD_TIER}
TIER_STEPS = {"easy": 500_000, "medium": 1_000_000, "hard": 2_000_000}

# MEASURED endpoints in this codebase (scripted expert vs uniform random,
# randomised goals, action_repeat 2). Useful for reading progress as a fraction
# of the achievable range rather than an absolute return:
#   reach       random  268*  expert 4841
#   door-open   random  268   expert 4492
#   hammer      random  460   expert 2512   (zero-action scores 486)
#   drawer-open random  613   expert 4057
#   button-press-topdown random 193  expert 3652
#   window-open random  192   expert 1997
#   peg-insert-side random 6  expert 3745
# (*reach random return measured at 716 with its own goal distribution.)

# Recommended suite: 15 tasks, ~5 per tier, chosen to span difficulty rather
# than to flatter the method. FIX THE TASK LIST BEFORE LOOKING AT RESULTS.
SUITE_15 = {
    "easy": ["button-press-topdown", "door-open", "drawer-open",
             "reach", "window-open"],
    "medium": ["basketball", "hammer", "peg-insert-side", "soccer",
               "sweep-into"],
    "hard": ["assembly", "pick-place", "pick-place-wall", "push",
             "stick-push"],
}
# Smallest defensible suite if compute is tight.
SUITE_6 = ["reach", "door-open", "hammer", "peg-insert-side", "assembly",
           "pick-place"]


def flat(suite):
    """Flatten a tiered suite dict into (task, tier, steps) triples."""
    return [(t, tier, TIER_STEPS[tier])
            for tier, tasks in suite.items() for t in tasks]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="suite15",
                    choices=["suite15", "suite6", "easy", "medium", "hard", "all"])
    ap.add_argument("--with-steps", action="store_true")
    a = ap.parse_args()
    if a.suite == "suite15":
        rows = flat(SUITE_15)
    elif a.suite == "suite6":
        rows = [(t, "hard", 2_000_000) for t in SUITE_6]
    elif a.suite == "all":
        rows = flat(TIERS)
    else:
        rows = [(t, a.suite, TIER_STEPS[a.suite]) for t in TIERS[a.suite]]
    for task, tier, steps in rows:
        print(f"{task}\t{tier}\t{steps}" if a.with_steps else task)
