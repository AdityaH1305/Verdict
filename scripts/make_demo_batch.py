"""
Build the pinned demo batch used for the pitch recording.

    python scripts/make_demo_batch.py

Searches candidate samples of the held-out set for one that tells the whole story
in a single screen, then writes it to data/demo/ as COMMITTED DATA.

Why commit the rows rather than just record a seed: a seed only reproduces the
same batch against a byte-identical test.csv. Regenerate the data on recording
day, or run on a machine whose pandas samples differently, and the seed silently
yields a different batch -- with different numbers than the ones being said out
loud. The CSV removes that failure mode entirely. The seed is still recorded in
the sidecar JSON so the selection can be audited and re-derived.

Selection criteria (hard filters, then a tie-break):
  1. All five actions present -- including `escalate`, which is ~0.6% of traffic
     and so is the one that can silently go missing.
  2. At least two `escalate` rows, so the rarest action survives a feed filter.
  3. At least one fraud-blocked row -- the architectural hard rule needs to be
     visible, not described.
  4. A visible card-vs-UPI skew in the designed direction: cards lean
     hard_decline, UPI leans customer_dropoff.
  5. Tie-break: recovered revenue closest to a round number, because the headline
     figure gets spoken aloud.
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.recovery_agent import RecoveryAgent  # noqa: E402
from src.features import TARGET_RECOVERY  # noqa: E402
from src.paths import ROOT, TEST_DATA  # noqa: E402

DEMO_DIR = os.path.join(ROOT, "data", "demo")
DEMO_CSV = os.path.join(DEMO_DIR, "demo_batch.csv")
DEMO_META = os.path.join(DEMO_DIR, "demo_batch.json")

BATCH_SIZE = 300
SEEDS_TO_SCAN = 200

ALL_ACTIONS = {"auto_retry_now", "retry_later", "customer_nudge",
               "escalate", "no_action"}
ACTING = {"auto_retry_now", "retry_later", "customer_nudge"}

MIN_ESCALATE = 2
MIN_FRAUD = 1
MIN_CARD_HARD_SHARE = 0.42     # cards leaning hard_decline
MIN_UPI_DROPOFF_SHARE = 0.40   # UPI leaning customer_dropoff

# The headline gets spoken aloud, so prefer a batch whose recovered revenue lands
# near a round figure. Not a correctness criterion -- purely presentational, and
# applied only as a tie-break among batches that already pass every filter above.
TARGET_RECOVERED = 100_000


def score_batch(decisions: pd.DataFrame) -> dict:
    actions = decisions["action"]
    acted = decisions[actions.isin(ACTING)]
    recovered = acted[acted[TARGET_RECOVERY]]

    shares = decisions.groupby("payment_method")["predicted_category"].value_counts(
        normalize=True)

    def share(method, category):
        try:
            return float(shares.loc[(method, category)])
        except KeyError:
            return 0.0

    return {
        "actions_present": set(actions.unique()),
        "escalate": int((actions == "escalate").sum()),
        "fraud_blocked": int(decisions["fraud_blocked"].sum()),
        "card_hard_share": share("card", "hard_decline"),
        "upi_dropoff_share": share("upi", "customer_dropoff"),
        "acted_count": int(len(acted)),
        "recovered_count": int(len(recovered)),
        "recovered_value": round(float(recovered["amount"].sum()), 2),
        "total_failed_value": round(float(decisions["amount"].sum()), 2),
        "action_mix": actions.value_counts().to_dict(),
    }


def passes(s: dict) -> bool:
    return (s["actions_present"] == ALL_ACTIONS
            and s["escalate"] >= MIN_ESCALATE
            and s["fraud_blocked"] >= MIN_FRAUD
            and s["card_hard_share"] >= MIN_CARD_HARD_SHARE
            and s["upi_dropoff_share"] >= MIN_UPI_DROPOFF_SHARE)


def main():
    if not os.path.exists(TEST_DATA):
        raise SystemExit("No test data. Run: python scripts/prepare_data.py")

    pool = pd.read_csv(TEST_DATA)
    agent = RecoveryAgent(explain=False)

    print(f"Scanning {SEEDS_TO_SCAN} candidate batches of {BATCH_SIZE} "
          f"from {len(pool)} held-out transactions...\n")

    candidates = []
    for seed in range(SEEDS_TO_SCAN):
        batch = pool.sample(n=BATCH_SIZE, random_state=seed)
        stats = score_batch(agent.decide_batch(batch))
        if passes(stats):
            stats["seed"] = seed
            stats["distance_to_target"] = abs(stats["recovered_value"] - TARGET_RECOVERED)
            candidates.append(stats)

    if not candidates:
        raise SystemExit(
            "No batch satisfied every criterion. Loosen the thresholds at the top "
            "of this file, or raise SEEDS_TO_SCAN."
        )

    candidates.sort(key=lambda s: s["distance_to_target"])
    best = candidates[0]

    print(f"{len(candidates)}/{SEEDS_TO_SCAN} batches passed every filter.")
    print("\nTop 5 by closeness of recovered revenue to a round figure:")
    print(f"  {'seed':>5} {'recovered':>12} {'esc':>4} {'fraud':>6} "
          f"{'card->hard':>11} {'upi->drop':>10}")
    for c in candidates[:5]:
        print(f"  {c['seed']:>5} {c['recovered_value']:>12,.0f} {c['escalate']:>4} "
              f"{c['fraud_blocked']:>6} {c['card_hard_share']:>9.1%} "
              f"{c['upi_dropoff_share']:>9.1%}")

    batch = pool.sample(n=BATCH_SIZE, random_state=best["seed"]).reset_index(drop=True)
    os.makedirs(DEMO_DIR, exist_ok=True)
    batch.to_csv(DEMO_CSV, index=False)

    meta = {
        "description": "Pinned demo batch for the pitch recording. The CSV is the "
                       "source of truth; the seed records how it was chosen.",
        "source_seed": best["seed"],
        "source_file": os.path.relpath(TEST_DATA, ROOT).replace("\\", "/"),
        "batch_size": BATCH_SIZE,
        "expected": {
            "action_mix": best["action_mix"],
            "recovered_count": best["recovered_count"],
            "recovered_value": best["recovered_value"],
            "acted_count": best["acted_count"],
            "total_failed_value": best["total_failed_value"],
            "fraud_blocked": best["fraud_blocked"],
            "card_hard_decline_share": round(best["card_hard_share"], 4),
            "upi_customer_dropoff_share": round(best["upi_dropoff_share"], 4),
        },
    }
    with open(DEMO_META, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSelected seed {best['seed']}:")
    print(f"  recovered      {best['recovered_value']:,.2f} "
          f"across {best['recovered_count']} transactions")
    print(f"  acted on       {best['acted_count']} of {BATCH_SIZE}")
    print(f"  action mix     {best['action_mix']}")
    print(f"  fraud blocked  {best['fraud_blocked']}")
    print(f"  skew           {best['card_hard_share']:.1%} of cards are hard "
          f"declines; {best['upi_dropoff_share']:.1%} of UPI is customer drop-off")
    print(f"\nWrote {DEMO_CSV}")
    print(f"Wrote {DEMO_META}")


if __name__ == "__main__":
    main()
