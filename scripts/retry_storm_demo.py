"""
Retry-storm demo: run the naive loop, show the storm, then run the fixed loop.

    python scripts/retry_storm_demo.py

Produces the before/after numbers for docs/failure_stories.md, Candidate 2, and
writes reports/retry_storm.json + reports/retry_storm.png.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.recovery_agent import RecoveryAgent  # noqa: E402
from src.agent.retry_simulator import (  # noqa: E402
    DEFAULT_BUDGET, DEFAULT_CYCLES, run_capped, run_uncapped,
)
from src.paths import REPORTS_DIR, TEST_DATA, ensure_dirs  # noqa: E402

STORM_JSON = os.path.join(REPORTS_DIR, "retry_storm.json")
STORM_PLOT = os.path.join(REPORTS_DIR, "retry_storm.png")


def plot(before: dict, after: dict, before_hist, after_hist, path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    labels = ["total\nattempts", "wasted\nattempts", "recovered"]
    b = [before["total_attempts"], before["wasted_attempts"], before["recovered"]]
    a = [after["total_attempts"], after["wasted_attempts"], after["recovered"]]
    x = np.arange(len(labels)); w = 0.38
    ax1.bar(x - w/2, b, w, label="uncapped", color="#e63946")
    ax1.bar(x + w/2, a, w, label="capped + backoff", color="#2a9d8f")
    for i, (bv, av) in enumerate(zip(b, a)):
        ax1.text(i - w/2, bv, f"{bv:,}", ha="center", va="bottom", fontsize=9)
        ax1.text(i + w/2, av, f"{av:,}", ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel("Count")
    ax1.set_title("Retry work done vs. revenue recovered")
    ax1.legend()
    ax1.set_ylim(0, max(b) * 1.15)

    bins = np.arange(0, max(before_hist + after_hist) + 2) - 0.5
    ax2.hist([before_hist, after_hist], bins=bins, label=["uncapped", "capped + backoff"],
             color=["#e63946", "#2a9d8f"])
    ax2.axvline(DEFAULT_BUDGET + 0.5, color="black", linestyle="--", linewidth=1)
    ax2.text(DEFAULT_BUDGET + 0.65, ax2.get_ylim()[1] * 0.9,
             f"budget = {DEFAULT_BUDGET}", fontsize=9)
    ax2.set_xlabel("Retry attempts on a single transaction")
    ax2.set_ylabel("Transactions")
    ax2.set_title("Attempts per transaction")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_dirs()
    test_df = pd.read_csv(TEST_DATA)

    agent = RecoveryAgent(explain=False)
    decisions = agent.decide_batch(test_df)

    print(f"Batch: {len(decisions)} failed transactions")
    print(f"Polling cycles simulated: {DEFAULT_CYCLES}\n")

    before = run_uncapped(decisions, cycles=DEFAULT_CYCLES)
    after = run_capped(decisions, cycles=DEFAULT_CYCLES, budget=DEFAULT_BUDGET)
    b, a = before.summary(), after.summary()

    print("=" * 74)
    print("BEFORE -- naive loop: re-queue anything that hasn't succeeded")
    print("=" * 74)
    print(f"  transactions queued for retry : {b['queued_transactions']}")
    print(f"  TOTAL RETRY ATTEMPTS          : {b['total_attempts']:,}")
    print(f"  attempts per transaction      : mean {b['attempts_per_txn_mean']}, "
          f"max {b['attempts_per_txn_max']}")
    print(f"  attempts that could never work: {b['wasted_attempts']:,} "
          f"({b['wasted_attempt_share']:.1%} of all attempts)")
    print(f"  recovered                     : {b['recovered']} txns, "
          f"{b['recovered_value']:,.0f}")
    print(f"  still queued after {b['cycles']} cycles : {b['still_queued_at_end']} "
          f"(these retry forever)")

    print("\n" + "=" * 74)
    print(f"AFTER -- per-transaction budget ({DEFAULT_BUDGET}) + exponential backoff")
    print("=" * 74)
    print(f"  transactions queued for retry : {a['queued_transactions']}")
    print(f"  TOTAL RETRY ATTEMPTS          : {a['total_attempts']:,}")
    print(f"  attempts per transaction      : mean {a['attempts_per_txn_mean']}, "
          f"max {a['attempts_per_txn_max']}")
    print(f"  attempts that could never work: {a['wasted_attempts']:,} "
          f"({a['wasted_attempt_share']:.1%} of all attempts)")
    print(f"  recovered                     : {a['recovered']} txns, "
          f"{a['recovered_value']:,.0f}")
    print(f"  gave up after budget exhausted: {a['budget_exhausted']}")
    print(f"  still queued after {a['cycles']} cycles : {a['still_queued_at_end']}")

    reduction = 1 - a["total_attempts"] / max(b["total_attempts"], 1)
    recovery_delta = a["recovered"] - b["recovered"]

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    print(f"  retry traffic cut by {reduction:.1%} "
          f"({b['total_attempts']:,} -> {a['total_attempts']:,} attempts)")
    print(f"  revenue recovered: {b['recovered_value']:,.0f} -> {a['recovered_value']:,.0f} "
          f"({recovery_delta:+d} transactions)")
    print(f"  worst-case load on a single transaction: "
          f"{b['attempts_per_txn_max']} -> {a['attempts_per_txn_max']} attempts")
    print(f"\n  The uncapped loop did {b['total_attempts'] / max(a['total_attempts'],1):.1f}x "
          f"the work for {'the same' if recovery_delta == 0 else 'comparable'} revenue.")

    payload = {"before": b, "after": a,
               "attempt_reduction": round(reduction, 4),
               "recovered_delta": recovery_delta}
    with open(STORM_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    plot(b, a, list(before.attempts_per_txn.values()),
         list(after.attempts_per_txn.values()), STORM_PLOT)

    print(f"\nWrote {STORM_JSON}")
    print(f"Wrote {STORM_PLOT}")


if __name__ == "__main__":
    main()
