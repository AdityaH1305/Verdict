"""
Run the agent over the whole test set and report where transactions land.

    python scripts/evaluate_agent.py

This is the end-to-end check on the decision layer: the action mix, how much
revenue each action actually recovers, and -- most importantly -- that no fraud
transaction ever receives an action other than no_action.

Writes reports/agent_action_mix.json + reports/action_mix.png.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.recovery_agent import RETRY_ACTIONS, Action, DecisionPolicy, RecoveryAgent  # noqa: E402
from src.features import TARGET_CATEGORY, TARGET_RECOVERY  # noqa: E402
from src.paths import REPORTS_DIR, TEST_DATA, ensure_dirs  # noqa: E402

ACTION_MIX_PATH = os.path.join(REPORTS_DIR, "agent_action_mix.json")
ACTION_MIX_PLOT = os.path.join(REPORTS_DIR, "action_mix.png")

# Actions where the agent actually does something to recover the transaction.
ACTING = {a.value for a in RETRY_ACTIONS} | {Action.CUSTOMER_NUDGE.value}


def plot_action_mix(summary: pd.DataFrame, path: str):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = {
        "auto_retry_now": "#2a9d8f", "retry_later": "#8ab17d",
        "customer_nudge": "#e9c46a", "escalate": "#f4a261", "no_action": "#adb5bd",
    }
    bar_colors = [colors.get(a, "#adb5bd") for a in summary.index]

    ax.barh(summary.index, summary["count"], color=bar_colors)
    for i, (n, share) in enumerate(zip(summary["count"], summary["share"])):
        ax.text(n + 6, i, f"{n}  ({share:.1%})", va="center", fontsize=9)
    ax.set_xlabel("Transactions")
    ax.set_title("Agent action mix (test set)")
    ax.set_xlim(0, summary["count"].max() * 1.28)

    ax2.barh(summary.index, summary["actual_recovery_rate"], color=bar_colors)
    for i, v in enumerate(summary["actual_recovery_rate"]):
        ax2.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=9)
    ax2.set_xlabel("Observed recovery rate of transactions in this bucket")
    ax2.set_title("Was the action justified?")
    ax2.set_xlim(0, max(summary["actual_recovery_rate"].max() * 1.25, 0.1))

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_dirs()
    test_df = pd.read_csv(TEST_DATA)

    # explain=False: the LLM narrates decisions, it doesn't make them, so a
    # 1,600-row evaluation shouldn't spend 1,600 API calls to measure routing.
    agent = RecoveryAgent(explain=False)
    decisions = agent.decide_batch(test_df)

    policy = agent.policy
    print(f"Thresholds: HIGH={policy.HIGH}  LOW={policy.LOW}")
    print(f"Test transactions: {len(decisions)}\n")

    summary = decisions.groupby("action").agg(
        count=("action", "size"),
        actual_recovery_rate=(TARGET_RECOVERY, "mean"),
        mean_recovery_prob=("recovery_prob", "mean"),
        total_amount=("amount", "sum"),
    )
    summary["share"] = summary["count"] / len(decisions)
    order = [a.value for a in Action if a.value in summary.index]
    summary = summary.loc[order]

    print("=== Action mix ===")
    print(summary[["count", "share", "actual_recovery_rate", "mean_recovery_prob"]]
          .round(4).to_string())

    # --- the hard rule, verified on real output ---
    #
    # Two distinct claims here, and they must not be conflated:
    #
    #   (1) ARCHITECTURAL GUARANTEE -- anything the system diagnoses as fraud is
    #       never acted on and never gets a recovery estimate. Absolute, and
    #       asserted below.
    #   (2) MODEL 1's FRAUD RECALL -- whether the classifier spots fraud at all.
    #       0.99, not 1.0. Reported, not asserted.
    #
    # The architecture guarantees behaviour *given* a diagnosis; it cannot
    # guarantee the diagnosis. A fraud transaction the classifier fails to
    # recognise is a detection miss, not a breach of the hard rule -- and saying
    # otherwise would overclaim what the design actually buys.
    flagged_fraud = decisions[decisions["fraud_blocked"]]
    flagged_and_acted = flagged_fraud[flagged_fraud["action"] != Action.NO_ACTION.value]
    scored_despite_flag = int(flagged_fraud["recovery_prob"].notna().sum())

    true_fraud = decisions[decisions[TARGET_CATEGORY] == "fraud_block"]
    missed_fraud = true_fraud[~true_fraud["fraud_blocked"]]
    missed_fraud_acted = missed_fraud[missed_fraud["action"] != Action.NO_ACTION.value]

    print("\n=== Fraud hard-block ===")
    print("(1) Architectural guarantee -- given the diagnosis:")
    print(f"    diagnosed as fraud and short-circuited:  {len(flagged_fraud)}")
    print(f"    of those, given a recovery action:       {len(flagged_and_acted)} (must be 0)")
    print(f"    of those, had a recovery prob computed:  {scored_despite_flag} (must be 0)")
    print("(2) Model 1 fraud detection -- separate concern, reported not asserted:")
    print(f"    true fraud transactions:                 {len(true_fraud)}")
    print(f"    missed by the classifier:                {len(missed_fraud)} "
          f"(recall {1 - len(missed_fraud)/max(len(true_fraud),1):.3f})")
    print(f"    of those misses, acted on:               {len(missed_fraud_acted)}")

    if len(flagged_and_acted) > 0 or scored_despite_flag > 0:
        raise SystemExit(
            "FAIL: a transaction diagnosed as fraud was acted on or scored. "
            "The architectural hard rule is broken."
        )

    # --- revenue ---
    acted = decisions[decisions["action"].isin(ACTING)]
    recovered = acted[acted[TARGET_RECOVERY]]
    missed = decisions[~decisions["action"].isin(ACTING) & decisions[TARGET_RECOVERY]]

    print("\n=== Revenue ===")
    print(f"total failed value:                {decisions['amount'].sum():>12,.0f}")
    print(f"agent acted on:                    {len(acted)} txns "
          f"({len(acted)/len(decisions):.1%}), {acted['amount'].sum():,.0f}")
    print(f"actually recovered:                {len(recovered)} txns, "
          f"{recovered['amount'].sum():,.0f}")
    print(f"recoverable but left alone:        {len(missed)} txns, "
          f"{missed['amount'].sum():,.0f}")
    print(f"precision of acting (acted & recovered / acted): "
          f"{len(recovered)/max(len(acted),1):.3f}")

    payload = {
        "thresholds": {"high": policy.HIGH, "low": policy.LOW},
        "n_transactions": int(len(decisions)),
        "action_mix": {
            action: {
                "count": int(row["count"]),
                "share": round(float(row["share"]), 4),
                "actual_recovery_rate": round(float(row["actual_recovery_rate"]), 4),
                "mean_recovery_prob": (None if pd.isna(row["mean_recovery_prob"])
                                        else round(float(row["mean_recovery_prob"]), 4)),
                "total_amount": round(float(row["total_amount"]), 2),
            }
            for action, row in summary.iterrows()
        },
        "fraud_hard_block": {
            "architectural_guarantee": {
                "diagnosed_as_fraud": int(len(flagged_fraud)),
                "given_a_recovery_action": int(len(flagged_and_acted)),
                "recovery_probability_computed": scored_despite_flag,
            },
            "model_1_fraud_detection": {
                "true_fraud_transactions": int(len(true_fraud)),
                "missed_by_classifier": int(len(missed_fraud)),
                "recall": round(1 - len(missed_fraud) / max(len(true_fraud), 1), 4),
                "missed_and_acted_on": int(len(missed_fraud_acted)),
            },
        },
        "revenue": {
            "total_failed_value": round(float(decisions["amount"].sum()), 2),
            "acted_on_count": int(len(acted)),
            "acted_on_value": round(float(acted["amount"].sum()), 2),
            "recovered_count": int(len(recovered)),
            "recovered_value": round(float(recovered["amount"].sum()), 2),
            "missed_recoverable_count": int(len(missed)),
            "missed_recoverable_value": round(float(missed["amount"].sum()), 2),
        },
    }
    with open(ACTION_MIX_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    plot_action_mix(summary, ACTION_MIX_PLOT)

    print(f"\nWrote {ACTION_MIX_PATH}")
    print(f"Wrote {ACTION_MIX_PLOT}")


if __name__ == "__main__":
    main()
