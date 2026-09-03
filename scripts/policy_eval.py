"""
Offline policy evaluation: was acting selectively actually worth it?

    python scripts/policy_eval.py

Scores four retry policies over the HELD-OUT TEST SET ONLY and writes
reports/policy_eval.json, which GET /reports/policy-eval serves to the dashboard.

    A  always_retry   retry every failed payment
    B  never_retry    retry nothing
    C  fixed_rule     retry when the bank's own decline code says a retry may work
    D  verdict        whatever the live agent actually decides

THE COUNTERFACTUAL ASSUMPTION, stated plainly because it is the thing that makes
this possible and also the thing that makes it a simulation rather than a
measurement:

    We assume the recovery outcome is known for EVERY test transaction -- that
    is, for each payment we know whether a retry would have worked, including
    for the payments a policy chose not to retry.

That holds here because `retry_success` is generated as a counterfactual
property of the transaction ("whether a retry/nudge would have recovered it",
src/data_generation/generate_transactions.py), not as a record of something that
happened. Real production data never has this: you only learn the outcome for
the payments you actually retried. So these numbers compare policies fairly
against each other, and should not be read as a forecast of live performance.

NO LABEL LEAKAGE. Every policy function receives a frame with `retry_success`
and `failure_category` already removed -- not "chooses not to look at them", but
cannot: the columns are not there. The outcome label is read in exactly one
place, `score()`, after every decision has already been made.
tests/test_policy_eval.py asserts both properties, including by flipping every
label and checking no decision moves.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.llm_adapter import TemplateAdapter  # noqa: E402
from src.agent.recovery_agent import Action, RecoveryAgent  # noqa: E402
from src.error_taxonomy import lookup  # noqa: E402
from src.paths import REPORTS_DIR, TEST_DATA, ensure_dirs  # noqa: E402

POLICY_EVAL_JSON = os.path.join(REPORTS_DIR, "policy_eval.json")

# Columns that exist only because this is a labelled dataset. Policies never see
# them; scoring reads retry_success and nothing else.
LABEL_COLUMNS = ["retry_success", "failure_category"]

# The actions that constitute a recovery attempt -- the same set
# /stats/recovered-revenue treats as "acted on". Built from the Action enum
# rather than from string literals, so renaming an action breaks loudly here
# instead of silently changing what gets counted.
ACTING = {Action.AUTO_RETRY_NOW.value, Action.RETRY_LATER.value,
          Action.CUSTOMER_NUDGE.value}

# --------------------------------------------------------------------------- #
# Cost model
# --------------------------------------------------------------------------- #

# Rupees per recovery attempt.
#
# THIS IS AN ASSUMPTION, NOT A MEASUREMENT. Nothing in this repository measures
# it, and it is the single number that decides which policy wins. The default is
# set in the region of card-network excessive-retry penalties -- Visa and
# Mastercard both charge acquirers per retry above a threshold -- because that is
# the only per-attempt charge a merchant reliably faces. Indian gateway fees are
# typically levied on success, not on attempts, so a retry that fails often costs
# nothing directly at all.
#
# Treat the default as a starting point for the sweep rather than as a finding.
# SWEEP below is the actual answer: it reports which policy wins at every cost in
# the range, so a reader who believes a different number can read off their own.
COST_PER_RETRY = 10.0

# Extra cost of nudging a customer, over and above a plain retry -- messaging
# spend, plus whatever a reminder costs in goodwill.
#
# Defaults to ZERO deliberately. Nothing here measures customer friction, and
# charging for it would mean inventing a number that moves the result. It is
# exposed as a parameter so it can be set on purpose rather than assumed, and
# only Verdict nudges, so a non-zero value counts against Verdict alone.
COST_PER_NUDGE_EXTRA = 0.0

# Wide enough to contain every crossover between the four policies.
SWEEP = np.arange(0.0, 501.0, 5.0)


# --------------------------------------------------------------------------- #
# Policies -- decisions from pre-retry information only
# --------------------------------------------------------------------------- #

def _blind(df: pd.DataFrame) -> pd.DataFrame:
    """The frame a policy is allowed to see: no outcome, no true category."""
    return df.drop(columns=LABEL_COLUMNS, errors="ignore")


def always_retry(df: pd.DataFrame) -> np.ndarray:
    """The naive baseline: try everything, including the fraud."""
    return np.ones(len(df), dtype=bool)


def never_retry(df: pd.DataFrame) -> np.ndarray:
    """The floor. Costs nothing, recovers nothing."""
    return np.zeros(len(df), dtype=bool)


def fixed_rule(df: pd.DataFrame) -> np.ndarray:
    """
    What a competent engineer writes without any machine learning: retry when
    the bank's own decline code says an unchanged retry can plausibly succeed.

    Reads src/error_taxonomy.py, which is a reference table of published ISO 8583
    and NPCI code meanings -- not a model, and not anything learned from this
    dataset. Codes whose retry outlook is unknown (`retryable is None`, e.g. the
    deliberately opaque card code 05) are NOT retried, because a rule engine that
    cannot tell has no basis to spend an attempt.
    """
    codes = df["error_code"].tolist()
    methods = df["payment_method"].tolist()
    out = np.zeros(len(df), dtype=bool)
    for i, (code, method) in enumerate(zip(codes, methods)):
        info = lookup(code, method)
        out[i] = info is not None and info.retryable is True
    return out


def _verdict_actions(agent: RecoveryAgent, df: pd.DataFrame) -> pd.Series:
    """The live agent's decision for every row, unmodified."""
    return agent.decide_batch(df)["action"]


def make_verdict(agent: RecoveryAgent):
    """
    Policy D: whatever the agent actually does.

    Calls decide_batch() rather than reimplementing the rule, so the thresholds
    cannot drift from DecisionPolicy.HIGH / .LOW -- there is nothing here to keep
    in sync with them, because nothing here restates them.
    """
    def verdict(df: pd.DataFrame) -> np.ndarray:
        return _verdict_actions(agent, df).isin(ACTING).values
    return verdict


def build_policies(agent: RecoveryAgent):
    """Ordered so the table reads naive, floor, rule engine, then Verdict."""
    return [
        ("always_retry", "Retry everything", always_retry),
        ("never_retry", "Retry nothing", never_retry),
        ("fixed_rule", "Retry what the bank says might work", fixed_rule),
        ("verdict", "What Verdict did", make_verdict(agent)),
    ]


# --------------------------------------------------------------------------- #
# Scoring -- the ONLY place the outcome label is read
# --------------------------------------------------------------------------- #

def score(key, label, mask, amount, won, nudged,
          cost_per_retry=COST_PER_RETRY, nudge_extra=COST_PER_NUDGE_EXTRA):
    attempts = int(mask.sum())
    recovered = int((mask & won).sum())
    revenue = float(amount[mask & won].sum())
    nudges = int((mask & nudged).sum())
    cost = attempts * cost_per_retry + nudges * nudge_extra
    n = len(amount)
    return {
        "key": key,
        "label": label,
        "attempts": attempts,
        "attempt_share": round(attempts / n, 4) if n else 0.0,
        "recovered": recovered,
        "recovered_revenue": round(revenue, 2),
        "wasted_retries": attempts - recovered,
        "nudges": nudges,
        "retry_cost": round(cost, 2),
        "net_value": round(revenue - cost, 2),
        "net_per_1000_failed": round((revenue - cost) / n * 1000, 2) if n else 0.0,
    }


def net_at(policy: dict, cost: float, nudge_extra: float = COST_PER_NUDGE_EXTRA) -> float:
    """Net value as a function of cost. Linear, which is why crossovers are exact."""
    return (policy["recovered_revenue"]
            - policy["attempts"] * cost
            - policy["nudges"] * nudge_extra)


def crossover(a: dict, b: dict):
    """
    The cost at which two policies' net values are equal.

    Net value is linear in cost, so this is solved rather than searched: no
    dependence on how finely the sweep happens to be sampled. Returns None when
    they never cross (equal attempt counts) or cross outside the swept range.
    """
    d_attempts = a["attempts"] - b["attempts"]
    if d_attempts == 0:
        return None
    c = (a["recovered_revenue"] - b["recovered_revenue"]) / d_attempts
    if c < SWEEP[0] or c > SWEEP[-1]:
        return None
    return round(float(c), 2)


# --------------------------------------------------------------------------- #

def main():
    ensure_dirs()
    if not os.path.exists(TEST_DATA):
        raise SystemExit("No test split. Run: python scripts/prepare_data.py")

    df = pd.read_csv(TEST_DATA)
    blind = _blind(df)

    amount = df["amount"].values.astype(float)
    won = df["retry_success"].values.astype(bool)          # scoring only
    n = len(df)

    agent = RecoveryAgent(adapter=TemplateAdapter(), explain=False)

    # Which rows Verdict nudges rather than retries. Needed so a non-zero
    # COST_PER_NUDGE_EXTRA lands on the policy that actually nudges.
    verdict_actions = _verdict_actions(agent, blind)
    nudged = (verdict_actions == Action.CUSTOMER_NUDGE.value).values

    results, masks = [], {}
    for key, label, fn in build_policies(agent):
        mask = fn(blind)
        masks[key] = mask
        is_verdict = key == "verdict"
        results.append(score(key, label, mask, amount, won,
                             nudged if is_verdict else np.zeros(n, dtype=bool)))

    by_key = {r["key"]: r for r in results}
    never, always, verdict = by_key["never_retry"], by_key["always_retry"], by_key["verdict"]

    # ---- sanity checks: raised, not papered over --------------------------
    if never["recovered_revenue"] != 0 or never["retry_cost"] != 0 or never["attempts"] != 0:
        raise AssertionError(
            f"never_retry produced attempts={never['attempts']}, "
            f"revenue={never['recovered_revenue']}, cost={never['retry_cost']}; "
            f"all three must be zero. The scorer is wrong."
        )
    best_recovery = max(r["recovered"] for r in results)
    if always["recovered"] != best_recovery:
        raise AssertionError(
            f"always_retry recovered {always['recovered']} but the best policy "
            f"recovered {best_recovery}. Retrying everything must recover at "
            f"least as much as any subset of it -- the outcome label is "
            f"misaligned with the rows."
        )
    recoverable = int(won.sum())
    if always["recovered"] != recoverable:
        raise AssertionError(
            f"always_retry recovered {always['recovered']} of {recoverable} "
            f"recoverable payments; retrying everything must recover all of them."
        )

    # ---- sensitivity sweep -------------------------------------------------
    sweep = []
    for c in SWEEP:
        nets = {r["key"]: round(net_at(r, float(c)), 2) for r in results}
        sweep.append({
            "cost": round(float(c), 2),
            "net_value": nets,
            "winner": max(nets, key=nets.get),
        })

    crossovers = []
    for other in (always, by_key["fixed_rule"]):
        c = crossover(verdict, other)
        if c is None:
            continue
        # Verdict makes more attempts than a policy it overtakes as cost RISES.
        verdict_wins_above = verdict["attempts"] < other["attempts"]
        crossovers.append({
            "against": other["key"],
            "against_label": other["label"],
            "cost": c,
            "verdict_wins": "above" if verdict_wins_above else "below",
            "note": (f"Verdict nets more than '{other['label'].lower()}' "
                     f"{'above' if verdict_wins_above else 'below'} Rs {c:,.2f} per attempt."),
        })

    # Derived from the solved crossovers, not from which grid points happen to
    # fall Verdict's way: a 5-rupee sweep step would otherwise report the band as
    # starting at 50 while the crossover sentence beside it says 46.84.
    _above = next((c["cost"] for c in crossovers if c["verdict_wins"] == "above"), None)
    _below = next((c["cost"] for c in crossovers if c["verdict_wins"] == "below"), None)
    wins_band = ([_above, _below] if _above is not None and _below is not None
                 and _above < _below else None)
    wins = [s["cost"] for s in sweep if s["winner"] == "verdict"]
    payload = {
        "assumptions": {
            "cost_per_retry": COST_PER_RETRY,
            "cost_per_nudge_extra": COST_PER_NUDGE_EXTRA,
            "sweep_min": float(SWEEP[0]),
            "sweep_max": float(SWEEP[-1]),
            "sweep_step": float(SWEEP[1] - SWEEP[0]),
            "cost_note": (
                "Cost per retry is an assumption, not a measurement -- nothing in "
                "this project measures it. It is the one number that decides which "
                "policy wins, which is why the whole range is reported rather than "
                "a single figure."
            ),
            "counterfactual_note": (
                "We assume we know, for every payment, whether a retry would have "
                "worked -- including the payments a policy chose not to retry. Real "
                "production data never tells you that: you only learn the outcome "
                "for the attempts you actually made. So this compares the policies "
                "fairly against each other, and is not a forecast."
            ),
            "single_attempt_note": (
                "One attempt per payment. The outcome label says whether a retry or "
                "a nudge would have worked, not how many tries it would take, so "
                "this does not model repeated attempts."
            ),
        },
        "dataset": {
            "split": "held-out test set",
            "n": n,
            "total_failed_value": round(float(amount.sum()), 2),
            "recoverable_count": recoverable,
            "recoverable_value": round(float(amount[won].sum()), 2),
        },
        "policies": results,
        "sweep": sweep,
        "crossovers": crossovers,
        "headline": {
            "recoverable_share_captured": round(
                verdict["recovered_revenue"] / float(amount[won].sum()), 4),
            "attempt_reduction_vs_always": round(
                1 - verdict["attempts"] / always["attempts"], 4),
            "verdict_wins_between": wins_band,
        },
    }

    with open(POLICY_EVAL_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    # ---- report ------------------------------------------------------------
    print(f"Held-out test set: {n} payments, Rs {amount.sum():,.0f} failed, "
          f"{recoverable} recoverable (Rs {amount[won].sum():,.0f})\n")
    print(f"{'policy':<16}{'attempts':>9}{'%':>7}{'recovered':>11}"
          f"{'revenue':>13}{'wasted':>8}{'net/1000':>12}")
    for r in results:
        print(f"{r['key']:<16}{r['attempts']:>9}{r['attempt_share']:>7.1%}"
              f"{r['recovered']:>11}{r['recovered_revenue']:>13,.0f}"
              f"{r['wasted_retries']:>8}{r['net_per_1000_failed']:>12,.0f}")

    print(f"\n  (net value at the default assumption of Rs {COST_PER_RETRY:,.2f} per attempt)")
    print("\nSensitivity")
    for c in crossovers:
        print(f"  {c['note']}")
    if wins_band:
        print(f"  Verdict is the best of the four between Rs {wins_band[0]:,.2f} "
              f"and Rs {wins_band[1]:,.2f} per attempt.")
    else:
        print("  Verdict is not the best policy anywhere in the swept range.")

    print(f"\n  Verdict captured {payload['headline']['recoverable_share_captured']:.1%} of "
          f"all recoverable revenue with "
          f"{payload['headline']['attempt_reduction_vs_always']:.0%} fewer attempts "
          f"than retrying everything.")
    print(f"\nWrote {POLICY_EVAL_JSON}")


if __name__ == "__main__":
    main()
