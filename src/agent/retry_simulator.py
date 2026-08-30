"""
Retry execution simulator -- deliberately built broken first.

See docs/failure_stories.md, Candidate 2. The uncapped loop below is the naive
implementation anyone writes on the first pass: a polling cycle that re-queues
any transaction still marked for retry. It is not a strawman -- it is what you
get when you build the retry loop without thinking about termination.

The pathology: a transaction whose retry can never succeed fails, is re-queued,
fails again, and is re-queued again, on every cycle, forever. The storm
concentrates precisely on the transactions that are least worth retrying, so the
system does unbounded work on the issuer's endpoint in exchange for nothing.

Retry outcomes are simulated against the ground-truth `retry_success` label:
  - not recoverable -> every attempt fails (this is what the label means)
  - recoverable     -> each attempt succeeds with probability RECOVERABLE_ATTEMPT_P,
                       so recoverable transactions usually land within a couple of
                       attempts but not always on the first. That is what makes a
                       retry budget a real tradeoff rather than a free win.
"""

import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.features import TARGET_RECOVERY  # noqa: E402

RETRYABLE_ACTIONS = {"auto_retry_now", "retry_later"}

RECOVERABLE_ATTEMPT_P = 0.6
DEFAULT_CYCLES = 12
DEFAULT_BUDGET = 3
DEFAULT_BACKOFF_BASE = 1  # cycles


@dataclass
class RetryStats:
    label: str
    cycles: int
    queued_transactions: int
    total_attempts: int = 0
    recovered: int = 0
    recovered_value: float = 0.0
    attempts_per_txn: dict = field(default_factory=dict)
    wasted_attempts: int = 0          # attempts on transactions that can never succeed
    still_queued_at_end: int = 0
    budget_exhausted: int = 0

    def summary(self) -> dict:
        counts = list(self.attempts_per_txn.values()) or [0]
        return {
            "label": self.label,
            "cycles": self.cycles,
            "queued_transactions": self.queued_transactions,
            "total_attempts": self.total_attempts,
            "attempts_per_txn_mean": round(float(np.mean(counts)), 2),
            "attempts_per_txn_max": int(np.max(counts)),
            "wasted_attempts": self.wasted_attempts,
            "wasted_attempt_share": (round(self.wasted_attempts / self.total_attempts, 4)
                                      if self.total_attempts else 0.0),
            "recovered": self.recovered,
            "recovered_value": round(self.recovered_value, 2),
            "still_queued_at_end": self.still_queued_at_end,
            "budget_exhausted": self.budget_exhausted,
        }


def _attempt_succeeds(is_recoverable: bool, rng: np.random.Generator) -> bool:
    if not is_recoverable:
        return False
    return rng.random() < RECOVERABLE_ATTEMPT_P


def run_uncapped(decisions: pd.DataFrame, cycles: int = DEFAULT_CYCLES,
                  seed: int = 42) -> RetryStats:
    """
    THE BUG, on purpose.

    Every cycle, re-queue everything that hasn't succeeded. No budget, no
    backoff, no termination condition other than success.
    """
    rng = np.random.default_rng(seed)
    queue = decisions[decisions["action"].isin(RETRYABLE_ACTIONS)].copy()
    stats = RetryStats("uncapped (no budget, no backoff)", cycles, len(queue))

    pending = {
        row["transaction_id"]: (bool(row[TARGET_RECOVERY]), float(row["amount"]))
        for _, row in queue.iterrows()
    }

    for _ in range(cycles):
        for txn_id in list(pending.keys()):
            recoverable, amount = pending[txn_id]

            stats.total_attempts += 1
            stats.attempts_per_txn[txn_id] = stats.attempts_per_txn.get(txn_id, 0) + 1
            if not recoverable:
                stats.wasted_attempts += 1

            if _attempt_succeeds(recoverable, rng):
                stats.recovered += 1
                stats.recovered_value += amount
                del pending[txn_id]
            # else: falls through and is retried again next cycle. Forever.

    stats.still_queued_at_end = len(pending)
    return stats


def run_capped(decisions: pd.DataFrame, cycles: int = DEFAULT_CYCLES,
                budget: int = DEFAULT_BUDGET, backoff_base: int = DEFAULT_BACKOFF_BASE,
                seed: int = 42) -> RetryStats:
    """
    THE FIX: a per-transaction retry budget plus exponential backoff.

    Budget bounds total work per transaction; backoff spreads the attempts across
    cycles instead of hammering the issuer on every poll.
    """
    rng = np.random.default_rng(seed)
    queue = decisions[decisions["action"].isin(RETRYABLE_ACTIONS)].copy()
    stats = RetryStats(f"capped (budget={budget}, exponential backoff)", cycles, len(queue))

    pending = {
        row["transaction_id"]: {
            "recoverable": bool(row[TARGET_RECOVERY]),
            "amount": float(row["amount"]),
            "attempts": 0,
            "next_eligible_cycle": 0,
        }
        for _, row in queue.iterrows()
    }

    for cycle in range(cycles):
        for txn_id in list(pending.keys()):
            state = pending[txn_id]

            if cycle < state["next_eligible_cycle"]:
                continue  # backoff: not due yet

            state["attempts"] += 1
            stats.total_attempts += 1
            stats.attempts_per_txn[txn_id] = state["attempts"]
            if not state["recoverable"]:
                stats.wasted_attempts += 1

            if _attempt_succeeds(state["recoverable"], rng):
                stats.recovered += 1
                stats.recovered_value += state["amount"]
                del pending[txn_id]
            elif state["attempts"] >= budget:
                # Budget is spent the moment the last attempt fails -- dequeue
                # here, not on a future visit. Enforcing it lazily leaves
                # transactions sitting in the queue whose backoff has pushed them
                # past the horizon: they look "still pending" forever while
                # actually being finished.
                stats.budget_exhausted += 1
                del pending[txn_id]      # give up; the agent escalates instead
            else:
                # exponential backoff: 2, 4, 8, ... cycles between attempts
                state["next_eligible_cycle"] = cycle + backoff_base * (2 ** state["attempts"])

    stats.still_queued_at_end = len(pending)
    return stats
