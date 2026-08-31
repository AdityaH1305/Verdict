"""
Recovery Agent: combines Model 1 + Model 2 outputs into a decision.

Decision flow:
  1. Run Model 1 -> failure_category
  2. If category == "fraud_block": return NO_ACTION immediately.
     Do not call Model 2. Do not call the LLM adapter. This is a hard
     architectural rule, not a learned preference -- see docs/decisions.md.
  3. Otherwise, run Model 2 -> recovery_success_probability
  4. Apply decision thresholds -> action
     (auto_retry_now | retry_later | customer_nudge | escalate | no_action)
  5. Call LLM adapter for a plain-language explanation of the decision
  6. Log the full decision (category, probability, action, explanation)
     for the dashboard's audit feed

On the fraud rule: step 2 returns before Model 2 or the adapter are reached at
all. It is not a filter applied to a computed result, and not "call them, then
ignore the answer" -- there is no execution path in `decide()` where a
fraud_block transaction reaches either. Model 2 independently raises if fraud
rows reach its training set (src/models/recovery_success_model.py), and the
adapters independently raise if invoked with a fraud category. Three guards, no
single point of failure.

Thresholds were derived from Model 2's measured score distribution, not guessed
-- see docs/decisions.md, "Day 3 -- threshold tuning".
"""

import os
import sys
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.agent.llm_adapter import LLMAdapter, get_adapter  # noqa: E402
from src.error_taxonomy import describe  # noqa: E402
from src.models.failure_classifier import FailureClassifier  # noqa: E402
from src.models.recovery_success_model import RecoverySuccessModel  # noqa: E402

FRAUD_CLASS = "fraud_block"

# Written here, not by a model. The adapter is never called for fraud, so this
# is the only explanation a fraud-blocked transaction can ever carry.
FRAUD_EXPLANATION = (
    "Blocked as suspected fraud. Recovery is not attempted for fraud-flagged "
    "transactions under any circumstances, and no recovery estimate is produced."
)


class Action(str, Enum):
    AUTO_RETRY_NOW = "auto_retry_now"
    RETRY_LATER = "retry_later"
    CUSTOMER_NUDGE = "customer_nudge"
    ESCALATE = "escalate"
    NO_ACTION = "no_action"


RETRY_ACTIONS = {Action.AUTO_RETRY_NOW, Action.RETRY_LATER}


class DecisionPolicy:
    """
    Threshold policy mapping (category, recovery probability) -> action.

    Both cutoffs come from the measured, calibrated score distribution on the
    test set rather than from round numbers picked in advance:

      HIGH = 0.50 -- "recovery is more likely than not, so just retry it."
          Also the median of the soft_decline population (0.495). Of the cutoffs
          evaluated, this one produced the highest observed recovery rate inside
          the auto-retry bucket (0.610).

      LOW = 0.25 -- the floor of the empirical valley between the two modes of a
          bimodal distribution. Below it sits the hard_decline mass (observed
          recovery 0.043); above it, everything worth acting on. Beats 0.30:
          fewer recoverable transactions stranded, half as many pointless
          escalations.

    The build plan's original 0.6/0.3 were written before any scores existed. At
    0.6 only 1.6% of transactions ever reached auto_retry_now.
    """

    HIGH = 0.50
    LOW = 0.25

    def decide_action(self, category: str, recovery_prob: float) -> Action:
        if category == FRAUD_CLASS:
            # Defence in depth -- decide() returns before ever reaching here.
            return Action.NO_ACTION

        if recovery_prob >= self.HIGH:
            return Action.AUTO_RETRY_NOW

        if recovery_prob >= self.LOW:
            # Same odds, different mechanism: a bank-side failure wants another
            # attempt; a human who walked away wants prompting, not a silent retry.
            return (Action.CUSTOMER_NUDGE if category == "customer_dropoff"
                    else Action.RETRY_LATER)

        # Below the floor. Escalating an unrecoverable hard decline just makes
        # work for a human who can't do anything about it either.
        return Action.NO_ACTION if category == "hard_decline" else Action.ESCALATE


class RecoveryAgent:
    def __init__(self, classifier: Optional[FailureClassifier] = None,
                 recovery_model: Optional[RecoverySuccessModel] = None,
                 adapter: Optional[LLMAdapter] = None,
                 policy: Optional[DecisionPolicy] = None,
                 explain: bool = True):
        self.classifier = classifier if classifier is not None else FailureClassifier.load()
        self.recovery_model = (recovery_model if recovery_model is not None
                                else RecoverySuccessModel.load())
        self.adapter = adapter if adapter is not None else get_adapter()
        self.policy = policy if policy is not None else DecisionPolicy()
        self.explain = explain
        self.audit_log = []

    def decide(self, transaction: dict) -> dict:
        df = pd.DataFrame([transaction])
        category_proba = self.classifier.predict_proba(df)
        category = self.classifier.predict(df)[0]
        proba_map = {c: round(float(p), 4)
                     for c, p in zip(self.classifier.classes_, category_proba[0])}

        # ---- HARD RULE ----------------------------------------------------
        # Return here. Model 2 is not called. The LLM adapter is not called.
        # Nothing below this block executes for a fraud_block transaction.
        if category == FRAUD_CLASS:
            return self._record(
                transaction, category, proba_map,
                recovery_prob=None,
                action=Action.NO_ACTION,
                explanation=FRAUD_EXPLANATION,
                fraud_blocked=True,
            )
        # -------------------------------------------------------------------

        recovery_prob = float(
            self.recovery_model.predict_proba(df, category_proba)[0]
        )
        action = self.policy.decide_action(category, recovery_prob)

        explanation = ""
        if self.explain:
            explanation = self.adapter.generate_explanation(
                transaction, category, recovery_prob, action.value
            )

        return self._record(transaction, category, proba_map, recovery_prob,
                            action, explanation, fraud_blocked=False)

    def decide_batch(self, transactions: pd.DataFrame, log: bool = False,
                      explain: bool = False) -> pd.DataFrame:
        """
        Vectorized decisions for a batch. Same rules as decide(), but the models
        run once over the whole frame instead of per row.

        Fraud rows are split out BEFORE Model 2 runs, so the model never sees
        them here either.

        `log` appends to the audit feed. Off by default: a 1,600-row evaluation
        run has no reason to build an audit trail, but the API's demo seed does.

        `explain` narrates each logged decision through `self.adapter`. Only the
        narration loops -- the models still run vectorized. Off by default so
        evaluation runs stay fast.

        Which adapter narrates is the caller's choice, deliberately: the agent
        explains with whatever it was given, and the API hands the bulk path a
        TemplateAdapter so dashboard use cannot spend a live provider's quota.
        """
        df = transactions.reset_index(drop=True)
        category_proba = self.classifier.predict_proba(df)
        categories = self.classifier.predict(df)

        is_fraud = categories == FRAUD_CLASS
        recovery = np.full(len(df), np.nan)

        if (~is_fraud).any():
            non_fraud = df.loc[~is_fraud].reset_index(drop=True)
            recovery[~is_fraud] = self.recovery_model.predict_proba(
                non_fraud, category_proba[~is_fraud]
            )

        actions = [
            Action.NO_ACTION if fraud else self.policy.decide_action(cat, p)
            for fraud, cat, p in zip(is_fraud, categories, recovery)
        ]

        out = df.copy()
        out["predicted_category"] = categories
        out["recovery_prob"] = recovery
        out["action"] = [a.value for a in actions]
        out["fraud_blocked"] = is_fraud

        if log:
            for i, (_, row) in enumerate(df.iterrows()):
                txn = row.to_dict()

                if is_fraud[i]:
                    # Same wording and same hard rule as decide(). The adapter is
                    # not called -- fraud never reaches a prompt.
                    explanation = FRAUD_EXPLANATION
                elif explain:
                    explanation = self.adapter.generate_explanation(
                        txn, categories[i], float(recovery[i]), actions[i].value
                    )
                else:
                    explanation = ""

                self._record(
                    txn, categories[i],
                    {c: round(float(p), 4)
                     for c, p in zip(self.classifier.classes_, category_proba[i])},
                    None if is_fraud[i] else float(recovery[i]),
                    actions[i],
                    explanation=explanation,
                    fraud_blocked=bool(is_fraud[i]),
                )
        return out

    @staticmethod
    def _clean(value):
        """
        NaN -> None.

        Audit records are serialized to JSON, and NaN is not valid JSON. It shows
        up legitimately here: `error_code` is absent for transactions the customer
        abandoned before the bank ever saw them.
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return value

    def _record(self, transaction, category, proba_map, recovery_prob, action,
                 explanation, fraud_blocked) -> dict:
        record = {
            "transaction_id": self._clean(transaction.get("transaction_id")),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payment_method": self._clean(transaction.get("payment_method")),
            "amount": self._clean(transaction.get("amount")),
            "error_code": self._clean(transaction.get("error_code")),
            "error_code_meaning": describe(transaction.get("error_code"),
                                            transaction.get("payment_method")),
            "predicted_category": category,
            "category_probabilities": proba_map,
            "recovery_probability": (round(recovery_prob, 4)
                                      if recovery_prob is not None else None),
            "action": action.value,
            "fraud_blocked": bool(fraud_blocked),
            "explanation": explanation,
        }
        self.audit_log.append(record)
        return record
