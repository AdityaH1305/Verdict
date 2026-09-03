"""
Guard tests for the agent decision layer.

The headline claim of this project is that the fraud block is ARCHITECTURAL --
Model 2 and the LLM are not merely ignored for fraud, they are never reached.
The spy objects below make that testable: they raise if anything touches them.
A post-hoc filter would pass a "the action was no_action" assertion but fail
these.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.llm_adapter import TemplateAdapter, _template_explanation  # noqa: E402
from src.agent.recovery_agent import Action, DecisionPolicy, RecoveryAgent  # noqa: E402
from src.agent.retry_simulator import (  # noqa: E402
    RETRYABLE_ACTIONS, run_capped, run_uncapped,
)
from src.error_taxonomy import describe, lookup  # noqa: E402
from src.features import TARGET_CATEGORY, TARGET_RECOVERY  # noqa: E402
from src.paths import CLASSIFIER_PATH, TEST_DATA  # noqa: E402

requires_models = pytest.mark.skipif(
    not (os.path.exists(CLASSIFIER_PATH) and os.path.exists(TEST_DATA)),
    reason="run scripts/train.py first",
)


class ExplodingRecoveryModel:
    """
    Model 2 stand-in that fails the test if it is ever consulted.

    Traps EVERY attribute rather than naming the methods the agent happens to
    call today. Naming them meant the guard depended on call order -- it only
    tripped because predict_proba was reached first -- so a refactor that changed
    which method the agent asks for first would have raised AttributeError and
    quietly stopped testing the hard rule.
    """

    def __getattr__(self, name):
        def refuse(*args, **kwargs):
            raise AssertionError(
                f"Model 2 was called ({name}) for a fraud_block transaction. The "
                f"recovery model must never be invoked for fraud -- see "
                f"docs/decisions.md, 'Hard rules'."
            )
        return refuse


class ExplodingAdapter:
    """LLM adapter stand-in that fails the test if it is ever consulted."""

    def generate_explanation(self, *args, **kwargs):
        raise AssertionError(
            "The LLM adapter was called for a fraud_block transaction. No prompt "
            "may ever see a fraud case."
        )


# --------------------------------------------------------------------------- #
# Decision policy (no models needed)
# --------------------------------------------------------------------------- #

class TestDecisionPolicy:
    def setup_method(self):
        self.policy = DecisionPolicy()

    def test_fraud_is_never_actioned_at_any_probability(self):
        for p in (0.0, 0.25, 0.5, 0.99, 1.0):
            assert self.policy.decide_action("fraud_block", p) == Action.NO_ACTION

    def test_high_threshold_is_inclusive(self):
        assert self.policy.decide_action("soft_decline", DecisionPolicy.HIGH) == Action.AUTO_RETRY_NOW
        # just below the cutoff must not auto-retry
        assert self.policy.decide_action("soft_decline", DecisionPolicy.HIGH - 1e-9) == Action.RETRY_LATER

    def test_low_threshold_is_inclusive(self):
        assert self.policy.decide_action("soft_decline", DecisionPolicy.LOW) == Action.RETRY_LATER
        assert self.policy.decide_action("soft_decline", DecisionPolicy.LOW - 1e-9) == Action.ESCALATE

    def test_mid_band_splits_on_category(self):
        mid = (DecisionPolicy.LOW + DecisionPolicy.HIGH) / 2
        # a human who walked away needs prompting, not a silent retry
        assert self.policy.decide_action("customer_dropoff", mid) == Action.CUSTOMER_NUDGE
        assert self.policy.decide_action("soft_decline", mid) == Action.RETRY_LATER

    def test_unrecoverable_hard_decline_is_not_escalated(self):
        # escalating an unrecoverable decline just makes work for a human who
        # cannot do anything about it either
        assert self.policy.decide_action("hard_decline", 0.01) == Action.NO_ACTION
        assert self.policy.decide_action("soft_decline", 0.01) == Action.ESCALATE


# --------------------------------------------------------------------------- #
# The fraud hard block, end to end
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def parts():
    """Loaded once: both artifacts plus one classifier pass over the test set."""
    agent = RecoveryAgent(adapter=TemplateAdapter())
    df = pd.read_csv(TEST_DATA)
    return agent, df, agent.classifier.predict_proba(df)


@requires_models
class TestFastPathsAreBitExact:
    """
    The serving path computes the class label and the recovery interval from
    work it has already done, instead of re-running each model a second time.
    That is only legitimate if it is the SAME number, so these compare exactly
    -- no tolerance -- over the whole held-out set.
    """

    def test_labels_from_proba_match_a_second_model_pass(self, parts):
        agent, df, proba = parts
        assert (agent.classifier.predict_from_proba(proba)
                == agent.classifier.predict(df)).all()

    def test_point_and_interval_from_one_pass_match_two(self, parts):
        agent, df, proba = parts
        labels = agent.classifier.predict_from_proba(proba)
        non_fraud = df.loc[labels != "fraud_block"].reset_index(drop=True)
        nf_proba = proba[labels != "fraud_block"]

        point, lo, hi = agent.recovery_model.predict_with_interval(non_fraud, nf_proba)
        was_point = agent.recovery_model.predict_proba(non_fraud, nf_proba)
        was_lo, was_hi = agent.recovery_model.predict_interval(non_fraud, nf_proba)

        # Exact, not approx: the point estimate is the mean of the same members
        # the interval is taken from, so any drift here is a real defect.
        assert (point == was_point).all()
        assert (lo == was_lo).all()
        assert (hi == was_hi).all()


@requires_models
class TestFraudHardBlock:
    def _fraud_agent(self):
        """Agent whose Model 2 and adapter both explode on contact."""
        from src.models.failure_classifier import FailureClassifier
        return RecoveryAgent(
            classifier=FailureClassifier.load(),
            recovery_model=ExplodingRecoveryModel(),
            adapter=ExplodingAdapter(),
        )

    def _a_transaction_predicted_fraud(self):
        df = pd.read_csv(TEST_DATA)
        from src.models.failure_classifier import FailureClassifier
        clf = FailureClassifier.load()
        preds = clf.predict(df)
        fraud_idx = [i for i, p in enumerate(preds) if p == "fraud_block"]
        assert fraud_idx, "expected at least one transaction predicted as fraud"
        return df.iloc[fraud_idx[0]].to_dict()

    def test_fraud_never_reaches_model_2_or_the_llm(self):
        """
        The core architectural claim. If either component is reached, the spies
        raise and this test fails.
        """
        decision = self._fraud_agent().decide(self._a_transaction_predicted_fraud())

        assert decision["action"] == Action.NO_ACTION.value
        assert decision["fraud_blocked"] is True
        # no estimate was produced, rather than produced and discarded
        assert decision["recovery_probability"] is None

    def test_non_fraud_does_reach_model_2(self):
        """
        Control for the test above: prove the spies would actually fire.

        Without this, test_fraud_never_reaches... could pass simply because the
        agent never calls Model 2 for anything.
        """
        df = pd.read_csv(TEST_DATA)
        from src.models.failure_classifier import FailureClassifier
        clf = FailureClassifier.load()
        preds = clf.predict(df)
        non_fraud_idx = [i for i, p in enumerate(preds) if p != "fraud_block"][0]

        with pytest.raises(AssertionError, match="Model 2 was called"):
            self._fraud_agent().decide(df.iloc[non_fraud_idx].to_dict())

    def test_adapters_refuse_fraud_directly(self):
        """Third guard: the adapters themselves refuse, independent of the agent."""
        with pytest.raises(AssertionError, match="fraud_block"):
            TemplateAdapter().generate_explanation({}, "fraud_block", 0.9, "auto_retry_now")

    def test_batch_path_also_blocks_fraud(self):
        agent = RecoveryAgent(explain=False)
        decisions = agent.decide_batch(pd.read_csv(TEST_DATA))
        flagged = decisions[decisions["fraud_blocked"]]

        assert len(flagged) > 0
        assert (flagged["action"] == Action.NO_ACTION.value).all()
        # no recovery estimate exists for any fraud-flagged transaction
        assert flagged["recovery_prob"].isna().all()


# --------------------------------------------------------------------------- #
# Retry budget
# --------------------------------------------------------------------------- #

@requires_models
class TestRetryBudget:
    def setup_method(self):
        agent = RecoveryAgent(explain=False)
        self.decisions = agent.decide_batch(pd.read_csv(TEST_DATA))

    def test_capped_run_never_exceeds_the_budget(self):
        budget = 3
        stats = run_capped(self.decisions, cycles=12, budget=budget)
        assert stats.attempts_per_txn, "expected some retries to have happened"
        assert max(stats.attempts_per_txn.values()) <= budget

    def test_uncapped_run_demonstrably_storms(self):
        """The bug must actually reproduce, or the failure story is fiction."""
        cycles = 12
        uncapped = run_uncapped(self.decisions, cycles=cycles)
        capped = run_capped(self.decisions, cycles=cycles, budget=3)

        assert max(uncapped.attempts_per_txn.values()) > 3
        assert uncapped.total_attempts > capped.total_attempts
        # the storm falls on transactions that can never succeed
        assert uncapped.wasted_attempts / uncapped.total_attempts > 0.5

    def test_capped_run_terminates(self):
        stats = run_capped(self.decisions, cycles=12, budget=3)
        assert stats.still_queued_at_end == 0

    def test_only_retry_actions_are_queued(self):
        stats = run_uncapped(self.decisions, cycles=1)
        expected = int(self.decisions["action"].isin(RETRYABLE_ACTIONS).sum())
        assert stats.queued_transactions == expected


# --------------------------------------------------------------------------- #
# Grounded explanations
# --------------------------------------------------------------------------- #

class TestGrounding:
    def test_known_codes_resolve_to_verified_meanings(self):
        assert "Insufficient funds" in describe("51", "card")
        assert "Invalid virtual payment address" in describe("ZH", "upi")

    def test_shared_code_resolves_per_payment_method(self):
        # 59 exists on both rails with different wording
        assert lookup("59", "card").method == "card"
        assert lookup("59", "upi").method == "upi"

    def test_unknown_code_refuses_to_guess(self):
        text = describe("NOT_A_REAL_CODE", "card")
        assert "Do not speculate" in text

    def test_missing_code_is_described_as_abandonment(self):
        assert "left the payment flow" in describe(None, "upi")

    def test_template_explanation_quotes_the_real_meaning(self):
        txn = {"payment_method": "card", "error_code": "54", "amount": 100, "retry_count": 0}
        text = _template_explanation(txn, "hard_decline", 0.05, "no_action")
        assert "Expired card" in text
        assert "54" in text
