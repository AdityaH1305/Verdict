"""
Uncertainty-aware decisions.

Two claims carry the whole design and are asserted first:

  1. The point estimate did not move. The interval is built from the members
     whose mean IS the point estimate, so calibration cannot have drifted.
  2. Passing no interval reproduces the original behaviour exactly, so anything
     that worked before still works.

The rest pin the decision rule and the row-alignment trap that made the first
version of the analysis wrong.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.recovery_agent import (  # noqa: E402
    Action, DecisionPolicy, RecoveryAgent,
)
from src.features import TARGET_RECOVERY  # noqa: E402
from src.models.failure_classifier import FailureClassifier  # noqa: E402
from src.models.recovery_success_model import RecoverySuccessModel  # noqa: E402
from src.paths import CLASSIFIER_PATH, TEST_DATA  # noqa: E402

requires_models = pytest.mark.skipif(
    not (os.path.exists(CLASSIFIER_PATH) and os.path.exists(TEST_DATA)),
    reason="run scripts/train.py first",
)

HIGH, LOW = DecisionPolicy.HIGH, DecisionPolicy.LOW
WIDE = DecisionPolicy.MIN_HEDGE_WIDTH
ACTING = {"auto_retry_now", "retry_later", "customer_nudge"}


@pytest.fixture(scope="module")
def decisions():
    agent = RecoveryAgent(explain=False)
    return agent, agent.decide_batch(pd.read_csv(TEST_DATA))


# --------------------------------------------------------------------------- #
# The two safety claims
# --------------------------------------------------------------------------- #

@requires_models
class TestNothingBroke:
    def test_point_estimate_is_the_mean_of_the_interval_members(self):
        """
        The calibration-safety argument, asserted rather than asserted-in-prose.

        CalibratedClassifierCV averages its fold models; the interval is the
        spread of those same models. So the point estimate cannot drift -- it is
        the identical arithmetic.
        """
        from src.models.recovery_success_model import drop_fraud_rows

        clf = FailureClassifier.load()
        model = RecoverySuccessModel.load()
        test = pd.read_csv(TEST_DATA)
        proba = clf.predict_proba(test)
        m2, m2_proba = drop_fraud_rows(test, proba)

        point = model.predict_proba(m2, m2_proba)
        members = model._member_probabilities(m2, m2_proba)

        assert members is not None and members.shape[1] >= 2
        assert np.allclose(point, members.mean(axis=1), atol=1e-12, rtol=0)

    def test_no_interval_matches_original_behaviour(self):
        """Omitting the interval must reproduce the pre-uncertainty rule exactly."""
        policy = DecisionPolicy()
        for category in ("hard_decline", "soft_decline", "customer_dropoff", "fraud_block"):
            for p in np.linspace(0.0, 1.0, 101):
                assert (policy.decide_action(category, p)
                        is policy._point_action(category, p))
                assert (policy.decide_action(category, p, lo=None, hi=None)
                        is policy._point_action(category, p))


# --------------------------------------------------------------------------- #
# The rule
# --------------------------------------------------------------------------- #

class TestDecisionRule:
    def setup_method(self):
        self.policy = DecisionPolicy()

    def test_wide_straddle_at_the_top_hedges_rather_than_withdrawing(self):
        """
        Measured, not assumed: transactions flagged here recover MORE often than
        confident ones, so escalating them would destroy revenue. Hedging to a
        backoff retry keeps them in the recovery path.
        """
        action = self.policy.decide_action(
            "soft_decline", 0.55, lo=HIGH - 0.15, hi=HIGH + 0.10)

        assert action is Action.RETRY_LATER
        assert action is not Action.ESCALATE, "hedging must not withdraw the retry"

    def test_narrow_straddle_at_the_top_is_left_alone(self):
        """80% of auto-retries straddle 0.50; only the genuinely wide ones hedge."""
        assert self.policy.decide_action(
            "soft_decline", 0.51, lo=0.49, hi=0.53) is Action.AUTO_RETRY_NOW

    def test_straddle_at_the_give_up_line_surfaces_for_review(self):
        assert self.policy.decide_action(
            "hard_decline", 0.20, lo=0.10, hi=0.35) is Action.ESCALATE

    def test_confident_give_up_stays_no_action(self):
        assert self.policy.decide_action(
            "hard_decline", 0.05, lo=0.03, hi=0.08) is Action.NO_ACTION

    def test_confident_auto_retry_stays_auto_retry(self):
        assert self.policy.decide_action(
            "soft_decline", 0.80, lo=0.72, hi=0.88) is Action.AUTO_RETRY_NOW

    def test_uncertainty_never_overrides_the_fraud_rule(self):
        """The hard rule outranks everything, at any interval."""
        for lo, hi in [(0.0, 1.0), (HIGH - 0.3, HIGH + 0.3), (LOW - 0.2, LOW + 0.2)]:
            assert self.policy.decide_action(
                "fraud_block", 0.9, lo=lo, hi=hi) is Action.NO_ACTION

    def test_mid_band_actions_are_untouched(self):
        """Only the two boundary rules exist; nudge/retry_later are not adjusted."""
        mid = (LOW + HIGH) / 2
        assert self.policy.decide_action(
            "customer_dropoff", mid, lo=0.0, hi=1.0) is Action.CUSTOMER_NUDGE

    def test_reason_is_reported_only_when_something_changed(self):
        _, unchanged = self.policy.apply_uncertainty(
            Action.AUTO_RETRY_NOW, "soft_decline", 0.72, 0.88)
        _, changed = self.policy.apply_uncertainty(
            Action.AUTO_RETRY_NOW, "soft_decline", HIGH - 0.15, HIGH + 0.10)

        assert unchanged is None
        assert changed == "uncertain_at_auto_retry_threshold"


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #

@requires_models
class TestEndToEnd:
    def test_intervals_bracket_the_point_estimate(self, decisions):
        _, d = decisions
        nf = d[~d["fraud_blocked"]]

        assert nf["recovery_lo"].notna().all()
        assert (nf["recovery_lo"] <= nf["recovery_prob"] + 1e-9).all()
        assert (nf["recovery_prob"] <= nf["recovery_hi"] + 1e-9).all()
        assert (nf["recovery_lo"] >= 0).all() and (nf["recovery_hi"] <= 1).all()

    def test_fraud_rows_get_no_interval(self, decisions):
        _, d = decisions
        fraud = d[d["fraud_blocked"]]

        assert len(fraud) > 0
        assert fraud["recovery_lo"].isna().all()
        assert fraud["recovery_hi"].isna().all()
        assert fraud["uncertainty_reason"].isna().all()
        assert (fraud["action"] == Action.NO_ACTION.value).all()

    def test_intervals_align_with_the_predicted_fraud_mask(self, decisions):
        """
        Regression on a trap that made the first analysis wrong.

        Intervals are computed on non-fraud rows and scattered back. The mask
        must be the PREDICTED fraud mask that decide_batch uses, not
        drop_fraud_rows()'s TRUE-label mask -- those differ (Model 1's fraud
        recall is 0.985), and mixing them attaches one transaction's uncertainty
        to another's decision without raising anything.
        """
        agent, d = decisions
        from src.models.recovery_success_model import drop_fraud_rows

        test = pd.read_csv(TEST_DATA)
        predicted_non_fraud = int((~d["fraud_blocked"]).sum())
        true_non_fraud = len(drop_fraud_rows(test)[0])

        # the two masks genuinely differ -- that is why this test exists
        assert predicted_non_fraud != true_non_fraud

        # every row with an interval is exactly a predicted-non-fraud row
        assert int(d["recovery_lo"].notna().sum()) == predicted_non_fraud

        # and re-deriving on the predicted mask reproduces the same values
        nf = test.loc[(~d["fraud_blocked"]).values].reset_index(drop=True)
        proba = agent.classifier.predict_proba(test)[(~d["fraud_blocked"]).values]
        lo, hi = agent.recovery_model.predict_interval(nf, proba)
        assert np.allclose(d.loc[~d["fraud_blocked"], "recovery_lo"].values, lo)
        assert np.allclose(d.loc[~d["fraud_blocked"], "recovery_hi"].values, hi)

    def test_uncertainty_is_revenue_neutral(self, decisions):
        """
        Both adjustments move between actions of the same kind: auto_retry_now
        and retry_later both act; no_action and escalate both do not. So the
        agent's recovered revenue is identical with the rule on or off.
        """
        agent, d = decisions
        point = [
            agent.policy._point_action(c, p).value if not f else Action.NO_ACTION.value
            for c, p, f in zip(d["predicted_category"], d["recovery_prob"], d["fraud_blocked"])
        ]
        before = pd.Series(point).isin(ACTING).values
        after = d["action"].isin(ACTING).values

        assert before.sum() == after.sum()
        assert (d.loc[before & d[TARGET_RECOVERY], "amount"].sum()
                == pytest.approx(d.loc[after & d[TARGET_RECOVERY], "amount"].sum(), abs=0.01))

    def test_the_rule_actually_fires_on_real_data(self, decisions):
        """A guardrail nobody triggers proves nothing."""
        _, d = decisions
        reasons = d["uncertainty_reason"].dropna()

        assert (reasons == "uncertain_at_auto_retry_threshold").sum() > 0
        assert (reasons == "uncertain_at_give_up_threshold").sum() > 0

    def test_hedging_stays_a_minority_of_auto_retries(self, decisions):
        """
        Without the width floor this hedged 80% of auto-retries and emptied the
        action out. Pin that it stays a minority.
        """
        agent, d = decisions
        point_auto = sum(
            1 for c, p, f in zip(d["predicted_category"], d["recovery_prob"], d["fraud_blocked"])
            if not f and agent.policy._point_action(c, p) is Action.AUTO_RETRY_NOW
        )
        hedged = int((d["uncertainty_reason"] == "uncertain_at_auto_retry_threshold").sum())

        assert 0 < hedged < point_auto * 0.5

    def test_surfaced_transactions_are_genuinely_more_recoverable(self, decisions):
        """
        The justification for the give-up rule, checked against ground truth
        rather than trusted: these recover several times more often than the
        confident give-ups they were separated from.
        """
        _, d = decisions
        surfaced = d[d["uncertainty_reason"] == "uncertain_at_give_up_threshold"]
        still_dropped = d[(d["action"] == Action.NO_ACTION.value) & ~d["fraud_blocked"]]

        assert len(surfaced) > 0
        assert surfaced[TARGET_RECOVERY].mean() > 2 * still_dropped[TARGET_RECOVERY].mean()
