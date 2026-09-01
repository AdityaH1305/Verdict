"""
Drift monitoring.

The claim this feature makes is "we would notice if the data moved". Two things
have to hold for that to mean anything, and both are asserted here:

  1. It fires on a real shift.
  2. It does NOT fire on sampling noise. A monitor that alarms on every 300-row
     batch is worse than no monitor, because people learn to ignore it.

The second nearly shipped broken: `error_code[upi]`, with 17 levels against ~135
UPI rows, had a MEDIAN PSI of 0.139 on same-distribution batches -- it would have
reported MODERATE half the time with no drift at all. The noise-floor tests below
exist to keep that fixed.

Also asserted: this layer is genuinely read-only and changes no decision.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.monitoring.baseline import (  # noqa: E402
    MIN_LEVEL_SHARE, build_baseline, load_baseline,
)
from src.monitoring.drift_detector import (  # noqa: E402
    MISSING_BUCKET, OTHER_BUCKET, PSI_MODERATE, PSI_STABLE, detect_drift,
    psi_from_proportions, psi_for_feature, verdict_for,
)
from src.paths import BASELINE_PATH, ROOT, TEST_DATA, TRAIN_DATA  # noqa: E402

DEMO_CSV = os.path.join(ROOT, "data", "demo", "demo_batch.csv")
DRIFT_CSV = os.path.join(ROOT, "data", "demo", "drift_demo_batch.csv")

requires_baseline = pytest.mark.skipif(
    not (os.path.exists(BASELINE_PATH) and os.path.exists(TRAIN_DATA)),
    reason="run scripts/train.py first",
)
requires_drift_demo = pytest.mark.skipif(
    not os.path.exists(DRIFT_CSV),
    reason="run scripts/make_drift_demo.py first",
)


@pytest.fixture(scope="module")
def baseline():
    return load_baseline(BASELINE_PATH)


# --------------------------------------------------------------------------- #
# PSI itself
# --------------------------------------------------------------------------- #

class TestPSI:
    def test_identical_distributions_score_zero(self):
        p = np.array([0.5, 0.3, 0.2])
        assert psi_from_proportions(p, p) == pytest.approx(0.0, abs=1e-12)

    def test_psi_is_non_negative(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            e = rng.dirichlet(np.ones(6))
            a = rng.dirichlet(np.ones(6))
            assert psi_from_proportions(e, a) >= 0

    def test_bigger_shift_scores_higher(self):
        e = np.array([0.5, 0.5])
        assert (psi_from_proportions(e, np.array([0.6, 0.4]))
                < psi_from_proportions(e, np.array([0.9, 0.1])))

    def test_empty_bin_stays_finite(self):
        """
        Guaranteed with small batches: a 300-row batch cannot contain all 23
        error codes. An empty bin must score high, not inf or nan.
        """
        psi = psi_from_proportions(np.array([0.5, 0.5]), np.array([1.0, 0.0]))
        assert np.isfinite(psi)
        assert psi > PSI_MODERATE

    def test_verdict_boundaries_are_the_conventional_cutoffs(self):
        assert verdict_for(PSI_STABLE - 1e-9) == "LOW"
        assert verdict_for(PSI_STABLE) == "MODERATE"
        assert verdict_for(PSI_MODERATE - 1e-9) == "MODERATE"
        assert verdict_for(PSI_MODERATE) == "HIGH"


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #

@requires_baseline
class TestBaseline:
    def test_is_deterministic(self):
        train = pd.read_csv(TRAIN_DATA)
        assert build_baseline(train) == build_baseline(train)

    def test_monitors_the_intended_features(self, baseline):
        assert set(baseline["features"]) == {
            "payment_method", "failure_category",
            "error_code[card]", "error_code[upi]",
            "amount", "seconds_to_failure",
        }

    def test_error_code_is_monitored_per_payment_method(self, baseline):
        """
        Otherwise a shift in the card/UPI mix would masquerade as a shift in the
        code distribution.
        """
        assert baseline["features"]["error_code[card]"]["payment_method"] == "card"
        assert baseline["features"]["error_code[upi]"]["payment_method"] == "upi"

    def test_missing_error_codes_are_a_real_bucket(self, baseline):
        """
        ~11% of rows have no error code BY DESIGN -- a customer who abandons
        never reaches the bank. Dropping nulls would hide a shift in abandonment.
        """
        levels = baseline["features"]["error_code[upi]"]["levels"]
        assert MISSING_BUCKET in levels

    def test_rare_levels_are_merged(self, baseline):
        spec = baseline["features"]["error_code[upi]"]
        assert OTHER_BUCKET in spec["levels"]
        assert spec["merged_rare_levels"]
        assert all(p >= MIN_LEVEL_SHARE
                   for level, p in zip(spec["levels"], spec["proportions"])
                   if level != OTHER_BUCKET)

    def test_proportions_sum_to_one(self, baseline):
        for name, spec in baseline["features"].items():
            assert sum(spec["proportions"]) == pytest.approx(1.0, abs=1e-3), name

    def test_records_the_label_drift_caveat(self, baseline):
        """failure_category is the label; that limitation must travel with the data."""
        assert "failure_category" in baseline["label_drift_features"]
        assert "production" in baseline["label_drift_note"]


# --------------------------------------------------------------------------- #
# Directional sanity
# --------------------------------------------------------------------------- #

@requires_baseline
class TestDirectionalSanity:
    def test_training_data_against_itself_is_zero(self, baseline):
        """The floor. If this is not ~0, the metric is broken."""
        result = detect_drift(baseline, pd.read_csv(TRAIN_DATA))
        assert result["max_psi"] == pytest.approx(0.0, abs=1e-6)
        assert result["overall_verdict"] == "LOW"

    def test_noise_floor_stays_below_the_stable_cutoff(self, baseline):
        """
        Same-distribution batches must not alarm.

        This is the test that would have caught the near-miss: before rare-level
        consolidation, error_code[upi] scored a median 0.139 here.
        """
        pool = pd.read_csv(TEST_DATA)
        for name, spec in baseline["features"].items():
            scores = [psi_for_feature(spec, pool.sample(n=300, random_state=s))["psi"]
                      for s in range(25)]
            assert np.median(scores) < PSI_STABLE, (
                f"{name} has a median PSI of {np.median(scores):.3f} on batches "
                f"drawn from the SAME distribution -- it is measuring sample "
                f"size, not drift"
            )

    @pytest.mark.skipif(not os.path.exists(DEMO_CSV), reason="no pinned batch")
    def test_representative_batch_reads_low(self, baseline):
        result = detect_drift(baseline, pd.read_csv(DEMO_CSV))
        assert result["overall_verdict"] == "LOW"

    @requires_drift_demo
    def test_constructed_shift_is_detected(self, baseline):
        result = detect_drift(baseline, pd.read_csv(DRIFT_CSV))
        assert result["overall_verdict"] in ("MODERATE", "HIGH")
        assert result["max_psi"] >= PSI_MODERATE

    @requires_drift_demo
    @pytest.mark.skipif(not os.path.exists(DEMO_CSV), reason="no pinned batch")
    def test_shifted_batch_scores_far_above_the_representative_one(self, baseline):
        """The separation is what makes the monitor useful, not the absolute number."""
        low = detect_drift(baseline, pd.read_csv(DEMO_CSV))["max_psi"]
        high = detect_drift(baseline, pd.read_csv(DRIFT_CSV))["max_psi"]
        assert high > 3 * low

    @requires_drift_demo
    def test_verdict_names_a_driver(self, baseline):
        result = detect_drift(baseline, pd.read_csv(DRIFT_CSV))
        assert result["driver"] in result["features"]
        assert result["driver_detail"]

    def test_unseen_levels_land_in_other_rather_than_vanishing(self, baseline):
        """A brand-new error code is a shift, and must register as one."""
        batch = pd.read_csv(TEST_DATA).sample(n=200, random_state=1).copy()
        batch.loc[batch["payment_method"] == "upi", "error_code"] = "NEW_CODE_XX"

        result = psi_for_feature(baseline["features"]["error_code[upi]"], batch)
        assert result["psi"] > PSI_MODERATE


# --------------------------------------------------------------------------- #
# Read-only
# --------------------------------------------------------------------------- #

@requires_baseline
class TestReadOnly:
    def test_drift_does_not_change_any_decision(self):
        """
        The whole feature is supposed to observe and report. Prove decisions are
        byte-identical whether or not drift has been computed over the batch.
        """
        from src.agent.recovery_agent import RecoveryAgent

        batch = pd.read_csv(TEST_DATA).sample(n=200, random_state=3)
        agent = RecoveryAgent(explain=False)

        before = agent.decide_batch(batch)
        detect_drift(load_baseline(BASELINE_PATH), batch)
        after = agent.decide_batch(batch)

        pd.testing.assert_frame_equal(before, after)

    def test_detector_does_not_mutate_the_batch(self, baseline):
        batch = pd.read_csv(TEST_DATA).sample(n=150, random_state=4)
        snapshot = batch.copy(deep=True)

        detect_drift(baseline, batch)

        pd.testing.assert_frame_equal(batch, snapshot)

    def test_monitoring_imports_nothing_from_the_decision_path(self):
        """
        Structural guard on the dependency direction: monitoring may read models
        and data, but the agent must never import monitoring, or "read-only"
        stops being enforceable.
        """
        agent_src = os.path.join(ROOT, "src", "agent")
        for name in os.listdir(agent_src):
            if not name.endswith(".py"):
                continue
            with open(os.path.join(agent_src, name), encoding="utf-8") as f:
                assert "monitoring" not in f.read(), (
                    f"src/agent/{name} imports monitoring -- the decision path "
                    f"must not depend on the observer"
                )


# --------------------------------------------------------------------------- #
# The constructed scenario is labelled as constructed
# --------------------------------------------------------------------------- #

@requires_drift_demo
def test_shifted_batch_is_documented_as_constructed():
    """
    It must never read as drift observed in production. This is an honesty
    guard, and it is cheap to keep.
    """
    meta_path = os.path.join(ROOT, "data", "demo", "drift_demo_batch.json")
    with open(meta_path) as f:
        meta = json.load(f)

    blob = json.dumps(meta).lower()
    assert "constructed" in blob
    assert "deliberately" in blob
    assert "not observed" in blob or "was not observed" in blob
