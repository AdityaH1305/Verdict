"""
Tests for the offline policy comparison.

Two things are being defended here.

The SANITY checks are arithmetic that must hold for the simulation to mean
anything: retrying nothing must cost nothing and recover nothing, and retrying
everything must recover every recoverable payment. If either fails, the scorer
and the outcome label have come apart and every number in the report is wrong.

The LEAKAGE checks are the load-bearing ones. A policy comparison where a policy
can see the outcome it is being scored on is worthless, and the failure is silent
-- the numbers still look plausible. So it is asserted twice, structurally and
behaviourally, and the behavioural one would catch a leak the structural one
cannot: flip every outcome label and demand that not a single decision moves.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.policy_eval import (  # noqa: E402
    ACTING, COST_PER_NUDGE_EXTRA, COST_PER_RETRY, LABEL_COLUMNS, _blind,
    build_policies, crossover, net_at, score,
)
from src.agent.llm_adapter import TemplateAdapter  # noqa: E402
from src.agent.recovery_agent import RecoveryAgent  # noqa: E402
from src.api import main as api_main  # noqa: E402
from src.paths import CLASSIFIER_PATH, TEST_DATA  # noqa: E402

requires_models = pytest.mark.skipif(
    not (os.path.exists(CLASSIFIER_PATH) and os.path.exists(TEST_DATA)),
    reason="run scripts/train.py first",
)


@pytest.fixture(scope="module")
def setup():
    """One agent and one pass over the test set, shared by every test here."""
    df = pd.read_csv(TEST_DATA)
    agent = RecoveryAgent(adapter=TemplateAdapter(), explain=False)
    policies = build_policies(agent)
    blind = _blind(df)
    masks = {key: fn(blind) for key, _, fn in policies}
    return df, agent, policies, blind, masks


@pytest.fixture(scope="module")
def client():
    with TestClient(api_main.app) as c:
        yield c


@requires_models
class TestNoLabelLeakage:
    def test_policies_never_receive_the_outcome_label(self, setup):
        """
        Structural: the columns are not withheld by convention, they are absent.

        A policy that reached for retry_success or failure_category would raise
        a KeyError here rather than quietly working.
        """
        df, _, policies, blind, _ = setup

        for column in LABEL_COLUMNS:
            assert column in df.columns, f"{column} should exist in the raw split"
            assert column not in blind.columns, f"{column} reached a policy"

        for key, _, fn in policies:
            mask = fn(blind)
            assert len(mask) == len(df), f"{key} returned a mask of the wrong length"
            assert mask.dtype == bool, f"{key} returned a non-boolean mask"

    def test_flipping_every_outcome_changes_no_decision(self, setup):
        """
        Behavioural, and the stronger of the two.

        Each policy is handed the RAW frame -- outcome label present -- and then
        the same frame with every outcome inverted. The two masks must be
        bit-identical.

        Deliberately NOT blinded, and that is the whole point. Blinding is the
        pipeline's protection and is asserted separately above; this test exists
        to catch a policy that WOULD misuse the label if it ever saw one, which
        is what a future refactor forgetting to blind would expose. An earlier
        version of this test passed the blinded frame, which made it vacuous: it
        would have passed for a policy that returned the outcome label verbatim.
        """
        df, _, policies, _, _ = setup

        flipped = df.copy()
        flipped["retry_success"] = ~df["retry_success"].values.astype(bool)
        assert not (flipped["retry_success"].values
                    == df["retry_success"].values).all(), "the flip did nothing"
        assert "retry_success" in df.columns, "the label must be visible to this test"

        for key, _, fn in policies:
            assert (fn(df) == fn(flipped)).all(), (
                f"policy '{key}' decided differently once the outcome label was "
                f"flipped, so it is reading the label it is scored against"
            )

    def test_the_pipeline_blinds_before_it_asks(self, setup):
        """
        Belt to the braces above: main() must call policies with the blinded
        frame, so even a policy that would misuse the label never receives one.
        """
        df, _, policies, blind, _ = setup

        for column in LABEL_COLUMNS:
            assert column not in blind.columns
        # and the blinded frame is otherwise intact
        assert len(blind) == len(df)
        assert len(blind.columns) == len(df.columns) - len(LABEL_COLUMNS)

    def test_scoring_is_the_only_consumer_of_the_label(self, setup):
        """
        The label changes the SCORE -- that much must be true, or the flip test
        above would pass for the trivial reason that nothing reads it at all.
        """
        df, _, _, _, masks = setup
        amount = df["amount"].values.astype(float)
        won = df["retry_success"].values.astype(bool)
        zero = np.zeros(len(df), dtype=bool)

        real = score("always_retry", "x", masks["always_retry"], amount, won, zero)
        inverted = score("always_retry", "x", masks["always_retry"], amount, ~won, zero)

        assert real["recovered"] != inverted["recovered"]


@requires_models
class TestSanity:
    def test_never_retry_costs_nothing_and_recovers_nothing(self, setup):
        df, _, _, _, masks = setup
        amount = df["amount"].values.astype(float)
        won = df["retry_success"].values.astype(bool)
        zero = np.zeros(len(df), dtype=bool)

        r = score("never_retry", "x", masks["never_retry"], amount, won, zero)

        assert r["attempts"] == 0
        assert r["recovered"] == 0
        assert r["recovered_revenue"] == 0
        assert r["retry_cost"] == 0
        assert r["net_value"] == 0

    def test_always_retry_recovers_the_most(self, setup):
        """
        Retrying everything is a superset of every other policy's attempts, so
        it cannot recover less than any of them.
        """
        df, _, _, _, masks = setup
        amount = df["amount"].values.astype(float)
        won = df["retry_success"].values.astype(bool)
        zero = np.zeros(len(df), dtype=bool)

        scored = {k: score(k, "x", m, amount, won, zero) for k, m in masks.items()}
        best = max(s["recovered"] for s in scored.values())

        assert scored["always_retry"]["recovered"] == best
        # and specifically: it recovers every recoverable payment there is
        assert scored["always_retry"]["recovered"] == int(won.sum())
        assert scored["always_retry"]["recovered_revenue"] == pytest.approx(
            float(amount[won].sum()), abs=0.01)

    def test_recovered_never_exceeds_attempts(self, setup):
        df, _, _, _, masks = setup
        amount = df["amount"].values.astype(float)
        won = df["retry_success"].values.astype(bool)
        zero = np.zeros(len(df), dtype=bool)

        for key, mask in masks.items():
            r = score(key, "x", mask, amount, won, zero)
            assert r["attempts"] == int(mask.sum())
            assert r["recovered"] <= r["attempts"], key
            assert r["wasted_retries"] == r["attempts"] - r["recovered"]

    def test_fraud_is_never_retried_by_verdict(self, setup):
        """
        The architectural rule, seen through this simulation: retrying everything
        spends attempts on fraud, and Verdict spends none.
        """
        df, agent, _, blind, masks = setup
        is_fraud = agent.decide_batch(blind)["fraud_blocked"].values

        assert is_fraud.sum() > 0, "expected some fraud in the test set"
        assert not (masks["verdict"] & is_fraud).any()
        assert (masks["always_retry"] & is_fraud).sum() == is_fraud.sum()


@requires_models
class TestVerdictPolicyIsTheLiveAgent:
    def test_it_reads_the_live_decision_and_does_not_restate_it(self, setup):
        """
        Policy D must BE the agent's decision, not a copy of its thresholds. If
        someone changes DecisionPolicy.HIGH, this simulation has to move with it.
        """
        df, agent, _, blind, masks = setup
        live = agent.decide_batch(blind)["action"].isin(ACTING).values

        assert (masks["verdict"] == live).all()

    def test_the_acting_set_matches_what_the_api_counts(self, setup):
        """
        The report and /stats/recovered-revenue must agree on what "acted on"
        means, or the dashboard would show two different numbers for one idea.
        Compared through behaviour rather than by matching string literals.
        """
        df, _, _, _, masks = setup

        # STATE is cleared when the app's lifespan exits, so the comparison
        # happens inside the client context rather than after it.
        with TestClient(api_main.app) as client:
            client.post("/simulate/seed?n=200&seed=7")
            stats = client.get("/stats/recovered-revenue?n=200").json()
            batch = api_main.STATE["batch"]
            agent = api_main.STATE["bulk_agent"]
            acted = int(agent.decide_batch(batch)["action"].isin(ACTING).sum())

        assert acted == stats["acted_on_count"]


@requires_models
class TestSensitivity:
    def test_net_value_falls_as_retries_get_more_expensive(self, setup):
        df, _, _, _, masks = setup
        amount = df["amount"].values.astype(float)
        won = df["retry_success"].values.astype(bool)
        zero = np.zeros(len(df), dtype=bool)
        verdict = score("verdict", "x", masks["verdict"], amount, won, zero)

        assert net_at(verdict, 0.0) > net_at(verdict, 50.0) > net_at(verdict, 500.0)

    def test_crossover_is_where_the_two_curves_actually_meet(self, setup):
        """The crossover is solved, not searched, so it should be exact."""
        df, _, _, _, masks = setup
        amount = df["amount"].values.astype(float)
        won = df["retry_success"].values.astype(bool)
        zero = np.zeros(len(df), dtype=bool)

        verdict = score("verdict", "x", masks["verdict"], amount, won, zero)
        always = score("always_retry", "x", masks["always_retry"], amount, won, zero)

        c = crossover(verdict, always)
        assert c is not None
        # c is reported to the paisa, and each paisa of rounding moves the two
        # curves apart by the difference in their attempt counts. Tolerating a
        # flat rupee here would be asserting a precision the rounding forbids.
        tolerance = abs(verdict["attempts"] - always["attempts"]) * 0.005 + 0.01
        assert net_at(verdict, c) == pytest.approx(net_at(always, c), abs=tolerance)
        # and the ordering genuinely swaps either side of it
        assert net_at(always, c - 5) > net_at(verdict, c - 5)
        assert net_at(verdict, c + 5) > net_at(always, c + 5)

    def test_a_policy_never_crosses_itself(self, setup):
        df, _, _, _, masks = setup
        amount = df["amount"].values.astype(float)
        won = df["retry_success"].values.astype(bool)
        zero = np.zeros(len(df), dtype=bool)
        v = score("verdict", "x", masks["verdict"], amount, won, zero)

        assert crossover(v, v) is None


@requires_models
class TestReportEndpoint:
    def test_policy_eval_report_is_served(self, client):
        r = client.get("/reports/policy-eval")
        if r.status_code == 404:
            pytest.skip("run scripts/policy_eval.py first")

        body = r.json()
        assert {p["key"] for p in body["policies"]} == {
            "always_retry", "never_retry", "fixed_rule", "verdict"}
        assert body["sweep"] and body["crossovers"]
        assert body["assumptions"]["cost_per_retry"] == COST_PER_RETRY
        assert body["assumptions"]["cost_per_nudge_extra"] == COST_PER_NUDGE_EXTRA
        # the limitation has to travel with the numbers
        assert "counterfactual_note" in body["assumptions"]

    def test_policy_eval_404s_cleanly_when_missing(self, client, monkeypatch):
        monkeypatch.setattr(api_main, "POLICY_EVAL_REPORT", "/nonexistent/policy_eval.json")
        r = client.get("/reports/policy-eval")

        assert r.status_code == 404
        assert "policy_eval.py" in r.json()["detail"]
