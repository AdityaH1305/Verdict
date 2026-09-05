"""
Tests for the endpoints the dashboard depends on.

The load-bearing one is `test_bulk_path_never_calls_a_live_provider`: the whole
quota strategy rests on ordinary dashboard use being served by the offline
template adapter, with live calls confined to a human clicking "explain live".
That is a claim about wiring, so it is asserted with a spy that raises on
contact rather than trusted.
"""

import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.llm_adapter import TemplateAdapter  # noqa: E402
from src.api import main as api_main  # noqa: E402
from src.paths import CLASSIFIER_PATH, TEST_DATA  # noqa: E402

requires_models = pytest.mark.skipif(
    not (os.path.exists(CLASSIFIER_PATH) and os.path.exists(TEST_DATA)),
    reason="run scripts/train.py first",
)


class ExplodingAdapter:
    """Fails the test if the dashboard's bulk path ever reaches a live provider."""

    provider = "exploding"
    model = "none"
    is_live = True

    def describe_self(self):
        return {"adapter": "ExplodingAdapter", "provider": self.provider,
                "model": self.model, "live": True}

    def generate_explanation(self, *args, **kwargs):
        raise AssertionError(
            "A looping dashboard endpoint called the live adapter. Bulk work must "
            "use TemplateAdapter -- see the bulk_agent wiring in src/api/main.py."
        )


@pytest.fixture(scope="module")
def client():
    """
    Module-scoped: the app's lifespan loads both XGBoost artifacts, which cost
    ~3s. Per-test that dominated the suite. Nothing here mutates shared state
    without restoring it, and every test seeds its own batch.
    """
    with TestClient(api_main.app) as c:
        yield c


@requires_models
class TestQuotaSafety:
    def test_bulk_agent_uses_the_template_adapter(self, client):
        assert isinstance(api_main.STATE["bulk_agent"].adapter, TemplateAdapter)

    def test_bulk_and_live_agents_share_loaded_models(self, client):
        """The second agent must not reload or duplicate the model artifacts."""
        live, bulk = api_main.STATE["agent"], api_main.STATE["bulk_agent"]
        assert live.classifier is bulk.classifier
        assert live.recovery_model is bulk.recovery_model

    def test_bulk_path_never_calls_a_live_provider(self, client):
        """A full dashboard load must make zero provider calls."""
        original = api_main.STATE["agent"].adapter
        api_main.STATE["agent"].adapter = ExplodingAdapter()
        try:
            assert client.post("/simulate/seed?n=25&seed=1").status_code == 200
            assert client.get("/stats/breakdown?n=25").status_code == 200
            assert client.get("/stats/recovered-revenue?n=25").status_code == 200
            assert client.get("/decisions?limit=25").status_code == 200
        finally:
            api_main.STATE["agent"].adapter = original


@requires_models
class TestLoadCost:
    """
    The load path must put the batch through the models exactly once.

    This is the regression that matters for page speed: the dashboard used to
    seed a batch and then have each stats panel re-decide the identical rows,
    which tripled the model work for numbers that could not possibly differ.
    Counted rather than timed, so it fails deterministically the moment a second
    pass is reintroduced.
    """

    def test_a_full_dashboard_load_decides_the_batch_once(self, client):
        bulk = api_main.STATE["bulk_agent"]
        original = bulk.decide_batch
        calls = []

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        bulk.decide_batch = counting
        try:
            # exactly what the page does on load, in order
            assert client.post("/simulate/seed?n=50&seed=3").status_code == 200
            assert client.get("/stats/breakdown?n=50").status_code == 200
            assert client.get("/stats/recovered-revenue?n=50").status_code == 200
            assert client.get("/decisions?limit=50").status_code == 200
            assert client.get("/stats/drift?n=50").status_code in (200, 503)
        finally:
            bulk.decide_batch = original

        assert len(calls) == 1, (
            f"a dashboard load ran the models {len(calls)} times over one batch; "
            f"the stats panels must read the memo in _decided(), not re-decide."
        )

    def test_a_new_batch_invalidates_the_memo(self, client):
        """Reuse must never outlive the rows it describes."""
        client.post("/simulate/seed?n=40&seed=11")
        first = client.get("/stats/recovered-revenue?n=40").json()
        client.post("/simulate/seed?n=40&seed=12")
        second = client.get("/stats/recovered-revenue?n=40").json()

        assert first["total_failed_value"] != second["total_failed_value"], (
            "the stats panel served a stale batch after a new seed"
        )


@requires_models
class TestDecisionFeed:
    def test_explanations_are_populated(self, client):
        """
        Regression: the batch path used to log explanation="" for every row,
        leaving the dashboard's "why" panel empty.
        """
        client.post("/simulate/seed?n=25&seed=1")
        decisions = client.get("/decisions?limit=25").json()["decisions"]

        assert decisions
        assert all(d["explanation"].strip() for d in decisions)

    def test_explanations_are_grounded_in_the_taxonomy(self, client):
        client.post("/simulate/seed?n=40&seed=1")
        decisions = client.get("/decisions?limit=40").json()["decisions"]

        coded = [d for d in decisions
                 if d["error_code"] and not d["fraud_blocked"]]
        assert coded, "expected some transactions with an error code"
        # the explanation cites the real code, not an invented reason
        assert all(str(d["error_code"]) in d["explanation"] for d in coded)

    def test_fraud_rows_carry_the_hard_rule_wording_and_no_score(self, client):
        client.post("/simulate/seed?n=60&seed=1")
        decisions = client.get("/decisions?limit=60").json()["decisions"]

        fraud = [d for d in decisions if d["fraud_blocked"]]
        assert fraud, "expected at least one fraud-blocked transaction"
        for d in fraud:
            assert d["recovery_probability"] is None
            assert d["action"] == "no_action"
            assert "fraud" in d["explanation"].lower()

    def test_seed_rows_carry_the_outcome_but_not_the_diagnosis_label(self, client):
        """
        The dashboard's money-flow needs the realised outcome per row to show
        which actions actually recovered anything. The diagnosis label stays
        server-side -- handing it over would let the page grade the classifier,
        which is not the page's job.
        """
        rows = client.post("/simulate/seed?n=40&seed=1").json()["decisions"]

        assert rows
        assert "retry_success" in rows[0]
        assert "failure_category" not in rows[0]

    def test_each_run_draws_a_fresh_sample(self, client):
        """
        Regression: /simulate/seed used `.head(n)`, so "Run simulation" replayed
        a byte-identical batch forever -- same transaction ids, same revenue,
        every click.
        """
        runs = []
        for _ in range(3):
            client.post("/simulate/seed?n=80")
            ids = [d["transaction_id"]
                   for d in client.get("/decisions?limit=80").json()["decisions"]]
            runs.append(tuple(ids))

        assert len(set(runs)) == 3, "consecutive runs returned identical batches"
        # sampling 80 of ~1600 should overlap only partially, never wholly
        assert set(runs[0]) != set(runs[1])

    def test_an_explicit_seed_is_reproducible(self, client):
        """The escape hatch: scripted demos and tests can pin a batch."""
        def batch():
            client.post("/simulate/seed?n=60&seed=99")
            return [d["transaction_id"]
                    for d in client.get("/decisions?limit=60").json()["decisions"]]

        assert batch() == batch()

    def test_stats_describe_the_same_batch_as_the_feed(self, client):
        """
        Once the batch became a random sample, the stats panels had to read the
        SAME sample -- otherwise the headline revenue would describe a different
        set of transactions than the rows shown underneath it.
        """
        client.post("/simulate/seed?n=80")
        feed = client.get("/decisions?limit=80").json()["decisions"]
        stats = client.get("/stats/recovered-revenue?n=80").json()

        feed_total = round(sum(d["amount"] for d in feed), 2)
        assert feed_total == pytest.approx(stats["total_failed_value"], abs=0.01)

    def test_seed_resets_the_feed_between_runs(self, client):
        """
        Each seed is one self-contained replay. Without the reset the log
        accumulated across runs and the feed showed several batches at once,
        with counts that no longer matched the stats panels.
        """
        client.post("/simulate/seed?n=30")
        first = client.get("/decisions?limit=500").json()["total"]
        client.post("/simulate/seed?n=30")
        second = client.get("/decisions?limit=500").json()["total"]

        assert first == 30
        assert second == 30, "audit log accumulated across seeds"


@requires_models
class TestDashboardRoutes:
    def test_root_serves_the_dashboard(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "Verdict" in r.text
        # The decision feed's heading. Merchant-facing wording -- the section
        # itself is unchanged, only what it is called on screen.
        assert "Every decision, on the record" in r.text

    def test_retry_storm_report_is_served(self, client):
        r = client.get("/reports/retry-storm")
        if r.status_code == 404:
            pytest.skip("run scripts/retry_storm_demo.py first")

        body = r.json()
        assert body["before"]["total_attempts"] > body["after"]["total_attempts"]
        assert body["after"]["attempts_per_txn_max"] <= body["before"]["attempts_per_txn_max"]
        assert body["after"]["still_queued_at_end"] == 0

    def test_retry_storm_404s_cleanly_when_missing(self, client, monkeypatch):
        monkeypatch.setattr(api_main, "RETRY_STORM_REPORT", "/nonexistent/retry_storm.json")
        r = client.get("/reports/retry-storm")

        assert r.status_code == 404
        assert "retry_storm_demo" in r.json()["detail"]

    def test_metrics_report_is_served(self, client):
        """The "how this works" panel quotes these rather than hard-coding them."""
        r = client.get("/reports/metrics")
        if r.status_code == 404:
            pytest.skip("run scripts/train.py first")

        body = r.json()
        assert "model_1_failure_classifier" in body
        assert "model_2_recovery_success" in body
        assert 0.0 < body["model_1_failure_classifier"]["accuracy"] <= 1.0

    def test_metrics_404s_cleanly_when_missing(self, client, monkeypatch):
        monkeypatch.setattr(api_main, "METRICS_REPORT", "/nonexistent/metrics.json")
        r = client.get("/reports/metrics")

        assert r.status_code == 404
        assert "train.py" in r.json()["detail"]

    def test_breakdown_exposes_the_card_vs_upi_split(self, client):
        """The dashboard's skew chart depends on this shape existing."""
        body = client.get("/stats/breakdown?n=200").json()

        assert set(body["category_by_method"]) <= {"card", "upi"}
        for method in body["category_by_method"].values():
            assert method, "expected per-category counts for each payment method"


@requires_models
class TestVendoredLibraries:
    """
    GSAP, ScrollTrigger and Lenis are committed and served locally rather than
    pulled from a CDN, so the page keeps working with no internet. These assert
    the wiring that makes that true.
    """

    LIBS = ["gsap.min.js", "ScrollTrigger.min.js", "lenis.min.js"]

    @pytest.mark.parametrize("name", LIBS)
    def test_each_library_is_served(self, client, name):
        r = client.get(f"/vendor/{name}")

        assert r.status_code == 200, f"/vendor/{name} is not being served"
        assert "javascript" in r.headers["content-type"].lower()

    @pytest.mark.parametrize("name", LIBS)
    def test_each_library_is_the_real_thing(self, name):
        """
        Guards against an error page saved under a .js name. The first Lenis URL
        tried during this work returned a 404 HTML document, which would have
        committed happily and failed only in the browser.
        """
        path = os.path.join(api_main.VENDOR_DIR, name)

        assert os.path.exists(path), f"{name} is not vendored"
        assert os.path.getsize(path) > 10_000, f"{name} is too small to be the library"
        with open(path, "rb") as f:
            head = f.read(200).lower()
        assert b"<html" not in head, f"{name} looks like an HTML error page"

    def test_the_page_references_them_by_relative_path(self):
        """
        The whole point: no absolute URL, so the offline test below still holds
        and the page never reaches the network for them.
        """
        with open(api_main.DASHBOARD_INDEX, encoding="utf-8") as f:
            html = f.read()

        for name in self.LIBS:
            assert f'src="vendor/{name}"' in html, f"{name} is not referenced"

    def test_the_dashboard_survives_a_missing_library(self, client):
        """
        The libraries are an enhancement, not a dependency. The page must still
        be served -- and still contain its data wiring -- if one goes missing.
        """
        r = client.get("/")

        assert r.status_code == 200
        # the guards that make graceful degradation real, not aspirational
        assert "HAS_GSAP" in r.text
        assert "HAS_LENIS" in r.text
        assert "scrollIntoView" in r.text, "native scroll fallback was removed"


@requires_models
def test_dashboard_html_has_no_external_dependencies():
    """
    The page must stay offline-safe: no CDN scripts, stylesheets, or fonts.
    A demo that breaks on a bad connection is worse than a plainer one.

    Scans index.html ONLY, deliberately. The vendored libraries in
    src/dashboard/vendor/ do contain https://gsap.com inside their required
    licence banners, but they are read from disk and served locally -- a URL in
    a comment is not a network dependency. TestVendoredLibraries above covers
    those instead.
    """
    with open(api_main.DASHBOARD_INDEX, encoding="utf-8") as f:
        html = f.read()

    for marker in ("http://", "https://", "//cdn", "unpkg", "jsdelivr"):
        assert marker not in html, f"external reference found: {marker}"
