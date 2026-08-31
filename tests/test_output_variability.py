"""
General safeguard against "looks fine, is structurally frozen" defects.

The `.head(n)` bug survived 71 passing tests because every test asserted on the
*shape* of one response. Nothing compared two responses to each other, so an
endpoint that could only ever return one answer looked perfectly healthy.

This file tests the property that class of bug violates: **does calling the same
thing twice produce what it should?** Both directions matter --

  VARYING  endpoints must differ across calls (a frozen demo is a lie)
  STABLE   endpoints must not differ across calls (unstable reports are a lie)

The load-bearing test is `test_every_route_is_classified`. Listing endpoints by
hand would rot the moment someone adds one; instead it walks the app's own route
table and fails if a route appears in neither bucket. A new endpoint cannot
silently escape the question "should this vary?".
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import main as api_main  # noqa: E402
from src.paths import CLASSIFIER_PATH, TEST_DATA  # noqa: E402

requires_models = pytest.mark.skipif(
    not (os.path.exists(CLASSIFIER_PATH) and os.path.exists(TEST_DATA)),
    reason="run scripts/train.py first",
)

CALLS = 3


def _fingerprint(response):
    """Compare payloads, ignoring fields that legitimately change every call."""
    if response.headers.get("content-type", "").startswith("application/json"):
        body = response.json()
        return _strip_volatile(body)
    return response.text


def _strip_volatile(obj):
    """Timestamps are expected to move; they are not what these tests are about."""
    volatile = {"timestamp"}
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items() if k not in volatile}
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


# --------------------------------------------------------------------------- #
# The classification. Adding an endpoint means adding it here.
# --------------------------------------------------------------------------- #

# Must produce a DIFFERENT answer across repeated calls. These drive the demo,
# and a frozen demo misrepresents the system as replaying live behaviour.
VARYING = [
    ("POST", "/simulate/seed?n=60", "each run must draw a fresh sample"),
]

# Must produce the SAME answer across repeated calls. Reports and static
# resources; instability here would mean the numbers quoted in the docs and the
# pitch cannot be trusted.
STABLE = [
    ("GET", "/reports/retry-storm", "a committed report, not a live computation"),
    ("GET", "/", "static dashboard page"),
    ("GET", "/health", "server state, not sampled data"),
    ("POST", "/simulate/demo",
     "the pinned pitch batch -- being perfectly repeatable is its entire purpose"),
]

# Endpoints whose variability is governed by another endpoint, so testing them
# in isolation would assert the wrong thing. Each needs a reason.
DERIVED = {
    "/stats/breakdown": "describes the batch chosen by /simulate/seed",
    "/stats/recovered-revenue": "describes the batch chosen by /simulate/seed",
    "/decisions": "replays the audit log written by /simulate/seed",
    "/decide": "caller supplies the transaction; nothing is sampled",
    "/simulate/batch": "caller supplies the transactions; nothing is sampled",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(api_main.app) as c:
        yield c


def _call(client, method, url):
    return client.post(url) if method == "POST" else client.get(url)


@requires_models
class TestVaryingEndpoints:
    @pytest.mark.parametrize("method,url,why", VARYING)
    def test_output_varies_across_calls(self, client, method, url, why):
        results = [_fingerprint(_call(client, method, url)) for _ in range(CALLS)]
        distinct = {repr(r) for r in results}

        assert len(distinct) == CALLS, (
            f"{method} {url} returned the same payload {CALLS} times -- {why}. "
            f"This is the .head(n) bug class: a fixed slice or a pinned seed in "
            f"the request path."
        )

    def test_a_pinned_seed_still_overrides_variation(self, client):
        """Varying by default must not mean 'impossible to reproduce'."""
        results = [_fingerprint(client.post("/simulate/seed?n=60&seed=5"))
                   for _ in range(CALLS)]
        assert len({repr(r) for r in results}) == 1


@requires_models
class TestStableEndpoints:
    @pytest.mark.parametrize("method,url,why", STABLE)
    def test_output_is_identical_across_calls(self, client, method, url, why):
        first = _call(client, method, url)
        if first.status_code == 404:
            pytest.skip(f"{url} not available in this environment")

        results = [_fingerprint(_call(client, method, url)) for _ in range(CALLS)]
        assert len({repr(r) for r in results}) == 1, (
            f"{method} {url} changed between identical calls -- {why}."
        )


@requires_models
class TestBatchCoherence:
    """
    The panels must all describe the same batch.

    This is the second-order version of the same defect: fixing the freshness of
    one endpoint while leaving others on their own sample would put a headline
    number above a table of different transactions.
    """

    def test_all_panels_describe_one_batch(self, client):
        client.post("/simulate/seed?n=120")

        feed = client.get("/decisions?limit=120").json()["decisions"]
        revenue = client.get("/stats/recovered-revenue?n=120").json()
        breakdown = client.get("/stats/breakdown?n=120").json()

        assert len(feed) == 120
        assert revenue["n"] == 120
        assert breakdown["n"] == 120
        assert round(sum(d["amount"] for d in feed), 2) == pytest.approx(
            revenue["total_failed_value"], abs=0.01)

    def test_mismatched_n_cannot_split_the_panels(self, client):
        """
        Regression: the reuse guard was `len(batch) != n`, so a stats call with a
        different n silently resampled and replaced the batch the feed was
        showing. Panels then described different transaction sets.
        """
        client.post("/simulate/seed?n=100")
        before = client.get("/stats/recovered-revenue?n=100").json()

        # a caller asking for a different size must not move the current batch
        client.get("/stats/breakdown?n=400")
        after = client.get("/stats/recovered-revenue?n=100").json()

        assert before == after, "a stats call with a different n replaced the batch"

    def test_pinned_demo_batch_matches_its_recorded_metadata(self, client):
        """
        The pitch quotes these numbers out loud. If the committed batch and its
        sidecar ever drift apart, the slide and the screen disagree -- so assert
        they still match rather than trusting the file.
        """
        import json

        meta_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "demo", "demo_batch.json")
        if not os.path.exists(meta_path):
            pytest.skip("run scripts/make_demo_batch.py first")

        with open(meta_path) as f:
            expected = json.load(f)["expected"]

        body = client.post("/simulate/demo").json()
        if body.get("detail"):
            pytest.skip("no pinned demo batch present")
        revenue = client.get("/stats/recovered-revenue").json()

        assert body["action_mix"] == expected["action_mix"]
        assert revenue["actual_recovered_value"] == pytest.approx(
            expected["recovered_value"], abs=0.01)
        assert revenue["acted_on_count"] == expected["acted_count"]

        # the story the batch was chosen to tell
        assert set(body["action_mix"]) == {
            "auto_retry_now", "retry_later", "customer_nudge", "escalate", "no_action"
        }, "the demo batch must exercise all five actions"
        assert body["action_mix"]["escalate"] >= 2

        feed = client.get("/decisions?limit=300").json()["decisions"]
        assert sum(d["fraud_blocked"] for d in feed) == expected["fraud_blocked"]

    def test_stats_follow_a_new_seed(self, client):
        """The batch must still be replaceable -- by seeding, the intended way."""
        client.post("/simulate/seed?n=100&seed=1")
        first = client.get("/stats/recovered-revenue?n=100").json()
        client.post("/simulate/seed?n=100&seed=2")
        second = client.get("/stats/recovered-revenue?n=100").json()

        assert first != second


@requires_models
def test_every_route_is_classified():
    """
    The safeguard that makes this file general rather than a list of one-offs.

    A new endpoint must be consciously placed in VARYING, STABLE, or DERIVED.
    Without this, the next `.head(n)` lands on a route nobody thought to check --
    which is exactly how the original bug survived.
    """
    classified = (
        {url.split("?")[0] for _, url, _ in VARYING}
        | {url.split("?")[0] for _, url, _ in STABLE}
        | set(DERIVED)
    )

    app_routes = {
        route.path for route in api_main.app.routes
        if getattr(route, "methods", None)
        and not route.path.startswith(("/openapi", "/docs", "/redoc"))
    }

    unclassified = app_routes - classified
    assert not unclassified, (
        f"Unclassified endpoint(s): {sorted(unclassified)}. Decide whether each "
        f"must vary across calls (VARYING), must not (STABLE), or inherits its "
        f"behaviour from another endpoint (DERIVED), and record it in "
        f"tests/test_output_variability.py."
    )
