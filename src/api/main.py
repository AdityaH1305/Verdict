"""
Backend API (FastAPI).

    uvicorn src.api.main:app --reload

Models and the agent are constructed once at startup, not per request -- loading
two XGBoost artifacts on every call would dominate latency.

The API is a thin transport layer over RecoveryAgent. It contains no decision
logic of its own, and in particular it does NOT re-implement the fraud rule:
that lives in the agent (src/agent/recovery_agent.py) so there is exactly one
place where the rule can be right or wrong.
"""

import json
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.agent.llm_adapter import TemplateAdapter, get_adapter, load_env  # noqa: E402
from src.agent.recovery_agent import RecoveryAgent  # noqa: E402
from src.paths import REPORTS_DIR, TEST_DATA  # noqa: E402

DASHBOARD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard"
)
DASHBOARD_INDEX = os.path.join(DASHBOARD_DIR, "index.html")
RETRY_STORM_REPORT = os.path.join(REPORTS_DIR, "retry_storm.json")

STATE: Dict[str, Any] = {"agent": None, "bulk_agent": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_env()   # so /health reports the keys that are actually available
    try:
        agent = RecoveryAgent(adapter=get_adapter())
        STATE["agent"] = agent
        # Bulk agent for anything that loops over many transactions (the demo
        # seed, the dashboard's stats). Same loaded model objects -- no second
        # load, no double memory -- but pinned to the deterministic template
        # adapter so ordinary dashboard use can never spend a live provider's
        # free-tier quota. Live calls stay confined to POST /decide.
        STATE["bulk_agent"] = RecoveryAgent(
            classifier=agent.classifier,
            recovery_model=agent.recovery_model,
            adapter=TemplateAdapter(),
            policy=agent.policy,
        )
    except FileNotFoundError as exc:
        # Serve /health with a useful message rather than failing to boot.
        STATE["error"] = f"model artifacts not found ({exc}). Run scripts/train.py"
    yield
    STATE.clear()


app = FastAPI(
    title="Revenue Recovery Agent",
    description="Diagnoses why a payment failed and decides the economically "
                "correct recovery action.",
    version="0.1.0",
    lifespan=lifespan,
)


def _agent() -> RecoveryAgent:
    """The live agent -- may narrate via a real provider. Single decisions only."""
    if STATE.get("agent") is None:
        raise HTTPException(status_code=503, detail=STATE.get("error") or "agent not ready")
    return STATE["agent"]


def _bulk_agent() -> RecoveryAgent:
    """Template-only agent for anything that loops. Never spends provider quota."""
    if STATE.get("bulk_agent") is None:
        raise HTTPException(status_code=503, detail=STATE.get("error") or "agent not ready")
    return STATE["bulk_agent"]


def _records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    DataFrame -> JSON-safe records.

    NaN is not valid JSON, and this data is full of legitimate NaNs: card columns
    are empty on UPI rows and vice versa, and recovery_prob is deliberately absent
    for fraud-blocked transactions. They serialize as null.
    """
    return df.astype(object).where(pd.notna(df), None).to_dict(orient="records")


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

class Transaction(BaseModel):
    """A failed transaction at the moment of failure."""

    transaction_id: Optional[str] = None
    amount: float
    payment_method: Literal["card", "upi"]
    time_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    retry_count: int = Field(default=0, ge=0)
    customer_txn_history_count: int = Field(default=1, ge=0)
    customer_past_failure_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    seconds_to_failure: float = Field(default=5.0, ge=0.0)
    merchant_category: Optional[str] = None
    error_code: Optional[str] = None

    # card-only
    card_network: Optional[str] = None
    issuing_bank: Optional[str] = None
    is_3ds_flow: Optional[bool] = None
    card_age_months: Optional[float] = None

    # UPI-only
    psp_app: Optional[str] = None
    time_since_app_switch: Optional[float] = None
    daily_limit_utilization: Optional[float] = None


class Decision(BaseModel):
    transaction_id: Optional[str]
    timestamp: str
    payment_method: Optional[str]
    amount: Optional[float]
    error_code: Optional[str]
    error_code_meaning: str
    predicted_category: str
    category_probabilities: Dict[str, float]
    recovery_probability: Optional[float]
    action: str
    fraud_blocked: bool
    explanation: str


class BatchRequest(BaseModel):
    transactions: List[Transaction]
    explain: bool = Field(
        default=False,
        description="Generate LLM explanations. Off by default -- a large batch "
                    "would otherwise issue one API call per transaction.",
    )


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/health")
def health():
    agent = STATE.get("agent")
    adapter = getattr(agent, "adapter", None)
    return {
        "status": "ok" if agent is not None else "degraded",
        "models_loaded": agent is not None,
        "error": STATE.get("error"),
        # Reported from the adapter itself rather than an isinstance chain, so
        # adding a provider doesn't mean editing the API layer.
        "llm": {
            **(adapter.describe_self() if adapter else
               {"adapter": None, "provider": None, "model": None, "live": False}),
            "keys_present": {
                "GEMINI_API_KEY": bool(os.environ.get("GEMINI_API_KEY")),
                "ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY")),
            },
            "note": (None if (adapter and adapter.is_live) else
                     "No provider key set -- explanations fall back to grounded "
                     "templates. Decisions are unaffected."),
        },
        "thresholds": (None if agent is None else
                       {"auto_retry_at": agent.policy.HIGH, "give_up_below": agent.policy.LOW}),
    }


@app.post("/decide", response_model=Decision)
def decide(transaction: Transaction):
    """Diagnose one failed transaction and decide what to do about it."""
    return _agent().decide(transaction.model_dump())


@app.post("/simulate/batch")
def simulate_batch(request: BatchRequest):
    """
    Run the agent over a caller-supplied batch of failed transactions.

    `explain` defaults to False, so the default path spends no provider quota.
    Setting it True deliberately opts into one live call per transaction -- an
    explicit caller choice, and not a path the dashboard uses (the dashboard
    goes through /simulate/seed, which is pinned to the template adapter).
    """
    if not request.transactions:
        raise HTTPException(status_code=400, detail="no transactions supplied")

    agent = _agent()
    if request.explain:
        decisions = [agent.decide(t.model_dump()) for t in request.transactions]
    else:
        df = pd.DataFrame([t.model_dump() for t in request.transactions])
        decisions = _records(agent.decide_batch(df))

    return {"count": len(decisions), "decisions": decisions}


@app.post("/simulate/seed")
def simulate_seed(n: int = Query(default=200, ge=1, le=2000)):
    """
    Replay n held-out transactions through the agent.

    Backs the demo scenario -- "n failed transactions in, here is what the agent
    did" -- without needing a client to post a payload.
    """
    if not os.path.exists(TEST_DATA):
        raise HTTPException(status_code=503,
                            detail="no test data; run scripts/prepare_data.py")

    agent = _bulk_agent()
    # Each seed is one self-contained replay, so start from a clean feed.
    # Without this the audit log accumulates across runs and the dashboard shows
    # several batches mixed together, with counts that no longer match the stats.
    agent.audit_log.clear()

    df = pd.read_csv(TEST_DATA).head(n)
    # bulk agent + template narration: populates the "why" in the feed without
    # spending provider quota on a whole batch
    decisions = agent.decide_batch(df, log=True, explain=True)
    return {
        "count": int(len(decisions)),
        "action_mix": decisions["action"].value_counts().to_dict(),
        "decisions": _records(decisions.drop(
            columns=["failure_category", "retry_success"], errors="ignore")),
    }


@app.get("/stats/breakdown")
def breakdown(n: int = Query(default=500, ge=1, le=2000)):
    """Failure-category and action breakdown for the dashboard charts."""
    if not os.path.exists(TEST_DATA):
        raise HTTPException(status_code=503, detail="no test data")

    decisions = _bulk_agent().decide_batch(pd.read_csv(TEST_DATA).head(n))
    return {
        "n": int(len(decisions)),
        "by_category": decisions["predicted_category"].value_counts().to_dict(),
        "by_action": decisions["action"].value_counts().to_dict(),
        "by_payment_method": decisions["payment_method"].value_counts().to_dict(),
        "category_by_method": {
            method: sub["predicted_category"].value_counts().to_dict()
            for method, sub in decisions.groupby("payment_method")
        },
    }


@app.get("/stats/recovered-revenue")
def recovered_revenue(n: int = Query(default=500, ge=1, le=2000)):
    """
    Recovered-revenue estimate.

    `expected_recovered_value` is the honest number to quote: the sum of
    amount * P(recovery) over transactions the agent chose to act on. Where
    ground-truth outcomes exist (held-out data), the realised figure is reported
    alongside it so the estimate can be checked rather than taken on faith.
    """
    if not os.path.exists(TEST_DATA):
        raise HTTPException(status_code=503, detail="no test data")

    decisions = _bulk_agent().decide_batch(pd.read_csv(TEST_DATA).head(n))
    acting = {"auto_retry_now", "retry_later", "customer_nudge"}
    acted = decisions[decisions["action"].isin(acting)]

    payload = {
        "n": int(len(decisions)),
        "total_failed_value": round(float(decisions["amount"].sum()), 2),
        "acted_on_count": int(len(acted)),
        "acted_on_value": round(float(acted["amount"].sum()), 2),
        "expected_recovered_value": round(
            float((acted["amount"] * acted["recovery_prob"]).sum()), 2),
    }
    if "retry_success" in decisions.columns:
        realised = acted[acted["retry_success"]]
        payload["actual_recovered_count"] = int(len(realised))
        payload["actual_recovered_value"] = round(float(realised["amount"].sum()), 2)
    return payload


@app.get("/decisions")
def decision_feed(limit: int = Query(default=50, ge=1, le=500)):
    """Live decision feed -- the audit log, most recent first."""
    log = _bulk_agent().audit_log
    return {"total": len(log), "decisions": list(reversed(log[-limit:]))}


@app.get("/reports/retry-storm")
def retry_storm_report():
    """
    Serve the retry-storm before/after produced by scripts/retry_storm_demo.py.

    A static passthrough, not a computation -- the browser cannot read a local
    file, and recomputing the simulation per request would be wasteful and would
    drift from the numbers quoted in the docs.
    """
    if not os.path.exists(RETRY_STORM_REPORT):
        raise HTTPException(
            status_code=404,
            detail="No retry-storm report yet. Run: python scripts/retry_storm_demo.py",
        )
    with open(RETRY_STORM_REPORT) as f:
        return json.load(f)


@app.get("/", include_in_schema=False)
def dashboard():
    """The dashboard itself, served from the same origin as the API (no CORS)."""
    if not os.path.exists(DASHBOARD_INDEX):
        raise HTTPException(status_code=404, detail="dashboard/index.html not found")
    return FileResponse(DASHBOARD_INDEX)
