# Revenue Recovery Agent

An AI system that diagnoses *why* payment transactions fail and decides the
economically correct recovery action — instead of treating "failure" as a
single undifferentiated bucket.

Built for the Razorpay AI Buildathon (deadline: Sept 5, 2026).

## The problem

Payment gateways lose meaningful revenue to failed transactions, but not all
failures are the same and not all failures deserve the same response:

- A bank-side timeout is often recoverable with a retry.
- Insufficient funds is not recoverable — retrying wastes a retry slot and
  annoys the customer.
- A fraud block should never be retried, under any circumstances.
- A customer who abandoned mid-flow needs a nudge, not a silent retry.

Most systems treat this as one failure rate number. This system treats it as
a diagnosis problem: classify the failure, estimate the odds a recovery
action would succeed, then decide.

## System design

Two ML models + one reasoning agent, in that order:

1. **Failure Classifier** — predicts failure category from transaction
   features at time of failure: `hard_decline`, `soft_decline`,
   `customer_dropoff`, `fraud_block`.
2. **Recovery Success Model** — predicts probability that a retry/nudge
   recovers the transaction, conditioned on the predicted category.
   **Never invoked for `fraud_block` transactions — this is an
   architectural rule, not a learned preference.**
3. **Recovery Agent** — combines both model outputs into an action
   (auto-retry now, retry later, customer nudge, escalate to merchant,
   no action) and generates a plain-language explanation via an LLM
   adapter (see below).

```
transaction ──▶ Failure Classifier ──▶ category
                                          │
                        (skip if fraud_block)
                                          ▼
                              Recovery Success Model ──▶ probability
                                          │
                                          ▼
                                   Recovery Agent
                                          │
                         ┌────────────────┼────────────────┐
                         ▼                ▼                ▼
                  action decision   explanation (LLM)   audit log
```

## Why cards + UPI, not just one

Card and UPI transactions fail for structurally different reasons:

- **Cards** fail mostly for issuer/network reasons — insufficient funds,
  do-not-honor, CVV mismatch, expired card, 3DS abandonment, fraud block.
  Skews toward `hard_decline` / `fraud_block`.
- **UPI** fails mostly for operational reasons — bank server downtime at
  the PSP/NPCI layer, PIN errors, timeouts, app-switch abandonment, daily
  limit caps. Skews toward `soft_decline` / `customer_dropoff`.

Modeling both means `payment_method` is a real, load-bearing feature, not
decoration — the classifier has to learn method-conditional failure
patterns, which is closer to a real production problem.

## LLM adapter (provider-swappable by design)

The agent never calls an LLM SDK directly. It calls a single interface:

```python
def generate_explanation(transaction, category, recovery_prob, action) -> str
```

Swapping providers is a one-file change, not a rewrite of agent logic. This also
enforces the fraud hard-block: the adapter is architecturally never called for
`fraud_block` cases, so no prompt can override that decision.

**This got tested for real.** Paid Anthropic access went away mid-build, so the
live path moved to Google Gemini's free tier. The swap touched exactly one file
(`src/agent/llm_adapter.py`) plus the `/health` reporting block — no agent logic,
no thresholds, no fraud rules, no model code.

| Adapter | Model | Picked when |
|---|---|---|
| `GeminiAdapter` | `gemini-2.5-flash-lite` | `GEMINI_API_KEY` set — **active** |
| `ClaudeAdapter` | `claude-haiku-4-5` | `ANTHROPIC_API_KEY` set, no Gemini key |
| `TemplateAdapter` | — | no key at all |

`ClaudeAdapter` is kept and still works; it's dormant only for lack of a key.
**Decisions are identical under all three** — the LLM narrates a decision the
threshold policy already made. On a free tier, rate limits are the steady state
rather than an edge case, so a 429 degrades to the grounded template instead of
failing the request.

Measured free-tier behaviour on `flash-lite`: about **11 live calls in a burst**,
after which the limit persists well beyond a 60-second window — a paced 10/min
run and a further two-minute wait both still returned 429s. Budget live
narration per demo, not per minute. Batch paths therefore default to
`explain=False`, and live narration is reserved for `POST /decide` — the
interactive path a demo actually drives.

## Repo structure

```
data/
  raw/            # generated synthetic transaction data
  processed/      # train/test splits, feature-engineered data
src/
  data_generation/  # synthetic data generator (cards + UPI, method-conditional)
  models/           # failure classifier + recovery success model
  agent/            # decision logic + LLM adapter interface
  api/              # backend serving layer
  dashboard/        # frontend
  features.py       # shared feature spec + encoder (one definition, both models)
  paths.py          # canonical project paths
models/           # saved trained model artifacts
reports/          # confusion matrix, calibration curve, metrics.json
notebooks/        # ad-hoc EDA
docs/             # architecture doc, decisions log, "what broke" writeup
scripts/          # prepare_data.py, train.py
tests/            # guard tests for the architectural hard rules
```

Evaluation artifacts are emitted as PNG/JSON by `scripts/train.py` rather than
living in notebooks, so they are reproducible, run headless, and drop straight
into the dashboard and pitch video.

## Running it

```bash
pip install -r requirements.txt
python src/data_generation/generate_transactions.py   # -> data/raw/transactions.csv
python scripts/prepare_data.py                        # -> data/processed/{train,test}.csv
python scripts/train.py                               # -> models/*.pkl, reports/*
python scripts/evaluate_agent.py                      # -> action mix + revenue
python scripts/retry_storm_demo.py                    # -> retry-storm before/after
pytest tests/
```

Then the API:

```bash
uvicorn src.api.main:app --reload
```

`POST /decide` returns a decision for one failed transaction; `POST /simulate/seed`
replays held-out transactions for the demo; `GET /stats/breakdown` and
`GET /stats/recovered-revenue` back the dashboard; `GET /decisions` is the audit
feed.

For LLM explanations, copy `.env.example` to `.env` and set `GEMINI_API_KEY`
(free tier — [get one here](https://aistudio.google.com/apikey)), then:

```bash
python scripts/check_llm.py
```

That makes one real call and verifies the answer stays grounded in the error-code
taxonomy.

## Model results

| | Model 1 (XGBoost) | Naive `error_code` lookup |
|---|---|---|
| Accuracy | **0.943** | 0.809 |
| Macro-F1 | **0.945** | 0.799 |

The lookup baseline is not decoration. An early version of the data generator
made `error_code` a near-deterministic map to the label, so the "classifier"
scored 96.6% by memorizing a lookup table. The generator was rebuilt so codes
overlap the way they do in production, and `scripts/train.py` now **fails the
build** if the model ever stops beating that baseline. See `docs/decisions.md`.

**82% of the model's remaining errors are the single `soft_decline` ↔
`customer_dropoff` pair** — the ambiguity this project was designed to expose.
`hard_decline` and `fraud_block` sit at 0.99 recall.

Model 2: ROC-AUC 0.784, isotonic-calibrated (Brier 0.175 → 0.169). Calibration
is a correctness requirement, not polish — the agent reads its economic cutoffs
directly off this probability scale.

## Agent decisions

Thresholds were derived from Model 2's measured score distribution, not guessed.
The distribution is bimodal, so the cutoffs sit at `0.50` ("recovery is more
likely than not — just retry it") and `0.25` (the valley floor between the
recoverable and unrecoverable modes). The build plan's original `0.6` would have
sent only 1.6% of transactions to `auto_retry_now`.

| Action | Count | Share | Observed recovery |
|---|---|---|---|
| `auto_retry_now` | 241 | 15.1% | 0.610 |
| `retry_later` | 248 | 15.5% | 0.520 |
| `customer_nudge` | 369 | 23.1% | 0.398 |
| `escalate` | 10 | 0.6% | 0.500 |
| `no_action` | 732 | 45.8% | 0.030 |

The agent acts on **53.6%** of failures and leaves only **27 recoverable
transactions** untouched. Note the mid band splits on *category*, not
probability: identical odds mean different things depending on whether a bank
refused or a human walked away.

### What the fraud hard-block guarantees

Two claims, deliberately kept separate:

- **Architectural, absolute** — anything diagnosed as `fraud_block` is never
  acted on and never scored. 207 diagnosed, **0 actioned, 0 scored**. Three
  independent guards: the agent returns before Model 2 or the LLM are reached,
  Model 2 raises if fraud rows enter training, and both adapters raise if invoked
  with a fraud category. `tests/test_agent_rules.py` proves this with spy objects
  that fail the build on contact — a post-hoc filter would pass a naive
  "action == no_action" assertion but fail these.
- **Model 1's fraud recall — 0.985, not 1.0.** The architecture guarantees
  behaviour *given* a diagnosis; it cannot guarantee the diagnosis. 3 of 204 true
  fraud transactions were not recognised as fraud. That's a detection miss, not a
  breach of the rule, and it's reported rather than asserted.

### Retry storm (see `docs/failure_stories.md`)

Built without a cap first, on purpose. Over 12 polling cycles on 489
retry-eligible transactions:

| | Uncapped | Budget of 3 + backoff |
|---|---|---|
| Retry attempts | 3,025 | **1,062** (−64.9%) |
| Max attempts on one transaction | 12 | **3** |
| Attempts that could never succeed | 84.5% | 60.2% |
| Still queued at end | 213 | **0** |

The uncapped loop did **2.8× the work for 8% more revenue** — and its worst case
is unbounded, because everything left in that queue is unrecoverable by
construction.

## Status

- [x] Data generator (cards + UPI, method-conditional failure logic)
- [x] Failure classifier (Model 1)
- [x] Recovery success model (Model 2)
- [x] Agent decision logic + LLM adapter
- [x] Backend API
- [ ] Dashboard
- [ ] Deployment
- [ ] Architecture doc + "what broke" writeup
- [ ] Pitch video

See `docs/decisions.md` for design rationale and `docs/failure_stories.md`
for deliberately-engineered failure scenarios being tracked for the
submission's required "what broke" narrative.
