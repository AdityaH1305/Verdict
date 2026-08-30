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

implemented today by a `ClaudeAdapter`. Swapping providers later is a
one-file change, not a rewrite of agent logic. This also enforces the
fraud hard-block: the adapter is architecturally never called for
`fraud_block` cases, so no prompt can override that decision.

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
pytest tests/
```

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

## Status

- [x] Data generator (cards + UPI, method-conditional failure logic)
- [x] Failure classifier (Model 1)
- [x] Recovery success model (Model 2)
- [ ] Agent decision logic + LLM adapter
- [ ] Backend API
- [ ] Dashboard
- [ ] Deployment
- [ ] Architecture doc + "what broke" writeup
- [ ] Pitch video

See `docs/decisions.md` for design rationale and `docs/failure_stories.md`
for deliberately-engineered failure scenarios being tracked for the
submission's required "what broke" narrative.
