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
models/           # saved trained model artifacts
notebooks/        # EDA, model evaluation, calibration plots
docs/             # architecture doc, decisions log, "what broke" writeup
scripts/          # one-off utility scripts (train, generate, eval)
tests/
```

## Status

- [ ] Data generator (cards + UPI, method-conditional failure logic)
- [ ] Failure classifier (Model 1)
- [ ] Recovery success model (Model 2)
- [ ] Agent decision logic + LLM adapter
- [ ] Backend API
- [ ] Dashboard
- [ ] Deployment
- [ ] Architecture doc + "what broke" writeup
- [ ] Pitch video

See `docs/decisions.md` for design rationale and `docs/failure_stories.md`
for deliberately-engineered failure scenarios being tracked for the
submission's required "what broke" narrative.
