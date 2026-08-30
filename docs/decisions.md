# Design decisions log

Keep this updated as you build — it's most of your architecture doc for
free at submission time, and it's the material for the pitch video's
"how we thought about this" section.

## Scope (locked for the 6-7 day solo build)

**In scope:**
- Synthetic dataset, cards + UPI, method-conditional failure generation
- Failure classifier (Model 1): hard_decline / soft_decline /
  customer_dropoff / fraud_block
- Recovery success model (Model 2): P(recovery succeeds | features, category)
- Agent decision logic combining both models
- LLM-generated natural-language explanations, provider-swappable adapter
- Dashboard: failure breakdown, recovered revenue estimate, live decision
  feed with reasoning shown per transaction

**Explicitly cut (simulated, not built):**
- Real WhatsApp/SMS/email sending — simulate the "nudge" as a logged action
- Real bank/gateway retry integration — simulate outcomes
- Auth/login, multi-tenant merchant accounts
- Real-time transaction ingestion — batch/simulated feed is enough

## Hard rules (architectural, not learned)

- `fraud_block` is never eligible for retry or nudge, ever. The recovery
  success model is not invoked for these — not "invoked but ignored."
  This is enforced before any model call, not as a post-hoc filter.
- The LLM adapter is never called for `fraud_block` cases either, so no
  prompt-level behavior can produce a fraud override.

## LLM choice

Claude API (Haiku-class model — this is high-volume per-transaction
reasoning, not a task needing a large model). Wrapped behind a single
adapter interface (`generate_explanation(...)`) so switching providers is
a one-file change. This is worth stating explicitly in the pitch as a
deliberate anti-vendor-lock-in design choice.

## Why two models instead of one

A single classifier tells you *what* went wrong. It doesn't tell you
whether acting on it is worth it. Splitting into (1) diagnose and (2)
estimate recovery odds means the agent's decision is an economic one —
category + probability — not a reflexive one. This is also just a more
interesting ML story for the judges than a single classifier.

## Why cards + UPI instead of one payment method

They fail for structurally different reasons — cards fail mostly at the
issuer/network layer (declines, fraud, expired cards), UPI fails mostly
operationally (bank server downtime, PIN errors, app abandonment). This
makes `payment_method` a load-bearing feature the models have to learn
around, not a cosmetic column. See README for the fuller breakdown.

## Deliberate failure scenarios (for the "what broke" requirement)

Tracked in detail in `failure_stories.md`. Summary:

1. **Classifier ambiguity** — soft_decline vs customer_dropoff overlap
   deliberately in the generated data (e.g. a UPI timeout caused by slow
   OTP entry — is that systemic or human?). Expect real confusion in the
   confusion matrix here; this is a feature of the story, not a bug to
   hide.
2. **Retry-storm risk** — build the agent first *without* a retry budget,
   let it over-retry in a test run, then add a per-transaction retry cap
   / backoff and log the before/after. Real production failure mode,
   cheap to build, clean to demo.
3. **LLM hallucinated reason codes** — early prompt versions without a
   grounded error-code lookup table may let the model invent plausible
   but wrong bank terminology. Fix by grounding the prompt strictly in
   the actual error code + a lookup table. Screenshot before/after if it
   happens naturally.

## Open decisions (fill in as you go)

- Exact classifier algorithm (leaning XGBoost/LightGBM — fast, explainable,
  defensible in a panel Q&A over a black-box net)
- Retry timing logic specifics (fixed backoff vs. learned timing)
- Dashboard framework choice
- Deployment target
