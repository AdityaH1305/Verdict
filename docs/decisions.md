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

## Day 1 — error code taxonomy verification

Per the build plan's action item, spot-checked the illustrative code tables
against public sources before generating data.

**Card (ISO 8583) codes — confirmed as originally listed**, cross-checked
against EBANX's ISO 8583 response code docs and general processor
references: 51 (insufficient funds), 05 (do not honor), 14 (invalid card
number), 54 (expired card), 61 (exceeds withdrawal limit) → hard_decline;
91 (issuer unavailable/timeout), 96 (system malfunction) → soft_decline;
59 (suspected fraud) → fraud_block. 3DS/OTP abandonment (no code) →
customer_dropoff.

**UPI (NPCI) codes — original table was wrong on several codes**, corrected
using a bank-published UPI error code reference (HDFC/Juspay integration
docs), since the raw NPCI PDF wasn't text-extractable. Corrected mapping
used in the generator:

| Code | Meaning | Category |
|------|---------|----------|
| Z9 | Insufficient funds in remitter account | hard_decline |
| ZH | Invalid virtual address (VPA) | hard_decline |
| Z8 / Z7 / ZU | Per-txn / frequency / bank daily limit exceeded | hard_decline |
| ZM | Invalid/incorrect MPIN entered | hard_decline |
| UT / BT / U28 / Y1 / XY | Remitter/beneficiary bank unavailable (timeout) | soft_decline |
| U67 / U68 | Debit / credit timeout | soft_decline |
| U30 | Debit failed — ambiguous: bank-server-side per some sources, but also documented as resulting from repeated incorrect PIN entry or hitting daily limits (customer-side). Used deliberately as the Candidate 1 ambiguity code. | soft_decline / customer_dropoff (overlap) |
| 59 / ZI | Suspected fraud, risk-score decline (remitter/beneficiary side) | fraud_block |
| (no code) | User abandons before PIN entry / mid app-switch | customer_dropoff |

Original table's `U69` (claimed "insufficient balance") and `ZM` (claimed
"transaction limit exceeded") don't hold up — `U69` isn't in the verified
source at all, and `ZM` is actually incorrect-MPIN, not a limit code.
Replaced with `Z9` and `ZU`/`Z8`/`Z7` respectively.

## Day 1 — data generator assumptions

- 8,000 rows, `payment_method` split ~55% card / 45% UPI (rough proxy for
  card vs. UPI mix on a mid-size Indian payment gateway).
- Card failure category mix: hard_decline 45%, soft_decline 20%,
  customer_dropoff 20%, fraud_block 15% — cards skew hard_decline/fraud
  per README's stated design.
- UPI failure category mix: soft_decline 35%, customer_dropoff 35%,
  hard_decline 20%, fraud_block 10% — UPI skews operational/dropoff per
  README.
- `retry_success` sampled per category with category-specific base rates,
  then nudged by features (e.g. higher `customer_past_failure_rate` lowers
  it, more `retry_count` lowers it slightly to reflect diminishing
  returns): hard_decline ~8% (small chance a retry after a payday/limit
  reset succeeds), soft_decline ~65% (mostly transient), customer_dropoff
  ~42% (nudge-recoverable), fraud_block always False by construction.
- Candidate 1 ambiguity: for ~12% of UPI soft_decline rows, force
  `error_code = U30` and give them customer_dropoff-like behavioral
  features (`time_since_app_switch` drawn from the dropoff distribution
  instead of the soft_decline one), while leaving the label mechanism the
  same probabilistic draw as any soft_decline row — i.e. the ambiguity is
  in the features, not a special label rule. A mirrored subset of
  customer_dropoff rows also gets `error_code = U30` instead of no code.
- `is_3ds_flow` more common for card `customer_dropoff` and `fraud_block`
  rows (3DS is where fraud checks and drop-off both concentrate in
  reality).
- `card_age_months` skews low for `fraud_block` (newer/less-established
  cards more associated with fraud patterns in this synthetic model).

## Day 2 — why the data was regenerated

Before training, I checked whether Model 1 was actually a learnable problem.
It was not.

- **7,057 of 8,000 rows (88%) had an `error_code` that mapped to exactly one
  `failure_category`.** Only `U30` was ambiguous.
- Benchmark: **96.6% accuracy with `error_code`, 60.8% without** (majority-class
  baseline 33.8%).

Model 1 was a lookup table wearing a classifier costume. That guts the whole
pitch — "diagnosis is a real problem" — and the first question a judge asks is
*"isn't that just the error code?"*

**Root cause:** the Day 1 generator gave each category a private, disjoint code
list. Reality is messier. `05` "do not honor" is a deliberately opaque catch-all
that issuers use to mask fraud suspicion and funds problems alike; an issuer
timeout (`91`, `U67`) is exactly where a stalled *customer* hides behind a
*system* code; a wrong MPIN (`ZM`) very often ends in the customer giving up.

**Fix:** categories now emit from *overlapping* code distributions
(`CARD_CODE_GIVEN_CATEGORY` / `UPI_CODE_GIVEN_CATEGORY`), so `error_code` is
**informative but not decisive**. Codes that genuinely are unambiguous in
production (`51` insufficient funds, `54` expired, `Z9`, `ZH`, `59`) stay
near-pure — the goal was realism, not uniform mud.

Result: deterministic rows fell 88% → 42%, and the naive lookup ceiling fell
96.4% → 80.4%.

**Critical second-order point:** overlap alone would only have added irreducible
noise, and no model could then beat the lookup. The overlap had to be
*resolvable from behaviour*, so category-conditional behavioural distributions
were sharpened at the same time (fraud → high amount / thin customer history /
young card / overnight; dropoff → long dwell; hard_decline → high daily-limit
utilization).

### New feature: `seconds_to_failure`

Added beyond the build plan's feature spec. Cards had no dwell-time analogue to
`time_since_app_switch`, which left card `customer_dropoff` vs `soft_decline` on
code `91` **structurally unresolvable** — the model could not have beaten the
lookup on the exact cases that matter. Every real gateway logs
time-from-initiation-to-failure.

One subtlety worth keeping: `soft_decline` is **slow** too. A bank-side timeout
is slow *by definition* — that is what a timeout is (gateway cutoffs sit around
30s). An intermediate version gave `soft_decline` a fast dwell, which made
drop-off trivially separable and quietly trained the Candidate 1 ambiguity out
of the problem. Both categories now overlap on dwell time, which is the honest
form of the ambiguity: a long wait looks the same whether the bank stalled or
the human did.

## Day 2 — model results

| | Model | Naive `error_code` lookup |
|---|---|---|
| Accuracy | **0.9431** | 0.8087 |
| Macro-F1 | **0.9451** | 0.7993 |

The lookup baseline is reported on **every** training run, and `scripts/train.py`
**fails the build** if the model stops beating it. If that gap ever collapses,
the classifier is not doing real work and the data needs another pass.

Where the difficulty lives: **75 of 91 total errors (82%) are the
`soft_decline` ↔ `customer_dropoff` pair.** `hard_decline` (F1 0.987) and
`fraud_block` (F1 0.978, only 3 rows leaked) are near-clean, which is correct —
insufficient funds genuinely *is* unambiguous. Candidate 1 is now structural
rather than a single-code artifact.

Model 2: ROC-AUC 0.784, PR-AUC 0.550, Brier 0.1745 → **0.1688** after isotonic
calibration.

## Day 2 — decisions locked

- **Algorithm: XGBoost** for both models. Native handling of the structural NaNs
  *and* the categorical columns, and feature importances that survive a panel
  Q&A better than a black-box net. (Closes the open decision below.)
- **Model 2's category input is Model 1's full predicted probability vector**,
  not a hard label and not ground truth. Ground truth is unavailable at serving
  time, and a hard label discards exactly the uncertainty the agent needs to
  hedge on ambiguous cases. Trained on **out-of-fold** predictions
  (`cross_val_predict`) so Model 2 learns against the noise level it will
  actually face — training on in-sample predictions would be train/serve skew.
- **Model 2 output is isotonic-calibrated.** The agent reads economic cutoffs
  straight off this probability scale, so a raw tree score of "0.6" that really
  means 0.75 would silently corrupt every decision. Calibration is a correctness
  requirement here, not polish.
- **Structural NaNs are preserved, not imputed.** A card feature is absent on a
  UPI row because it does not exist, and "absent" is information.
- **One shared feature encoder** (`src/features.py`), bundled into each model
  artifact, so the serving path cannot drift from the training path.

## Day 2 → Day 3 handoff: the thresholds in build_plan.md are wrong

The build plan asked to log Model 2's actual score distribution before finalizing
agent cutoffs. Doing so shows the proposed thresholds do not fit the data:

| Bucket | Planned action | Share of transactions |
|---|---|---|
| `>= 0.6` | `auto_retry_now` | **1.9%** |
| `0.3 – 0.6` | `retry_later` / `customer_nudge` | 58.5% |
| `< 0.3` | `escalate` / `no_action` | 39.7% |

At a 0.6 cutoff the agent would essentially **never** take its flagship action.
This is not a model defect — it is correct. True recovery rates cap out around
0.61 for `soft_decline` and 0.38 for `customer_dropoff`, so no population in this
data honestly deserves >70% recovery odds; isotonic calibration correctly refuses
to invent confidence that isn't there.

The score distribution is also **bimodal** (see `reports/calibration_curve.png`):
a spike near 0 (`hard_decline`, genuinely unrecoverable) and a broad mass around
0.40–0.55. **Day 3 should place the cutoffs in the valley between those modes**,
not at the round numbers guessed before any data existed. Retune and record the
final values here.

## Open decisions (fill in as you go)

- ~~Exact classifier algorithm~~ — **locked Day 2: XGBoost** (see above)
- Agent decision thresholds — **must be retuned**, see the handoff section above
- Retry timing logic specifics (fixed backoff vs. learned timing)
- Dashboard framework choice
- Deployment target
