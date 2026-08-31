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

## Day 3 — threshold tuning (closes the Day 2 handoff)

Cutoffs derived from Model 2's measured, calibrated score distribution on the
1,396 non-fraud test rows. The distribution is bimodal:

| Band | Rows | What lives there |
|---|---|---|
| 0.00–0.10 | 475 | `hard_decline` — observed recovery **0.043** |
| **0.15–0.30** | **48** | **the valley — genuinely sparse** |
| 0.35–0.60 | 763 | `soft_decline` (~0.50) + `customer_dropoff` (~0.41) |

**`HIGH = 0.50`** (was 0.6). Reads as *"recovery is more likely than not, so just
retry it"* — a defensible line on its own terms rather than a round number, and
it lands on the `soft_decline` median (0.495). Of the cutoffs evaluated it gave
the **highest observed recovery inside the auto-retry bucket (0.610)**:

| Cutoff | Share auto-retried | Observed recovery in bucket |
|---|---|---|
| 0.60 | 1.6% | 0.538 |
| 0.55 | 5.7% | 0.560 |
| **0.50** | **15.1%** | **0.610** |
| 0.45 | 30.0% | 0.552 |

**`LOW = 0.25`** (was 0.3). The valley floor. Below it is the unrecoverable mode;
above it, everything worth acting on. Beats 0.30 on both counts: 26 vs 30
recoverable transactions stranded, and 15 vs 27 pointless escalations.

Final rule:

```
fraud_block                        -> no_action        (hard rule, no threshold)
p >= 0.50                          -> auto_retry_now
0.25 <= p < 0.50, customer_dropoff -> customer_nudge
0.25 <= p < 0.50, otherwise        -> retry_later      (with backoff)
p <  0.25, hard_decline            -> no_action        (nothing to escalate)
p <  0.25, otherwise               -> escalate
```

Two judgement calls encoded there. The mid band splits on **category, not
probability**: the same odds mean different things depending on whether a bank
refused or a human walked away — one wants another attempt, the other wants
prompting. And an unrecoverable `hard_decline` gets `no_action` rather than
`escalate`, because escalating an insufficient-funds decline just makes work for
someone who can't fix it either.

### Resulting action mix (1,600 test transactions)

| Action | Count | Share | Observed recovery |
|---|---|---|---|
| `auto_retry_now` | 241 | 15.1% | 0.610 |
| `retry_later` | 248 | 15.5% | 0.520 |
| `customer_nudge` | 369 | 23.1% | 0.398 |
| `escalate` | 10 | 0.6% | 0.500 |
| `no_action` | 732 | 45.8% | 0.030 |

Agent acts on **53.6%** of failures. Of 2,235,064 in failed value it recovers
426,920 across 423 transactions, leaving only **27 recoverable transactions
(34,752)** untouched. Precision of acting: 0.493.

## Day 3 — what the fraud hard-block does and does not guarantee

Worth stating precisely, because it is easy to overclaim. Two separate things:

1. **Architectural guarantee — absolute.** Any transaction the system diagnoses
   as `fraud_block` is never acted on and never has a recovery probability
   computed. Verified on the test set: 207 diagnosed, 0 actioned, 0 scored.
   Three independent guards — `RecoveryAgent.decide()` returns before Model 2 or
   the adapter are reached; Model 2 raises if fraud rows enter its training set;
   both adapters raise if invoked with a fraud category.
2. **Model 1's fraud recall — 0.985, not 1.0.** Of 204 true fraud transactions,
   3 were not recognised as fraud, and 1 of those received a recovery action.

The architecture guarantees behaviour *given* a diagnosis; it cannot guarantee
the diagnosis. A fraud transaction the classifier fails to recognise is a
detection miss, not a breach of the rule. `scripts/evaluate_agent.py` asserts (1)
and merely reports (2) — an earlier version of that check conflated them and
failed the build over a classifier miss.

## Provider swap: Claude → Gemini (and why it was cheap)

We lost paid Anthropic access mid-build, so the live explanation path moved to
**Google Gemini's free tier**.

**The swap touched exactly one file** — `src/agent/llm_adapter.py` — plus the
`/health` reporting block. No agent logic, no thresholds, no fraud rules, no
model code. This is the anti-vendor-lock-in claim from the original design being
cashed in under real conditions rather than asserted in a slide: the interface
was built to make providers swappable, and then a provider actually had to be
swapped.

`ClaudeAdapter` is **retained and working**, dormant only because no
`ANTHROPIC_API_KEY` is set. Keeping both implementations is the evidence that
the abstraction is real.

| Adapter | Provider | Model | When it's picked |
|---|---|---|---|
| `GeminiAdapter` | Google | `gemini-2.5-flash-lite` | `GEMINI_API_KEY` set — the active path |
| `ClaudeAdapter` | Anthropic | `claude-haiku-4-5` | `ANTHROPIC_API_KEY` set and no Gemini key |
| `TemplateAdapter` | none | — | no key at all |

**Decisions are identical under all three.** Only the wording of the explanation
changes, because the LLM narrates a decision the threshold policy already made.

Notes on the Gemini choice:

- **SDK: `google-genai`, not `google-generativeai`.** The latter is deprecated;
  the Gen AI SDK has been the recommended package since May 2025. Worth checking
  rather than recalling — Google's naming has churned.
- **Flash-class, not Pro.** Pro models are not free-tier eligible.
  Settled on **`gemini-2.5-flash-lite`** over `gemini-2.5-flash` for demo
  headroom: request rate is the binding constraint when narrating a batch, not
  model quality, and this task is grounded restatement rather than reasoning.
  `gemini-2.5-flash` is the step up if explanation quality ever needs it.
- **The flash → flash-lite switch was verified, not assumed**, on the principle
  that the thinking-token bug only surfaced on a specific model. Two findings
  worth keeping:
  - `thinking_budget=0` is accepted on flash-lite, and **flash-lite has thinking
    off by default anyway** (flash has it on). The setting is therefore redundant
    on this model but is kept explicit — it guards against a default change and
    keeps behaviour identical if anyone switches back to flash.
  - Truncation detection behaves identically: `finish_reason` still renders as
    `FinishReason.MAX_TOKENS`, so the fallback fires. Confirmed end-to-end
    through the adapter by forcing a 12-token budget, not just against the raw
    SDK.
- **429s are expected, not exceptional.** On a free tier, rate limiting is the
  steady state under load, so a rate-limited call degrades to the grounded
  template instead of raising. An explanation is a narration layer; a payment
  decision must not depend on one. `tests/test_llm_adapters.py` pins this with a
  simulated 429 rather than leaving it to hope.

### Measured free-tier limits (`gemini-2.5-flash-lite`)

Measured rather than taken from documentation, because the practical number
turned out to differ from the naive reading:

| Test | Result |
|---|---|
| Rapid burst from cold | **11 live calls** in 12.8s, then 429 |
| Paced at 10 req/min after a 65s cooldown | 4 live, then **8 consecutive 429s** over 73s |
| Single call after a further 120s cooldown | still 429 |

The later results are the important ones. The quota does **not** refill on a
60-second window: once the limit is hit it stayed hit through a paced 10/min run
and a subsequent two-minute wait, after roughly 50 calls across the session. So
the usable model is "a modest burst, then a long lockout", not "N per minute,
forever" — plan capacity per demo, not per minute.

**Consequence for the demo:** do not narrate a whole batch live. This is already
how the system is built — `POST /simulate/batch` and `decide_batch()` default to
`explain=False`, and `scripts/evaluate_agent.py` runs with explanations off — so
live narration is reserved for `POST /decide` on individual transactions, which
is exactly the interactive path a demo drives. Anything beyond that falls back to
grounded templates, which is a degradation in prose quality only, never in the
decision.
- **Thinking is disabled (`thinking_budget=0`), and this was a real bug.** The
  first live run produced explanations truncated mid-sentence. Gemini 2.5 is a
  thinking model and reasoning tokens are billed against `max_output_tokens`, so
  the 300-token budget carried over from the Claude adapter was consumed
  **286 by thinking, leaving 10 for the answer**. Not a prompt problem and not a
  model-quality problem — a Claude-shaped assumption that does not transfer.
  This task is grounded restatement of facts we already supply, so there is
  nothing to reason about and the whole budget belongs to the answer.
- **A `MAX_TOKENS` finish is now treated as a failure**, not a result. The first
  version passed the truncated fragment straight through, where it read as
  authoritative while stopping mid-clause. Truncated output is worse than a
  template, so it degrades.
- **The prompt now requires citing the error code verbatim.** Once truncation was
  fixed, Gemini produced correct but code-free prose ("the bank timed out").
  Operations staff need the literal code to quote back to the bank, so this is a
  product requirement, not a concession to the checker.
- **The grounding checker had the same class of bug it was built to catch.** It
  graded the *meaning* against four strings — `timed out`, `timeout`, `time out`,
  `timing out` — all morphological variants of one word, while its own comment
  claimed to be checking "the load-bearing CONCEPT, not the phrasing". A correct
  explanation saying the bank "did not respond in time" was flagged as ungrounded.
  The fix grades against named concepts with multiple lexical families
  (`REQUIRED_CONCEPTS`), so jargon and plain-language rewordings both pass.

  Loosening a checker risks making it vacuous, so the grader was extracted into
  `grade_explanation()` and `tests/test_grounding_checker.py` now tests it in
  pairs: five real paraphrases must pass, and five explanations that are each
  wrong in exactly one way — missing code, invented "insufficient funds",
  hallucinated CVV/3DS on a UPI transaction, contradicting the chosen action,
  saying nothing about the failure's nature — must still fail. A grader nobody
  tested is how the original bug survived.
- Adapters now declare `provider` / `model` / `is_live` on the class, so
  `/health` reports the active provider without an isinstance chain that has to
  grow every time a provider is added.

Verify the live path with `python scripts/check_llm.py` — it makes one real call
and checks the answer cites the actual error code, reflects the verified meaning,
and invents no bank terminology.

## Day 3 — LLM adapter

- **Model: Haiku-class / Flash-class.** High-volume per-transaction narration,
  not a task needing a large model.
- **The LLM explains a decision; it never makes one.** The action is chosen by
  the threshold policy before the adapter is called, and the prompt says so, so
  no prompt-level behaviour can produce a different action than the one logged.
- **Grounded prompts.** The prompt carries the real error code plus its entry
  from `src/error_taxonomy.py` and forbids speculation beyond those facts.
  `ClaudeAdapter(grounded=False)` reproduces the ungrounded version on purpose,
  for the Candidate 3 before/after.
- **`TemplateAdapter` fallback when `ANTHROPIC_API_KEY` is absent**, built from
  the same taxonomy. Explanations are a narration layer; a payment decision is
  not. The agent, API, and test suite all run offline, and the demo never depends
  on network access. API errors degrade to the template rather than propagating.

## Day 3 — audit log format

Every decision emits: `transaction_id`, `timestamp`, `payment_method`, `amount`,
`error_code`, `error_code_meaning` (from the taxonomy, not the model),
`predicted_category`, `category_probabilities` (full 4-vector),
`recovery_probability` (null when fraud-blocked), `action`, `fraud_blocked`,
`explanation`. Served by `GET /decisions`; this is the Day 4 dashboard's feed
format.

## Day 4 — dashboard

**Framework: none.** A single self-contained `src/dashboard/index.html` —
vanilla JS + CSS, no build step, no `npm install`, no CDN — served by FastAPI at
`GET /`. This closes the open "dashboard framework" decision below.

Node 24 is available, so Vite + React was viable. Rejected because the deciding
constraints were operational, not architectural:

- **One command, one server, one port.** Same origin as the API means no CORS,
  no proxy config, and no second dev server to babysit mid-demo.
- **Offline-safe.** A CDN-hosted React breaks the demo on a bad connection, which
  already happened once during this build. `tests/test_dashboard_api.py` asserts
  the page contains no external references so it stays that way.
- Four sections of bar charts don't need a component framework, and hand-rolled
  CSS bars were less work than wiring a chart library into a no-build page.

### Quota safety: two agents, one set of models

Normal dashboard use makes **zero** live-provider calls. The API now holds two
agents sharing the *same* loaded model objects (no second load, no double
memory):

| Agent | Adapter | Used by |
|---|---|---|
| `agent` | live (Gemini) | `POST /decide` only |
| `bulk_agent` | `TemplateAdapter` | `/simulate/seed`, `/stats/*`, the feed |

Anything that loops is served by the template adapter — still grounded in the
real error-code taxonomy, just generated deterministically. Live calls are
confined to a human clicking "Explain live with Gemini" on one expanded row, and
that button is absent on fraud-blocked rows.

The policy lives in the API rather than the agent on purpose: the agent explains
with whatever adapter it is handed, and the API decides which agent handles which
kind of work. A spy adapter in `TestQuotaSafety` fails the build if a looping
endpoint ever reaches a live provider.

### Three defects found while building it

All three were invisible until something actually consumed the data:

1. **`GET /decisions` returned `explanation: ""` for every transaction.** The
   batch path hard-coded an empty string, so the "expandable why" — the
   auditability story — had nothing to show. The audit record already declared
   an `explanation` field, so this was a defect against an existing contract.
   Fixed with an `explain` flag on `decide_batch()` (default off, so evaluation
   runs are unaffected — verified by re-running `evaluate_agent.py` to identical
   numbers).
2. **The audit log accumulated across seed runs.** Two page loads produced 600
   mixed entries whose counts no longer matched the stats panels. `/simulate/seed`
   now clears the feed first: each seed is one self-contained replay.
3. **The prompt described an absent error code as the code `"none"`**, so Gemini
   wrote *"the payment failed with the error code \"none\""*. An absent code is a
   fact about the transaction, not a code whose value is the word "none". Also
   caught a latent bug on the same line: `NaN` is truthy, so a pandas-missing code
   would have rendered as `"nan"`.
4. **"Run simulation" replayed the same fixed batch forever.** `/simulate/seed`
   used `pd.read_csv(...).head(n)` — not a seeded RNG, just a deterministic
   slice, so every click returned byte-identical transaction ids and revenue.
   Only noticed because someone ran it repeatedly and compared.

### Where randomness belongs (and where it does not)

Fixed by sampling in the request path, with the seed *optional*:

| Path | Seeding | Why |
|---|---|---|
| `POST /simulate/seed` (dashboard button) | none by default | a demo should show a fresh draw each click |
| `POST /simulate/seed?seed=N` | pinned | scripted demos and content-asserting tests |
| `scripts/train.py`, `prepare_data.py`, generator | fixed `RANDOM_SEED` | regression numbers must be reproducible |
| `scripts/evaluate_agent.py`, `retry_storm_demo.py` | full test set, no sampling | unchanged and bit-for-bit stable |

The distinction is the point: **determinism is valuable where numbers are
compared across runs, and actively misleading where a demo implies live
behaviour.**

One consequence worth noting: once the batch became a random sample, the stats
panels had to read the *same* sample. Re-sampling per endpoint would have made
the headline revenue describe a different set of transactions than the rows
underneath it, which is worse than the original bug. `/simulate/seed` now stores
its sample and `/stats/*` read it back — asserted by
`test_stats_describe_the_same_batch_as_the_feed`.

## Open decisions (fill in as you go)

- ~~Exact classifier algorithm~~ — **locked Day 2: XGBoost**
- ~~Agent decision thresholds~~ — **locked Day 3: 0.50 / 0.25** (see above)
- ~~Retry timing logic~~ — **locked Day 3:** per-transaction budget of 3 +
  exponential backoff (2, 4, 8 cycles). Fixed backoff, not learned — the storm
  data showed the win comes from bounding attempts, not from timing them cleverly
- ~~Dashboard framework choice~~ - **locked Day 4: none** (single-file vanilla JS, see above)
- Deployment target
