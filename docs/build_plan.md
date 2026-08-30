# Build plan

Solo build, 6-7 days, deadline Sept 5. This is the concrete execution plan
-- read this alongside README.md (architecture) and decisions.md (rationale).

## Day-by-day

**Day 1 -- Data + problem framing**
Design and generate the synthetic dataset (see feature spec + error code
taxonomy below). Write down generation assumptions in decisions.md as you
make them. Target ~5,000-10,000 rows.

**Day 2 -- ML models**
Train Model 1 (failure classifier) and Model 2 (recovery success
probability). Evaluate properly: confusion matrix + per-class
precision/recall for Model 1, AUC + calibration curve for Model 2.

**Day 3 -- Agent + backend**
FastAPI backend. Agent decision logic (fraud hard-block first, then
threshold logic on category + probability -> action). LLM adapter wired
to Claude API. Build the retry loop WITHOUT a cap first (deliberately --
see failure_stories.md Candidate 2), observe retry-storm behavior, then
add a per-transaction retry budget + backoff and log before/after.

**Day 4 -- Frontend dashboard**
Failure category breakdown chart, recovered revenue estimate, live
decision feed with expandable "why" per transaction.

**Day 5 -- Integration + demo scenario**
Wire everything together. Seed a demo scenario that tells a clear story
(e.g. "500 failed transactions in, agent recovers X%, here's the
breakdown"). Fix integration bugs.

**Day 6 -- Deploy + failure story**
Deploy publicly (Vercel/Render/Railway or similar -- pick whatever is
fastest to stand up). Confirm which of the three candidate failure
stories in failure_stories.md actually happened, document it with real
numbers/logs. Write the architecture doc (README.md + decisions.md are
most of it already).

**Day 7 (buffer) -- Pitch video + polish**
Record the 5-minute pitch. Clean repo/README. Final polish.

## Feature spec for the data generator

### Shared features (all transactions)
- `amount`
- `time_of_day`
- `day_of_week`
- `retry_count` (retries so far on this transaction)
- `customer_txn_history_count`
- `customer_past_failure_rate`
- `merchant_category`
- `payment_method` (card | upi)

### Card-specific features
- `card_network` (Visa | Mastercard | Rupay | Amex)
- `issuing_bank`
- `error_code` (from card taxonomy below)
- `is_3ds_flow` (bool)
- `card_age_months`

### UPI-specific features
- `psp_app` (GPay | PhonePe | Paytm | other)
- `issuing_bank`
- `error_code` (from UPI taxonomy below)
- `time_since_app_switch`
- `daily_limit_utilization`

### Labels
- `failure_category`: hard_decline | soft_decline | customer_dropoff | fraud_block
- `retry_success`: bool (always False for fraud_block by construction)

## Error code taxonomy (ground the generator in these -- real, public codes)

These are illustrative reference points, not exhaustive. Verify/expand
against current published ISO 8583 card response codes and NPCI UPI
response codes before finalizing -- codes and descriptions can be revised
by networks, so don't hardcode from memory without a spot-check.

**Card decline codes (ISO 8583-style, commonly used across networks):**
| Code | Meaning | Typical category |
|------|---------|-------------------|
| 51 | Insufficient funds | hard_decline |
| 05 | Do not honor | hard_decline |
| 14 | Invalid card number | hard_decline |
| 54 | Expired card | hard_decline |
| 61 | Exceeds withdrawal limit | hard_decline |
| 91 | Issuer unavailable/timeout | soft_decline |
| 96 | System malfunction | soft_decline |
| 59 | Suspected fraud | fraud_block |
| N/A (user abandons at 3DS/OTP) | -- | customer_dropoff |

**UPI decline codes (NPCI-style, commonly referenced):**
| Code | Meaning | Typical category |
|------|---------|-------------------|
| U69 | Insufficient balance | hard_decline |
| U30 | Bank server not responding / timeout | soft_decline |
| U16 | Invalid VPA | hard_decline |
| ZM | Transaction limit exceeded | hard_decline |
| U91 | Issuer unavailable | soft_decline |
| -- | User abandons before PIN entry | customer_dropoff |
| -- | Flagged by risk engine | fraud_block |

**Action item for Day 1:** search for the current official ISO 8583
response code list and NPCI UPI error code documentation to confirm/expand
this table before generating data -- do not ship the demo on
unverified codes if avoidable.

## Deliberate overlap for Candidate 1 (classifier ambiguity)

For a subset of UPI transactions, generate cases where a timeout occurs
because the customer was slow entering their PIN (not a genuine bank-side
timeout). These should have soft_decline-like features (error_code = U30
range) but customer_dropoff-like behavioral features (long
time_since_app_switch). Don't label these cleanly -- let the ambiguity be
real in the generated ground truth, not just in the model's confusion.

## Decision thresholds (agent) -- starting point, tune during Day 3

Not finalized -- pick reasonable starting cutoffs, log actual score
distributions from Model 2, then adjust:

- `fraud_block` -> always `no_action` (hard rule, no threshold involved)
- `recovery_prob >= 0.6` -> `auto_retry_now`
- `0.3 <= recovery_prob < 0.6` -> `retry_later` (apply backoff) or
  `customer_nudge`, depending on category (soft_decline leans retry,
  customer_dropoff leans nudge)
- `recovery_prob < 0.3` -> `escalate` (merchant dashboard flag) or
  `no_action` if category is hard_decline (no point escalating an
  unrecoverable decline)

Document whatever you actually land on in decisions.md once tuned.
