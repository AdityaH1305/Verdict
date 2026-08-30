# "What broke" tracking

The buildathon explicitly wants a story about what broke during
development and how you recovered. Fill these in as they actually happen
— don't fabricate, but these three are designed into the plan so at least
one is very likely to occur naturally. Pick the most honest, most
technically interesting one for the pitch video.

## Candidate 1: Classifier confusion on soft_decline vs customer_dropoff

**Designed trap:** the data generator gives these two categories
overlapping feature signatures for certain UPI cases (e.g. a timeout that
happens because the customer was slow at PIN entry — is that a system
issue or a human one?).

**Status after Day 2: confirmed, and it is the dominant error mode.**

- [x] **Confusion matrix showing the actual confusion rate** —
      `reports/confusion_matrix.png`. **75 of 91 total test errors (82%) are
      this one pair**: 53 `customer_dropoff` → `soft_decline`, 22 the other
      way. Recall is 0.88 / 0.94 for the pair versus 0.99 for both
      `hard_decline` and `fraud_block`. Restricted to **cards only**, where
      there is no `time_since_app_switch` signal, accuracy on this pair drops
      to **82.8%**.
- [x] **Explanation of *why* it's genuinely ambiguous** — a bank-side timeout
      is slow *by definition*; that is what a timeout is. So a long
      time-to-failure looks identical whether the bank stalled or the human
      did, and both surface under the *same* wire codes (`91`, `U30`, `U67`,
      `ZM`). The label is a statement about *cause*, but the gateway only
      observes *symptoms*, and here the symptoms genuinely coincide. On cards
      it is worse: without the UPI app-switch signal there is no feature that
      separates them at all. This is not a modelling failure — a perfect model
      could not do much better on this slice, because the information required
      is not in the data. It is not in production either.
      (An intermediate version of the generator accidentally made
      `soft_decline` *fast*, which made drop-off trivially separable and
      quietly trained the ambiguity out of the problem. Caught and fixed —
      see `decisions.md`.)
- [ ] How the agent hedges: uses recovery probability as a tiebreaker
      instead of trusting the hard category label alone
      — **Day 3.** The plumbing is already in place: Model 2 consumes Model 1's
      full predicted *probability vector* rather than a hard label, so an
      ambiguous 0.55/0.45 split reaches the decision layer as uncertainty
      instead of a coin-flip argmax. Worth noting the economics: the two
      categories differ in true recovery rate (0.61 vs 0.38) but *both* are
      worth acting on, just differently (retry vs nudge) — so the hedge is
      cheap. Confusing either with `hard_decline` would be expensive, and the
      model almost never does.

## Candidate 2: Retry-storm

**Designed trap:** build the agent's retry loop first without any cap or
backoff. Run it against a batch of soft_decline transactions and see if
it re-queues the same transaction on every polling cycle.

**Status: confirmed.** Reproduced on 489 retry-eligible transactions from the
held-out set over 12 polling cycles. Run it with
`python scripts/retry_storm_demo.py`; numbers in `reports/retry_storm.json`,
chart in `reports/retry_storm.png`.

- [x] **Log/metric showing retry count before the fix**

  | | Uncapped |
  |---|---|
  | Total retry attempts | **3,025** |
  | Attempts per transaction | mean 6.19, **max 12** |
  | Attempts that could never succeed | **2,556 (84.5% of all attempts)** |
  | Still queued when the run ended | **213** — these retry forever |

  Max 12 on a 12-cycle run means exactly what it looks like: those transactions
  were retried on *every single cycle*. The storm concentrates on the
  transactions least worth retrying — 84.5% of all traffic went to transactions
  that could never succeed, because the ones that *can* succeed leave the queue
  and the ones that can't never do. The failure mode is self-reinforcing.

- [x] **The fix:** per-transaction retry budget of 3 + exponential backoff
      (2, 4, 8 cycles between attempts). Implemented in
      `src/agent/retry_simulator.py`.

      One subtlety that bit during implementation: the budget has to be spent
      the moment the last attempt *fails*, not lazily on the transaction's next
      visit. Enforcing it lazily left transactions whose backoff had pushed them
      past the horizon sitting in the queue looking permanently pending while
      actually being finished — 234 phantom entries and a `budget_exhausted`
      count of 0 that should have been 234.

- [x] **Log/metric showing the same batch after the fix**

  | | Uncapped | Capped + backoff |
  |---|---|---|
  | Total retry attempts | 3,025 | **1,062** (−64.9%) |
  | Max attempts on one transaction | 12 | **3** |
  | Wasted attempts | 2,556 (84.5%) | 639 (60.2%) |
  | Transactions recovered | 276 | 255 |
  | Value recovered | 283,980 | 242,145 |
  | Still queued at end | 213 | **0** |

  **The honest tradeoff:** the budget costs 21 recoveries (−41,835) to save 65%
  of retry traffic. That is not free, and the writeup should not pretend it is —
  three attempts at ~60% each recovers ~94% of what's recoverable, while the
  uncapped loop eventually gets ~100% by hammering the issuer indefinitely. The
  uncapped loop did **2.8× the work for 8% more revenue**, and its worst case is
  unbounded: on a longer run the ratio keeps getting worse while the extra
  revenue stays flat, because everything still in that queue is unrecoverable.

## Candidate 3: LLM hallucinating reason codes

**Designed trap:** first prompt version asks the LLM to explain the
decline reason without passing it a grounded lookup table — just the raw
error code. Check whether it invents plausible-but-wrong bank
terminology.

- [ ] Example of a hallucinated explanation
- [ ] The fix: pass the actual error code + taxonomy lookup table into
      the prompt, constrain the explanation to grounded facts
- [ ] Example of the corrected explanation for the same case

## Actual story used in submission

(Fill in after the build — which one happened, what the real numbers/logs
were, and the exact fix.)
