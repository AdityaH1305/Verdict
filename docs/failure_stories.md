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

- [ ] Log/metric showing retry count before the fix
- [ ] The fix: per-transaction retry budget + backoff
- [ ] Log/metric showing the same batch after the fix

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
