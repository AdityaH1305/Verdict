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

- [ ] Confusion matrix showing the actual confusion rate
- [ ] Explanation of *why* it's genuinely ambiguous, not just a modeling
      failure
- [ ] How the agent hedges: uses recovery probability as a tiebreaker
      instead of trusting the hard category label alone

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
