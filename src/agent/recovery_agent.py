"""
Recovery Agent: combines Model 1 + Model 2 outputs into a decision.

Decision flow:
  1. Run Model 1 -> failure_category
  2. If category == "fraud_block": return NO_ACTION immediately.
     Do not call Model 2. Do not call the LLM adapter. This is a hard
     architectural rule, not a learned preference -- see docs/decisions.md.
  3. Otherwise, run Model 2 -> recovery_success_probability
  4. Apply decision thresholds -> action
     (auto_retry_now | retry_later | customer_nudge | escalate | no_action)
  5. Call LLM adapter for a plain-language explanation of the decision
  6. Log the full decision (category, probability, action, explanation)
     for the dashboard's audit feed

TODO:
  - Load Model 1 + Model 2 artifacts
  - Implement decision thresholds (design these deliberately -- this is
    the crux of the pitch: the agent decides economically, not reflexively)
  - Implement per-transaction retry budget / backoff
    (see docs/failure_stories.md, Candidate 2)
  - Wire up LLMAdapter call
  - Structured logging for dashboard consumption
"""

class RecoveryAgent:
    def decide(self, transaction: dict) -> dict:
        raise NotImplementedError
