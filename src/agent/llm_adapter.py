"""
LLM adapter interface -- provider-swappable by design.

The agent must never call an LLM SDK directly. All reasoning/explanation
generation goes through this interface so swapping providers later is a
one-file change.

IMPORTANT: this adapter must never be called for fraud_block transactions.
That rule is enforced by the caller (recovery_agent.py), but keep it in
mind when writing prompts -- there should be no code path where a fraud
case reaches generate_explanation().
"""

from abc import ABC, abstractmethod


class LLMAdapter(ABC):
    @abstractmethod
    def generate_explanation(self, transaction: dict, category: str,
                              recovery_prob: float, action: str) -> str:
        """Return a plain-language explanation of the decision."""
        raise NotImplementedError


class ClaudeAdapter(LLMAdapter):
    """
    TODO:
      - Wire up Claude API call (Haiku-class model)
      - Ground the prompt in the actual error code + taxonomy lookup table
        (see docs/failure_stories.md, Candidate 3 -- do not let the model
        guess at bank terminology)
      - Keep prompts strict: no speculation beyond provided facts
    """

    def generate_explanation(self, transaction: dict, category: str,
                              recovery_prob: float, action: str) -> str:
        raise NotImplementedError
