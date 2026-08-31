"""
LLM adapter interface -- provider-swappable by design.

The agent must never call an LLM SDK directly. All reasoning/explanation
generation goes through this interface so swapping providers later is a
one-file change.

IMPORTANT: this adapter must never be called for fraud_block transactions.
That rule is enforced by the caller (recovery_agent.py), but keep it in
mind when writing prompts -- there should be no code path where a fraud
case reaches generate_explanation().

Two things this module is careful about:

1. **The LLM explains a decision; it never makes one.** The action is already
   chosen by the agent's threshold policy before the adapter is called. The
   prompt says so explicitly, so no prompt-level behaviour can produce a
   different action than the one that was logged.

2. **Explanations are grounded in the real error code.** The prompt carries the
   actual code plus its entry from src/error_taxonomy.py, and forbids
   speculation beyond those facts. Handed a bare code like "U67", a model will
   happily invent plausible-sounding bank terminology -- see
   docs/failure_stories.md, Candidate 3. `grounded=False` reproduces that
   failure mode on purpose so the before/after can be captured.

Providers
---------
Three implementations, all behind the same interface:

  GeminiAdapter   -- active path (Google, free tier)
  ClaudeAdapter   -- retained and working, dormant without an ANTHROPIC_API_KEY
  TemplateAdapter -- deterministic offline fallback

The provider swap from Claude to Gemini touched this file and nothing else --
no agent logic, no thresholds, no fraud rules. That was the point of putting an
interface here in the first place, and it is now demonstrated rather than
merely claimed.
"""

import math
import os
import sys
from abc import ABC, abstractmethod
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.error_taxonomy import describe, lookup  # noqa: E402

# Haiku-class: this is high-volume per-transaction reasoning, not a task that
# needs a large model (see docs/decisions.md, "LLM choice").
DEFAULT_MODEL = "claude-haiku-4-5"

# Gemini free tier. Flash-class for the same reason as Haiku-class above; Pro
# models are not free-tier eligible. flash-lite over flash for demo headroom --
# the free-tier request rate is the binding constraint when narrating a batch,
# not model quality, and this task is grounded restatement rather than reasoning.
# "gemini-2.5-flash" is the step up if explanation quality ever needs it.
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"

MAX_TOKENS = 300


def load_env():
    """
    Load .env so keys can live in a file rather than the shell.

    Optional dependency: a missing python-dotenv degrades to real environment
    variables instead of breaking the agent.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()

SYSTEM_PROMPT = """You explain payment-failure recovery decisions to merchant \
operations staff.

The decision has ALREADY been made by a separate rules-and-models layer. Your \
only job is to explain it clearly. You must never suggest, second-guess, or \
imply a different action.

Rules:
- Use ONLY the facts given to you. Never invent bank terminology, error code \
meanings, or reasons that were not provided.
- ALWAYS cite the bank's error code verbatim when one was returned. Operations \
staff need the exact code to quote back to the bank, so it must appear in your \
explanation.
- Describe the failure using the verified meaning supplied to you. You may \
reword it into plain language, but you may not change what it says or add \
causes it does not mention.
- If the provided facts do not explain something, say so plainly rather than \
speculating.
- 2-3 sentences, plain language, no bullet points, no preamble.
- Write for someone who understands payments but not machine learning. Refer to \
the recovery probability as a likelihood, not a "model score"."""


class LLMAdapter(ABC):
    # Declared on the class so callers (e.g. the API's /health) can report which
    # provider is live without an isinstance chain that grows with every new one.
    provider = "unknown"
    model = None
    is_live = False

    @abstractmethod
    def generate_explanation(self, transaction: dict, category: str,
                              recovery_prob: float, action: str) -> str:
        """Return a plain-language explanation of the decision."""
        raise NotImplementedError

    def describe_self(self) -> dict:
        return {"adapter": type(self).__name__, "provider": self.provider,
                "model": self.model, "live": self.is_live}


def _reject_fraud(category: str) -> None:
    """
    Defence in depth, shared by every adapter.

    The agent already guarantees this is unreachable. If it ever becomes
    reachable, fail loudly rather than let a model write prose that rationalises
    a fraud recovery.
    """
    if category == "fraud_block":
        raise AssertionError(
            "LLM adapter invoked for a fraud_block transaction -- this path "
            "must not exist. See docs/decisions.md, 'Hard rules'."
        )


def _build_facts(transaction: dict, category: str, recovery_prob: float,
                  action: str, grounded: bool = True) -> str:
    """Assemble the grounded fact sheet handed to the model."""
    method = transaction.get("payment_method")
    code = transaction.get("error_code")

    lines = [
        f"Payment method: {method}",
        f"Amount: {transaction.get('amount')}",
        f"Diagnosed failure category: {category}",
        f"Estimated likelihood a recovery attempt succeeds: {recovery_prob:.0%}",
        f"Action chosen by the decision layer: {action}",
        f"Retries already attempted: {transaction.get('retry_count', 0)}",
    ]

    # An absent code is a fact about the transaction, not a code whose value is
    # the word "none". Rendering it that way made the model write: 'the payment
    # failed with the error code "none"'. Note NaN is truthy, so a pandas-missing
    # value has to be caught explicitly or it renders as "nan".
    has_code = not (code is None or (isinstance(code, float) and math.isnan(code))
                    or str(code).strip().lower() in ("", "nan", "none"))

    if has_code:
        lines.append(f"Error code returned by the bank: {code}")
        if grounded:
            # The whole point: the code's meaning comes from the verified
            # taxonomy, not from whatever the model associates with the string.
            lines.append(f"VERIFIED MEANING OF THIS ERROR CODE: {describe(code, method)}")
        # else: Candidate 3 reproduction -- bare code, no grounding.
    else:
        lines.append("The bank returned NO error code for this transaction.")
        if grounded:
            lines.append(f"WHAT THAT ABSENCE MEANS: {describe(None, method)}")

    return "\n".join(lines)


class GeminiAdapter(LLMAdapter):
    """
    Gemini-backed explanations -- the active provider.

    Same interface, same grounding rules, same fraud refusal as ClaudeAdapter.
    Free-tier quotas are real, so a rate-limited call degrades to the
    deterministic template rather than taking down /decide: an explanation is a
    narration layer, and a payment decision must not depend on it.

    Set `grounded=False` only to reproduce the Candidate 3 hallucination story;
    production paths must leave it True.
    """

    provider = "google"
    is_live = True

    def __init__(self, model: str = DEFAULT_GEMINI_MODEL, grounded: bool = True,
                 api_key: Optional[str] = None):
        from google import genai  # lazy import: optional offline

        self.model = model
        self.grounded = grounded
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is not set")
        self._client = genai.Client(api_key=key)

    def generate_explanation(self, transaction: dict, category: str,
                              recovery_prob: float, action: str) -> str:
        from google.genai import errors, types

        _reject_fraud(category)

        facts = _build_facts(transaction, category, recovery_prob, action,
                              grounded=self.grounded)
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=f"Explain this decision:\n\n{facts}",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=MAX_TOKENS,
                    # low temperature: this is grounded restatement, not writing
                    temperature=0.2,
                    # Gemini 2.5 is a thinking model and reasoning tokens are
                    # billed against max_output_tokens. Left on, thinking ate 286
                    # of 300 tokens here and the answer was truncated mid-sentence
                    # after 10. This task is a grounded restatement of facts we
                    # already supply -- there is nothing to reason about, so the
                    # budget belongs entirely to the answer.
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
        except errors.APIError as exc:
            code = getattr(exc, "code", None)
            note = ("(rate limited, used template)" if code == 429
                    else f"(LLM unavailable: {code}, used template)")
            return _template_explanation(transaction, category, recovery_prob,
                                          action, note=note)
        except Exception:
            # network failure, malformed response, anything else -- the decision
            # still has to be explainable
            return _template_explanation(transaction, category, recovery_prob,
                                          action, note="(LLM unavailable, used template)")

        # A truncated explanation is worse than no explanation -- it reads as
        # authoritative while stopping mid-sentence. Treat it as a failure rather
        # than passing the fragment through, which is what shipped the first time.
        finish_reason = None
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish_reason = getattr(candidates[0], "finish_reason", None)
        if finish_reason is not None and str(finish_reason).endswith("MAX_TOKENS"):
            return _template_explanation(transaction, category, recovery_prob,
                                          action, note="(LLM response truncated, used template)")

        text = (response.text or "").strip()
        if not text:
            # e.g. the response was blocked before producing any text
            return _template_explanation(transaction, category, recovery_prob,
                                          action, note="(empty LLM response, used template)")
        return text


class ClaudeAdapter(LLMAdapter):
    """
    Claude-backed explanations.

    Retained and working, but dormant: with no ANTHROPIC_API_KEY the selection
    logic picks Gemini instead. Kept deliberately -- the interface exists so the
    provider can be swapped, and keeping both implementations is the evidence
    that it can be.

    Set `grounded=False` only to reproduce the Candidate 3 hallucination story;
    production paths must leave it True.
    """

    provider = "anthropic"
    is_live = True

    def __init__(self, model: str = DEFAULT_MODEL, grounded: bool = True,
                 api_key: Optional[str] = None):
        import anthropic  # imported lazily so the package is optional offline

        self.model = model
        self.grounded = grounded
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def generate_explanation(self, transaction: dict, category: str,
                              recovery_prob: float, action: str) -> str:
        import anthropic

        _reject_fraud(category)

        facts = _build_facts(transaction, category, recovery_prob, action,
                              grounded=self.grounded)
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"Explain this decision:\n\n{facts}",
                }],
            )
        except anthropic.APIStatusError as exc:
            # An explanation is a nice-to-have; a payment decision is not. Never
            # let the narration layer take down the decision path.
            return _template_explanation(transaction, category, recovery_prob,
                                          action, note=f"(LLM unavailable: {exc.status_code})")
        except anthropic.APIConnectionError:
            return _template_explanation(transaction, category, recovery_prob,
                                          action, note="(LLM unavailable: connection error)")

        return "".join(b.text for b in response.content if b.type == "text").strip()


def _template_explanation(transaction: dict, category: str, recovery_prob: float,
                           action: str, note: str = "") -> str:
    """Deterministic explanation built from the same grounded taxonomy."""
    method = transaction.get("payment_method")
    code = transaction.get("error_code")
    info = lookup(code, method)

    if info is not None:
        cause = f"The bank returned code {info.code} - {info.meaning.rstrip('.')} - raised by {info.who_declined}."
    elif code in (None, "", "nan") or (isinstance(code, float)):
        cause = ("No decline code was returned; the customer left the payment flow "
                 "before the transaction reached the bank.")
    else:
        cause = f"The bank returned code {code}, which is not in the known taxonomy."

    reason = {
        "hard_decline": "This is a hard decline, so the underlying condition must change before another attempt can work.",
        "soft_decline": "This is a transient, bank-side failure, so the same transaction can plausibly succeed on another attempt.",
        "customer_dropoff": "The customer abandoned the flow rather than the bank refusing it, so the customer needs prompting rather than a silent retry.",
        "fraud_block": "Flagged as fraud.",
    }.get(category, "")

    action_text = {
        "auto_retry_now": "Retrying immediately.",
        "retry_later": "Scheduling a retry with backoff.",
        "customer_nudge": "Sending the customer a reminder to complete the payment.",
        "escalate": "Flagging this to the merchant for manual review.",
        "no_action": "Taking no further action.",
    }.get(action, f"Action: {action}.")

    body = (f"{cause} {reason} Recovery was estimated at roughly "
            f"{recovery_prob:.0%}, so: {action_text}")
    return f"{body} {note}".strip()


class TemplateAdapter(LLMAdapter):
    """
    Offline fallback. Same grounded taxonomy, no API call.

    Keeps the agent, API, and test suite runnable with no provider key at all,
    so the fraud rules and threshold logic can be tested deterministically and
    the demo never depends on network access.
    """

    provider = "none (deterministic template)"
    is_live = False

    def generate_explanation(self, transaction: dict, category: str,
                              recovery_prob: float, action: str) -> str:
        _reject_fraud(category)
        return _template_explanation(transaction, category, recovery_prob, action)


def get_adapter(prefer_llm: bool = True) -> LLMAdapter:
    """
    Pick the best adapter available in this environment.

    Order: Gemini (free tier, the active path) -> Claude (retained, needs a paid
    key) -> deterministic template. Falls through on a missing SDK or key rather
    than failing, so pytest and the demo run offline.

    Decisions are identical under all three -- only the wording of the
    explanation changes -- because the LLM narrates a decision that has already
    been made.
    """
    if not prefer_llm:
        return TemplateAdapter()

    load_env()

    if os.environ.get("GEMINI_API_KEY"):
        try:
            return GeminiAdapter()
        except Exception:
            pass

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return ClaudeAdapter()
        except Exception:
            pass

    return TemplateAdapter()
