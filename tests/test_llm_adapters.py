"""
Adapter tests -- provider swapping, grounding, and graceful degradation.

The free-tier Gemini quota is real, so "what happens on a 429" is not a
hypothetical: it is the expected steady state under load. These tests pin the
behaviour that a rate-limited explanation degrades to the grounded template
rather than taking down the decision endpoint.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.llm_adapter import (  # noqa: E402
    DEFAULT_GEMINI_MODEL, ClaudeAdapter, GeminiAdapter, LLMAdapter,
    TemplateAdapter, get_adapter,
)

TXN = {"payment_method": "upi", "error_code": "U67", "amount": 1200.0, "retry_count": 1}


def _fake_response(text, finish_reason="STOP"):
    candidate = type("Cand", (), {"finish_reason": finish_reason})()
    return type("Resp", (), {"text": text, "candidates": [candidate]})()


class _FakeModels:
    """Stands in for client.models -- raises what we tell it to, or returns text."""

    def __init__(self, raise_exc=None, text=None, finish_reason="STOP"):
        self._raise = raise_exc
        self._text = text
        self._finish_reason = finish_reason
        self.calls = 0
        self.last_kwargs = None

    def generate_content(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if self._raise is not None:
            raise self._raise
        return _fake_response(self._text, self._finish_reason)


def _adapter_with(raise_exc=None, text=None, finish_reason="STOP") -> GeminiAdapter:
    adapter = GeminiAdapter.__new__(GeminiAdapter)   # skip __init__/network
    adapter.model = DEFAULT_GEMINI_MODEL
    adapter.grounded = True
    adapter._client = type(
        "C", (), {"models": _FakeModels(raise_exc, text, finish_reason)}
    )()
    return adapter


def _api_error(code: int):
    from google.genai import errors
    return errors.APIError(code, {"error": {"message": f"simulated {code}"}})


class TestGeminiAdapter:
    def test_implements_the_interface(self):
        assert issubclass(GeminiAdapter, LLMAdapter)
        assert GeminiAdapter.provider == "google"
        # flash-class only -- Pro models are not free-tier eligible
        assert "flash" in DEFAULT_GEMINI_MODEL

    def test_returns_model_text_on_success(self):
        adapter = _adapter_with(text="The bank timed out; retrying now.")
        out = adapter.generate_explanation(TXN, "soft_decline", 0.55, "auto_retry_now")
        assert out == "The bank timed out; retrying now."

    def test_rate_limit_falls_back_to_template(self):
        """A 429 must not surface as an exception to /decide."""
        adapter = _adapter_with(raise_exc=_api_error(429))
        out = adapter.generate_explanation(TXN, "soft_decline", 0.55, "auto_retry_now")

        assert "rate limited" in out
        # still a real, grounded explanation -- not an error string
        assert "Debit timed out on the remitter side" in out

    def test_other_api_errors_fall_back_to_template(self):
        adapter = _adapter_with(raise_exc=_api_error(500))
        out = adapter.generate_explanation(TXN, "soft_decline", 0.55, "auto_retry_now")
        assert "used template" in out
        assert "Debit timed out on the remitter side" in out

    def test_network_failure_falls_back_to_template(self):
        adapter = _adapter_with(raise_exc=ConnectionError("no route to host"))
        out = adapter.generate_explanation(TXN, "soft_decline", 0.55, "auto_retry_now")
        assert "used template" in out

    def test_empty_response_falls_back_to_template(self):
        adapter = _adapter_with(text="")
        out = adapter.generate_explanation(TXN, "soft_decline", 0.55, "auto_retry_now")
        assert "empty LLM response" in out

    def test_truncated_response_falls_back_to_template(self):
        """
        Regression: a MAX_TOKENS finish must not pass the fragment through.

        Gemini 2.5's thinking tokens are billed against max_output_tokens, so an
        overspent budget returns a partial sentence that reads as authoritative
        while stopping mid-clause. The first version shipped exactly that.
        """
        adapter = _adapter_with(text="This UPI payment failed due to a debit timeout on the",
                                 finish_reason="FinishReason.MAX_TOKENS")
        out = adapter.generate_explanation(TXN, "soft_decline", 0.55, "auto_retry_now")

        assert "truncated" in out
        assert out.endswith("(LLM response truncated, used template)")
        # a complete, grounded explanation replaced the fragment
        assert "Debit timed out on the remitter side" in out

    def test_thinking_is_disabled(self):
        """
        Root cause of the truncation bug: reasoning tokens ate the answer budget.

        This task is grounded restatement of supplied facts -- there is nothing
        to reason about, so the whole budget belongs to the answer.
        """
        adapter = _adapter_with(text="ok")
        adapter.generate_explanation(TXN, "soft_decline", 0.55, "auto_retry_now")

        config = adapter._client.models.last_kwargs["config"]
        assert config.thinking_config is not None
        assert config.thinking_config.thinking_budget == 0

    def test_prompt_requires_citing_the_error_code(self):
        """Ops staff need the literal code to quote back to the bank."""
        adapter = _adapter_with(text="ok")
        adapter.generate_explanation(TXN, "soft_decline", 0.55, "auto_retry_now")

        instruction = adapter._client.models.last_kwargs["config"].system_instruction
        assert "cite the bank's error code" in instruction.lower()

    def test_refuses_fraud_before_any_api_call(self):
        adapter = _adapter_with(text="should never be produced")
        with pytest.raises(AssertionError, match="fraud_block"):
            adapter.generate_explanation(TXN, "fraud_block", 0.9, "auto_retry_now")
        assert adapter._client.models.calls == 0, "no request may be made for fraud"

    def test_prompt_carries_the_grounded_taxonomy_entry(self):
        """The verified meaning must reach the model, not just the bare code."""
        captured = {}

        adapter = _adapter_with(text="ok")

        def spy(**kwargs):
            captured.update(kwargs)
            return type("Resp", (), {"text": "ok"})()

        adapter._client.models.generate_content = spy
        adapter.generate_explanation(TXN, "soft_decline", 0.55, "auto_retry_now")

        contents = captured["contents"]
        assert "U67" in contents
        assert "Debit timed out on the remitter side" in contents
        assert "VERIFIED MEANING" in contents
        # the model is told the decision is already made
        assert "ALREADY been made" in captured["config"].system_instruction

    def test_ungrounded_mode_withholds_the_taxonomy(self):
        """Candidate 3 reproduction must actually remove the grounding."""
        captured = {}
        adapter = _adapter_with(text="ok")
        adapter.grounded = False

        def spy(**kwargs):
            captured.update(kwargs)
            return type("Resp", (), {"text": "ok"})()

        adapter._client.models.generate_content = spy
        adapter.generate_explanation(TXN, "soft_decline", 0.55, "auto_retry_now")

        assert "U67" in captured["contents"]
        assert "VERIFIED MEANING" not in captured["contents"]


class TestAdapterSelection:
    def setup_method(self):
        self._saved = {k: os.environ.pop(k, None)
                       for k in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY")}

    def teardown_method(self):
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_no_keys_uses_template(self, monkeypatch):
        monkeypatch.setattr("src.agent.llm_adapter.load_env", lambda: None)
        assert isinstance(get_adapter(), TemplateAdapter)

    def test_gemini_wins_when_both_keys_present(self, monkeypatch):
        monkeypatch.setattr("src.agent.llm_adapter.load_env", lambda: None)
        monkeypatch.setenv("GEMINI_API_KEY", "fake")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
        assert isinstance(get_adapter(), GeminiAdapter)

    def test_falls_back_to_claude_when_only_anthropic_key(self, monkeypatch):
        monkeypatch.setattr("src.agent.llm_adapter.load_env", lambda: None)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
        assert isinstance(get_adapter(), ClaudeAdapter)

    def test_prefer_llm_false_forces_template(self):
        assert isinstance(get_adapter(prefer_llm=False), TemplateAdapter)


class TestProviderParity:
    """Every adapter must honour the same contract."""

    @pytest.mark.parametrize("cls", [GeminiAdapter, ClaudeAdapter, TemplateAdapter])
    def test_declares_itself_for_health_reporting(self, cls):
        assert issubclass(cls, LLMAdapter)
        assert cls.provider != "unknown"
        assert hasattr(cls, "is_live")

    @pytest.mark.parametrize("cls", [GeminiAdapter, ClaudeAdapter, TemplateAdapter])
    def test_signature_is_identical(self, cls):
        import inspect
        params = list(inspect.signature(cls.generate_explanation).parameters)
        assert params == ["self", "transaction", "category", "recovery_prob", "action"]
