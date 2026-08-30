"""
Tests for the grounding grader in scripts/check_llm.py.

A grader nobody tests is how the original bug survived: it claimed to check the
CONCEPT of a timeout while only accepting morphological variants of the word
"timeout", so it failed a correct explanation that said the bank "did not respond
in time".

The risk when loosening a checker is over-correcting into one that passes
everything. So these tests come in pairs: real paraphrases must PASS, and
genuinely wrong explanations must still FAIL.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from check_llm import REQUIRED_CONCEPTS, grade_explanation  # noqa: E402


# Real model output. The first is verbatim what Gemini produced and the old
# checker wrongly rejected.
GOOD_EXPLANATIONS = [
    "The payment failed because the remitting bank's system did not respond in "
    "time (error code U67). Given the 55% likelihood of success, we are retrying "
    "this payment automatically now.",

    "The UPI payment for 2450.00 INR failed with error code U67, indicating a "
    "debit timeout on the remitter's bank side. We are automatically retrying "
    "this payment now, as there is a 55% likelihood that a retry will succeed.",

    "This UPI payment failed with code U67 because the customer's bank never "
    "responded to the debit request. Since a retry has a 55% likelihood of "
    "going through, it is being retried right away.",

    "Error code U67 means the remitting bank took too long to confirm the debit. "
    "We're retrying immediately given the 55% chance of success.",

    "The bank's systems were unresponsive when the debit was attempted "
    "(U67), so the payment did not complete. A retry is being attempted now.",
]

# Each of these is wrong in exactly one way the grader must still catch.
BAD_EXPLANATIONS = {
    "no error code cited": (
        "The payment failed because the remitting bank did not respond in time. "
        "We are retrying now given the 55% likelihood of success."
    ),
    "wrong meaning -- invents insufficient funds": (
        "The payment failed with code U67 because there were insufficient funds "
        "in the customer's account. We are retrying now."
    ),
    "hallucinated card terminology on a UPI transaction": (
        "The payment failed with code U67 after the CVV check failed during 3DS "
        "verification and the bank did not respond. Retrying now."
    ),
    "contradicts the chosen action": (
        "Code U67 shows the bank did not respond in time, but we will take no "
        "action on this payment."
    ),
    "says nothing about the nature of the failure": (
        "The payment with error code U67 was unsuccessful. Given the 55% "
        "likelihood of recovery, we are retrying it automatically."
    ),
}


class TestAcceptsParaphrase:
    @pytest.mark.parametrize("text", GOOD_EXPLANATIONS)
    def test_good_explanations_pass(self, text):
        assert grade_explanation(text) == []

    def test_the_exact_text_the_old_checker_rejected(self):
        """Regression: the literal case that prompted the fix."""
        text = ("The payment failed because the remitting bank's system did not "
                "respond in time (error code U67), and there is a 55% likelihood "
                "a retry succeeds, so it is being retried now.")

        assert grade_explanation(text) == []

        # ...and prove the OLD logic would have rejected it, so this test is
        # actually pinning the fix rather than restating current behaviour.
        old_forms = ("timed out", "timeout", "time out", "timing out")
        assert not any(f in text.lower() for f in old_forms)


class TestStillCatchesFailures:
    """The looser checker must not have become vacuous."""

    @pytest.mark.parametrize("label,text", list(BAD_EXPLANATIONS.items()))
    def test_bad_explanations_fail(self, label, text):
        problems = grade_explanation(text)
        assert problems, f"grader accepted a bad explanation ({label})"

    def test_missing_code_is_reported_specifically(self):
        problems = grade_explanation(BAD_EXPLANATIONS["no error code cited"])
        assert any("does not cite" in p for p in problems)

    def test_invented_terminology_is_reported_specifically(self):
        problems = grade_explanation(
            BAD_EXPLANATIONS["hallucinated card terminology on a UPI transaction"])
        assert any("never given" in p for p in problems)

    def test_missing_concept_is_reported_specifically(self):
        problems = grade_explanation(
            BAD_EXPLANATIONS["says nothing about the nature of the failure"])
        assert any("does not convey" in p for p in problems)

    def test_empty_text_fails_everything(self):
        problems = grade_explanation("")
        assert len(problems) >= 2   # no code, no concept


class TestConceptSpec:
    def test_concepts_cover_both_jargon_and_plain_language(self):
        """
        The bug was a concept list containing only one lexical family. Guard it:
        the timeout concept must accept both the jargon and a plain-language
        rewording that shares no words with it.
        """
        forms = REQUIRED_CONCEPTS["the failure was a timeout / the bank did not respond"]
        assert any("timeout" in f or "timed out" in f for f in forms), "jargon missing"
        assert any("respond" in f for f in forms), "plain-language form missing"
