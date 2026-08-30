"""
Verify the live LLM path end to end.

    python scripts/check_llm.py

Makes ONE real API call with whichever provider key is configured, prints the
explanation, and checks it stayed grounded in the error-code taxonomy. Kept as a
separate script because the rest of the test suite must run offline and free.

Exit code is non-zero if the active adapter is not live, so this doubles as a
smoke check before a demo.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.llm_adapter import get_adapter, load_env  # noqa: E402
from src.error_taxonomy import lookup  # noqa: E402

# A UPI debit timeout: unambiguous code, so the explanation has something
# concrete to be graded against.
SAMPLE_TXN = {
    "transaction_id": "txn_demo",
    "payment_method": "upi",
    "error_code": "U67",
    "amount": 2450.0,
    "retry_count": 1,
}
CATEGORY = "soft_decline"
RECOVERY_PROB = 0.55
ACTION = "auto_retry_now"

# Terminology the model was never given. If any of it appears, the explanation
# is not grounded -- it is free-associating bank vocabulary (Candidate 3).
INVENTED_TERMS = [
    "cvv", "3ds", "chargeback", "acquirer bank", "issuer bank",
    "insufficient funds", "expired", "stolen", "blocked card", "do not honor",
]

# What the explanation must CONVEY about SAMPLE_TXN's failure.
#
# Grounding means "says what the taxonomy says", not "repeats its wording".
# Rewording the verified meaning into plain language is the job, so paraphrase
# has to pass: a bank that "did not respond in time" has timed out, and for a
# merchant-ops reader that phrasing is arguably clearer than the jargon.
#
# Each entry is one required concept and the lexical families that express it.
# The explanation must hit at least one form. An earlier version listed only
# morphological variants of the single word "timeout" while claiming to check
# the concept -- it failed a correct explanation for using different words.
REQUIRED_CONCEPTS = {
    "the failure was a timeout / the bank did not respond": [
        # the jargon
        "timed out", "timeout", "time out", "timing out",
        # plain-language equivalents
        "did not respond", "didn't respond", "not respond", "no response",
        "never responded", "failed to respond", "without responding",
        "unresponsive", "no reply", "did not reply", "took too long",
        "exceeded the time", "not answer", "no answer",
    ],
}


def grade_explanation(explanation: str) -> list:
    """
    Return a list of grounding problems with `explanation`; empty means it passed.

    Two kinds of check, deliberately held to different standards:

      The error CODE is matched literally, because the system prompt explicitly
      requires citing it -- operations staff need the exact string to quote back
      to the bank.

      The MEANING is matched semantically (see REQUIRED_CONCEPTS), because
      restating it in plain language is the job. Demanding the taxonomy's own
      wording would fail good explanations, which is exactly what happened: a
      correct "the bank did not respond in time" was flagged because the checker
      only accepted variants of the word "timeout".

    Separated from main() so the grader can be tested against known-good and
    known-bad text without spending an API call -- a grader nobody tests is how
    the lexical-vs-semantic bug survived in the first place.
    """
    truth = lookup(SAMPLE_TXN["error_code"], SAMPLE_TXN["payment_method"])
    lowered = explanation.lower()
    problems = []

    if SAMPLE_TXN["error_code"].lower() not in lowered:
        problems.append(f"does not cite the actual error code {SAMPLE_TXN['error_code']}")

    for concept, forms in REQUIRED_CONCEPTS.items():
        if not any(f in lowered for f in forms):
            problems.append(
                f"does not convey that {concept} "
                f"(verified meaning: {truth.meaning})"
            )

    invented = [t for t in INVENTED_TERMS if t in lowered]
    if invented:
        problems.append(f"introduces terminology it was never given: {invented}")

    # The action was decided before the model was called; it must not propose
    # a different one.
    contradictions = [a for a in ("no action", "escalate", "nudge", "do not retry")
                      if a in lowered]
    if contradictions:
        problems.append(f"contradicts the chosen action ({ACTION}): {contradictions}")

    return problems


def main():
    load_env()
    adapter = get_adapter()
    info = adapter.describe_self()

    print(f"adapter : {info['adapter']}")
    print(f"provider: {info['provider']}")
    print(f"model   : {info['model']}")
    print(f"live    : {info['live']}")

    if not info["live"]:
        print("\nNo provider key found. Set GEMINI_API_KEY in .env "
              "(https://aistudio.google.com/apikey) and re-run.")
        return 1

    print(f"\nSending one real request to {info['provider']}...\n")
    explanation = adapter.generate_explanation(
        SAMPLE_TXN, CATEGORY, RECOVERY_PROB, ACTION
    )
    print("-" * 72)
    print(explanation)
    print("-" * 72)

    if "used template" in explanation or "empty LLM response" in explanation:
        print("\nThe call did not reach the model -- it degraded to the template. "
              "Check the key and any rate limits.")
        return 1

    problems = grade_explanation(explanation)

    print()
    if problems:
        print("GROUNDING PROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("Grounded: cites the real code, reflects the verified meaning, "
          "invents no bank terminology.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
