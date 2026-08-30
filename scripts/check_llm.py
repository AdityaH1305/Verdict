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

    # --- grounding checks ---
    #
    # Grounding means "says only what the taxonomy says", not "quotes it
    # verbatim". Rewording the verified meaning into plain language is the job;
    # demanding an exact substring would fail good explanations. So the meaning
    # check looks for the load-bearing CONCEPT (a timeout), not the phrasing.
    #
    # The error-code citation is checked literally, because the system prompt
    # explicitly requires it -- operations staff need the exact code to quote
    # back to the bank.
    truth = lookup(SAMPLE_TXN["error_code"], SAMPLE_TXN["payment_method"])
    lowered = explanation.lower()
    problems = []

    if SAMPLE_TXN["error_code"].lower() not in lowered:
        problems.append(f"does not cite the actual error code {SAMPLE_TXN['error_code']}")

    if not any(w in lowered for w in ("timed out", "timeout", "time out", "timing out")):
        problems.append(f"does not reflect the verified meaning ({truth.meaning})")

    invented = [t for t in INVENTED_TERMS if t in lowered]
    if invented:
        problems.append(f"introduces terminology it was never given: {invented}")

    # The action was decided before the model was called; it must not propose
    # a different one.
    contradictions = [a for a in ("no action", "escalate", "nudge", "do not retry")
                      if a in lowered]
    if contradictions:
        problems.append(f"contradicts the chosen action ({ACTION}): {contradictions}")

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
