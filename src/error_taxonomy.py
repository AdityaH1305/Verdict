"""
Grounded error-code taxonomy: code -> what it actually means.

This is the factual base the LLM explanation is constrained to. Without it the
model is left to free-associate plausible-sounding bank terminology from a bare
code like "U67" -- see docs/failure_stories.md, Candidate 3. The adapter passes
the matching entry into the prompt and instructs the model to explain only what
is here.

Distinct from the generator's CARD_CODE_GIVEN_CATEGORY / UPI_CODE_GIVEN_CATEGORY
dicts, which are *emission probabilities*. This table is *meaning*.

Codes verified against public sources (see docs/decisions.md, "Day 1 -- error
code taxonomy verification"): ISO 8583 processor references for cards, and a
bank-published NPCI UPI error code reference for UPI.
"""

from typing import Optional

CARD = "card"
UPI = "upi"


class ErrorCodeInfo:
    __slots__ = ("code", "method", "meaning", "who_declined", "retryable")

    def __init__(self, code, method, meaning, who_declined, retryable):
        self.code = code
        self.method = method
        self.meaning = meaning
        # which party in the chain produced the failure -- keeps the explanation
        # from blaming the merchant for an issuer-side event, or vice versa
        self.who_declined = who_declined
        # whether retrying the SAME transaction unchanged could plausibly work
        self.retryable = retryable

    def as_prompt_fact(self) -> str:
        retry_note = {
            True: "retrying unchanged can plausibly succeed",
            False: "retrying unchanged cannot succeed; the underlying condition must change first",
            None: "whether a retry helps depends on why it occurred",
        }[self.retryable]
        return (f"{self.code} ({self.method.upper()}): {self.meaning}. "
                f"Raised by: {self.who_declined}. Retry outlook: {retry_note}.")


_ENTRIES = [
    # --- Card, ISO 8583 ---
    ErrorCodeInfo("51", CARD, "Insufficient funds in the cardholder's account",
                  "the card issuer", False),
    ErrorCodeInfo("05", CARD,
                  "Do not honor -- a generic refusal the issuer returns without "
                  "stating a specific reason. It is deliberately opaque and can "
                  "cover risk, funds, or issuer policy",
                  "the card issuer", None),
    ErrorCodeInfo("14", CARD, "Invalid card number", "the card issuer", False),
    ErrorCodeInfo("54", CARD, "Expired card", "the card issuer", False),
    ErrorCodeInfo("61", CARD, "Amount exceeds the issuer's withdrawal/activity limit",
                  "the card issuer", False),
    ErrorCodeInfo("91", CARD, "Issuer unavailable or timed out before responding",
                  "the issuer's systems", True),
    ErrorCodeInfo("96", CARD, "System malfunction in the processing chain",
                  "the payment network", True),
    ErrorCodeInfo("59", CARD, "Suspected fraud -- the issuer declined on risk grounds",
                  "the issuer's risk system", False),

    # --- UPI, NPCI ---
    ErrorCodeInfo("Z9", UPI, "Insufficient funds in the remitter's account",
                  "the remitting bank", False),
    ErrorCodeInfo("ZH", UPI, "Invalid virtual payment address (VPA)",
                  "the remitting bank", False),
    ErrorCodeInfo("Z8", UPI, "Per-transaction limit exceeded, as set by the remitting bank",
                  "the remitting bank", False),
    ErrorCodeInfo("Z7", UPI, "Transaction frequency limit exceeded, as set by the remitting bank",
                  "the remitting bank", False),
    ErrorCodeInfo("ZU", UPI, "Limit exceeded for the remitting/issuing bank",
                  "the remitting bank", False),
    ErrorCodeInfo("ZM", UPI, "Invalid or incorrect UPI PIN entered",
                  "the remitting bank", False),
    ErrorCodeInfo("UT", UPI, "Remitter/issuer unavailable -- request timed out",
                  "the remitting bank", True),
    ErrorCodeInfo("BT", UPI, "Acquirer/beneficiary unavailable -- request timed out",
                  "the beneficiary bank", True),
    ErrorCodeInfo("U28", UPI, "Remitter bank not available",
                  "the remitting bank", True),
    ErrorCodeInfo("Y1", UPI, "Beneficiary core banking system offline",
                  "the beneficiary bank", True),
    ErrorCodeInfo("XY", UPI, "Remitter core banking system offline",
                  "the remitting bank", True),
    ErrorCodeInfo("U67", UPI, "Debit timed out on the remitter side",
                  "the remitting bank", True),
    ErrorCodeInfo("U68", UPI, "Credit timed out on the beneficiary side",
                  "the beneficiary bank", True),
    ErrorCodeInfo("U30", UPI,
                  "Debit failed. This code is genuinely ambiguous in production: it "
                  "is returned both for bank-side debit failures and for cases "
                  "involving repeated incorrect PIN entry or daily limits",
                  "the remitting bank", None),
    ErrorCodeInfo("ZI", UPI,
                  "Suspected fraud -- declined on risk score by the beneficiary side",
                  "the beneficiary bank's risk system", False),
    ErrorCodeInfo("59", UPI,
                  "Suspected fraud -- declined on risk score by the remitter side",
                  "the remitting bank's risk system", False),
]

# 59 is used by both rails; key on (code, method) and fall back to code alone.
ERROR_CODES = {(e.code, e.method): e for e in _ENTRIES}
_BY_CODE = {}
for _e in _ENTRIES:
    _BY_CODE.setdefault(_e.code, _e)

ABANDONED_NO_CODE = (
    "No decline code was returned -- the customer left the payment flow before "
    "the transaction reached the bank for authorization."
)


def lookup(error_code: Optional[str], payment_method: Optional[str] = None) -> Optional[ErrorCodeInfo]:
    """
    Resolve a code to its grounded meaning, preferring the payment-method-specific
    entry. Returns None for unknown or absent codes -- callers must handle that
    rather than inventing a meaning.
    """
    if error_code is None or (isinstance(error_code, float)) or error_code == "":
        return None
    code = str(error_code).strip()
    if payment_method:
        hit = ERROR_CODES.get((code, str(payment_method).lower()))
        if hit is not None:
            return hit
    return _BY_CODE.get(code)


def describe(error_code: Optional[str], payment_method: Optional[str] = None) -> str:
    """Plain-language fact for the prompt / dashboard. Never guesses."""
    info = lookup(error_code, payment_method)
    if info is not None:
        return info.as_prompt_fact()
    if error_code is None or str(error_code) in ("", "nan"):
        return ABANDONED_NO_CODE
    return (f"{error_code}: no description available for this code in the "
            f"taxonomy. Do not speculate about what it means.")
