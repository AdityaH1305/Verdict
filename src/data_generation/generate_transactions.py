"""
Synthetic transaction failure data generator.

Generates cards + UPI transactions with method-conditional failure logic,
grounded in real public bank/NPCI response code taxonomies.

Two labels per row:
  - failure_category: hard_decline | soft_decline | customer_dropoff | fraud_block
  - retry_success: bool, whether a retry/nudge would have recovered the transaction
                   (always False for fraud_block by construction)

Design notes (see docs/decisions.md):
  - Cards skew toward hard_decline / fraud_block (issuer/network-driven failures)
  - UPI skews toward soft_decline / customer_dropoff (operational failures)
  - error_code is INFORMATIVE BUT NOT DECISIVE. Categories emit from overlapping
    code distributions, mirroring reality: "05 / do not honor" is a deliberately
    opaque catch-all, and an issuer timeout (91, U67) is exactly where a stalled
    *customer* hides behind a *system* code. The first version of this generator
    gave each category a private, disjoint code list, which made Model 1 a lookup
    table (96.6% accuracy from one column). See docs/decisions.md, "Day 2 --
    why the data was regenerated".
  - The overlap is deliberately RESOLVABLE from behavioural features (dwell time,
    card age, customer history, amount) -- otherwise it would just be irreducible
    noise and the classifier could never beat the naive lookup baseline.

Error code taxonomies below were spot-checked against public sources on
2026-08-30 (see docs/decisions.md, "Day 1 -- error code taxonomy
verification") -- the UPI table in particular differs from the build
plan's illustrative version, which had several codes wrong.
"""

import os

import numpy as np
import pandas as pd

RNG_SEED = 42
N_ROWS = 8000
OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "raw", "transactions.csv",
)

CATEGORIES = ["hard_decline", "soft_decline", "customer_dropoff", "fraud_block"]

CARD_CATEGORY_MIX = {
    "hard_decline": 0.45,
    "soft_decline": 0.20,
    "customer_dropoff": 0.20,
    "fraud_block": 0.15,
}

UPI_CATEGORY_MIX = {
    "soft_decline": 0.35,
    "customer_dropoff": 0.35,
    "hard_decline": 0.20,
    "fraud_block": 0.10,
}

# P(error_code | failure_category) for cards. ISO 8583 codes verified against
# processor references. Overlap is intentional and grounded:
#   05 "do not honor" -- issuers use it to mask fraud suspicion and funds issues
#   61 "exceeds limit" -- velocity caps double as risk controls
#   91/96 timeouts    -- a user stalling at 3DS surfaces as an issuer timeout
#   51/14/54          -- genuinely unambiguous in reality, kept near-pure
CARD_CODE_GIVEN_CATEGORY = {
    "hard_decline":     {"51": 0.24, "05": 0.28, "14": 0.15, "54": 0.17, "61": 0.14, "91": 0.02},
    "soft_decline":     {"91": 0.48, "96": 0.37, "05": 0.11, "61": 0.04},
    "customer_dropoff": {None: 0.52, "91": 0.30, "96": 0.13, "05": 0.05},
    "fraud_block":      {"59": 0.52, "05": 0.33, "61": 0.11, "14": 0.04},
}

# P(error_code | failure_category) for UPI. NPCI codes verified against a
# bank-published reference (see docs/decisions.md). Overlap is intentional:
#   ZM "wrong MPIN"   -- very often ends in the customer abandoning
#   U30              -- the original Candidate 1 trap, kept genuinely even
#   UT/BT/U67/...    -- same stalled-user-behind-a-timeout effect as cards
UPI_CODE_GIVEN_CATEGORY = {
    "hard_decline":     {"Z9": 0.26, "ZH": 0.16, "Z8": 0.14, "Z7": 0.09, "ZU": 0.13,
                         "ZM": 0.19, "U30": 0.03},
    "soft_decline":     {"UT": 0.17, "BT": 0.11, "U28": 0.13, "Y1": 0.12, "XY": 0.10,
                         "U67": 0.12, "U68": 0.10, "U30": 0.15},
    "customer_dropoff": {None: 0.34, "U30": 0.26, "ZM": 0.15, "U67": 0.10, "UT": 0.08,
                         "U68": 0.07},
    "fraud_block":      {"59": 0.48, "ZI": 0.37, "Z8": 0.09, "ZU": 0.06},
}

CARD_NETWORKS = ["Visa", "Mastercard", "Rupay", "Amex"]
CARD_ISSUING_BANKS = ["HDFC", "ICICI", "SBI", "Axis", "Kotak", "IDFC First"]
UPI_ISSUING_BANKS = ["HDFC", "ICICI", "SBI", "Axis", "Kotak", "Yes Bank", "PNB"]
PSP_APPS = ["GPay", "PhonePe", "Paytm", "other"]
MERCHANT_CATEGORIES = [
    "ecommerce", "food_delivery", "travel", "utilities", "subscriptions", "gaming",
]


def _sample_category_mix(mix: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    cats = list(mix.keys())
    probs = list(mix.values())
    return rng.choice(cats, size=n, p=probs)


def _emit_error_codes(categories: np.ndarray, code_dists: dict,
                       rng: np.random.Generator) -> np.ndarray:
    """Draw an error_code per row from P(code | category), with shared codes."""
    codes = np.empty(len(categories), dtype=object)
    for cat, dist in code_dists.items():
        mask = categories == cat
        if not mask.any():
            continue
        options = list(dist.keys())
        probs = np.array(list(dist.values()), dtype=float)
        probs = probs / probs.sum()
        # rng.choice can't hold None in a numeric/str array -- draw indices instead
        idx = rng.choice(len(options), size=int(mask.sum()), p=probs)
        codes[mask] = [options[i] for i in idx]
    return codes


def _category_conditional_behaviour(categories: np.ndarray, n: int,
                                     rng: np.random.Generator) -> dict:
    """
    Behavioural signals that make the error_code overlap resolvable.

    Without these, sharing codes across categories would only add irreducible
    noise and no classifier could beat the naive lookup baseline.
    """
    is_fraud = categories == "fraud_block"
    is_dropoff = categories == "customer_dropoff"
    is_hard = categories == "hard_decline"

    # amount: fraud skews to the high tail
    amount = rng.lognormal(mean=6.5, sigma=1.0, size=n)
    amount = np.where(is_fraud, rng.lognormal(mean=7.6, sigma=1.1, size=n), amount)

    # customer history: fraud tends to hit thin/new customer profiles
    history = rng.geometric(p=0.05, size=n)
    history = np.where(is_fraud, rng.geometric(p=0.35, size=n), history)

    # time_of_day: fraud over-represented overnight (00:00-05:00)
    time_of_day = rng.integers(0, 24, size=n)
    overnight = rng.integers(0, 6, size=n)
    time_of_day = np.where(is_fraud & (rng.random(n) < 0.45), overnight, time_of_day)

    # seconds_to_failure: the dwell-time signal. Applies to BOTH methods -- cards
    # previously had no analogue to time_since_app_switch, leaving card dropoff vs
    # soft_decline on code 91 structurally unresolvable. Real gateways log this.
    #
    # Crucially, soft_decline is SLOW too: a bank-side timeout is slow by
    # definition -- that is what a timeout is (gateway cutoffs sit around 30s).
    # So soft_decline and customer_dropoff genuinely OVERLAP on dwell time, which
    # is the real-world form of the Candidate 1 ambiguity: a long wait looks the
    # same whether the bank stalled or the human did. An earlier version of this
    # generator gave soft_decline a fast dwell, which made drop-off trivially
    # separable and trained the ambiguity out of the problem.
    is_soft = categories == "soft_decline"
    seconds_to_failure = rng.exponential(scale=5.0, size=n)                    # fraud
    seconds_to_failure = np.where(is_hard, rng.exponential(scale=4.0, size=n),
                                   seconds_to_failure)                         # instant decline
    seconds_to_failure = np.where(is_soft, rng.gamma(6.0, 6.5, size=n),
                                   seconds_to_failure)                         # ~39s timeout
    seconds_to_failure = np.where(is_dropoff, rng.gamma(3.0, 22.0, size=n),
                                   seconds_to_failure)                         # ~66s, wide

    return {
        "amount": np.round(amount, 2),
        "time_of_day": time_of_day,
        "customer_txn_history_count": history.clip(1, 500),
        "seconds_to_failure": np.round(seconds_to_failure, 1).clip(0.3, 600),
    }


def _shared_features(categories: np.ndarray, n: int, rng: np.random.Generator) -> pd.DataFrame:
    behaviour = _category_conditional_behaviour(categories, n, rng)
    return pd.DataFrame({
        "amount": behaviour["amount"],
        "time_of_day": behaviour["time_of_day"],
        "day_of_week": rng.integers(0, 7, size=n),
        "retry_count": rng.poisson(lam=0.6, size=n).clip(0, 5),
        "customer_txn_history_count": behaviour["customer_txn_history_count"],
        "customer_past_failure_rate": np.round(rng.beta(2, 8, size=n), 3),
        "seconds_to_failure": behaviour["seconds_to_failure"],
        "merchant_category": rng.choice(MERCHANT_CATEGORIES, size=n),
    })


def _retry_success(categories: np.ndarray, past_failure_rate: np.ndarray,
                    retry_count: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    base_rate = {
        "hard_decline": 0.08,
        "soft_decline": 0.65,
        "customer_dropoff": 0.42,
        "fraud_block": 0.0,
    }
    p = np.array([base_rate[c] for c in categories], dtype=float)
    # more prior failures / more retries already spent -> diminishing returns
    p = p - 0.15 * past_failure_rate - 0.03 * retry_count
    p = p.clip(0.0, 0.95)
    p[categories == "fraud_block"] = 0.0
    return rng.random(len(categories)) < p


def generate_card_transactions(n: int, rng: np.random.Generator) -> pd.DataFrame:
    categories = _sample_category_mix(CARD_CATEGORY_MIX, n, rng)
    df = _shared_features(categories, n, rng)
    df["payment_method"] = "card"
    df["failure_category"] = categories
    df["error_code"] = _emit_error_codes(categories, CARD_CODE_GIVEN_CATEGORY, rng)

    df["card_network"] = rng.choice(CARD_NETWORKS, size=n, p=[0.45, 0.30, 0.20, 0.05])
    df["issuing_bank"] = rng.choice(CARD_ISSUING_BANKS, size=n)

    # 3DS is where fraud checks and drop-off both concentrate in reality
    is_dropoff_or_fraud = np.isin(categories, ["customer_dropoff", "fraud_block"])
    p_3ds = np.where(is_dropoff_or_fraud, 0.78, 0.32)
    df["is_3ds_flow"] = rng.random(n) < p_3ds

    # fraud skews toward newer/less-established cards
    card_age = rng.gamma(shape=4.0, scale=10.0, size=n)
    card_age = np.where(categories == "fraud_block", rng.gamma(1.8, 3.5, size=n), card_age)
    df["card_age_months"] = card_age.round(1).clip(1, 240)

    df["retry_success"] = _retry_success(categories, df["customer_past_failure_rate"].values,
                                          df["retry_count"].values, rng)
    return df


def generate_upi_transactions(n: int, rng: np.random.Generator) -> pd.DataFrame:
    categories = _sample_category_mix(UPI_CATEGORY_MIX, n, rng)
    df = _shared_features(categories, n, rng)
    df["payment_method"] = "upi"
    df["failure_category"] = categories
    df["error_code"] = _emit_error_codes(categories, UPI_CODE_GIVEN_CATEGORY, rng)

    df["psp_app"] = rng.choice(PSP_APPS, size=n, p=[0.42, 0.38, 0.15, 0.05])
    df["issuing_bank"] = rng.choice(UPI_ISSUING_BANKS, size=n)

    # time_since_app_switch (seconds): short for soft_decline (bank-side, not
    # user-driven), long for customer_dropoff (user stalled/abandoned). Correlated
    # with seconds_to_failure but not identical -- app-switch is a UPI-only leg.
    time_since_switch = np.empty(n)
    soft_mask = categories == "soft_decline"
    dropoff_mask = categories == "customer_dropoff"
    other_mask = ~(soft_mask | dropoff_mask)
    time_since_switch[soft_mask] = rng.exponential(scale=8, size=soft_mask.sum())
    time_since_switch[dropoff_mask] = rng.exponential(scale=90, size=dropoff_mask.sum())
    time_since_switch[other_mask] = rng.exponential(scale=20, size=other_mask.sum())
    df["time_since_app_switch"] = time_since_switch.round(1)

    df["daily_limit_utilization"] = np.round(rng.beta(2, 3, size=n), 3)
    # hard_decline rows plausibly sit near the daily cap
    hard_mask = categories == "hard_decline"
    df.loc[hard_mask, "daily_limit_utilization"] = np.round(
        rng.beta(6, 2, size=hard_mask.sum()), 3
    )

    df["retry_success"] = _retry_success(categories, df["customer_past_failure_rate"].values,
                                          df["retry_count"].values, rng)
    return df


def _report(df: pd.DataFrame) -> None:
    print(f"Wrote {len(df)} rows to {OUT_PATH}")
    print("\nfailure_category by payment_method:")
    print(pd.crosstab(df["payment_method"], df["failure_category"]))
    print("\nretry_success rate by failure_category:")
    print(df.groupby("failure_category")["retry_success"].mean().round(3))

    # The check that caught the Day 1 leakage -- keep reporting it every run.
    codes = df["error_code"].fillna("<none>")
    ct = pd.crosstab(codes, df["failure_category"])
    purity = ct.max(axis=1) / ct.sum(axis=1)
    deterministic = int(ct.sum(axis=1)[purity == 1.0].sum())
    print("\nerror_code purity (max class share per code):")
    print(purity.sort_values().round(3).to_string())
    print(f"\nrows whose code maps to exactly one category: {deterministic} / {len(df)} "
          f"({deterministic / len(df):.1%})")
    print(f"naive lookup-table ceiling (majority class per code): "
          f"{ct.max(axis=1).sum() / len(df):.1%}")


def main():
    rng = np.random.default_rng(RNG_SEED)

    n_card = int(round(N_ROWS * 0.55))
    n_upi = N_ROWS - n_card

    card_df = generate_card_transactions(n_card, rng)
    upi_df = generate_upi_transactions(n_upi, rng)

    df = pd.concat([card_df, upi_df], ignore_index=True)
    df = df.sample(frac=1, random_state=RNG_SEED).reset_index(drop=True)
    df.insert(0, "transaction_id", [f"txn_{i:06d}" for i in range(len(df))])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    _report(df)


if __name__ == "__main__":
    main()
