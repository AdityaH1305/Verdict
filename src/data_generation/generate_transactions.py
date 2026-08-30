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
  - Deliberate overlap is introduced between soft_decline and customer_dropoff
    for a subset of UPI transactions (see docs/failure_stories.md, Candidate 1),
    via a shared, genuinely ambiguous error code (U30).

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

# error_code -> category, verified against ISO 8583 processor references.
CARD_ERROR_CODES = {
    "hard_decline": ["51", "05", "14", "54", "61"],
    "soft_decline": ["91", "96"],
    "fraud_block": ["59"],
    "customer_dropoff": [None],  # no decline code -- abandoned before completion
}

# error_code -> category, verified against a bank-published NPCI UPI error
# code reference (see docs/decisions.md). U30 is deliberately shared between
# soft_decline and customer_dropoff -- it's a genuinely ambiguous code.
UPI_ERROR_CODES = {
    "hard_decline": ["Z9", "ZH", "Z8", "Z7", "ZU", "ZM"],
    "soft_decline": ["UT", "BT", "U28", "Y1", "XY", "U67", "U68", "U30"],
    "fraud_block": ["59", "ZI"],
    "customer_dropoff": [None, "U30"],
}

CARD_NETWORKS = ["Visa", "Mastercard", "Rupay", "Amex"]
CARD_ISSUING_BANKS = ["HDFC", "ICICI", "SBI", "Axis", "Kotak", "IDFC First"]
UPI_ISSUING_BANKS = ["HDFC", "ICICI", "SBI", "Axis", "Kotak", "Yes Bank", "PNB"]
PSP_APPS = ["GPay", "PhonePe", "Paytm", "other"]
MERCHANT_CATEGORIES = [
    "ecommerce", "food_delivery", "travel", "utilities", "subscriptions", "gaming",
]

AMBIGUOUS_U30_FRACTION = 0.12  # of UPI soft_decline / customer_dropoff rows


def _sample_category_mix(mix: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    cats = list(mix.keys())
    probs = list(mix.values())
    return rng.choice(cats, size=n, p=probs)


def _shared_features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "amount": np.round(rng.lognormal(mean=6.5, sigma=1.0, size=n), 2),
        "time_of_day": rng.integers(0, 24, size=n),
        "day_of_week": rng.integers(0, 7, size=n),
        "retry_count": rng.poisson(lam=0.6, size=n).clip(0, 5),
        "customer_txn_history_count": rng.geometric(p=0.05, size=n).clip(1, 500),
        "customer_past_failure_rate": np.round(rng.beta(2, 8, size=n), 3),
        "merchant_category": rng.choice(MERCHANT_CATEGORIES, size=n),
    })


def _assign_error_codes(categories: np.ndarray, taxonomy: dict, rng: np.random.Generator) -> np.ndarray:
    codes = np.empty(len(categories), dtype=object)
    for cat in np.unique(categories):
        mask = categories == cat
        options = taxonomy[cat]
        codes[mask] = rng.choice(options, size=mask.sum())
    return codes


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
    df = _shared_features(n, rng)
    df["payment_method"] = "card"

    categories = _sample_category_mix(CARD_CATEGORY_MIX, n, rng)
    df["failure_category"] = categories
    df["error_code"] = _assign_error_codes(categories, CARD_ERROR_CODES, rng)

    df["card_network"] = rng.choice(CARD_NETWORKS, size=n, p=[0.45, 0.30, 0.20, 0.05])
    df["issuing_bank"] = rng.choice(CARD_ISSUING_BANKS, size=n)

    # 3DS abandonment concentrates fraud checks + drop-off in reality
    is_dropoff_or_fraud = np.isin(categories, ["customer_dropoff", "fraud_block"])
    p_3ds = np.where(is_dropoff_or_fraud, 0.75, 0.35)
    df["is_3ds_flow"] = rng.random(n) < p_3ds

    # fraud skews toward newer/less-established cards
    card_age = rng.gamma(shape=4.0, scale=10.0, size=n)
    card_age = np.where(categories == "fraud_block", rng.gamma(2.0, 4.0, size=n), card_age)
    df["card_age_months"] = card_age.round(1).clip(1, 240)

    df["retry_success"] = _retry_success(categories, df["customer_past_failure_rate"].values,
                                          df["retry_count"].values, rng)
    return df


def generate_upi_transactions(n: int, rng: np.random.Generator) -> pd.DataFrame:
    df = _shared_features(n, rng)
    df["payment_method"] = "upi"

    categories = _sample_category_mix(UPI_CATEGORY_MIX, n, rng)
    df["failure_category"] = categories
    df["error_code"] = _assign_error_codes(categories, UPI_ERROR_CODES, rng)

    df["psp_app"] = rng.choice(PSP_APPS, size=n, p=[0.42, 0.38, 0.15, 0.05])
    df["issuing_bank"] = rng.choice(UPI_ISSUING_BANKS, size=n)

    # time_since_app_switch (seconds): short for soft_decline (bank-side,
    # not user-driven), long for customer_dropoff (user stalled/abandoned)
    time_since_switch = np.empty(n)
    soft_mask = categories == "soft_decline"
    dropoff_mask = categories == "customer_dropoff"
    other_mask = ~(soft_mask | dropoff_mask)
    time_since_switch[soft_mask] = rng.exponential(scale=8, size=soft_mask.sum())
    time_since_switch[dropoff_mask] = rng.exponential(scale=90, size=dropoff_mask.sum())
    time_since_switch[other_mask] = rng.exponential(scale=20, size=other_mask.sum())
    df["time_since_app_switch"] = time_since_switch.round(1)

    df["daily_limit_utilization"] = np.round(rng.beta(2, 3, size=n), 3)
    # hard_decline rows hit the limit code more plausibly with high utilization
    hard_mask = categories == "hard_decline"
    df.loc[hard_mask, "daily_limit_utilization"] = np.round(
        rng.beta(6, 2, size=hard_mask.sum()), 3
    )

    # Candidate 1 (see docs/failure_stories.md): force a subset of soft_decline
    # rows to genuinely ambiguous U30 + dropoff-like behavioral features (long
    # time_since_app_switch), so the ambiguity lives in the features, not a
    # special-cased label.
    soft_idx = np.flatnonzero(soft_mask)
    n_ambiguous = int(round(len(soft_idx) * AMBIGUOUS_U30_FRACTION))
    if n_ambiguous:
        amb_idx = rng.choice(soft_idx, size=n_ambiguous, replace=False)
        df.loc[amb_idx, "error_code"] = "U30"
        df.loc[amb_idx, "time_since_app_switch"] = np.round(
            rng.exponential(scale=90, size=n_ambiguous), 1
        )

    df["retry_success"] = _retry_success(categories, df["customer_past_failure_rate"].values,
                                          df["retry_count"].values, rng)
    return df


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

    print(f"Wrote {len(df)} rows to {OUT_PATH}")
    print("\nfailure_category by payment_method:")
    print(pd.crosstab(df["payment_method"], df["failure_category"]))
    print("\nretry_success rate by failure_category:")
    print(df.groupby("failure_category")["retry_success"].mean().round(3))


if __name__ == "__main__":
    main()
