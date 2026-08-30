"""
Shared feature spec + encoder, used by BOTH models and by the agent/API at
inference time.

There is deliberately one definition of "what the features are" in this repo.
The fitted encoder is bundled into each saved model artifact, so the serving
path cannot drift from the training path -- no separate preprocessing step for
the API to reimplement, no train/serve skew.

Structural NaNs are meaningful and are preserved, not imputed: card columns are
NaN on UPI rows and vice versa, because the feature genuinely does not exist for
that payment method. XGBoost learns a default split direction for missing values,
so "absent" is information rather than a hole to patch.
"""

import numpy as np
import pandas as pd

TARGET_CATEGORY = "failure_category"
TARGET_RECOVERY = "retry_success"

CATEGORY_CLASSES = ["customer_dropoff", "fraud_block", "hard_decline", "soft_decline"]

NUMERIC_FEATURES = [
    "amount",
    "time_of_day",
    "day_of_week",
    "retry_count",
    "customer_txn_history_count",
    "customer_past_failure_rate",
    "seconds_to_failure",
    # card-only (NaN on UPI rows)
    "card_age_months",
    # UPI-only (NaN on card rows)
    "time_since_app_switch",
    "daily_limit_utilization",
]

# Booleans are numeric 0/1, NOT categoricals -- XGBoost's categorical handling
# rejects boolean category levels.
BOOLEAN_FEATURES = ["is_3ds_flow"]

CATEGORICAL_FEATURES = [
    "merchant_category",
    "payment_method",
    "error_code",
    "card_network",
    "issuing_bank",
    "psp_app",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES


class FeatureEncoder:
    """
    Learns the categorical level sets at fit time and applies them consistently
    at transform time.

    Unseen categories at inference map to NaN rather than raising -- a new
    issuing bank or a novel error code should degrade the prediction, not take
    down the payment path.
    """

    def __init__(self):
        self.categories_ = {}

    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        for col in CATEGORICAL_FEATURES:
            values = df[col].dropna().unique()
            self.categories_[col] = pd.Index(sorted(values.tolist()))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.categories_:
            raise RuntimeError("FeatureEncoder.transform() called before fit()")

        out = pd.DataFrame(index=df.index)

        for col in NUMERIC_FEATURES:
            out[col] = pd.to_numeric(df.get(col), errors="coerce")

        for col in BOOLEAN_FEATURES:
            # keep NaN as NaN (structurally absent), map True/False -> 1.0/0.0
            out[col] = df.get(col).map(
                {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}
            ).astype(float) if col in df else np.nan

        for col in CATEGORICAL_FEATURES:
            values = df.get(col)
            if values is None:
                values = pd.Series([np.nan] * len(df), index=df.index)
            out[col] = pd.Categorical(values, categories=self.categories_[col])

        return out[FEATURE_COLUMNS]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def encode_labels(series: pd.Series) -> np.ndarray:
    """failure_category -> integer codes, using a fixed class order."""
    return pd.Categorical(series, categories=CATEGORY_CLASSES).codes


def decode_labels(codes) -> np.ndarray:
    return np.asarray(CATEGORY_CLASSES, dtype=object)[np.asarray(codes)]
