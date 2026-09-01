"""
Reference statistics from the training data, for drift comparison.

Computed from the SAME `train.csv` the models are fitted on, and written by
`scripts/train.py` so the baseline and the models can never describe different
data. `models/training_baseline.json` is gitignored alongside `*.pkl` for that
reason: it is a build artifact of one training run, and committing it would let
it go silently stale if the data were regenerated without retraining.

WHICH FEATURES ARE MONITORED, AND WHY

Measured against the generator rather than picked by hand. Features split
cleanly by whether they vary with `failure_category` -- i.e. whether they carry
any information about the failure process at all:

  carries signal            spread     drawn uniformly       spread
  seconds_to_failure        2.90       customer_past_fail    0.08
  amount                    1.76       issuing_bank          0.059
  payment_method            0.344      psp_app               0.059
  error_code                entropy    card_network          0.044
  failure_category          (label)    merchant_category     0.032
                                       retry_count           0.033
                                       day_of_week           0.024

The right column is drawn independently of everything in the generator. Those
features would only move if someone shifted them deliberately, and putting seven
no-signal features into a combined score dilutes it.

Also excluded: `card_age_months`, `time_since_app_switch`,
`daily_limit_utilization`, `is_3ds_flow`. These DO carry signal, but each exists
for only one rail (45-55% populated), so a change in the card/UPI mix would
surface as drift in all four at once -- one shift counted five times.
`error_code` avoids that by being monitored PER PAYMENT METHOD.

CAVEAT, stated plainly: `failure_category` is the ground-truth label. It can be
monitored here only because these batches are held-out labelled rows. In
production it would not exist at scoring time, so this is label/concept drift,
not input drift. It is reported as such.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.monitoring.drift_detector import MISSING_BUCKET, OTHER_BUCKET  # noqa: E402

N_QUANTILE_BINS = 10

# Categorical levels below this training share are merged into OTHER_BUCKET.
#
# Not cosmetic -- without it the metric measures sample size instead of drift.
# Measured on 60 same-distribution 300-row batches, `error_code[upi]` (17 levels
# against ~135 UPI rows, so 3-6 expected observations in the rare bins) scored a
# MEDIAN PSI of 0.139 and a p90 of 0.252: it would have reported MODERATE half
# the time and HIGH one time in ten with no drift whatsoever. Merging rare levels
# brings that to median 0.043 / p90 0.086, below the stable cutoff.
#
# 5% of a rail in a 300-row batch is roughly 7 expected observations, which is
# the conventional minimum for a stable PSI bin.
MIN_LEVEL_SHARE = 0.05


def _categorical_spec(df: pd.DataFrame, column: str, payment_method=None) -> dict:
    frame = df if payment_method is None else df[df["payment_method"] == payment_method]
    values = frame[column].astype(object).where(frame[column].notna(), MISSING_BUCKET)
    counts = values.value_counts()
    total = max(len(values), 1)

    shares = {str(level): counts[level] / total for level in counts.index}
    kept = sorted(level for level, share in shares.items() if share >= MIN_LEVEL_SHARE)
    other = sum(share for level, share in shares.items() if share < MIN_LEVEL_SHARE)

    levels = kept + ([OTHER_BUCKET] if other > 0 else [])
    proportions = [shares[level] for level in kept] + ([other] if other > 0 else [])

    return {
        "kind": "categorical",
        "column": column,
        "payment_method": payment_method,
        "levels": levels,
        "proportions": [round(float(p), 6) for p in proportions],
        "merged_rare_levels": sorted(
            level for level, share in shares.items() if share < MIN_LEVEL_SHARE),
        "min_level_share": MIN_LEVEL_SHARE,
        "n_train_rows": int(len(frame)),
    }


def _continuous_spec(df: pd.DataFrame, column: str) -> dict:
    """
    Quantile bin edges from training, stored so the detector never needs
    train.csv at runtime -- and so both distributions are always bucketed the
    same way. Re-deriving quantiles per batch would make every batch look
    identical to itself and report no drift ever.
    """
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    quantiles = np.linspace(0, 1, N_QUANTILE_BINS + 1)[1:-1]
    edges = np.unique(np.quantile(values, quantiles))

    counts, _ = np.histogram(values, bins=edges)
    below = int((values < edges[0]).sum())
    above = int((values > edges[-1]).sum())
    proportions = np.concatenate([[below], counts, [above]]) / max(len(values), 1)

    labels = ([f"<{edges[0]:.4g}"]
              + [f"{edges[i]:.4g}-{edges[i+1]:.4g}" for i in range(len(edges) - 1)]
              + [f">{edges[-1]:.4g}"])

    return {
        "kind": "continuous",
        "column": column,
        "payment_method": None,
        "bin_edges": [float(e) for e in edges],
        "bin_labels": labels,
        "proportions": [round(float(p), 6) for p in proportions],
        "n_train_rows": int(len(values)),
    }


def build_baseline(train_df: pd.DataFrame) -> dict:
    """Reference distributions for every monitored feature. Deterministic."""
    features = {
        "payment_method": _categorical_spec(train_df, "payment_method"),
        "failure_category": _categorical_spec(train_df, "failure_category"),
        "error_code[card]": _categorical_spec(train_df, "error_code", "card"),
        "error_code[upi]": _categorical_spec(train_df, "error_code", "upi"),
        "amount": _continuous_spec(train_df, "amount"),
        "seconds_to_failure": _continuous_spec(train_df, "seconds_to_failure"),
    }
    return {
        "description": (
            "Reference distributions from the training split, for PSI drift "
            "comparison. Regenerated by scripts/train.py on every training run."
        ),
        "n_train_rows": int(len(train_df)),
        "quantile_bins": N_QUANTILE_BINS,
        "label_drift_features": ["failure_category"],
        "label_drift_note": (
            "failure_category is the ground-truth label. Monitoring it is "
            "label/concept drift and is only possible here because batches are "
            "held-out labelled rows; production scoring would not have it."
        ),
        "features": features,
    }


def write_baseline(train_df: pd.DataFrame, path: str) -> dict:
    baseline = build_baseline(train_df)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)
    return baseline


def load_baseline(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
