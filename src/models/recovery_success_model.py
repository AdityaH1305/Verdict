"""
Model 2: Recovery Success Model

Predicts probability that a retry/nudge recovers a failed transaction,
conditioned on features + Model 1's predicted failure_category distribution.

IMPORTANT: never invoked for fraud_block transactions. This is enforced in the
agent layer (src/agent/), not here -- and this model is neither trained nor
evaluated on fraud_block rows. That is not a technicality: fraud_block rows have
retry_success == False by construction, so including them would teach the model
"fraud => 0.0" as a LEARNED PREFERENCE. The architecture deliberately makes that
an unbreakable rule instead of a soft weight a model could someday drift on.
See docs/decisions.md, "Hard rules".

Two design choices worth defending:

1. The category input is Model 1's full predicted PROBABILITY VECTOR, not a hard
   label and not the ground-truth label. Ground truth is unavailable at serving
   time; a hard label throws away exactly the uncertainty the agent needs. Passing
   the vector is what lets the agent hedge on ambiguous soft_decline/customer_dropoff
   cases (docs/failure_stories.md, Candidate 1) rather than trusting a coin-flip
   argmax.

2. The output is CALIBRATED (isotonic). The agent reads economic cutoffs straight
   off this probability scale (0.6 -> retry now, 0.3 -> escalate). A raw boosted-tree
   score of "0.6" that really means 0.75 would silently corrupt every decision the
   agent makes, so calibration here is a correctness requirement, not polish.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.features import (  # noqa: E402
    CATEGORY_CLASSES, TARGET_CATEGORY, TARGET_RECOVERY, FeatureEncoder,
)
from src.paths import RANDOM_SEED, RECOVERY_MODEL_PATH  # noqa: E402

FRAUD_CLASS = "fraud_block"
CATEGORY_PROBA_FEATURES = [f"p_{c}" for c in CATEGORY_CLASSES]

PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.06,
    subsample=0.9,
    colsample_bytree=0.9,
    enable_categorical=True,
    tree_method="hist",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)


def drop_fraud_rows(df: pd.DataFrame, category_proba: np.ndarray = None):
    """
    Remove fraud_block rows. Model 2 must never see them -- in training or eval.

    Returns the filtered frame and, if given, the matching rows of the category
    probability matrix.
    """
    mask = (df[TARGET_CATEGORY] != FRAUD_CLASS).values
    if category_proba is None:
        return df.loc[mask].reset_index(drop=True), None
    return df.loc[mask].reset_index(drop=True), category_proba[mask]


class RecoverySuccessModel:
    """Model 2. Bundles its own fitted feature encoder (see src/features.py)."""

    def __init__(self):
        self.encoder = FeatureEncoder()
        self.calibrated = None
        self.uncalibrated = None

    def _build_X(self, df: pd.DataFrame, category_proba: np.ndarray,
                  fit: bool = False) -> pd.DataFrame:
        X = self.encoder.fit_transform(df) if fit else self.encoder.transform(df)
        proba = pd.DataFrame(category_proba, columns=CATEGORY_PROBA_FEATURES,
                              index=X.index)
        return pd.concat([X, proba], axis=1)

    def fit(self, df: pd.DataFrame, category_proba: np.ndarray) -> "RecoverySuccessModel":
        if (df[TARGET_CATEGORY] == FRAUD_CLASS).any():
            raise ValueError(
                "fraud_block rows reached Model 2's training set. This model is "
                "architecturally never invoked for fraud -- see docs/decisions.md."
            )

        X = self._build_X(df, category_proba, fit=True)
        y = df[TARGET_RECOVERY].astype(int).values

        self.uncalibrated = xgb.XGBClassifier(**PARAMS).fit(X, y)
        # prefit would refit on the same rows; cv=5 keeps the calibrator honest
        self.calibrated = CalibratedClassifierCV(
            xgb.XGBClassifier(**PARAMS), method="isotonic", cv=5
        ).fit(X, y)
        return self

    def predict_proba(self, df: pd.DataFrame, category_proba: np.ndarray) -> np.ndarray:
        """Calibrated P(recovery succeeds)."""
        X = self._build_X(df, category_proba)
        return self.calibrated.predict_proba(X)[:, 1]

    def predict_proba_uncalibrated(self, df: pd.DataFrame,
                                    category_proba: np.ndarray) -> np.ndarray:
        X = self._build_X(df, category_proba)
        return self.uncalibrated.predict_proba(X)[:, 1]

    def predict_interval(self, df: pd.DataFrame, category_proba: np.ndarray):
        """
        Per-transaction uncertainty interval, as (lo, hi) arrays.

        `CalibratedClassifierCV(cv=5)` fits five sub-models, each trained on a
        different fold and each with its own isotonic calibrator, and
        `predict_proba` returns their MEAN. So the spread across those five is
        available with no retraining, no second model, and -- critically -- no
        possible drift in the point estimate, because the point estimate is
        literally the mean of the members this interval is built from. Verified
        on the full test set: max |point - mean(members)| == 0.0.

        That is why this is preferred over bootstrapping (which would retrain N
        models, each acquiring its own calibration) or quantile regression
        (which is a regression formulation; the target here is binary).

        What it measures, stated honestly: disagreement between calibration
        folds -- epistemic uncertainty about the calibrated mapping. It is NOT a
        frequentist confidence interval and is not labelled as one anywhere.

        Returns None when the underlying members are unavailable (e.g. a model
        artifact pickled before this existed), so callers fall back to
        point-estimate behaviour rather than failing.
        """
        members = self._member_probabilities(df, category_proba)
        if members is None:
            return None
        return members.min(axis=1), members.max(axis=1)

    def _member_probabilities(self, df: pd.DataFrame,
                               category_proba: np.ndarray):
        """The five fold-calibrated predictions per row, or None."""
        subs = getattr(self.calibrated, "calibrated_classifiers_", None)
        if not subs:
            return None
        X = self._build_X(df, category_proba)
        return np.column_stack([s.predict_proba(X)[:, 1] for s in subs])

    def save(self, path: str = RECOVERY_MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = RECOVERY_MODEL_PATH) -> "RecoverySuccessModel":
        return joblib.load(path)


def evaluate(model: RecoverySuccessModel, test_df: pd.DataFrame,
             test_category_proba: np.ndarray) -> dict:
    """test_df must already have fraud_block rows removed."""
    assert not (test_df[TARGET_CATEGORY] == FRAUD_CLASS).any(), \
        "fraud_block rows must not reach Model 2 evaluation"

    y_true = test_df[TARGET_RECOVERY].astype(int).values
    p_cal = model.predict_proba(test_df, test_category_proba)
    p_raw = model.predict_proba_uncalibrated(test_df, test_category_proba)

    metrics = {
        "n_test": int(len(test_df)),
        "positive_rate": round(float(y_true.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y_true, p_cal)), 4),
        "pr_auc": round(float(average_precision_score(y_true, p_cal)), 4),
        "brier_calibrated": round(float(brier_score_loss(y_true, p_cal)), 4),
        "brier_uncalibrated": round(float(brier_score_loss(y_true, p_raw)), 4),
        "roc_auc_uncalibrated": round(float(roc_auc_score(y_true, p_raw)), 4),
    }

    print("\n=== Model 2: Recovery Success ===")
    print(f"test rows (fraud_block excluded): {metrics['n_test']}")
    print(f"base recovery rate:               {metrics['positive_rate']:.4f}")
    print(f"ROC-AUC:                          {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:                           {metrics['pr_auc']:.4f}")
    print(f"Brier (uncalibrated -> isotonic): {metrics['brier_uncalibrated']:.4f} -> "
          f"{metrics['brier_calibrated']:.4f}")

    # Score distribution against the agent's planned cutoffs, so Day 3 tunes
    # thresholds against real numbers instead of guesses (build_plan.md asks
    # for exactly this).
    buckets = {
        ">=0.6 (auto_retry_now)": float((p_cal >= 0.6).mean()),
        "0.3-0.6 (retry_later / nudge)": float(((p_cal >= 0.3) & (p_cal < 0.6)).mean()),
        "<0.3 (escalate / no_action)": float((p_cal < 0.3).mean()),
    }
    print("\nCalibrated score distribution vs. planned agent thresholds:")
    for name, share in buckets.items():
        print(f"  {name:32} {share:6.1%}")
    metrics["threshold_buckets"] = {k: round(v, 4) for k, v in buckets.items()}

    print("\nActual recovery rate by predicted-probability decile "
          "(should climb monotonically if calibrated):")
    deciles = pd.DataFrame({"p": p_cal, "y": y_true})
    deciles["bucket"] = pd.qcut(deciles["p"], 10, duplicates="drop")
    observed = deciles.groupby("bucket", observed=True).agg(
        predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size")
    )
    print(observed.round(3).to_string())

    frac_pos, mean_pred = calibration_curve(y_true, p_cal, n_bins=10, strategy="quantile")
    metrics["calibration_curve"] = {
        "mean_predicted": [round(float(v), 4) for v in mean_pred],
        "fraction_positive": [round(float(v), 4) for v in frac_pos],
    }
    return metrics
