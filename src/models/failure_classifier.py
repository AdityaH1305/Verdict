"""
Model 1: Failure Classifier

Predicts failure_category from transaction features at time of failure.
Output classes: customer_dropoff | fraud_block | hard_decline | soft_decline

Algorithm: XGBoost (locked in docs/decisions.md) -- native handling of both the
structural NaNs and the categorical columns, and feature importances that hold up
in a panel Q&A better than a black-box net.

Evaluated against a NAIVE ERROR-CODE LOOKUP BASELINE. That baseline is the whole
point: an early version of the data generator made error_code a near-deterministic
map to the label, so the "classifier" was a lookup table scoring 96.6%. The
generator was rebuilt so codes overlap the way they do in production, and this
model now has to earn its accuracy over the lookup. If that gap ever collapses,
the model is not doing real work -- so the baseline is reported on every run.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.features import (  # noqa: E402
    CATEGORY_CLASSES, FEATURE_COLUMNS, TARGET_CATEGORY, FeatureEncoder, encode_labels,
)
from src.paths import CLASSIFIER_PATH, RANDOM_SEED  # noqa: E402

PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    enable_categorical=True,
    tree_method="hist",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)


class FailureClassifier:
    """Model 1. Bundles its own fitted feature encoder (see src/features.py)."""

    def __init__(self):
        self.encoder = FeatureEncoder()
        self.model = xgb.XGBClassifier(**PARAMS)
        self.classes_ = CATEGORY_CLASSES

    def fit(self, df: pd.DataFrame) -> "FailureClassifier":
        X = self.encoder.fit_transform(df)
        y = encode_labels(df[TARGET_CATEGORY])
        self.model.fit(X, y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.encoder.transform(df))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        codes = self.model.predict(self.encoder.transform(df))
        return np.asarray(CATEGORY_CLASSES, dtype=object)[codes]

    def oof_predict_proba(self, df: pd.DataFrame, n_splits: int = 5) -> np.ndarray:
        """
        Out-of-fold predicted probabilities on the training set.

        Model 2 consumes Model 1's output as an input feature. Training it on
        in-sample predictions (or on true labels) would give it a cleaner signal
        than it will ever see in production -- classic train/serve skew. Fitting
        on out-of-fold predictions makes Model 2 learn against the noise level it
        will actually face.
        """
        X = self.encoder.fit_transform(df)
        y = encode_labels(df[TARGET_CATEGORY])
        return cross_val_predict(
            xgb.XGBClassifier(**PARAMS), X, y, cv=n_splits, method="predict_proba"
        )

    def save(self, path: str = CLASSIFIER_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = CLASSIFIER_PATH) -> "FailureClassifier":
        return joblib.load(path)


def error_code_lookup_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """
    The bar to clear: majority failure_category per error_code, learned on train.

    This is what a competent engineer would write in ten minutes without any ML.
    """
    codes_train = train_df["error_code"].fillna("<none>")
    lookup = train_df.groupby(codes_train)[TARGET_CATEGORY].agg(
        lambda s: s.value_counts().idxmax()
    )
    fallback = train_df[TARGET_CATEGORY].value_counts().idxmax()
    return test_df["error_code"].fillna("<none>").map(lookup).fillna(fallback).values


def evaluate(clf: FailureClassifier, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    y_true = test_df[TARGET_CATEGORY].values
    y_pred = clf.predict(test_df)

    baseline_pred = error_code_lookup_baseline(train_df, test_df)

    cm = confusion_matrix(y_true, y_pred, labels=CATEGORY_CLASSES)
    report = classification_report(
        y_true, y_pred, labels=CATEGORY_CLASSES, target_names=CATEGORY_CLASSES,
        digits=3, output_dict=True, zero_division=0,
    )

    model_acc = accuracy_score(y_true, y_pred)
    model_f1 = f1_score(y_true, y_pred, average="macro")
    base_acc = accuracy_score(y_true, baseline_pred)
    base_f1 = f1_score(y_true, baseline_pred, average="macro", labels=CATEGORY_CLASSES)

    # The Candidate 1 story (docs/failure_stories.md): how much of the model's
    # remaining error is the genuinely ambiguous soft_decline <-> customer_dropoff
    # pair, rather than scattered noise?
    idx = {c: i for i, c in enumerate(CATEGORY_CLASSES)}
    total_errors = int(cm.sum() - np.trace(cm))
    ambiguous_pair = int(
        cm[idx["soft_decline"], idx["customer_dropoff"]]
        + cm[idx["customer_dropoff"], idx["soft_decline"]]
    )
    fraud_leakage = int(
        cm[idx["fraud_block"]].sum() - cm[idx["fraud_block"], idx["fraud_block"]]
    )

    print("\n=== Model 1: Failure Classifier ===")
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(pd.DataFrame(cm, index=CATEGORY_CLASSES, columns=CATEGORY_CLASSES))
    print("\nPer-class metrics:")
    print(classification_report(y_true, y_pred, labels=CATEGORY_CLASSES,
                                 target_names=CATEGORY_CLASSES, digits=3, zero_division=0))
    print(f"XGBoost         accuracy={model_acc:.4f}  macro-F1={model_f1:.4f}")
    print(f"error-code LUT  accuracy={base_acc:.4f}  macro-F1={base_f1:.4f}   <- baseline to beat")
    print(f"lift over baseline: {model_acc - base_acc:+.4f} acc, {model_f1 - base_f1:+.4f} macro-F1")
    print(f"\nsoft_decline <-> customer_dropoff confusion: {ambiguous_pair}/{total_errors} "
          f"of all errors ({ambiguous_pair / max(total_errors, 1):.1%})")
    print(f"fraud_block misclassified as something else: {fraud_leakage}")

    importances = pd.Series(clf.model.feature_importances_, index=FEATURE_COLUMNS)
    print("\nTop features:")
    print(importances.sort_values(ascending=False).head(8).round(4).to_string())

    return {
        "accuracy": round(float(model_acc), 4),
        "macro_f1": round(float(model_f1), 4),
        "baseline_error_code_lookup": {
            "accuracy": round(float(base_acc), 4),
            "macro_f1": round(float(base_f1), 4),
        },
        "lift_over_baseline": {
            "accuracy": round(float(model_acc - base_acc), 4),
            "macro_f1": round(float(model_f1 - base_f1), 4),
        },
        "per_class": {
            c: {k: round(float(v), 4) for k, v in report[c].items()}
            for c in CATEGORY_CLASSES
        },
        "confusion_matrix": {
            "labels": CATEGORY_CLASSES,
            "matrix": cm.tolist(),
        },
        "total_errors": total_errors,
        "soft_vs_dropoff_errors": ambiguous_pair,
        "fraud_block_leakage": fraud_leakage,
        "feature_importances": importances.sort_values(ascending=False).round(4).to_dict(),
    }
