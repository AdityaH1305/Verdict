"""
Train both models end to end and write evaluation artifacts.

    python scripts/train.py

Writes:
  models/failure_classifier.pkl
  models/recovery_success_model.pkl
  reports/metrics.json
  reports/confusion_matrix.png
  reports/calibration_curve.png

Reports are emitted as PNG/JSON rather than notebooks so they are reproducible,
run headless, and drop straight into the Day 4 dashboard and the pitch video.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import CATEGORY_CLASSES, TARGET_CATEGORY, TARGET_RECOVERY  # noqa: E402
from src.models.failure_classifier import FailureClassifier  # noqa: E402
from src.models.failure_classifier import evaluate as evaluate_classifier  # noqa: E402
from src.models.recovery_success_model import (  # noqa: E402
    RecoverySuccessModel, drop_fraud_rows,
)
from src.models.recovery_success_model import evaluate as evaluate_recovery  # noqa: E402
from src.monitoring.baseline import write_baseline  # noqa: E402
from src.paths import (  # noqa: E402
    BASELINE_PATH, CALIBRATION_CURVE_PATH, CONFUSION_MATRIX_PATH, METRICS_PATH,
    TEST_DATA, TRAIN_DATA, ensure_dirs,
)


def plot_confusion_matrix(cm: np.ndarray, path: str):
    normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for col, (ax, data, title, fmt) in enumerate((
        (axes[0], cm, "Counts", "d"),
        (axes[1], normalized, "Row-normalized (recall)", ".2f"),
    )):
        im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(CATEGORY_CLASSES)))
        ax.set_yticks(range(len(CATEGORY_CLASSES)))
        ax.set_xticklabels(CATEGORY_CLASSES, rotation=30, ha="right")
        # y tick labels only on the left panel -- on the right they collide with
        # the left panel's cells
        ax.set_yticklabels(CATEGORY_CLASSES if col == 0 else [])
        ax.set_xlabel("Predicted")
        if col == 0:
            ax.set_ylabel("True")
        ax.set_title(f"Model 1 — {title}")
        for i in range(len(CATEGORY_CLASSES)):
            for j in range(len(CATEGORY_CLASSES)):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if normalized[i, j] > 0.5 else "black", fontsize=10)
    fig.subplots_adjust(wspace=0.08)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(model, test_df, test_proba, path: str):
    from sklearn.calibration import calibration_curve

    y = test_df[TARGET_RECOVERY].astype(int).values
    p_cal = model.predict_proba(test_df, test_proba)
    p_raw = model.predict_proba_uncalibrated(test_df, test_proba)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfectly calibrated")
    for probs, label in ((p_raw, "uncalibrated XGBoost"), (p_cal, "isotonic-calibrated")):
        frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", label=label)
    # the agent's economic cutoffs are read off this axis
    for cut in (0.3, 0.6):
        ax.axvline(cut, color="crimson", linestyle=":", linewidth=1)
    ax.text(0.605, 0.05, "0.6 auto_retry", color="crimson", fontsize=8)
    ax.text(0.305, 0.05, "0.3 escalate", color="crimson", fontsize=8)
    ax.set_xlabel("Predicted probability of recovery")
    ax.set_ylabel("Observed recovery rate")
    ax.set_title("Model 2 — calibration")
    ax.legend(loc="upper left", fontsize=9)

    ax2.hist(p_cal, bins=40, color="steelblue", edgecolor="white")
    for cut in (0.3, 0.6):
        ax2.axvline(cut, color="crimson", linestyle=":", linewidth=1.5)
    ax2.set_xlabel("Calibrated P(recovery)")
    ax2.set_ylabel("Transactions")
    ax2.set_title("Score distribution vs. agent thresholds")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_dirs()

    if not (os.path.exists(TRAIN_DATA) and os.path.exists(TEST_DATA)):
        raise SystemExit("Missing splits. Run: python scripts/prepare_data.py")

    train_df = pd.read_csv(TRAIN_DATA)
    test_df = pd.read_csv(TEST_DATA)

    # ---- Model 1 ----
    clf = FailureClassifier()
    # Out-of-fold probabilities BEFORE the final fit, so Model 2 trains against
    # the noise level it will actually see at serving time.
    oof_proba = clf.oof_predict_proba(train_df)
    clf.fit(train_df)
    clf.save()
    classifier_metrics = evaluate_classifier(clf, train_df, test_df)
    plot_confusion_matrix(
        np.array(classifier_metrics["confusion_matrix"]["matrix"]), CONFUSION_MATRIX_PATH
    )

    # ---- Model 2 ----
    # fraud_block rows are dropped from BOTH train and test before Model 2 exists.
    m2_train, m2_train_proba = drop_fraud_rows(train_df, oof_proba)
    test_proba = clf.predict_proba(test_df)
    m2_test, m2_test_proba = drop_fraud_rows(test_df, test_proba)

    print(f"\nModel 2 training rows: {len(m2_train)} "
          f"({len(train_df) - len(m2_train)} fraud_block rows excluded)")

    recovery = RecoverySuccessModel().fit(m2_train, m2_train_proba)
    recovery.save()
    recovery_metrics = evaluate_recovery(recovery, m2_test, m2_test_proba)
    plot_calibration(recovery, m2_test, m2_test_proba, CALIBRATION_CURVE_PATH)

    metrics = {
        "dataset": {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "category_balance": train_df[TARGET_CATEGORY]
                .value_counts(normalize=True).round(4).to_dict(),
        },
        "model_1_failure_classifier": classifier_metrics,
        "model_2_recovery_success": recovery_metrics,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # Drift reference stats, from the SAME frame the models were just fitted on,
    # so the baseline and the models can never describe different data.
    # Read-only monitoring: nothing in the decision path reads this.
    write_baseline(train_df, BASELINE_PATH)

    print(f"\nWrote {METRICS_PATH}")
    print(f"Wrote {BASELINE_PATH}")
    print(f"Wrote {CONFUSION_MATRIX_PATH}")
    print(f"Wrote {CALIBRATION_CURVE_PATH}")

    # Verification gate from the Day 2 plan: if the classifier ever stops beating
    # the naive lookup, the model is not doing real work and the data needs a pass.
    lift = classifier_metrics["lift_over_baseline"]["macro_f1"]
    if lift <= 0:
        raise SystemExit(
            f"\nFAIL: Model 1 does not beat the error-code lookup baseline "
            f"(macro-F1 lift {lift:+.4f}). The classifier is not adding value "
            f"over a ten-line lookup table."
        )
    print(f"\nOK: Model 1 beats the error-code lookup baseline by {lift:+.4f} macro-F1.")


if __name__ == "__main__":
    main()
