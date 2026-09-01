"""Canonical project paths, so scripts don't each reinvent them."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA = os.path.join(ROOT, "data", "raw", "transactions.csv")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
TRAIN_DATA = os.path.join(PROCESSED_DIR, "train.csv")
TEST_DATA = os.path.join(PROCESSED_DIR, "test.csv")

MODELS_DIR = os.path.join(ROOT, "models")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "failure_classifier.pkl")
RECOVERY_MODEL_PATH = os.path.join(MODELS_DIR, "recovery_success_model.pkl")
# Drift reference stats. A build artifact of one training run, like the models
# it sits beside -- written by scripts/train.py, gitignored for the same reason.
BASELINE_PATH = os.path.join(MODELS_DIR, "training_baseline.json")

REPORTS_DIR = os.path.join(ROOT, "reports")
METRICS_PATH = os.path.join(REPORTS_DIR, "metrics.json")
CONFUSION_MATRIX_PATH = os.path.join(REPORTS_DIR, "confusion_matrix.png")
CALIBRATION_CURVE_PATH = os.path.join(REPORTS_DIR, "calibration_curve.png")

RANDOM_SEED = 42


def ensure_dirs():
    for d in (PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
        os.makedirs(d, exist_ok=True)
