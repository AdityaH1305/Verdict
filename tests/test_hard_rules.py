"""
Guard tests for the architectural hard rules.

The project's central claim is that the fraud block is an ARCHITECTURAL rule,
not a learned preference (README.md, docs/decisions.md). A claim like that
belongs in CI, not only in prose -- these tests fail loudly if a future change
quietly lets fraud_block rows into a recovery path.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import TARGET_CATEGORY, TARGET_RECOVERY  # noqa: E402
from src.models.recovery_success_model import (  # noqa: E402
    FRAUD_CLASS, RecoverySuccessModel, drop_fraud_rows,
)
from src.paths import RAW_DATA, TEST_DATA, TRAIN_DATA  # noqa: E402

import numpy as np  # noqa: E402


requires_data = pytest.mark.skipif(
    not os.path.exists(RAW_DATA),
    reason="run src/data_generation/generate_transactions.py first",
)
requires_splits = pytest.mark.skipif(
    not (os.path.exists(TRAIN_DATA) and os.path.exists(TEST_DATA)),
    reason="run scripts/prepare_data.py first",
)


@requires_data
def test_fraud_block_rows_are_never_recoverable():
    """No fraud_block transaction may be labelled as recoverable, ever."""
    df = pd.read_csv(RAW_DATA)
    fraud = df[df[TARGET_CATEGORY] == FRAUD_CLASS]

    assert len(fraud) > 0, "expected fraud_block rows in the dataset"
    assert not fraud[TARGET_RECOVERY].any(), (
        f"{int(fraud[TARGET_RECOVERY].sum())} fraud_block rows are marked "
        f"retry_success=True; fraud is unrecoverable by construction"
    )


@requires_splits
def test_drop_fraud_rows_removes_every_fraud_row():
    train_df = pd.read_csv(TRAIN_DATA)
    proba = np.zeros((len(train_df), 4))

    filtered, filtered_proba = drop_fraud_rows(train_df, proba)

    assert (train_df[TARGET_CATEGORY] == FRAUD_CLASS).any(), "fixture should contain fraud rows"
    assert not (filtered[TARGET_CATEGORY] == FRAUD_CLASS).any()
    assert len(filtered) == len(filtered_proba), "row filter must stay aligned with proba matrix"


@requires_splits
def test_model_2_refuses_to_train_on_fraud_rows():
    """
    Model 2 must reject fraud_block rows rather than quietly learning 'fraud -> 0'.

    Learning it would make the block a soft weight that could drift. The
    architecture requires it to be an unbreakable rule.
    """
    train_df = pd.read_csv(TRAIN_DATA)
    proba = np.zeros((len(train_df), 4))

    with pytest.raises(ValueError, match="fraud_block"):
        RecoverySuccessModel().fit(train_df, proba)
