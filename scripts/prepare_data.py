"""
Split raw transactions into train/test.

Both models train off this single split, so an end-to-end evaluation of the
agent on the test set stays honest -- no row that trained Model 1 can leak into
Model 2's test set, or into the agent demo.
"""

import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import TARGET_CATEGORY  # noqa: E402
from src.paths import (  # noqa: E402
    RANDOM_SEED, RAW_DATA, TEST_DATA, TRAIN_DATA, ensure_dirs,
)

TEST_SIZE = 0.2


def main():
    ensure_dirs()

    if not os.path.exists(RAW_DATA):
        raise SystemExit(
            f"No raw data at {RAW_DATA}\n"
            "Run: python src/data_generation/generate_transactions.py"
        )

    df = pd.read_csv(RAW_DATA)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET_CATEGORY],
        random_state=RANDOM_SEED,
    )

    train_df.to_csv(TRAIN_DATA, index=False)
    test_df.to_csv(TEST_DATA, index=False)

    print(f"train: {len(train_df)} rows -> {TRAIN_DATA}")
    print(f"test:  {len(test_df)} rows -> {TEST_DATA}")
    print("\nfailure_category balance:")
    balance = pd.DataFrame({
        "train": train_df[TARGET_CATEGORY].value_counts(normalize=True),
        "test": test_df[TARGET_CATEGORY].value_counts(normalize=True),
    })
    print(balance.round(4))


if __name__ == "__main__":
    main()
