"""
Build a DELIBERATELY SHIFTED batch to demonstrate the drift monitor firing.

    python scripts/make_drift_demo.py

READ THIS BEFORE QUOTING ANY NUMBER FROM IT
-------------------------------------------
This batch is CONSTRUCTED. The drift it exhibits was introduced on purpose by
this script; it was not observed in production, in real traffic, or from any
live system. Synthetic transactions cannot drift on their own -- the generator
draws every batch from one fixed distribution -- so demonstrating that the
monitor works at all requires manufacturing something for it to catch.

The honest claim is: "here is the monitor detecting a shift we introduced
deliberately", never "here is drift we observed".

HOW THE SHIFT IS MADE

By re-weighting which held-out rows get sampled, not by inventing values. Every
row in the output is a real generated transaction with its own internally
consistent features; only the POPULATION MIX differs from training. That keeps
the shift a genuine distribution change rather than a fabricated one, and it is
the same kind of shift a real gateway would see if its traffic mix moved --
a UPI-heavy month, a fraud wave, a shift into higher-value payments.

Writes:
  data/demo/drift_demo_batch.csv   -- the shifted batch (committed)
  data/demo/drift_demo_batch.json  -- provenance and the shift recipe
  reports/drift_comparison.json    -- pinned vs shifted, side by side
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.monitoring.baseline import load_baseline  # noqa: E402
from src.monitoring.drift_detector import detect_drift  # noqa: E402
from src.paths import BASELINE_PATH, REPORTS_DIR, ROOT, TEST_DATA, ensure_dirs  # noqa: E402

DEMO_DIR = os.path.join(ROOT, "data", "demo")
PINNED_CSV = os.path.join(DEMO_DIR, "demo_batch.csv")
SHIFTED_CSV = os.path.join(DEMO_DIR, "drift_demo_batch.csv")
SHIFTED_META = os.path.join(DEMO_DIR, "drift_demo_batch.json")
COMPARISON = os.path.join(REPORTS_DIR, "drift_comparison.json")

BATCH_SIZE = 300
SEED = 7

# The constructed shift. Sampling weights, applied multiplicatively per row.
SHIFT = {
    "upi_weight": 6.0,          # UPI-heavy traffic (training is ~45% UPI)
    "fraud_weight": 3.0,        # a fraud wave
    "high_amount_weight": 3.0,  # movement into higher-value payments
    "high_amount_threshold": 1500.0,
}


def build_shifted_batch(pool: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Re-weight which real rows are drawn. No values are invented."""
    w = np.ones(len(pool), dtype=float)
    w[(pool["payment_method"] == "upi").values] *= SHIFT["upi_weight"]
    w[(pool["failure_category"] == "fraud_block").values] *= SHIFT["fraud_weight"]
    w[(pool["amount"] >= SHIFT["high_amount_threshold"]).values] *= SHIFT["high_amount_weight"]

    # The three multipliers stack: a row matching all three conditions gets
    # 6.0 * 3.0 * 3.0 = 54x the weight of a row matching none, a 54:1 ratio.
    # pandas' weighted-without-replacement sampler (`replace=False`) can fail
    # to satisfy a ratio that extreme for this pool size (1,600) and draw count
    # (300) -- it raised "Weighted sampling cannot be achieved with
    # replace=False" on Render's pandas build, though not on every pandas
    # version. sqrt-dampening the combined weight brings the ratio down to
    # ~7.3:1 (still comfortably reproduces the intended shift -- see the
    # regression test in tests/test_drift.py) while keeping every row's
    # RELATIVE ranking identical, since sqrt is monotonic.
    w = np.sqrt(w)

    return pool.sample(
        n=BATCH_SIZE, replace=False, weights=w / w.sum(), random_state=seed
    ).reset_index(drop=True)


def _summarise(batch: pd.DataFrame) -> dict:
    return {
        "n": int(len(batch)),
        "upi_share": round(float((batch["payment_method"] == "upi").mean()), 4),
        "fraud_share": round(float((batch["failure_category"] == "fraud_block").mean()), 4),
        "median_amount": round(float(batch["amount"].median()), 2),
    }


def main():
    ensure_dirs()
    baseline = load_baseline(BASELINE_PATH)
    if baseline is None:
        raise SystemExit("No baseline. Run: python scripts/train.py")
    if not os.path.exists(TEST_DATA):
        raise SystemExit("No test data. Run: python scripts/prepare_data.py")

    pool = pd.read_csv(TEST_DATA)
    shifted = build_shifted_batch(pool)
    os.makedirs(DEMO_DIR, exist_ok=True)
    shifted.to_csv(SHIFTED_CSV, index=False)

    shifted_drift = detect_drift(baseline, shifted)

    pinned_drift = None
    if os.path.exists(PINNED_CSV):
        pinned = pd.read_csv(PINNED_CSV)
        pinned_drift = detect_drift(baseline, pinned)

    meta = {
        "WARNING": "CONSTRUCTED FOR DEMONSTRATION. This drift was introduced "
                   "deliberately by scripts/make_drift_demo.py. It was not "
                   "observed in production or in real traffic.",
        "how": "Held-out rows re-weighted during sampling; no feature values "
               "were invented. Only the population mix differs from training.",
        "shift_recipe": SHIFT,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "composition": _summarise(shifted),
        "expected_verdict": shifted_drift["overall_verdict"],
        "expected_max_psi": shifted_drift["max_psi"],
    }
    with open(SHIFTED_META, "w") as f:
        json.dump(meta, f, indent=2)

    comparison = {
        "note": "Side-by-side drift comparison. The 'shifted' batch is "
                "CONSTRUCTED for demonstration -- see data/demo/drift_demo_batch.json.",
        "baseline_source": "models/training_baseline.json (from the training split)",
        "pinned": {
            "label": "Pinned demo batch (representative sample)",
            "constructed": False,
            "composition": _summarise(pd.read_csv(PINNED_CSV)) if pinned_drift else None,
            "drift": pinned_drift,
        },
        "shifted": {
            "label": "Deliberately shifted batch (constructed for demonstration)",
            "constructed": True,
            "composition": _summarise(shifted),
            "drift": shifted_drift,
        },
    }
    with open(COMPARISON, "w") as f:
        json.dump(comparison, f, indent=2)

    print("CONSTRUCTED demo batch -- drift introduced deliberately, not observed.\n")
    print(f"  {'':22} {'pinned':>12} {'shifted':>12}")
    if pinned_drift:
        p, s = _summarise(pd.read_csv(PINNED_CSV)), _summarise(shifted)
        for key in ("upi_share", "fraud_share", "median_amount"):
            print(f"  {key:22} {p[key]:>12} {s[key]:>12}")
        print(f"  {'verdict':22} {pinned_drift['overall_verdict']:>12} "
              f"{shifted_drift['overall_verdict']:>12}")
        print(f"  {'max PSI':22} {pinned_drift['max_psi']:>12} "
              f"{shifted_drift['max_psi']:>12}")

    print("\nPer-feature PSI on the shifted batch:")
    for name, feature in shifted_drift["features"].items():
        if feature.get("psi") is not None:
            print(f"  {name:22} {feature['psi']:>8.4f}  {feature['verdict']}")
    print(f"\n  {shifted_drift['driver_detail']}")

    print(f"\nWrote {SHIFTED_CSV}")
    print(f"Wrote {SHIFTED_META}")
    print(f"Wrote {COMPARISON}")


if __name__ == "__main__":
    main()
