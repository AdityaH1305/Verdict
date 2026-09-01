"""
Population Stability Index drift detection.

READ-ONLY. Nothing here is imported by the agent, the models, or the decision
path -- it observes batches and reports, and cannot change what the agent does.

Why PSI for every feature, rather than PSI for categoricals and a KS test for
continuous ones:

  * One scale. PSI is comparable across mixed types, so a single combined
    verdict actually means something. A KS statistic and a PSI score are not
    on the same scale and cannot be maxed or averaged together honestly.
  * KS is continuous-only. Using it would force a second metric for
    `payment_method`, `error_code` and `failure_category` -- three of the five
    monitored features -- and then a rule for reconciling two scales.
  * PSI with the 0.10 / 0.25 cutoffs is the standing convention in credit-risk
    and fraud model monitoring, which is the domain this system sits in. Using
    the conventional metric means the numbers are readable by anyone who
    already monitors scorecards, without a translation step.

    PSI = sum over bins of (actual_i - expected_i) * ln(actual_i / expected_i)

Thresholds below are the conventional cutoffs, not values invented here.
"""

import numpy as np
import pandas as pd

# Conventional PSI cutoffs from credit-risk / scorecard monitoring practice.
PSI_STABLE = 0.10        # < 0.10  -- no meaningful shift
PSI_MODERATE = 0.25      # 0.10-0.25 -- moderate shift, worth watching
                         # > 0.25  -- significant shift, investigate

VERDICT_LOW = "LOW"
VERDICT_MODERATE = "MODERATE"
VERDICT_HIGH = "HIGH"

# Floor for zero bins. A 300-row batch cannot contain all 23 error codes, so
# empty bins are guaranteed rather than exceptional. Flooring (the conventional
# treatment) keeps PSI finite while still scoring a vanished category as large
# drift; dropping empty bins instead would hide exactly the shift worth catching.
EPSILON = 1e-4

# Nulls are a real signal here, not missing data to be discarded: `error_code`
# is ~11% null BY DESIGN, because a customer who abandons never reaches the bank
# for a decline code. Dropping nulls would mask a shift in abandonment rate.
MISSING_BUCKET = "__missing__"

# Rare training levels are merged here by the baseline builder, and any level the
# batch contains that training never saw lands here too. That second case is a
# feature, not a fallback: a brand-new error code appearing in production is a
# shift, and it should register as one rather than be silently dropped.
OTHER_BUCKET = "__other__"


def verdict_for(psi: float) -> str:
    if psi < PSI_STABLE:
        return VERDICT_LOW
    if psi < PSI_MODERATE:
        return VERDICT_MODERATE
    return VERDICT_HIGH


def psi_from_proportions(expected: np.ndarray, actual: np.ndarray) -> float:
    """
    PSI between two proportion vectors aligned on the same bins.

    Both are floored at EPSILON so an empty bin contributes a large finite
    number rather than inf or nan.
    """
    e = np.clip(np.asarray(expected, dtype=float), EPSILON, None)
    a = np.clip(np.asarray(actual, dtype=float), EPSILON, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _categorical_proportions(series: pd.Series, levels) -> np.ndarray:
    """
    Proportion per baseline level.

    Nulls go to MISSING_BUCKET. Anything the baseline does not list -- a rare
    level it merged, or a level training never saw at all -- goes to
    OTHER_BUCKET when the baseline has one, so proportions always sum to 1 and
    a novel category still moves the score.
    """
    values = series.astype(object).where(series.notna(), MISSING_BUCKET)
    counts = values.value_counts()
    total = max(len(values), 1)

    known = [level for level in levels if level != OTHER_BUCKET]
    proportions = [counts.get(level, 0) / total for level in known]

    if OTHER_BUCKET in levels:
        matched = sum(counts.get(level, 0) for level in known)
        other = (len(values) - matched) / total
        # insert at the baseline's own position for OTHER_BUCKET
        proportions.insert(list(levels).index(OTHER_BUCKET), other)

    return np.array(proportions, dtype=float)


def _binned_proportions(series: pd.Series, edges) -> np.ndarray:
    """
    Proportion per bin, using bin edges learned from the TRAINING data.

    Reusing the training edges is what makes the two distributions comparable;
    re-deriving quantiles from the current batch would make every batch look
    identical to itself and report no drift ever.
    """
    values = pd.to_numeric(series, errors="coerce").dropna()
    total = max(len(values), 1)
    # len(edges) - 1 interior bins, plus the open tails on each side
    counts, _ = np.histogram(values, bins=np.asarray(edges, dtype=float))
    below = int((values < edges[0]).sum())
    above = int((values > edges[-1]).sum())
    return np.concatenate([[below], counts, [above]]).astype(float) / total


def psi_for_feature(baseline_feature: dict, batch: pd.DataFrame) -> dict:
    """
    Score one monitored feature. Returns its PSI, verdict, and top contributors.

    `baseline_feature` is one entry from models/training_baseline.json.
    """
    kind = baseline_feature["kind"]
    column = baseline_feature["column"]

    if column not in batch.columns:
        return {"psi": None, "verdict": None,
                "note": f"column '{column}' not present in this batch"}

    series = batch[column]
    # error_code is monitored per rail, so a shift in the card/UPI mix does not
    # masquerade as a shift in the code distribution.
    if baseline_feature.get("payment_method"):
        series = series[batch["payment_method"] == baseline_feature["payment_method"]]

    if len(series) == 0:
        return {"psi": None, "verdict": None,
                "note": "no rows of this payment method in the batch"}

    if kind == "categorical":
        bins = baseline_feature["levels"]
        expected = np.array(baseline_feature["proportions"], dtype=float)
        actual = _categorical_proportions(series, bins)
    else:
        edges = baseline_feature["bin_edges"]
        bins = baseline_feature["bin_labels"]
        expected = np.array(baseline_feature["proportions"], dtype=float)
        actual = _binned_proportions(series, edges)

    psi = psi_from_proportions(expected, actual)

    # Per-bin contributions, so the report can say WHICH bucket moved rather
    # than only that something did.
    e = np.clip(expected, EPSILON, None)
    a = np.clip(actual, EPSILON, None)
    contributions = (a - e) * np.log(a / e)
    order = np.argsort(contributions)[::-1][:3]

    return {
        "psi": round(psi, 4),
        "verdict": verdict_for(psi),
        "n_rows": int(len(series)),
        "top_contributors": [
            {"bin": str(bins[i]),
             "expected": round(float(expected[i]), 4),
             "actual": round(float(actual[i]), 4),
             "contribution": round(float(contributions[i]), 4)}
            for i in order if contributions[i] > 1e-6
        ],
    }


def detect_drift(baseline: dict, batch: pd.DataFrame) -> dict:
    """
    Score every monitored feature and combine into one verdict.

    The overall verdict is the MAXIMUM per-feature PSI, not the average.
    PSI thresholds are defined per feature, so averaging would let one severely
    drifted feature hide behind several stable ones -- the precise failure a
    monitor exists to prevent. The driving feature is named so the verdict is
    actionable rather than just a colour.
    """
    features = {}
    for name, spec in baseline["features"].items():
        features[name] = psi_for_feature(spec, batch)

    scored = {n: f for n, f in features.items() if f.get("psi") is not None}
    if not scored:
        return {"overall_verdict": None, "features": features,
                "note": "no monitored feature could be scored on this batch"}

    driver = max(scored, key=lambda n: scored[n]["psi"])
    max_psi = scored[driver]["psi"]

    return {
        "overall_verdict": verdict_for(max_psi),
        "max_psi": max_psi,
        "driver": driver,
        "driver_detail": _describe_driver(driver, scored[driver]),
        "n_rows": int(len(batch)),
        "features": features,
        "thresholds": {"stable_below": PSI_STABLE, "significant_above": PSI_MODERATE},
    }


def _describe_driver(name: str, feature: dict) -> str:
    """One line naming what actually moved."""
    if feature["verdict"] == VERDICT_LOW:
        return f"No feature shows meaningful drift; '{name}' is the highest at PSI {feature['psi']}."
    top = feature.get("top_contributors") or []
    if not top:
        return f"'{name}' drifted (PSI {feature['psi']})."
    b = top[0]
    direction = "more" if b["actual"] > b["expected"] else "less"
    return (f"'{name}' drifted (PSI {feature['psi']}), driven by bin {b['bin']}: "
            f"{b['actual']:.1%} of the batch vs {b['expected']:.1%} in training "
            f"({direction} frequent).")
