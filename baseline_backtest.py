from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

FOLDS = [
    ("fold_1", "2017-06-28"),
    ("fold_2", "2017-07-14"),
]

# Fold 3 remains untouched until model choices are fixed.
HORIZON = 16
KEYS = ["store_nbr", "family"]

train = pd.read_csv(
    ROOT / "data" / "raw" / "train.csv",
    parse_dates=["date"],
)

if train.duplicated(["date", *KEYS]).any():
    raise ValueError("Duplicate date–store–family records found.")

# Each column is one store–family series.
wide = train.pivot(
    index="date",
    columns=KEYS,
    values="sales",
).sort_index()

# Restore calendar dates without inventing sales for missing days.
calendar = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
wide = wide.reindex(calendar)
wide.index.name = "date"


def metrics(actual, predicted):
    """Bias > 0 means overforecasting; bias < 0 means underforecasting."""
    error = predicted - actual

    return {
        "rmsle": np.sqrt(
            np.mean((np.log1p(predicted) - np.log1p(actual)) ** 2)
        ),
        "mae": np.mean(np.abs(error)),
        "bias": np.mean(error),
    }


overall_results = []
family_results = []

for fold_name, origin_text in FOLDS:
    origin = pd.Timestamp(origin_text)
    target_dates = pd.date_range(
        origin + pd.Timedelta(days=1),
        periods=HORIZON,
        freq="D",
    )

    # This slice is the only sales history available to the forecaster.
    history = wide.loc[:origin]
    actual = wide.loc[target_dates].to_numpy(dtype=float)

    if not np.isfinite(actual).all():
        raise ValueError(f"{fold_name}: missing evaluation targets.")

    predictions = {
        "seasonal_naive": [],
        "weekday_mean_4": [],
    }

    for target_date in target_dates:
        # Find the latest matching weekday at or before the origin.
        # This repeats the observed week even for horizons 8–16.
        days_back = (origin.weekday() - target_date.weekday()) % 7
        latest_same_weekday = origin - pd.Timedelta(days=days_back)

        reference_dates = [
            latest_same_weekday - pd.Timedelta(weeks=week)
            for week in range(4)
        ]

        reference_values = history.loc[reference_dates].to_numpy(
            dtype=float
        )

        # Fail visibly instead of silently replacing unknown sales.
        if not np.isfinite(reference_values).all():
            raise ValueError(
                f"{fold_name}: missing baseline reference observations."
            )

        predictions["seasonal_naive"].append(reference_values[0])
        predictions["weekday_mean_4"].append(
            reference_values.mean(axis=0)
        )

    families = wide.columns.get_level_values("family")

    for model_name, rows in predictions.items():
        predicted = np.asarray(rows)

        if predicted.shape != actual.shape:
            raise ValueError("Prediction and target shapes differ.")

        overall_results.append({
            "fold": fold_name,
            "model": model_name,
            "n_predictions": actual.size,
            **metrics(actual, predicted),
        })

        for family in families.unique():
            mask = families == family

            family_results.append({
                "fold": fold_name,
                "model": model_name,
                "family": family,
                **metrics(actual[:, mask], predicted[:, mask]),
            })

overall = pd.DataFrame(overall_results)
by_family = pd.DataFrame(family_results)

overall.to_csv(OUTPUT / "baseline_metrics.csv", index=False)
by_family.to_csv(OUTPUT / "baseline_metrics_by_family.csv", index=False)

print("\nBASELINE BACKTEST RESULTS")
print(overall.round(4).to_string(index=False))
print("\nSaved overall and family-level metrics in outputs/")

# Compare family-level performance across development folds.
family_comparison = (
    by_family.groupby(["family", "model"])["rmsle"]
    .mean()
    .unstack("model")
)

family_comparison["improvement_pct"] = (
    (
        family_comparison["seasonal_naive"]
        - family_comparison["weekday_mean_4"]
    )
    / family_comparison["seasonal_naive"].replace(0, np.nan)
    * 100
)

print("\nHARDEST FAMILIES — FOUR-WEEK BASELINE")
print(
    family_comparison
    .sort_values("weekday_mean_4", ascending=False)
    .head(10)
    .round(4)
    .to_string()
)

print("\nFAMILIES WHERE FOUR-WEEK AVERAGING IS WORSE")
worse = family_comparison[
    family_comparison["weekday_mean_4"]
    > family_comparison["seasonal_naive"]
]

print(worse.round(4).to_string())

family_comparison.to_csv(
    OUTPUT / "baseline_family_comparison.csv"
)