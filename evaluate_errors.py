from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"


def metrics(actual, predicted):
    error = predicted - actual

    return {
        "rmsle": np.sqrt(
            np.mean((np.log1p(predicted) - np.log1p(actual)) ** 2)
        ),
        "mae": np.mean(np.abs(error)),
        "bias": np.mean(error),
        "total_error": error.sum(),
    }


tables = {"family": [], "horizon": []}

for fold in ["fold_1", "fold_2"]:
    predictions = pd.read_csv(
        OUTPUT / f"{fold}_xgb_predictions.csv"
    )

    for dimension in tables:
        for group, data in predictions.groupby(dimension):
            actual = data["sales"].to_numpy()

            baseline = metrics(
                actual, data["baseline"].to_numpy()
            )
            model = metrics(
                actual, data["prediction"].to_numpy()
            )

            tables[dimension].append({
                "fold": fold,
                dimension: group,
                "baseline_rmsle": baseline["rmsle"],
                "xgb_rmsle": model["rmsle"],
                # Positive means XGBoost improves over the baseline.
                "rmsle_gain": baseline["rmsle"] - model["rmsle"],
                "baseline_mae": baseline["mae"],
                "xgb_mae": model["mae"],
                "baseline_bias": baseline["bias"],
                "xgb_bias": model["bias"],
                "xgb_total_error": model["total_error"],
            })

family_results = pd.DataFrame(tables["family"])
horizon_results = pd.DataFrame(tables["horizon"])

family_results.to_csv(
    OUTPUT / "xgb_v1_errors_by_family.csv", index=False
)
horizon_results.to_csv(
    OUTPUT / "xgb_v1_errors_by_horizon.csv", index=False
)

print("\nFOLD 2: LARGEST NET UNDERFORECASTS BY FAMILY")
print(
    family_results[
        (family_results["fold"] == "fold_2")
        & (family_results["xgb_total_error"] < 0)
    ]
    .sort_values("xgb_total_error")
    .head(10)
    .round(4)
    .to_string(index=False)
)

print("\nFAMILY/FOLD COMBINATIONS WHERE XGBOOST HAS WORSE RMSLE")
worse = family_results[family_results["rmsle_gain"] < 0]

if worse.empty:
    print("None.")
else:
    print(
        worse.sort_values("rmsle_gain")
        .round(4)
        .to_string(index=False)
    )

print("\nERRORS BY FORECAST HORIZON")
print(horizon_results.round(4).to_string(index=False))

