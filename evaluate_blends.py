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
    }


results = []

for fold in ["fold_1", "fold_2"]:
    df = pd.read_csv(OUTPUT / f"{fold}_xgb_predictions.csv")

    actual = df["sales"].to_numpy()
    baseline = df["baseline"].to_numpy()
    model = df["prediction"].to_numpy()

    # Endpoints reproduce the baseline and XGBoost-only results.
    for weight in [0.0, 0.25, 0.50, 0.75, 1.0]:
        prediction = (
            weight * model
            + (1 - weight) * baseline
        )

        results.append({
            "fold": fold,
            "xgb_weight": weight,
            **metrics(actual, prediction),
        })

results = pd.DataFrame(results)

print("\nRESULTS BY FOLD")
print(results.round(4).to_string(index=False))

summary = (
    results.groupby("xgb_weight")
    .agg(
        mean_fold_rmsle=("rmsle", "mean"),
        worst_fold_rmsle=("rmsle", "max"),
        mean_fold_mae=("mae", "mean"),
        mean_absolute_fold_bias=(
            "bias", lambda values: values.abs().mean()
        ),
    )
    .sort_values("mean_fold_rmsle")
)

print("\nBLEND COMPARISON")
print(summary.round(4).to_string())

results.to_csv(OUTPUT / "blend_metrics.csv", index=False)
summary.to_csv(OUTPUT / "blend_summary.csv")