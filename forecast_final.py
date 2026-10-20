from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from features import load_sales
from dataset import build_forecast_rows
from train_xgb import attach_targets

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data/raw"
OUTPUT = ROOT / "outputs"
MODELS = ROOT / "models"

OUTPUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

KEYS = ["store_nbr", "family"]


if __name__ == "__main__":
    train = pd.read_csv(RAW / "train.csv", parse_dates=["date"])
    test = pd.read_csv(RAW / "test.csv", parse_dates=["date"])
    stores = pd.read_csv(RAW / "stores.csv")
    sample = pd.read_csv(RAW / "sample_submission.csv")
    wide = load_sales()

    origin = pd.Timestamp("2017-08-15")
    expected_dates = pd.date_range("2017-08-16", periods=16)

    if train["date"].max() != origin:
        raise ValueError("Unexpected training cutoff.")

    if set(test["date"]) != set(expected_dates):
        raise ValueError("Unexpected test dates.")

    if sample.columns.tolist() != ["id", "sales"]:
        raise ValueError("Unexpected submission columns.")

    if sample["id"].duplicated().any() or test["id"].duplicated().any():
        raise ValueError("Duplicate submission/test IDs.")

    if set(sample["id"]) != set(test["id"]):
        raise ValueError("Submission IDs do not match test IDs.")

    promotions = pd.concat(
        [
            train[["date", *KEYS, "onpromotion"]],
            test[["date", *KEYS, "onpromotion"]],
        ],
        ignore_index=True,
    ).rename(columns={"date": "target_date"})

    targets = train[
        ["date", *KEYS, "sales"]
    ].rename(columns={"date": "target_date"})

    # Same 12-origin training recipe used during evaluation.
    origins = pd.date_range(
        end=origin - pd.Timedelta(days=16),
        periods=12,
        freq="7D",
    )

    blocks = []

    for number, historical_origin in enumerate(origins, start=1):
        rows = build_forecast_rows(
            wide, promotions, stores, historical_origin
        )
        rows = attach_targets(rows, targets)
        blocks.append(rows)
        print(f"Training origin {number}/12 complete", flush=True)

    training = pd.concat(blocks, ignore_index=True)

    if training["target_date"].max() > origin:
        raise ValueError("Training targets extend beyond the cutoff.")

    future = build_forecast_rows(
        wide, promotions, stores, origin
    )

    excluded = {"origin", "target_date", "sales"}
    feature_columns = [
        column for column in training.columns
        if column not in excluded
    ]

    categorical = [
        "store_nbr", "family", "city", "state", "type", "cluster"
    ]
    numeric = [
        column for column in feature_columns
        if column not in categorical
    ]

    encoder = ColumnTransformer(
        transformers=[
            (
                "categories",
                OneHotEncoder(
                    handle_unknown="ignore",
                    dtype=np.float32,
                ),
                categorical,
            ),
            ("numeric", "passthrough", numeric),
        ],
        sparse_threshold=1.0,
    )

    X_train = encoder.fit_transform(
        training[feature_columns]
    ).astype(np.float32)

    X_future = encoder.transform(
        future[feature_columns]
    ).astype(np.float32)

    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=4,
    )

    print(f"Training final model on {len(training):,} rows...", flush=True)

    model.fit(
        X_train,
        np.log1p(training["sales"].to_numpy()),
    )

    predictions = np.maximum(
        np.expm1(model.predict(X_future)), 0
    )

    if not np.isfinite(predictions).all():
        raise ValueError("Non-finite forecasts.")

    forecast = future[
        ["target_date", *KEYS, "horizon", "weekday_mean_4"]
    ].copy()

    forecast["sales"] = predictions

    # Match predictions by date/store/family—not by row position.
    mapped = test.rename(columns={"date": "target_date"}).merge(
        forecast,
        on=["target_date", *KEYS],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not mapped["_merge"].eq("both").all():
        raise ValueError("Forecast keys do not exactly match test keys.")

    mapped = mapped.drop(columns="_merge")

    # Explicitly restore the sample submission's ordering.
    submission = (
        mapped.set_index("id")[["sales"]]
        .reindex(sample["id"])
        .reset_index()
    )

    if submission["sales"].isna().any():
        raise ValueError("Missing submission predictions.")

    if not submission["id"].equals(sample["id"]):
        raise ValueError("Submission ID order mismatch.")

    if not submission["sales"].ge(0).all():
        raise ValueError("Negative submission predictions.")

    submission.to_csv(OUTPUT / "submission.csv", index=False)
    mapped.sort_values("id").to_csv(
        OUTPUT / "final_forecast_details.csv",
        index=False,
    )

    model.save_model(MODELS / "xgboost_v1.json")
    joblib.dump(encoder, MODELS / "encoder.joblib")

    metadata = {
        "forecast_origin": str(origin.date()),
        "horizon_days": 16,
        "training_origins": [str(day.date()) for day in origins],
        "training_rows": len(training),
        "feature_columns": feature_columns,
        "categorical_columns": categorical,
        "target_transform": "log1p",
        "prediction_transform": "maximum(expm1(prediction), 0)",
        "model_parameters": model.get_params(),
    }

    with open(MODELS / "metadata.json", "w") as file:
        json.dump(metadata, file, indent=2)

    # Verify the exported file can be read with the correct structure.
    saved = pd.read_csv(OUTPUT / "submission.csv")
    assert saved.columns.tolist() == ["id", "sales"]
    assert saved["id"].equals(sample["id"])
    assert np.isfinite(saved["sales"]).all()
    assert saved["sales"].ge(0).all()

    print("\nFORECAST COMPLETE")
    print(f"Submission rows: {len(saved):,}")
    print("Dates: August 16–31, 2017")
    print("Submission checks passed.")
    print("Forecasts saved in outputs/; model and encoder saved in models/")
    print("\nPreview:")
    print(saved.head().to_string(index=False))