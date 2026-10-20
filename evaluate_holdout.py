from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from features import load_sales
from dataset import build_forecast_rows

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

KEYS = ["store_nbr", "family"]
FOLDS = [
    ("fold_3", "2017-07-30"),
]


def score(actual, predicted):
    error = predicted - actual
    return {
        "rmsle": np.sqrt(
            np.mean((np.log1p(predicted) - np.log1p(actual)) ** 2)
        ),
        "mae": np.mean(np.abs(error)),
        "bias": np.mean(error),
    }


def attach_targets(rows, targets):
    result = rows.merge(
        targets,
        on=["target_date", *KEYS],
        how="left",
        validate="many_to_one",
    )

    if result["sales"].isna().any():
        raise ValueError("Missing training or evaluation targets.")

    return result


if __name__ == "__main__":
    wide = load_sales()

    train = pd.read_csv(
        ROOT / "data/raw/train.csv",
        parse_dates=["date"],
    )
    stores = pd.read_csv(ROOT / "data/raw/stores.csv")

    promotions = train[
        ["date", *KEYS, "onpromotion"]
    ].rename(columns={"date": "target_date"})

    targets = train[
        ["date", *KEYS, "sales"]
    ].rename(columns={"date": "target_date"})

    results = []

    for fold_name, cutoff_text in FOLDS:
        cutoff = pd.Timestamp(cutoff_text)
        print(f"\n{fold_name}: building training examples...", flush=True)

        # Even the final origin's horizon-16 target must be <= cutoff.
        last_training_origin = cutoff - pd.Timedelta(days=16)
        origins = pd.date_range(
            end=last_training_origin,
            periods=12,
            freq="7D",
        )

        # Prevent training-target lookup from seeing beyond cutoff.
        allowed_targets = targets[
            targets["target_date"] <= cutoff
        ]

        training_blocks = []

        for number, origin in enumerate(origins, start=1):
            rows = build_forecast_rows(
                wide, promotions, stores, origin
            )
            rows = attach_targets(rows, allowed_targets)
            training_blocks.append(rows)
            print(f"  Origin {number}/12 complete", flush=True)

        training = pd.concat(training_blocks, ignore_index=True)

        assert training["target_date"].max() <= cutoff

        validation = build_forecast_rows(
            wide, promotions, stores, cutoff
        )
        validation = attach_targets(validation, targets)

        assert validation["target_date"].min() > cutoff

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

        X_valid = encoder.transform(
            validation[feature_columns]
        ).astype(np.float32)

        y_train = np.log1p(training["sales"].to_numpy())
        actual = validation["sales"].to_numpy()

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

        print(
            f"Training XGBoost on {len(training):,} rows...",
            flush=True,
        )

        # Fixed configuration: no tuning on the final holdout.
        model.fit(X_train, y_train)

        predicted = np.maximum(
            np.expm1(model.predict(X_valid)), 0
        )

        if not np.isfinite(predicted).all():
            raise ValueError("Non-finite model predictions.")

        baseline = validation["weekday_mean_4"].to_numpy()

        for name, values in [
            ("weekday_mean_4", baseline),
            ("xgboost_v1", predicted),
        ]:
            results.append({
                "fold": fold_name,
                "model": name,
                **score(actual, values),
            })

        # Save individual predictions for family/horizon diagnostics.
        predictions = validation[
            ["origin", "target_date", "horizon", *KEYS, "sales"]
        ].copy()

        predictions["baseline"] = baseline
        predictions["prediction"] = predicted
        predictions.to_csv(
            OUTPUT / f"{fold_name}_xgb_predictions.csv",
            index=False,
        )

        # Save completed results after each fold.
        pd.DataFrame(results).to_csv(
            OUTPUT / "xgb_v1_holdout_metrics.csv",
            index=False,
        )

        print(f"{fold_name} finished.", flush=True)

    print("\nMODEL COMPARISON")
    print(pd.DataFrame(results).round(4).to_string(index=False))
