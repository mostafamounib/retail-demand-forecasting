from pathlib import Path

import numpy as np
import pandas as pd

from features import load_sales, build_origin_features

ROOT = Path(__file__).resolve().parent
KEYS = ["store_nbr", "family"]


def build_forecast_rows(wide, promotions, stores, origin):
    """Build prediction features without attaching target sales."""
    origin = pd.Timestamp(origin)
    base = build_origin_features(wide, origin)

    blocks = []

    for horizon in range(1, 17):
        block = base.copy()
        target_date = origin + pd.Timedelta(days=horizon)

        block["horizon"] = horizon
        block["target_date"] = target_date
        block["target_weekday"] = target_date.dayofweek
        block["target_month"] = target_date.month
        block["target_day"] = target_date.day
        block["target_year"] = target_date.year
        block["target_weekend"] = int(target_date.dayofweek >= 5)

        # Include the strong baseline as an input to XGBoost.
        # All reference dates remain at or before the origin.
        days_back = (origin.weekday() - target_date.weekday()) % 7
        reference_date = origin - pd.Timedelta(days=days_back)

        reference_dates = [
            reference_date - pd.Timedelta(weeks=i)
            for i in range(4)
        ]

        weekday_mean = wide.loc[reference_dates].mean()
        series_index = pd.MultiIndex.from_frame(block[KEYS])

        block["weekday_mean_4"] = (
            weekday_mean.reindex(series_index).to_numpy()
        )

        blocks.append(block)

    rows = pd.concat(blocks, ignore_index=True)

    # A validated join prevents accidental row multiplication.
    rows = rows.merge(
        stores,
        on="store_nbr",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    if not rows["_merge"].eq("both").all():
        raise ValueError("Missing store metadata.")

    rows = rows.drop(columns="_merge")

    rows = rows.merge(
        promotions,
        on=["target_date", *KEYS],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    if not rows["_merge"].eq("both").all():
        raise ValueError("Missing target-date promotion records.")

    rows = rows.drop(columns="_merge")

    if rows.duplicated(["origin", "horizon", *KEYS]).any():
        raise ValueError("Duplicate forecast examples.")

    if not rows["target_date"].gt(rows["origin"]).all():
        raise ValueError("Target dates must follow forecast origins.")

    return rows


if __name__ == "__main__":
    wide = load_sales()

    train = pd.read_csv(
        ROOT / "data/raw/train.csv",
        parse_dates=["date"],
    )
    stores = pd.read_csv(ROOT / "data/raw/stores.csv")

    # Deliberately exclude sales from this feature input.
    promotions = train[
        ["date", *KEYS, "onpromotion"]
    ].rename(columns={"date": "target_date"})

    origin = pd.Timestamp("2017-06-28")

    rows = build_forecast_rows(
        wide, promotions, stores, origin
    )

    # Check the expanded features, including the weekday baseline.
    altered = wide.copy()
    altered.loc[altered.index > origin] = 999999.0

    altered_rows = build_forecast_rows(
        altered, promotions, stores, origin
    )

    pd.testing.assert_frame_equal(rows, altered_rows)

    expected_rows = len(wide.columns) * 16
    assert len(rows) == expected_rows
    assert "sales" not in rows.columns

    print("Future-sales leakage check passed.")
    print("Forecast rows:", len(rows))
    print("Horizons:", rows["horizon"].min(), "to", rows["horizon"].max())
    print(
        "Target dates:",
        rows["target_date"].min().date(),
        "to",
        rows["target_date"].max().date(),
    )

    print("\nEXAMPLE ROWS")
    print(
        rows[
            [
                "store_nbr", "family", "origin", "target_date",
                "horizon", "weekday_mean_4", "onpromotion", "city",
            ]
        ].head().to_string(index=False)
    )

    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    rows.to_csv(output / "forecast_rows_sample.csv", index=False)
    