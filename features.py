from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def load_sales():
    """Return a daily matrix: rows = dates, columns = store/family."""
    train = pd.read_csv(
        ROOT / "data/raw/train.csv",
        parse_dates=["date"],
    )

    if train.duplicated(["date", "store_nbr", "family"]).any():
        raise ValueError("Duplicate sales keys found.")

    wide = train.pivot(
        index="date",
        columns=["store_nbr", "family"],
        values="sales",
    ).sort_index()

    calendar = pd.date_range(
        wide.index.min(),
        wide.index.max(),
        freq="D",
    )

    return wide.reindex(calendar)


def build_origin_features(wide, origin):
    """Build one feature row per series using history through origin."""
    origin = pd.Timestamp(origin)

    if origin not in wide.index:
        raise ValueError("Forecast origin is outside the sales calendar.")

    # Restrict access before computing any feature.
    history = wide.loc[:origin]

    if len(history) < 56:
        raise ValueError("At least 56 calendar days of history required.")

    features = pd.DataFrame(index=wide.columns)

    # Days before the forecast origin, not before the target date.
    for days_back in [0, 1, 6, 7, 13, 14, 27, 28]:
        reference_date = origin - pd.Timedelta(days=days_back)
        features[f"sales_origin_minus_{days_back}"] = (
            history.loc[reference_date]
        )

    for window in [7, 14, 28, 56]:
        recent = history.tail(window)

        features[f"mean_{window}d"] = recent.mean()
        features[f"std_{window}d"] = recent.std(ddof=0)
        features[f"observed_days_{window}d"] = recent.count()

        # Unknown dates must not count as observed nonzero sales.
        observed_count = recent.count().replace(0, np.nan)
        features[f"zero_share_{window}d"] = (
            recent.eq(0).sum() / observed_count
        )

    # Difference rather than a ratio: stable when earlier sales are zero.
    features["recent_week_change"] = (
        history.tail(7).mean()
        - history.iloc[-14:-7].mean()
    )

    features = features.reset_index()
    features["origin"] = origin

    return features


if __name__ == "__main__":
    wide = load_sales()
    origin = pd.Timestamp("2017-06-28")

    features = build_origin_features(wide, origin)

    # Leakage test: change all future sales.
    # Features at this origin must remain exactly the same.
    altered = wide.copy()
    altered.loc[altered.index > origin] = 999999.0

    altered_features = build_origin_features(altered, origin)

    pd.testing.assert_frame_equal(features, altered_features)

    print("Leakage check passed: future sales do not change features.")
    print(f"Feature table shape: {features.shape}")

    print("\nEXAMPLE FEATURE ROWS")
    print(features.head().to_string(index=False))

    print("\nCOLUMNS CONTAINING MISSING VALUES")
    missing = features.isna().sum()
    print(missing[missing > 0].to_string())

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    features.to_csv(
        output_dir / "origin_features_sample.csv",
        index=False,
    )

    print("\nSaved outputs/origin_features_sample.csv")