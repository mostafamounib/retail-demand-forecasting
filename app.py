from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"

st.set_page_config(page_title="Retail Forecast Explorer", layout="wide")
st.title("Retail Forecast Explorer")
st.caption(
    "XGBoost forecasting • 54 stores • 33 product families • 16-day horizon"
)

holdout = pd.read_csv(
    OUTPUT / "fold_3_xgb_predictions.csv",
    parse_dates=["target_date"],
)
future = pd.read_csv(
    OUTPUT / "final_forecast_details.csv",
    parse_dates=["target_date"],
)

store = st.sidebar.selectbox(
    "Store", sorted(holdout["store_nbr"].unique())
)
family = st.sidebar.selectbox(
    "Product family", sorted(holdout["family"].unique())
)

selected_holdout = holdout[
    (holdout["store_nbr"] == store)
    & (holdout["family"] == family)
].sort_values("target_date")

selected_future = future[
    (future["store_nbr"] == store)
    & (future["family"] == family)
].sort_values("target_date")


def calculate_metrics(actual, predicted):
    return {
        "RMSLE": np.sqrt(
            np.mean((np.log1p(predicted) - np.log1p(actual)) ** 2)
        ),
        "MAE": np.mean(np.abs(predicted - actual)),
        "Bias": np.mean(predicted - actual),
    }


tab1, tab2, tab3 = st.tabs(
    ["Holdout inspection", "Future forecasts", "Error diagnostics"]
)

with tab1:
    st.subheader(f"Store {store} — {family}")
    st.caption(
        "Untouched evaluation window: July 31–August 15, 2017. "
        "Metrics below apply only to the selected series."
    )

    actual = selected_holdout["sales"].to_numpy()

    comparison = pd.DataFrame({
        "Baseline": calculate_metrics(
            actual, selected_holdout["baseline"].to_numpy()
        ),
        "XGBoost": calculate_metrics(
            actual, selected_holdout["prediction"].to_numpy()
        ),
    }).T

    st.dataframe(comparison.round(4))

    chart = selected_holdout.set_index("target_date")[
        ["sales", "baseline", "prediction"]
    ].rename(columns={
        "sales": "Actual sales",
        "baseline": "Four-week baseline",
        "prediction": "XGBoost",
    })

    st.line_chart(chart)
    st.caption("Positive bias means overprediction; negative means underprediction.")

with tab2:
    st.subheader(f"Future forecast — Store {store}, {family}")
    st.info(
        "August 16–31 actual sales are unavailable. "
        "This view displays forecasts, not measured accuracy."
    )

    st.line_chart(
        selected_future.set_index("target_date")[
            ["sales", "weekday_mean_4"]
        ].rename(columns={
            "sales": "XGBoost forecast",
            "weekday_mean_4": "Four-week baseline",
        })
    )

    st.dataframe(
        selected_future[["target_date", "horizon", "sales"]]
        .rename(columns={"sales": "forecast"}),
        hide_index=True,
    )

with tab3:
    st.subheader("Final holdout diagnostics")
    st.caption(
        "Uses all stores and families in the final holdout. "
        "Sidebar selections do not affect this tab."
    )

    dimension = st.selectbox("Group errors by", ["family", "horizon"])
    records = []

    for group, data in holdout.groupby(dimension):
        actual = data["sales"].to_numpy()
        baseline = calculate_metrics(
            actual, data["baseline"].to_numpy()
        )
        model = calculate_metrics(
            actual, data["prediction"].to_numpy()
        )

        records.append({
            dimension: group,
            "Baseline RMSLE": baseline["RMSLE"],
            "XGBoost RMSLE": model["RMSLE"],
            "RMSLE gain": baseline["RMSLE"] - model["RMSLE"],
            "Baseline MAE": baseline["MAE"],
            "XGBoost MAE": model["MAE"],
            "XGBoost bias": model["Bias"],
        })

    diagnostics = pd.DataFrame(records)

    if dimension == "family":
        diagnostics = diagnostics.sort_values(
            "XGBoost RMSLE", ascending=False
        )
    else:
        diagnostics = diagnostics.sort_values("horizon")

    st.dataframe(diagnostics.round(4), hide_index=True)
    st.caption(
        "Positive RMSLE gain means XGBoost beats the baseline. "
        "These are descriptive diagnostics, not a new tuning set."
    )