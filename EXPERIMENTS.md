# Experiment Results

## Task
Forecast daily sales for 54 stores and 33 product families
over a 16-day horizon.

## Selection procedure
Used two chronological development folds to compare:
- Four-week weekday-average baseline
- XGBoost trained on log1p(sales)
- Three blends of XGBoost and the baseline

Selected XGBoost alone using mean development-fold RMSLE.
Model settings were fixed before final holdout evaluation.

## Final holdout
Forecast origin: July 30, 2017
Evaluation period: July 31–August 15, 2017
Predictions: 28,512

| Model | RMSLE | MAE | Bias |
|---|---:|---:|---:|
| Four-week weekday average | 0.5311 | 82.2121 | 2.9593 |
| XGBoost v1 | 0.4379 | 75.8240 | 17.1328 |

XGBoost reduced holdout RMSLE by approximately 17.5%
and MAE by 7.8%, but had greater positive bias.

## Assumptions and limitations
- Target-date promotion counts are assumed known in advance.
- Sales history features use observations through the forecast origin.
- Training targets cannot extend beyond the training cutoff.
- Missing calendar dates are not automatically treated as zero sales.
- Observed sales do not necessarily equal unconstrained demand.
- Results cover three summer evaluation windows, not all seasons.
- Overlapping historical forecast examples are not independent samples.
- Future-sales mutation checks cover feature generation, not every
  possible source of leakage.


## Kaggle Submission

- Model: XGBoost v1
- Submission: submission.csv
- Date: October 20, 2026
- Kaggle RMSLE: 0.40592
- Local final-holdout RMSLE: 0.43788

The model configuration was selected using local chronological
validation before submission. Kaggle and local holdout scores
evaluate different data and are reported separately.