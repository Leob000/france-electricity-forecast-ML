# Net‑Load Forecasting During the “Soberty” Period (R + Python)

> TL;DR: End‑to‑end pipeline to forecast daily Net demand (France) during 2013–2023 with feature‑rich preprocessing, multiple models (GAM, Random Forest, XGBoost, Temporal Fusion Transformer), and light ensembling (incl. an MLP meta‑learner). Evaluated with Pinball Loss at τ = 0.8 (competition metric).

Kaggle competition: [Net-Load Forecasting During the "Soberty" Period](https://www.kaggle.com/competitions/net-load-soberty-period)

## What this repository does

* Goal: Forecast daily Net demand from mid‑2022 to mid‑2023 for the Kaggle challenge *“Net‑Load Forecasting During the ‘Soberty’ Period”*.
* Why it matters: Power providers need adaptive forecasts as consumption habits change (price shocks, savings behavior) and generation mix becomes more intermittent.
* How:

  * R for data processing and classic statistical learning (GAM, Random Forest, diagnostics).
  * Python for gradient‑boosting (XGBoost), sequence modeling (Temporal Fusion Transformer) and ensembling.
  * Unified metric: Pinball loss at quantile 0.8 (plus RMSE/MAE/bias for analysis).

## Highlights

* End‑to‑end pipeline from raw CSVs → engineered features → multiple models → ensemble → Kaggle‑ready submission.
* Feature engineering

  * Calendar effects (weekdays/holidays, DST, school breaks) with cyclical encodings (day‑of‑year & day‑of‑week as sin/cos).
  * Lagged loads (Net\_demand.1/.7, Load.1/.7) and trend (7‑day rolling mean of Net\_demand.1).
  * Weather: temperature summaries (s95/s99, min/max), wind, nebulosity; stabilization of pre‑2018 nebulosity and weighted ratios.
  * Exogenous signals: COVID policy indices (Oxford), simple tariff/behavior proxies.
  * Careful train‑only normalization to prevent leakage; preserved mean/SD to de‑standardize predictions.
* Models

  * GAM (mgcv): smooth seasonality + interactions (tensor products), holiday effects, weather, lags.
  * Random Forest (R): strong tabular baseline, OOB diagnostics.
  * XGBoost (Python): tuned hist‑GBDT on engineered features.
  * Temporal Fusion Transformer (PyTorch Forecasting): encoder length ≈ 1 year (366d), multi‑quantile outputs.
  * Ensembling: simple means/weighted means and an MLP meta‑learner over model predictions and a few stable features.
  * Experimental: state‑space / Kalman online learning (viking) for adaptive coefficients.
* Validation

  * Time‑aware splits (e.g., last‑year validation inside train; optional blocked CV for GAM residuals).
  * Example internal score: Pinball(τ=0.8) ≈ 225 on 2021‑09‑02 → 2022‑09‑01 using the MLP meta‑learner (see `ensemble.py`).

## Getting started

> The repo mixes R and Python. Install both environments first.

### Prerequisites

* R ≥ 4.2 with packages: `tidyverse`, `mgcv`, `randomForest`, `forecast`, `yarrr`, `magrittr`, `corrplot`, `zoo`, `viking`.
* Python ≥ 3.10 with packages: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `xgboost`, `lightning` (PyTorch Lightning), `pytorch-forecasting`, `torch`.

### Data

Place the competition CSVs under `Data/` and ensure the filenames used in the scripts match (see the commands below). The pipeline will also write intermediate artifacts to `Data/`.

### One‑line view of the pipeline

```text
Data/train.csv + Data/test.csv
        │
        ├── R/data_processing.r      # feature engineering + normalization → Data/treated_data.csv
        │
        ├── R/model_RF.r             # Random Forest → Data/preds_rf_{train,test}.csv
        ├── python/model_xgboost.py  # XGBoost      → Data/preds_xgb_{train,test}.csv
        ├── python/model_TFT.py      # TFT (multi‑quantile) → Data/preds_tft_{train,test}_new.csv
        ├── R/GAM_exploration.r      # GAM fits, residual diagnostics, optional Kalman experiment
        │
        └── python/ensemble.py       # Simple/weighted blends + MLP meta‑learner → Data/preds_mlp.csv (submission)
```

### Quickstart (commands)

1. Preprocess

```sh
Rscript data_processing.r
```

2. Train baselines

```sh
Rscript model_RF.r          # writes rf preds (uncomment write_csv lines if needed)
python model_xgboost.py     # writes xgb preds (uncomment to_csv lines if needed)
```

3. Train TFT (PyTorch Forecasting)

```sh
python model_TFT.py         # produces multi‑quantile preds for train/test
```

4. Blend / Meta‑learn & export

```sh
python ensemble.py          # generates Data/preds_mlp.csv (Id, Net_demand)
```

> Notes
>
> * Scripts default to CPU; switch to GPU in `model_TFT.py` if available.
> * Some `write_*` lines are commented to avoid accidental overwrites; enable them as needed.

## Repository structure (selected files)

```
R/
  score.R           # metrics: RMSE, MAPE, bias, pinball loss (τ‑generic)
  data_processing.r # feature engineering, normalization, joins (→ Data/treated_data.csv)
  model_RF.r        # Random Forest baseline (R)
  GAM_exploration.r # GAM with smooth terms, interactions, residual diagnostics,
                    # and optional Kalman state‑space experiment
python/
  model_xgboost.py  # XGBoost regressor on engineered features
  model_TFT.py      # Temporal Fusion Transformer (pytorch‑forecasting)
  ensemble.py       # simple blends + MLP meta‑learner; pinball(τ=0.8) evaluation
Data/
  schema.png        # quick diagram of files
  (train/test/covid CSVs; generated preds live here)
README.md           # you are here
pyproject.toml / pyprojet.toml (linting config)
```

## Modeling details

* Metrics

  * Primary: Pinball Loss at quantile τ = 0.8 (competition metric).
  * Secondary: RMSE, MAE (absolute loss), and bias for sanity checks.
* Why τ = 0.8?

  * The challenge evaluates point predictions as estimates of the 0.8 quantile; models and ensembling are tuned to minimize this objective.
* TFT quantiles & ensembling

  * TFT outputs 7 quantiles; explored alignment with GAM/XGB and used simple/weighted blends.
  * A small MLP meta‑learner over stable features and model predictions provided the best internal validation.
* Diagnostics

  * Blocked residual analysis, ACF of residuals, weekday/holiday patterns, histogram checks for approximate Gaussianity.
* Adaptivity (experimental)

  * State‑space Kalman approach (via `viking`) on GAM term predictions to adapt coefficients over time.

## Reproducibility

* Seeds are set where possible (`set.seed(42)` in R, fixed `random_state` / `seed_everything` in Python).
* Train‑only statistics for normalization; de‑standardization uses saved mean/SD.
* Time‑based splits (no leakage across the forecast horizon).

## What to look at first

* `data_processing.r`: feature engineering decisions and leakage‑safe scaling.
* `GAM_exploration.r`: interpretable baseline + residual diagnostics.
* `model_TFT.py`: sequence modeling setup (encoder length = 366 days) and quantile outputs.
* `ensemble.py`: simple yet effective meta‑learning for the τ=0.8 objective.
