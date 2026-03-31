# Crude Oil Forecasting Engine

This repository contains the curated research code, selected artifacts, and documentation for a crude-oil forecasting pipeline built from news headlines and market data.

The core idea is:

1. collect crude-oil-related news headlines,
2. score each headline for market relevance and map it into six economically meaningful channels,
3. score headline sentiment,
4. aggregate those headline-level signals into daily indices,
5. test whether those indices improve crude-oil forecasting relative to price-history baselines.

## Research Goal

The project is designed to test whether richer news-derived features improve forecasting for crude oil. Instead of using a single sentiment score, the news signal is decomposed into six channels:

- Supply Shock and availability risk
- Transport logistics and chokepoints
- Demand and macro economy
- Inventory SPR and refinery
- OPEC producer policy
- Geopolitical Normalisation and Peace

These channels are then used as features in time-series and machine-learning models.

## Repository Structure

```text
scripts/
  01_data_collection/
  02_relevance_classification/
  03_sentiment_scoring/
  04_feature_engineering/
  05_modeling/

assets/
  crudebert/

notebooks/

artifacts/
  sample_inputs/
  model_inputs/
  derived_features/
  results/

docs/
```

## Pipeline Overview

### 1. Data Collection

The scripts in `scripts/01_data_collection/` retrieve daily GDELT news records for crude-oil-related keyword queries.

- `gdelt_yearly_downloader.py` is the main bulk downloader for year-by-year daily CSV retrieval.
- `gdelt_headlines.py` and `gdelt_json.py` provide JSON-based request variants.
- `gdelt_scraper.py` automates the GDELT UI with Playwright for CSV download through the web interface.

Primary output:

- daily raw headline files like `artifacts/sample_inputs/raw_headlines_2026-01-26.csv`

### 2. Relevance Classification

The scripts in `scripts/02_relevance_classification/` take raw headline CSVs and score them with zero-shot NLI.

- `relevance_scores_full_batch.py` is the main batch pipeline.
- `nli_single_file_probs.py` is the single-file version for spot runs and debugging.
- `scan_missing_files.py` checks for missing or undersized day files.

The relevance model produces:

- `prob_relevant`
- six per-channel relevance probabilities `p_<channel>`

Representative output:

- `artifacts/sample_inputs/relevance_scores_2026-01-26_nli_probs.csv`

### 3. Sentiment Scoring

The script in `scripts/03_sentiment_scoring/` runs a local BERT-based sentiment model called CrudeBert.

- `crudebert_batch_run.py` scores each headline as positive, negative, or neutral.
- `assets/crudebert/crude_bert_config.json` stores the model configuration used by that script.
- `notebooks/CrudeBert_Test.ipynb` is the exploratory notebook used to test loading and inference.

Important note:

The original workspace contained the model configuration and notebook, but the full binary weights file was not added here because it is large and not suitable for a lightweight code repository. The pipeline code still reflects how the model was used.

### 4. Feature Engineering

The scripts in `scripts/04_feature_engineering/` convert headline-level outputs into daily features.

- `step1_mass_thresholds.py` computes daily channel mass and estimates train-period evidence thresholds.
- `aggregate_daily_channel_index.py` aggregates headline-level relevance and polarity into daily channel indices.
- `impute_train_locf_cap5.py` applies low-evidence handling and capped LOCF imputation.
- `apply_decay.py` applies exponential decay smoothing to the channel series.
- `merge_sentiment.py` aligns daily global sentiment to returns data using `t-1` sentiment.
- `build_targets.py` creates market targets from OHLC data, including returns, Parkinson volatility, and jump flags.

Key daily aggregation logic:

- `mass_day(channel) = sum of relevance weights across headlines`
- `numerator_day(channel) = sum of relevance * polarity`
- `z_day(channel) = numerator / mass`

Included feature artifacts:

- `artifacts/derived_features/train_2017_2023_daily_channel_index.csv`
- `artifacts/derived_features/test_2024_2025_daily_channel_index.csv`
- `artifacts/derived_features/aux_2026_daily_channel_index.csv`
- `artifacts/derived_features/train_2017_2023_daily_channel_index_imputed_locf5.csv`
- `artifacts/derived_features/train_2017_2023_decayed_L10_lambda1_5.csv`

### 5. Modeling

The scripts in `scripts/05_modeling/` evaluate whether the engineered signals help forecasting.

#### ARIMAX

Located in `scripts/05_modeling/arimax/`.

- `arimax_price.py` runs price-only baseline models on multiple targets.
- `arimax_price_return.py` runs a return-only ARIMA baseline.
- `arimax_price_gsent.py` runs ARIMAX with lagged global sentiment as exogenous input.

Included result artifacts:

- `artifacts/results/arimax_price_summary_metrics.csv`
- `artifacts/results/arimax_return_summary_metrics.csv`
- `artifacts/results/return_arimax_with_sent_tminus1_report.json`

#### XGBoost

Located in `scripts/05_modeling/xgboost/`.

- `xgb_returns_hypothesis_runner.py` compares three scenarios:
  - lagged returns only
  - lagged returns plus global sentiment
  - lagged returns plus six channel indices

Included result artifacts:

- `artifacts/results/xgboost_metrics_summary.csv`
- `artifacts/results/shap_global_importance_returns_plus_global_polarity.csv`
- `artifacts/results/shap_global_importance_returns_plus_6channels.csv`

## Included Artifacts

This repository intentionally includes a representative subset of the original workspace so the research flow is understandable without carrying the full raw corpus.

### `artifacts/sample_inputs/`

Small examples of:

- a raw daily headline file,
- the corresponding NLI relevance output,
- a channel-labeled day-level export.

### `artifacts/model_inputs/`

Compact tabular inputs used by the downstream modeling scripts, including:

- price files,
- global sentiment files,
- target files,
- target files with lagged sentiment merged in.

### `artifacts/derived_features/`

Daily aggregated channel index files and processed variants used for train, test, and auxiliary periods.

### `artifacts/results/`

Selected model outputs and summary metrics showing the current empirical results.

## Current Findings

The included outputs suggest that the news-derived signals add only modest incremental forecasting value relative to price-history baselines.

### ARIMAX

From the included summary files:

- return-only ARIMA RMSE is approximately `0.016915`
- ARIMAX with lagged global sentiment improves slightly to approximately `0.016895`

### XGBoost

From `artifacts/results/xgboost_metrics_summary.csv`:

- lagged returns only RMSE is approximately `0.016994`
- lagged returns plus global sentiment RMSE is approximately `0.016977`
- lagged returns plus six channels RMSE is approximately `0.016977`

Interpretation:

- the six-channel design is methodologically useful and economically interpretable,
- the incremental predictive lift is currently small,
- the project is stronger as a research pipeline than as evidence of large forecasting gains.

## What Is Included vs Omitted

Included:

- the main research scripts,
- selected inputs and outputs,
- representative derived feature files,
- summary model results,
- the CrudeBert configuration and test notebook.

Omitted:

- the full raw headline archive,
- the full per-day relevance and sentiment corpora,
- the large CrudeBert binary weights file,
- literature PDFs and unrelated workspace material.

The goal of this repository is to preserve the research logic in a readable and portable form.

## Running The Code

Most scripts were originally written as research utilities and expect file paths and parameters to be edited at the top of each file before execution.

Typical execution order:

1. run the data collection scripts
2. run relevance classification
3. run sentiment scoring
4. build daily channel indices and derived features
5. build market targets
6. run ARIMAX and XGBoost experiments

Because the scripts still contain environment-specific paths from the original workspace, they should be treated as research code rather than drop-in production software.

## Suggested Next Improvements

If this repository is extended further, the highest-value cleanup steps would be:

1. replace hardcoded local paths with CLI arguments or config files,
2. add a reproducible environment file,
3. add one orchestrating pipeline script,
4. document the missing headline-level polarity merge step more explicitly,
5. add train-test evaluation notebooks and ablation summaries.

## Dependencies

See `requirements.txt` for a practical package list inferred from the included code.
