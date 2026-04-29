# Crude Oil Forecasting Engine

This repository contains the final code, selected artifacts, and reporting outputs for a crude-oil forecasting project built from news headlines and market data.

The project asks a narrow question:

Can interpretable news signals improve crude-oil forecasting beyond lagged market data?

The answer from the current experiments is:

- yes, news features help in some setups,
- the improvement is real but modest,
- the strongest gains come from careful channel selection and walk-forward evaluation rather than from a single global sentiment score.

## What This Repo Now Contains

This is no longer just a mid-pipeline prototype. The repository now includes:

- a full news-to-features pipeline,
- six interpretable news channels,
- daily raw / imputed / decayed feature variants,
- return, volatility, and jump targets,
- walk-forward benchmarking across `XGBoost`, `LightGBM`, and `CatBoost`,
- SHAP exports for the best scenarios,
- final report and presentation-ready assets.

## Final Results

The strongest walk-forward results currently included in the repository are:

| Target | Best Model | Best Scenario | Main Metric | Secondary Metric |
|---|---|---|---:|---:|
| Return | CatBoost | `raw__market_plus_all6_channels` | RMSE `0.057499` | MAE `0.043308` |
| Volatility | XGBoost | `raw_decay__single__opec_producer_policy` | RMSE `0.007516` | MAE `0.005480` |
| Jump | CatBoost | `imputed_locf5_decay__market_plus_all6_channels` | PR-AUC `0.749896` | F1 `0.704545` |

Model-specific winners from the full benchmark:

| Model | Return Winner | Volatility Winner | Jump Winner |
|---|---|---|---|
| XGBoost | `imputed_locf5_decay__single__demand_and_macro_economy` | `raw_decay__single__opec_producer_policy` | `imputed_locf5_decay__single__opec_producer_policy` |
| LightGBM | `imputed_locf5_decay__single__demand_and_macro_economy` | `raw__market_only` | `imputed_locf5_decay__single__opec_producer_policy` |
| CatBoost | `raw__market_plus_all6_channels` | `raw_decay__single__opec_producer_policy` | `imputed_locf5_decay__market_plus_all6_channels` |

What those results mean:

- return prediction benefits most from richer news structure in the CatBoost setup,
- volatility prediction is especially sensitive to the `OPEC producer policy` channel,
- jump prediction gets the clearest lift from news-aware classification setups,
- all six channels together are not always best; single-channel ablations matter.

## Honest Takeaway

The project is strongest as an interpretable research pipeline with disciplined evaluation.

What is already convincing:

- the news signal is decomposed into economically meaningful channels,
- the feature engineering supports multiple preprocessing choices,
- evaluation moved from single-split comparisons to walk-forward backtesting,
- different targets clearly prefer different model / feature combinations.

What is not yet justified:

- claiming large forecasting gains,
- claiming that one universal news feature set wins for every task,
- treating the results as production-grade trading signals.

## Where To Look First

If you want the final outputs without reading the whole codebase, start here:

- final report: `docs/final_results.md`
- comparison table: `docs/assets/final_results/final_comparison_table.png`
- architecture diagram: `docs/assets/final_results/architecture_diagram.png`
- worked example day: `docs/assets/final_results/example_day_2026-01-26.png`
- SHAP manifest: `docs/assets/final_results/shap_manifest.csv`

Important benchmark summaries:

- `artifacts/results/tree_model_walkforward_xgboost_full/metrics_summary.csv`
- `artifacts/results/tree_model_walkforward_lightgbm_full/metrics_summary.csv`
- `artifacts/results/tree_model_walkforward_catboost_full/metrics_summary.csv`
- `artifacts/results/tree_model_walkforward_best_shap/`

## Pipeline In One Pass

The pipeline is:

1. collect crude-oil-related GDELT headlines,
2. score each headline for relevance,
3. map relevant headlines into six channels,
4. score headline sentiment with CrudeBert,
5. aggregate headline-level signals into daily channel indices,
6. build raw, imputed, and decayed feature variants,
7. construct return, Parkinson volatility, and jump targets,
8. benchmark forecasting models with walk-forward validation.

## Six News Channels

The project uses these interpretable channels:

- Supply Shock and availability risk
- Transport logistics and chokepoints
- Demand and macro economy
- Inventory SPR and refinery
- OPEC producer policy
- Geopolitical Normalisation and Peace

## Repository Layout

```text
scripts/
  01_data_collection/
  02_relevance_classification/
  03_sentiment_scoring/
  04_feature_engineering/
  05_modeling/
  06_reporting/

artifacts/
  sample_inputs/
  model_inputs/
  derived_features/
  results/

docs/
  final_results.md
  assets/final_results/

assets/
  crudebert/

notebooks/
```

## Key Scripts

### Data collection

- `scripts/01_data_collection/gdelt_yearly_downloader.py`
- `scripts/01_data_collection/gdelt_headlines.py`
- `scripts/01_data_collection/gdelt_json.py`
- `scripts/01_data_collection/gdelt_scraper.py`

### Relevance and channels

- `scripts/02_relevance_classification/relevance_scores_full_batch.py`
- `scripts/02_relevance_classification/nli_single_file_probs.py`

### Sentiment

- `scripts/03_sentiment_scoring/crudebert_batch_run.py`

### Feature engineering

- `scripts/04_feature_engineering/aggregate_daily_channel_index.py`
- `scripts/04_feature_engineering/impute_train_locf_cap5.py`
- `scripts/04_feature_engineering/apply_decay.py`
- `scripts/04_feature_engineering/merge_sentiment.py`
- `scripts/04_feature_engineering/build_targets.py`

### Modeling

- `scripts/05_modeling/arimax/arimax_price.py`
- `scripts/05_modeling/arimax/arimax_price_gsent.py`
- `scripts/05_modeling/arimax/arimax_price_return.py`
- `scripts/05_modeling/xgboost/xgb_returns_hypothesis_runner.py`
- `scripts/05_modeling/xgboost/xgb_walkforward_multitask.py`
- `scripts/05_modeling/boosting/tree_walkforward_benchmark.py`

### Reporting

- `scripts/06_reporting/build_final_results_report.py`
- `scripts/06_reporting/build_presentation_assets.py`

## Included Artifacts

Representative sample inputs:

- `artifacts/sample_inputs/raw_headlines_2026-01-26.csv`
- `artifacts/sample_inputs/relevance_scores_2026-01-26_nli_probs.csv`
- `artifacts/sample_inputs/channel_labels_2026-01-26.csv`

Daily engineered features:

- `artifacts/derived_features/train_2017_2023_daily_channel_index.csv`
- `artifacts/derived_features/test_2024_2025_daily_channel_index.csv`
- `artifacts/derived_features/aux_2026_daily_channel_index.csv`
- `artifacts/derived_features/train_2017_2023_daily_channel_index_imputed_locf5.csv`
- `artifacts/derived_features/train_2017_2023_decayed_L10_lambda1_5.csv`

Final benchmark outputs:

- `artifacts/results/tree_model_walkforward_xgboost_full/`
- `artifacts/results/tree_model_walkforward_lightgbm_full/`
- `artifacts/results/tree_model_walkforward_catboost_full/`
- `artifacts/results/tree_model_walkforward_best_shap/`

## Reproducing The Final Outputs

A lighter benchmark environment is included for the final results workflow.

```bash
python3 -m venv .venv-bench
.venv-bench/bin/pip install -r requirements-benchmark.txt
```

Full benchmark runs:

```bash
python scripts/05_modeling/boosting/tree_walkforward_benchmark.py \
  --models xgboost \
  --tasks return volatility jump \
  --scenario-set full \
  --disable-shap \
  --outdir artifacts/results/tree_model_walkforward_xgboost_full

python scripts/05_modeling/boosting/tree_walkforward_benchmark.py \
  --models lightgbm \
  --tasks return volatility jump \
  --scenario-set full \
  --disable-shap \
  --outdir artifacts/results/tree_model_walkforward_lightgbm_full

python scripts/05_modeling/boosting/tree_walkforward_benchmark.py \
  --models catboost \
  --tasks return volatility jump \
  --scenario-set full \
  --disable-shap \
  --outdir artifacts/results/tree_model_walkforward_catboost_full
```

Winner-scenario SHAP reruns:

```bash
python scripts/05_modeling/boosting/tree_walkforward_benchmark.py \
  --models catboost \
  --tasks return \
  --scenario-set full \
  --scenario-names raw__market_plus_all6_channels \
  --shap-scenarios raw__market_plus_all6_channels \
  --outdir artifacts/results/tree_model_walkforward_best_shap/catboost_return
```

Reporting:

```bash
python scripts/06_reporting/build_presentation_assets.py
python scripts/06_reporting/build_final_results_report.py
```

## Dependencies

- `requirements.txt` is the broader inferred dependency list.
- `requirements-benchmark.txt` is the lighter path for the benchmark and reporting workflow.

## Limitations

- some original research scripts still reflect workspace-style path assumptions and are not packaged as a clean CLI application,
- the full CrudeBert binary weights are not stored in this repository,
- the repository is a curated final-state workspace, not a polished production package,
- the benchmark outputs are meaningful research results, but they do not justify claims of large predictive edge.
