# Repository Layout

This repository was assembled from a larger research workspace and reorganized into a pipeline-oriented layout.

## Layout Logic

- `scripts/01_data_collection/`
  - raw GDELT retrieval scripts
- `scripts/02_relevance_classification/`
  - NLI-based relevance and channel scoring
- `scripts/03_sentiment_scoring/`
  - CrudeBert sentiment scoring
- `scripts/04_feature_engineering/`
  - daily aggregation, thresholding, imputation, smoothing, and target construction
- `scripts/05_modeling/`
  - ARIMAX, XGBoost, and tree-model benchmark experiments
- `scripts/06_reporting/`
  - final-results report generation and presentation helpers

## Artifacts

- `artifacts/sample_inputs/`
  - representative daily raw and scored files
- `artifacts/model_inputs/`
  - compact train/test modeling tables
- `artifacts/derived_features/`
  - daily channel indices and processed variants
- `artifacts/results/`
  - summary metrics and selected reports

## Notes

- The full original research workspace contained many more daily files and bulk outputs than are included here.
- The large CrudeBert model binary is intentionally excluded from this repository.
- The included files are sufficient to understand the research design and inspect the produced outputs.
