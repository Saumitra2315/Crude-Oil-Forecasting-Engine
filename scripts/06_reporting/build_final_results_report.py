from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
RESULTS_DIR = ARTIFACTS_DIR / "results"
WALKFORWARD_DIR = RESULTS_DIR / "xgboost_walkforward"
TREE_BENCH_DIR = RESULTS_DIR / "tree_model_walkforward"
BEST_SHAP_DIR = RESULTS_DIR / "tree_model_walkforward_best_shap"
XGBOOST_FULL_DIR = RESULTS_DIR / "tree_model_walkforward_xgboost_full"
LIGHTGBM_FULL_DIR = RESULTS_DIR / "tree_model_walkforward_lightgbm_full"
CATBOOST_FULL_DIR = RESULTS_DIR / "tree_model_walkforward_catboost_full"
DOCS_PATH = REPO_ROOT / "docs" / "final_results.md"
DOC_ASSET_DIR = REPO_ROOT / "docs" / "assets" / "final_results"

SAMPLE_DAY = "2026-01-26"


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _fmt_num(raw: Optional[str]) -> str:
    if raw in (None, "", "nan", "NaN"):
        return "-"
    try:
        val = float(raw)
    except Exception:
        return str(raw)
    if abs(val) >= 1000:
        return f"{val:,.2f}"
    return f"{val:.6f}"


def _md_cell(text: Optional[str]) -> str:
    if text is None:
        return "-"
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _pick_shap_figure() -> Optional[Path]:
    if BEST_SHAP_DIR.exists():
        preferred_best = sorted(BEST_SHAP_DIR.glob("catboost_return/shap_summary_bar__*.png"))
        if preferred_best:
            return preferred_best[0]
        fallback_best = sorted(BEST_SHAP_DIR.glob("*/shap_summary_bar__*.png"))
        if fallback_best:
            return fallback_best[0]
    if TREE_BENCH_DIR.exists():
        preferred_tree = sorted(TREE_BENCH_DIR.glob("shap_summary_bar__xgboost__return__*market_plus_all6_channels*.png"))
        if preferred_tree:
            return preferred_tree[0]
        fallback_tree = sorted(TREE_BENCH_DIR.glob("shap_summary_bar__*.png"))
        if fallback_tree:
            return fallback_tree[0]
    if not WALKFORWARD_DIR.exists():
        return None
    preferred = sorted(WALKFORWARD_DIR.glob("shap_summary_bar__return__*market_plus_all6_channels*.png"))
    if preferred:
        return preferred[0]
    fallback = sorted(WALKFORWARD_DIR.glob("shap_summary_bar__*.png"))
    return fallback[0] if fallback else None


def _baseline_table() -> List[str]:
    lines = ["| Model | Target | RMSE | MAE | Notes |", "|---|---:|---:|---:|---|"]

    arimax_return = _read_csv_rows(RESULTS_DIR / "arimax_return_summary_metrics.csv")
    for row in arimax_return:
        lines.append(
            f"| ARIMA baseline | {row.get('target_col', '-') } | {_fmt_num(row.get('reg_RMSE'))} | {_fmt_num(row.get('reg_MAE'))} | order={row.get('order', '-')} |"
        )

    arimax_price = _read_csv_rows(RESULTS_DIR / "arimax_price_summary_metrics.csv")
    for row in arimax_price:
        if row.get("task") == "regression":
            lines.append(
                f"| ARIMAX price-only | {row.get('target_col', '-')} | {_fmt_num(row.get('reg_RMSE'))} | {_fmt_num(row.get('reg_MAE'))} | order={row.get('order', '-')} |"
            )
        else:
            lines.append(
                f"| ARIMAX jump baseline | {row.get('target_col', '-')} | - | - | acc={_fmt_num(row.get('jump_Accuracy'))}, F1={_fmt_num(row.get('jump_F1'))} |"
            )

    xgb = _read_csv_rows(RESULTS_DIR / "xgboost_metrics_summary.csv")
    for row in xgb:
        lines.append(
            f"| XGBoost single-split | {row.get('scenario', '-') } | {_fmt_num(row.get('RMSE'))} | {_fmt_num(row.get('MAE'))} | used_gpu={row.get('used_gpu', '-')} |"
        )

    walkforward = _read_csv_rows(WALKFORWARD_DIR / "metrics_summary.csv")
    for row in walkforward[:12]:
        if row.get("task_type") == "regression":
            score = f"RMSE={_fmt_num(row.get('RMSE'))}, MAE={_fmt_num(row.get('MAE'))}"
        else:
            score = f"PR_AUC={_fmt_num(row.get('PR_AUC'))}, F1={_fmt_num(row.get('F1'))}"
        lines.append(
            f"| XGBoost walk-forward | {row.get('task', '-') } / {row.get('scenario', '-') } | {_fmt_num(row.get('RMSE'))} | {_fmt_num(row.get('MAE'))} | {score} |"
        )

    tree_bench = _read_csv_rows(TREE_BENCH_DIR / "metrics_summary.csv")
    for row in tree_bench[:18]:
        if row.get("task_type") == "regression":
            score = f"RMSE={_fmt_num(row.get('RMSE'))}, MAE={_fmt_num(row.get('MAE'))}"
        else:
            score = f"PR_AUC={_fmt_num(row.get('PR_AUC'))}, F1={_fmt_num(row.get('F1'))}"
        lines.append(
            f"| Tree walk-forward | {row.get('model', '-') } / {row.get('task', '-') } / {row.get('scenario', '-') } | {_fmt_num(row.get('RMSE'))} | {_fmt_num(row.get('MAE'))} | {score} |"
        )

    return lines


def _best_model_rows(section_title: str, model_label: str, summary_path: Path, run_hint: str) -> List[str]:
    rows = _read_csv_rows(summary_path)
    if not rows:
        return [
            section_title,
            "",
            f"Run the full {model_label} benchmark first:",
            "",
            f"`{run_hint}`",
        ]

    by_task: Dict[str, Dict[str, str]] = {}
    task_rows: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        task_rows.setdefault(row.get("task", "-"), []).append(row)

    def _safe_float(row: Dict[str, str], key: str, default: float) -> float:
        raw = row.get(key, "")
        try:
            return float(raw)
        except Exception:
            return default

    for task, group in task_rows.items():
        if task == "jump":
            best = max(group, key=lambda r: _safe_float(r, "PR_AUC", float("-inf")))
        else:
            best = min(group, key=lambda r: _safe_float(r, "RMSE", float("inf")))
        by_task[task] = best

    lines = [
        section_title,
        "",
        f"Best-performing {model_label} scenario per task from `{summary_path.relative_to(REPO_ROOT)}`.",
        "",
        "| Task | Best Scenario | Main Metric | Secondary Metric |",
        "|---|---|---:|---:|",
    ]

    order = ["return", "volatility", "jump"]
    for task in order:
        row = by_task.get(task)
        if not row:
            continue
        if task == "jump":
            main_metric = f"PR_AUC={_fmt_num(row.get('PR_AUC'))}"
            secondary = f"F1={_fmt_num(row.get('F1'))}"
        else:
            main_metric = f"RMSE={_fmt_num(row.get('RMSE'))}"
            secondary = f"MAE={_fmt_num(row.get('MAE'))}"
        lines.append(
            f"| {task} | `{row.get('scenario', '-')}` | {main_metric} | {secondary} |"
        )

    return lines


def _example_day_section() -> List[str]:
    raw_rows = _read_csv_rows(ARTIFACTS_DIR / "sample_inputs" / "raw_headlines_2026-01-26.csv")
    rel_rows = _read_csv_rows(ARTIFACTS_DIR / "sample_inputs" / "relevance_scores_2026-01-26_nli_probs.csv")
    label_rows = _read_csv_rows(ARTIFACTS_DIR / "sample_inputs" / "channel_labels_2026-01-26.csv")
    aux_rows = _read_csv_rows(ARTIFACTS_DIR / "derived_features" / "aux_2026_daily_channel_index.csv")

    day_row = next((row for row in aux_rows if row.get("date") == SAMPLE_DAY), None)

    lines = [f"## Example Day: {SAMPLE_DAY}", "", "### Headline Samples", ""]
    for row in raw_rows[:3]:
        lines.append(f"- `{row.get('Date', '-')}`: {_md_cell(row.get('Title', '-'))}")

    lines += ["", "### Relevance / Channel Mapping", "", "| Title | prob_relevant | strongest channel |", "|---|---:|---|"]
    label_lookup = {
        (row.get("URL", ""), row.get("Title", "")): row.get("Channels", "-")
        for row in label_rows
    }
    for row in rel_rows[:5]:
        key = (row.get("\ufeffURL", row.get("URL", "")), row.get("Title", ""))
        lines.append(
            f"| {_md_cell(row.get('Title', '-')[:90])} | {_fmt_num(row.get('prob_relevant'))} | {_md_cell(label_lookup.get(key, '-'))} |"
        )

    lines += ["", "### Daily Index Snapshot", ""]
    if day_row:
        lines += [
            "| Channel | z_day | low_evidence |",
            "|---|---:|---:|",
            f"| Supply Shock and availability risk | {_fmt_num(day_row.get('z_Supply Shock and availability risk'))} | {_fmt_num(day_row.get('low_evidence_Supply Shock and availability risk'))} |",
            f"| Transport logistics and chokepoints | {_fmt_num(day_row.get('z_Transport logistics and chokepoints'))} | {_fmt_num(day_row.get('low_evidence_Transport logistics and chokepoints'))} |",
            f"| Demand and macro economy | {_fmt_num(day_row.get('z_Demand and macro economy'))} | {_fmt_num(day_row.get('low_evidence_Demand and macro economy'))} |",
            f"| Inventory SPR and refinery | {_fmt_num(day_row.get('z_Inventory SPR and refinery'))} | {_fmt_num(day_row.get('low_evidence_Inventory SPR and refinery'))} |",
            f"| OPEC producer policy | {_fmt_num(day_row.get('z_OPEC producer policy'))} | {_fmt_num(day_row.get('low_evidence_OPEC producer policy'))} |",
            f"| Geopolitical Normalisation and Peace | {_fmt_num(day_row.get('z_Geopolitical Normalisation and Peace'))} | {_fmt_num(day_row.get('low_evidence_Geopolitical Normalisation and Peace'))} |",
        ]
    else:
        lines.append("No aux daily-index row found for the sample day.")

    return lines


def main() -> None:
    shap_figure = _pick_shap_figure()
    comparison_png = DOC_ASSET_DIR / "final_comparison_table.png"
    architecture_png = DOC_ASSET_DIR / "architecture_diagram.png"
    example_png = DOC_ASSET_DIR / "example_day_2026-01-26.png"
    shap_manifest = DOC_ASSET_DIR / "shap_manifest.csv"

    lines: List[str] = [
        "# Final Results Pack",
        "",
        "This note collects the current pipeline story plus the outputs needed for the final review.",
        "",
        "## Architecture",
        "",
        "```mermaid",
        "flowchart TD",
        "    A[Raw GDELT headlines] --> B[Zero-shot relevance scoring]",
        "    B --> C[Six channel probabilities]",
        "    A --> D[CrudeBert sentiment scoring]",
        "    C --> E[Daily channel aggregation]",
        "    D --> E",
        "    E --> F[Raw / imputed / decayed channel features]",
        "    G[OHLC crude price data] --> H[Return volatility and jump targets]",
        "    F --> I[Walk-forward tree-model benchmarks]",
        "    H --> I",
        "    I --> J[Metrics SHAP and ablation summaries]",
        "```",
        "",
        "## Model Comparison Table",
        "",
    ]

    lines.extend(_baseline_table())

    lines += [""] + _best_model_rows(
        "## Best XGBoost Rows",
        "XGBoost",
        XGBOOST_FULL_DIR / "metrics_summary.csv",
        "python scripts/05_modeling/boosting/tree_walkforward_benchmark.py --models xgboost --tasks return volatility jump --scenario-set full --disable-shap --outdir artifacts/results/tree_model_walkforward_xgboost_full",
    )

    lines += [""] + _best_model_rows(
        "## Best LightGBM Rows",
        "LightGBM",
        LIGHTGBM_FULL_DIR / "metrics_summary.csv",
        "python scripts/05_modeling/boosting/tree_walkforward_benchmark.py --models lightgbm --tasks return volatility jump --scenario-set full --disable-shap --outdir artifacts/results/tree_model_walkforward_lightgbm_full",
    )

    lines += [""] + _best_model_rows(
        "## Best CatBoost Rows",
        "CatBoost",
        CATBOOST_FULL_DIR / "metrics_summary.csv",
        "python scripts/05_modeling/boosting/tree_walkforward_benchmark.py --models catboost --tasks return volatility jump --scenario-set full --disable-shap --outdir artifacts/results/tree_model_walkforward_catboost_full",
    )

    lines += ["", "## SHAP Figure", ""]
    if shap_figure:
        rel = shap_figure.relative_to(REPO_ROOT)
        lines.append(f"Latest bar-summary figure: `{rel}`")
    else:
        lines.append(
            "Run `python scripts/05_modeling/boosting/tree_walkforward_benchmark.py` first. "
            "Expected SHAP output will be written under `artifacts/results/tree_model_walkforward/`."
        )

    lines += [
        "",
        "## Deck Assets",
        "",
        "- Comparison table PNG: `docs/assets/final_results/final_comparison_table.png`" if comparison_png.exists() else "- Comparison table PNG: not generated yet",
        "- Architecture diagram PNG: `docs/assets/final_results/architecture_diagram.png`" if architecture_png.exists() else "- Architecture diagram PNG: not generated yet",
        "- Example day PNG: `docs/assets/final_results/example_day_2026-01-26.png`" if example_png.exists() else "- Example day PNG: not generated yet",
        "- SHAP manifest CSV: `docs/assets/final_results/shap_manifest.csv`" if shap_manifest.exists() else "- SHAP manifest CSV: not generated yet",
    ]

    lines += ["", "## Ablation Coverage", "", "The benchmark runner defines these experiment families:", "", "- `market_only`", "- `market_plus_global_sentiment`", "- `market_plus_top3_channels`", "- `market_plus_all6_channels`", "- six single-channel ablations", "- channel variants: `raw`, `raw_decay`, `imputed_locf5`, `imputed_locf5_decay`", "- model families: `xgboost`, `lightgbm`, `catboost`", ""]

    lines.extend(_example_day_section())

    lines += [
        "",
        "## How To Regenerate",
        "",
        "```bash",
        "python3 -m venv .venv-bench",
        ".venv-bench/bin/pip install -r requirements-benchmark.txt",
        "python scripts/05_modeling/boosting/tree_walkforward_benchmark.py --scenario-set full",
        "python scripts/05_modeling/xgboost/xgb_walkforward_multitask.py --scenario-set full",
        "python scripts/06_reporting/build_final_results_report.py",
        "```",
        "",
        "## Notes",
        "",
        "- The walk-forward runner uses repo artifacts rather than the old external XLSX paths.",
        "- News features are lagged by one day before merging into market targets.",
        "- Jump experiments are handled as binary classification with probability outputs and PR-AUC/F1 reporting.",
        "- The multi-model benchmark uses the same folds and scenarios for XGBoost, LightGBM, and CatBoost.",
    ]

    DOCS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {DOCS_PATH}")


if __name__ == "__main__":
    main()
