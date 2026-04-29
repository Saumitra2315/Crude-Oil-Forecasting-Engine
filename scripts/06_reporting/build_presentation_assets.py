from __future__ import annotations

import csv
import shutil
import textwrap
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
RESULTS_DIR = ARTIFACTS_DIR / "results"
DOC_ASSET_DIR = REPO_ROOT / "docs" / "assets" / "final_results"
BEST_SHAP_DIR = RESULTS_DIR / "tree_model_walkforward_best_shap"
MPLCONFIG_DIR = REPO_ROOT / ".cache" / "matplotlib"

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SUMMARY_PATHS = {
    "XGBoost": RESULTS_DIR / "tree_model_walkforward_xgboost_full" / "metrics_summary.csv",
    "LightGBM": RESULTS_DIR / "tree_model_walkforward_lightgbm_full" / "metrics_summary.csv",
    "CatBoost": RESULTS_DIR / "tree_model_walkforward_catboost_full" / "metrics_summary.csv",
}

SAMPLE_DAY = "2026-01-26"
RAW_HEADLINES = ARTIFACTS_DIR / "sample_inputs" / "raw_headlines_2026-01-26.csv"
REL_SCORES = ARTIFACTS_DIR / "sample_inputs" / "relevance_scores_2026-01-26_nli_probs.csv"
CHANNEL_LABELS = ARTIFACTS_DIR / "sample_inputs" / "channel_labels_2026-01-26.csv"
AUX_DAILY = ARTIFACTS_DIR / "derived_features" / "aux_2026_daily_channel_index.csv"

BG = "#f7f1e5"
INK = "#1f2933"
MUTED = "#5b6b79"
ACCENT = "#b45309"
ACCENT_2 = "#0f766e"
ACCENT_3 = "#1d4ed8"
GRID = "#d8cdbd"
PANEL = "#fffaf2"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(raw: Optional[str], default: float) -> float:
    try:
        return float(raw) if raw not in (None, "", "nan", "NaN") else default
    except Exception:
        return default


def _best_rows_for_model(model_name: str, path: Path) -> List[Dict[str, str]]:
    rows = _read_rows(path)
    if not rows:
        return []

    out: List[Dict[str, str]] = []
    for task in ["return", "volatility", "jump"]:
        group = [r for r in rows if r.get("task") == task]
        if not group:
            continue
        if task == "jump":
            best = max(group, key=lambda r: _safe_float(r.get("PR_AUC"), float("-inf")))
            metric = f"PR_AUC={_safe_float(best.get('PR_AUC'), np.nan):.6f}"
            secondary = f"F1={_safe_float(best.get('F1'), np.nan):.6f}"
            score = _safe_float(best.get("PR_AUC"), float("-inf"))
        else:
            best = min(group, key=lambda r: _safe_float(r.get("RMSE"), float("inf")))
            metric = f"RMSE={_safe_float(best.get('RMSE'), np.nan):.6f}"
            secondary = f"MAE={_safe_float(best.get('MAE'), np.nan):.6f}"
            score = _safe_float(best.get("RMSE"), float("inf"))

        out.append(
            {
                "model": model_name,
                "task": task,
                "scenario": best.get("scenario", "-"),
                "metric": metric,
                "secondary_metric": secondary,
                "_score": score,
            }
        )
    return out


def _comparison_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    best_rows: List[Dict[str, str]] = []
    for model_name, path in SUMMARY_PATHS.items():
        best_rows.extend(_best_rows_for_model(model_name, path))

    by_model_task = pd.DataFrame(best_rows)

    overall_rows: List[Dict[str, str]] = []
    for task in ["return", "volatility", "jump"]:
        group = by_model_task[by_model_task["task"] == task].copy()
        if group.empty:
            continue
        ascending = task != "jump"
        group = group.sort_values("_score", ascending=ascending).reset_index(drop=True)
        winner = group.iloc[0]
        runner_up = group.iloc[1] if len(group) > 1 else None
        overall_rows.append(
            {
                "task": task,
                "winner_model": winner["model"],
                "winner_scenario": winner["scenario"],
                "winner_metric": winner["metric"],
                "winner_secondary_metric": winner["secondary_metric"],
                "runner_up_model": runner_up["model"] if runner_up is not None else "-",
                "runner_up_metric": runner_up["metric"] if runner_up is not None else "-",
            }
        )

    return by_model_task, pd.DataFrame(overall_rows)


def _save_table_assets(by_model_task: pd.DataFrame, overall: pd.DataFrame) -> None:
    by_model_task = by_model_task.drop(columns=["_score"])
    by_model_task.to_csv(DOC_ASSET_DIR / "best_per_model_task.csv", index=False)
    overall.to_csv(DOC_ASSET_DIR / "final_comparison_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(14, 5.6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    ax.text(0.02, 0.95, "Final Results Model Winners", fontsize=22, fontweight="bold", color=INK, transform=ax.transAxes)
    ax.text(
        0.02,
        0.90,
        "Best-performing model/scenario per target from the full walk-forward benchmark.",
        fontsize=11,
        color=MUTED,
        transform=ax.transAxes,
    )

    display = overall.rename(
        columns={
            "task": "Target",
            "winner_model": "Winner",
            "winner_scenario": "Best Scenario",
            "winner_metric": "Main Metric",
            "winner_secondary_metric": "Secondary Metric",
            "runner_up_model": "Runner-Up",
            "runner_up_metric": "Runner-Up Metric",
        }
    )
    for col in display.columns:
        display[col] = display[col].astype(str).str.replace("_", " ")

    table = ax.table(
        cellText=display.values.tolist(),
        colLabels=display.columns.tolist(),
        loc="center",
        cellLoc="left",
        colLoc="left",
        bbox=[0.02, 0.08, 0.96, 0.72],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_facecolor("#ead7b7")
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_facecolor(PANEL if row % 2 else "#f4ead8")
            if col == 1:
                cell.set_text_props(weight="bold", color=ACCENT_3)
            else:
                cell.set_text_props(color=INK)

    plt.tight_layout()
    plt.savefig(DOC_ASSET_DIR / "final_comparison_table.png", dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _box(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str, fc: str, ec: str = "none", color: str = INK) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        edgecolor=ec or fc,
        facecolor=fc,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12, color=color, wrap=True, transform=ax.transAxes)


def _arrow(ax: plt.Axes, start: Tuple[float, float], end: Tuple[float, float], color: str = INK) -> None:
    patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, linewidth=2, color=color, transform=ax.transAxes)
    ax.add_patch(patch)


def _save_architecture_diagram() -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    ax.text(0.04, 0.95, "Crude Oil Forecasting Engine", fontsize=24, fontweight="bold", color=INK, transform=ax.transAxes)
    ax.text(0.04, 0.91, "Final results architecture view: from raw headlines to walk-forward model evaluation.", fontsize=12, color=MUTED, transform=ax.transAxes)

    _box(ax, 0.05, 0.70, 0.22, 0.13, "GDELT Headlines\nraw event stream", "#f7d9a8")
    _box(ax, 0.37, 0.78, 0.24, 0.11, "Zero-shot relevance\nand 6-channel routing", "#c8e6dc")
    _box(ax, 0.37, 0.61, 0.24, 0.11, "CrudeBert\nsentiment scoring", "#c9daf8")
    _box(ax, 0.70, 0.70, 0.22, 0.13, "Daily channel indices\nraw / imputed / decayed", "#f3e2c7")

    _box(ax, 0.05, 0.34, 0.22, 0.13, "OHLC market data\nreturns, vol, jump", "#d6dee8")
    _box(ax, 0.37, 0.34, 0.24, 0.13, "Lagged feature frame\nnews lagged by 1 day", "#fff0cf")
    _box(ax, 0.70, 0.34, 0.22, 0.13, "Walk-forward models\nXGBoost / LightGBM / CatBoost", "#d8eadf")
    _box(ax, 0.37, 0.08, 0.24, 0.13, "Metrics + SHAP + ablations\ninterpretability and model selection", "#dbe7ff")

    _arrow(ax, (0.27, 0.765), (0.37, 0.835), ACCENT)
    _arrow(ax, (0.27, 0.765), (0.37, 0.665), ACCENT)
    _arrow(ax, (0.61, 0.835), (0.70, 0.765), ACCENT_2)
    _arrow(ax, (0.61, 0.665), (0.70, 0.765), ACCENT_3)
    _arrow(ax, (0.27, 0.405), (0.37, 0.405), INK)
    _arrow(ax, (0.81, 0.70), (0.81, 0.47), INK)
    _arrow(ax, (0.61, 0.405), (0.70, 0.405), INK)
    _arrow(ax, (0.49, 0.70), (0.49, 0.47), INK)
    _arrow(ax, (0.81, 0.34), (0.49, 0.21), ACCENT_2)

    ax.text(0.05, 0.23, "Core research question", fontsize=12, fontweight="bold", color=ACCENT, transform=ax.transAxes)
    ax.text(
        0.05,
        0.17,
        "Do interpretable news channels add forecasting value beyond lagged market features,\n"
        "and which channels matter for return, volatility, and jump behaviour?",
        fontsize=11,
        color=INK,
        transform=ax.transAxes,
    )

    plt.tight_layout()
    plt.savefig(DOC_ASSET_DIR / "architecture_diagram.png", dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False, break_on_hyphens=False))


def _example_rows() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    raw = pd.read_csv(RAW_HEADLINES)
    rel = pd.read_csv(REL_SCORES)
    lbl = pd.read_csv(CHANNEL_LABELS)
    aux = pd.read_csv(AUX_DAILY)

    raw.columns = [str(c).replace("\ufeff", "").strip() for c in raw.columns]
    rel.columns = [str(c).replace("\ufeff", "").strip() for c in rel.columns]
    lbl.columns = [str(c).replace("\ufeff", "").strip() for c in lbl.columns]
    aux.columns = [str(c).replace("\ufeff", "").strip() for c in aux.columns]

    rel = rel.sort_values("prob_relevant", ascending=False).head(5).copy()
    lbl_map = {(r["URL"], r["Title"]): r["Channels"] for _, r in lbl.iterrows()}
    rel["Channels"] = [lbl_map.get((row["URL"], row["Title"]), "-") for _, row in rel.iterrows()]

    day_row = aux.loc[aux["date"] == SAMPLE_DAY].iloc[0]
    return raw.head(3).copy(), rel, day_row


def _save_example_day_slide() -> None:
    raw_top, rel_top, day_row = _example_rows()
    fig = plt.figure(figsize=(15, 9), facecolor=BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.28, wspace=0.18)

    ax_head = fig.add_subplot(gs[0, 0])
    ax_rel = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, :])

    for ax in [ax_head, ax_rel]:
        ax.set_facecolor(PANEL)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax_head.text(0.02, 0.93, f"Worked Example: {SAMPLE_DAY}", fontsize=22, fontweight="bold", color=INK, transform=ax_head.transAxes)
    ax_head.text(0.02, 0.85, "Step 1: Raw headlines entering the pipeline", fontsize=12, color=MUTED, transform=ax_head.transAxes)

    y = 0.72
    for idx, (_, row) in enumerate(raw_top.iterrows(), start=1):
        box = FancyBboxPatch((0.02, y - 0.16), 0.94, 0.14, boxstyle="round,pad=0.012,rounding_size=0.02", facecolor="#fff4df", edgecolor=GRID, transform=ax_head.transAxes)
        ax_head.add_patch(box)
        ax_head.text(0.04, y - 0.04, f"{idx}. {_wrap(row['Title'], 54)}", fontsize=11, color=INK, va="top", transform=ax_head.transAxes)
        y -= 0.19

    ax_rel.text(0.03, 0.93, "Step 2: Relevance + strongest channel", fontsize=12, color=MUTED, transform=ax_rel.transAxes)

    table_rows = []
    for _, row in rel_top.head(5).iterrows():
        table_rows.append(
            [
                _wrap(row["Title"], 36),
                f"{float(row['prob_relevant']):.3f}",
                _wrap(str(row["Channels"]).replace("Normalisation and relief", "Normalisation / relief"), 26),
            ]
        )

    table = ax_rel.table(
        cellText=table_rows,
        colLabels=["Headline", "Rel.", "Assigned Channel"],
        cellLoc="left",
        colLoc="left",
        bbox=[0.03, 0.08, 0.94, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_facecolor("#ead7b7")
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_facecolor(PANEL if row % 2 else "#f4ead8")
            cell.set_text_props(color=INK)

    channels = [
        "Supply Shock and availability risk",
        "Transport logistics and chokepoints",
        "Demand and macro economy",
        "Inventory SPR and refinery",
        "OPEC producer policy",
        "Geopolitical Normalisation and Peace",
    ]
    z_values = [float(day_row[f"z_{c}"]) if pd.notna(day_row[f"z_{c}"]) else 0.0 for c in channels]
    colors = [ACCENT if v >= 0 else ACCENT_3 for v in z_values]

    ax_bar.set_facecolor(PANEL)
    ax_bar.bar(range(len(channels)), z_values, color=colors, alpha=0.9)
    ax_bar.axhline(0.0, color=GRID, linewidth=1.2)
    ax_bar.set_xticks(range(len(channels)))
    ax_bar.set_xticklabels(
        [
            "Supply",
            "Transport",
            "Demand",
            "Inventory",
            "OPEC",
            "Geo-Relief",
        ],
        fontsize=10,
    )
    ax_bar.set_ylabel("Daily z-index", color=INK)
    ax_bar.set_title("Step 3: Aggregated daily channel signal", fontsize=13, color=INK, loc="left")
    ax_bar.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax_bar.tick_params(colors=INK)
    for spine in ax_bar.spines.values():
        spine.set_color(GRID)

    ax_bar.text(
        0.01,
        1.04,
        "This turns many headline-level scores into one interpretable daily news state used by the forecasting models.",
        fontsize=10.5,
        color=MUTED,
        transform=ax_bar.transAxes,
    )

    plt.savefig(DOC_ASSET_DIR / "example_day_2026-01-26.png", dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _copy_shap_assets() -> None:
    target_dir = DOC_ASSET_DIR / "shap"
    _ensure_dir(target_dir)
    manifest_rows: List[Dict[str, str]] = []

    for run_dir in sorted(BEST_SHAP_DIR.glob("*")):
        if not run_dir.is_dir():
            continue

        row = {"run_dir": str(run_dir.relative_to(REPO_ROOT))}
        for pattern, key in [
            ("shap_summary_bar__*.png", "bar_png"),
            ("shap_summary_beeswarm__*.png", "beeswarm_png"),
            ("shap_global_importance__*.csv", "global_csv"),
            ("shap_local__*.csv", "local_csv"),
        ]:
            matches = sorted(run_dir.glob(pattern))
            if not matches:
                row[key] = ""
                continue
            src = matches[0]
            dst = target_dir / src.name
            shutil.copy2(src, dst)
            row[key] = str(dst.relative_to(REPO_ROOT))
        manifest_rows.append(row)

    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(DOC_ASSET_DIR / "shap_manifest.csv", index=False)


def main() -> None:
    _ensure_dir(DOC_ASSET_DIR)
    by_model_task, overall = _comparison_tables()
    _save_table_assets(by_model_task, overall)
    _save_architecture_diagram()
    _save_example_day_slide()
    _copy_shap_assets()
    print(f"[OK] Wrote presentation assets to {DOC_ASSET_DIR}")


if __name__ == "__main__":
    main()
