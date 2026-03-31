from __future__ import annotations

import os
import json
from dataclasses import dataclass
from datetime import datetime, timezone 
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Optional: diagnostics
try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    _HAS_DIAG = True
except Exception:
    _HAS_DIAG = False

# Optional: interactive plotting
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

# Optional: classification metrics for jump-risk
try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        average_precision_score, confusion_matrix
    )
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


# ============================================================
# ✅✅✅ EDIT ONLY THIS BLOCK ✅✅✅
# ============================================================

TRAIN_PATH = r"D:\ACADS\4-2\FIN SOP\Data\Code\Models\targets_out\train_with_targets.csv"
TEST_PATH  = r"D:\ACADS\4-2\FIN SOP\Data\Code\Models\targets_out\test_with_targets.csv"

OUTDIR = r"D:\ACADS\4-2\FIN SOP\Data\Code\Models\ARIMAX\arimax_outputs"

DATE_COL = "Date"  # present in your files

# Your pre-built targets (already in files)
TARGETS = [
    # (run_name, target_col, task_type)
    ("price_only_logret",  "target_return",         "regression"),
    ("price_only_parvol",  "target_vol_parkinson",  "regression"),
    ("price_only_jump",    "target_jump_flag",      "jump"),
]

# Price-only baseline: keep EXOG_COLS empty.
# Later you can add sentiment/channel columns here, e.g. ["sentiment"] or 6 channel cols.
EXOG_COLS: List[str] = []

# If you add exog later, set this True to use t-1 exog for predicting target at t (avoids leakage).
SHIFT_EXOG_BY_1 = True

# ARIMA/ARIMAX model selection
AUTO_ORDER = True     # grid search by AIC on TRAIN only
P_MAX = 3
Q_MAX = 3
D_VALUES = [0, 1]     # differencing candidates
TREND = "c"           # "c" constant, "n" none

# Jump-risk classification conversion threshold (from predicted score to {0,1})
JUMP_SCORE_THRESHOLD = 0.5

# ============================================================
# (Do not edit below unless changing logic)
# ============================================================

@dataclass
class FitConfig:
    order: Tuple[int, int, int]
    trend: str = "c"
    enforce_stationarity: bool = False
    enforce_invertibility: bool = False


def _load_table(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".csv"):
        return pd.read_csv(path)
    if p.endswith(".xlsx") or p.endswith(".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path} (use .csv/.xlsx/.xls)")


def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce", dayfirst=False)
    out = out.sort_values(DATE_COL).reset_index(drop=True)
    return out


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _safe_shift_exog(df: pd.DataFrame, exog_cols: List[str]) -> pd.DataFrame:
    if not exog_cols or not SHIFT_EXOG_BY_1:
        return df
    out = df.copy()
    out[exog_cols] = out[exog_cols].shift(1)
    return out


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred

    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))

    # MAPE: skip actual == 0
    denom = np.abs(y_true)
    nz = denom > 0
    if np.any(nz):
        mape = float(np.mean(np.abs(err[nz]) / denom[nz]) * 100.0)
        mape_n = int(np.sum(nz))
        mape_skipped = int(np.sum(~nz))
    else:
        mape = float("nan")
        mape_n = 0
        mape_skipped = int(len(y_true))

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE_percent": mape,
        "MAPE_n": mape_n,
        "MAPE_skipped_zero_actual": mape_skipped,
    }


def _jump_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, Any]:
    if not _HAS_SKLEARN:
        raise RuntimeError("Install scikit-learn for jump metrics: pip install scikit-learn")

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    # ARIMA output isn’t a probability; we treat it as a score and clip for PR metrics.
    y_score_clip = np.clip(y_score, 0.0, 1.0)
    y_pred = (y_score_clip >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    pr_auc = float(average_precision_score(y_true, y_score_clip))
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "threshold": float(threshold),
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "PR_AUC": pr_auc,
        "confusion_matrix_rows_true_0_1_cols_pred_0_1": cm,
        "positive_rate": float(np.mean(y_true)),
    }


def _select_order_by_aic(
    y_train: np.ndarray,
    exog_train: Optional[np.ndarray],
    p_max: int,
    d_values: List[int],
    q_max: int,
    trend: str,
) -> Tuple[int, int, int]:
    best_order = None
    best_aic = float("inf")

    for d in d_values:
        for p in range(p_max + 1):
            for q in range(q_max + 1):
                try:
                    model = SARIMAX(
                        y_train,
                        exog=exog_train,
                        order=(p, d, q),
                        trend=trend,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    res = model.fit(disp=False)
                    if not getattr(res, "mle_retvals", {}).get("converged", True):
                        continue
                    if np.isfinite(res.aic) and res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p, d, q)
                except Exception:
                    continue

    if best_order is None:
        raise RuntimeError("Order selection failed for all candidates. Try smaller P_MAX/Q_MAX or check data.")
    return best_order


def _fit_sarimax(y_train: np.ndarray, exog_train: Optional[np.ndarray], cfg: FitConfig):
    model = SARIMAX(
        y_train,
        exog=exog_train,
        order=cfg.order,
        trend=cfg.trend,
        enforce_stationarity=cfg.enforce_stationarity,
        enforce_invertibility=cfg.enforce_invertibility,
    )
    return model.fit(disp=False, method="lbfgs", maxiter=500)


def _walk_forward_predict(res, y_test: np.ndarray, exog_test: Optional[np.ndarray]) -> np.ndarray:
    """
    1-step-ahead walk-forward forecasts through the test set.
    Parameters are fixed; state is updated with actual y_t.
    """
    preds = []
    cur = res

    for i in range(len(y_test)):
        x_t = None
        if exog_test is not None:
            x_t = np.asarray(exog_test[i:i+1], dtype=float)

        yhat = cur.forecast(steps=1, exog=x_t)
        preds.append(float(yhat[0]))

        # extend state using realized observation
        cur = cur.extend(endog=np.asarray([y_test[i]], dtype=float), exog=x_t)

    return np.asarray(preds, dtype=float)


def _make_plot(df: pd.DataFrame, run_name: str, target_col: str, exog_cols: List[str], out_html: str) -> None:
    if not _HAS_PLOTLY:
        return

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x = df[DATE_COL] if DATE_COL in df.columns else np.arange(len(df))

    fig.add_trace(go.Scatter(x=x, y=df[target_col], name=f"Actual: {target_col}", mode="lines"), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=df["y_pred"], name=f"Predicted: {target_col}", mode="lines"), secondary_y=False)

    for c in exog_cols:
        if c in df.columns:
            fig.add_trace(go.Scatter(x=x, y=df[c], name=f"Exog: {c}", mode="lines"), secondary_y=True)

    fig.update_layout(
        title=f"{run_name} — {target_col}",
        hovermode="x unified",
        legend=dict(orientation="h"),
        margin=dict(l=50, r=50, t=60, b=40),
    )
    fig.update_yaxes(title_text="Target / Prediction", secondary_y=False)
    fig.update_yaxes(title_text="Inputs (exog)", secondary_y=True)

    # Dropdown views
    exog_n = len([c for c in exog_cols if c in df.columns])
    buttons = [
        dict(label="Actual+Pred", method="update",
             args=[{"visible": [True, True] + [False]*exog_n}]),
        dict(label="Inputs only", method="update",
             args=[{"visible": [False, False] + [True]*exog_n}]),
        dict(label="Show all", method="update",
             args=[{"visible": [True, True] + [True]*exog_n}]),
    ]
    fig.update_layout(updatemenus=[dict(type="dropdown", x=1.0, y=1.15, showactive=True, buttons=buttons)])

    fig.write_html(out_html)


def _write_md_report(path: str, report: Dict[str, Any]) -> None:
    def fmt(v):
        if isinstance(v, float):
            if np.isnan(v):
                return "NaN"
            return f"{v:.6g}"
        return str(v)

    lines = []
    lines.append(f"# ARIMAX Report — {report['run_name']}\n")
    lines.append(f"- **timestamp_utc**: {report['timestamp_utc']}")
    lines.append("")

    lines.append("## Data\n")
    for k, v in report["data"].items():
        lines.append(f"- **{k}**: {fmt(v)}")
    lines.append("")

    lines.append("## Model\n")
    for k, v in report["model"].items():
        if k != "params":
            lines.append(f"- **{k}**: {fmt(v)}")
    lines.append("")

    lines.append("### Parameters\n")
    lines.append("| param | value | std_err | p_value |")
    lines.append("|---|---:|---:|---:|")
    for r in report["model"]["params"]:
        lines.append(
            f"| {r.get('param','')} | {fmt(r.get('value', float('nan')))} | "
            f"{fmt(r.get('std_err', float('nan')))} | {fmt(r.get('p_value', float('nan')))} |"
        )
    lines.append("")

    lines.append("## Metrics\n")
    for section, vals in report["metrics"].items():
        lines.append(f"### {section}\n")
        for k, v in vals.items():
            lines.append(f"- **{k}**: {fmt(v)}")
        lines.append("")

    if "diagnostics" in report:
        lines.append("## Diagnostics\n")
        for k, v in report["diagnostics"].items():
            lines.append(f"- **{k}**: {fmt(v)}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _param_rows(res) -> List[Dict[str, Any]]:
    """
    Robust across statsmodels versions where res.params/bse/pvalues
    may be numpy arrays or pandas Series.
    """
    # get param names
    names = getattr(res, "param_names", None)
    if names is None:
        names = getattr(res.model, "param_names", None)
    if names is None:
        # last resort
        names = [f"param_{i}" for i in range(len(np.asarray(res.params)))]

    params = np.asarray(res.params, dtype=float)

    bse = getattr(res, "bse", None)
    pvals = getattr(res, "pvalues", None)

    # Convert bse and pvals to aligned arrays if possible
    def _to_aligned_array(x):
        if x is None:
            return None
        try:
            # pandas Series/DataFrame-like
            import pandas as pd
            if isinstance(x, (pd.Series, pd.Index)):
                return np.asarray(pd.Series(x).reindex(names), dtype=float)
        except Exception:
            pass
        # fallback: assume it's positional ndarray-like
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 0:
            return None
        if len(arr) != len(names):
            # can't align safely
            return None
        return arr

    bse_arr = _to_aligned_array(bse)
    pval_arr = _to_aligned_array(pvals)

    rows = []
    for i, nm in enumerate(names):
        row = {"param": str(nm), "value": float(params[i]) if i < len(params) else float("nan")}
        if bse_arr is not None and i < len(bse_arr) and np.isfinite(bse_arr[i]):
            row["std_err"] = float(bse_arr[i])
        if pval_arr is not None and i < len(pval_arr) and np.isfinite(pval_arr[i]):
            row["p_value"] = float(pval_arr[i])
        rows.append(row)

    return rows


def run_one(train_df: pd.DataFrame, test_df: pd.DataFrame, run_name: str, target_col: str, task: str) -> Dict[str, Any]:
    # Shift exog (if any) after sorting, before dropping NaNs
    train_use = _safe_shift_exog(train_df, EXOG_COLS)
    test_use = _safe_shift_exog(test_df, EXOG_COLS)

    needed_cols = [DATE_COL, target_col] + EXOG_COLS
    for c in needed_cols:
        if c not in train_use.columns:
            raise ValueError(f"[{run_name}] Missing column in TRAIN: {c}")
        if c not in test_use.columns:
            raise ValueError(f"[{run_name}] Missing column in TEST: {c}")

    # Coerce numeric for target+exog
    train_use = _coerce_numeric(train_use, [target_col] + EXOG_COLS)
    test_use = _coerce_numeric(test_use, [target_col] + EXOG_COLS)

    # Drop rows with NaN in required columns (this is the key "no issues with blanks" rule)
    train_before = len(train_use)
    test_before = len(test_use)
    train_use = train_use.dropna(subset=[target_col] + EXOG_COLS).reset_index(drop=True)
    test_use = test_use.dropna(subset=[target_col] + EXOG_COLS).reset_index(drop=True)

    if len(train_use) < 50:
        raise RuntimeError(f"[{run_name}] Too few train rows after NaN drop: {len(train_use)}")
    if len(test_use) < 10:
        raise RuntimeError(f"[{run_name}] Too few test rows after NaN drop: {len(test_use)}")

    y_train = train_use[target_col].astype(float).to_numpy()
    y_test = test_use[target_col].astype(float).to_numpy()

    exog_train = train_use[EXOG_COLS].astype(float).to_numpy() if EXOG_COLS else None
    exog_test = test_use[EXOG_COLS].astype(float).to_numpy() if EXOG_COLS else None

    # Order selection
    if AUTO_ORDER:
        order = _select_order_by_aic(y_train, exog_train, P_MAX, D_VALUES, Q_MAX, TREND)
    else:
        order = (1, 0, 1)

    cfg = FitConfig(order=order, trend=TREND)
    res = _fit_sarimax(y_train, exog_train, cfg)
    print("[INFO] Final model converged:", getattr(res, "mle_retvals", {}).get("converged", None))
    y_pred = _walk_forward_predict(res, y_test, exog_test)

    out = test_use[[DATE_COL, target_col] + EXOG_COLS].copy()
    out["y_pred"] = y_pred
    out["y_err"] = out[target_col].astype(float) - out["y_pred"].astype(float)

    # Metrics
    metrics: Dict[str, Any] = {}
    if task == "regression":
        metrics["regression"] = _regression_metrics(out[target_col].to_numpy(), out["y_pred"].to_numpy())
    elif task == "jump":
        # Ensure jump flag is {0,1}
        y_true = out[target_col].astype(int).to_numpy()
        metrics["jump"] = _jump_metrics(y_true, out["y_pred"].to_numpy(), JUMP_SCORE_THRESHOLD)
    else:
        raise ValueError(f"Unknown task: {task}")

    # Diagnostics (optional)
    diagnostics: Dict[str, Any] = {}
    if _HAS_DIAG and len(res.resid) > 20:
        try:
            lb = acorr_ljungbox(res.resid, lags=[10], return_df=True)
            diagnostics["ljung_box_pvalue_lag10"] = float(lb["lb_pvalue"].iloc[0])
        except Exception:
            pass

    # Report
    report = {
        "run_name": run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data": {
            "train_file": TRAIN_PATH,
            "test_file": TEST_PATH,
            "date_col": DATE_COL,
            "target_col": target_col,
            "exog_cols": EXOG_COLS,
            "shift_exog_by_1": bool(SHIFT_EXOG_BY_1),
            "train_rows_raw": int(train_before),
            "train_rows_used": int(len(train_use)),
            "test_rows_raw": int(test_before),
            "test_rows_used": int(len(test_use)),
            "train_date_min": str(train_use[DATE_COL].min()),
            "train_date_max": str(train_use[DATE_COL].max()),
            "test_date_min": str(test_use[DATE_COL].min()),
            "test_date_max": str(test_use[DATE_COL].max()),
        },
        "model": {
            "model_type": "SARIMAX (ARIMA/ARIMAX)",
            "order_p_d_q": order,
            "trend": TREND,
            "aic": float(res.aic),
            "bic": float(res.bic),
            "llf": float(res.llf),
            "nobs_train_used": int(res.nobs),
            "params": _param_rows(res),
        },
        "metrics": metrics,
        "diagnostics": diagnostics,
        "notes": [
            "Targets are taken directly from the provided target files (no recomputation).",
            "NaN/blank handling: rows with NaN in required columns are dropped separately within train/test.",
            "Predictions are 1-step-ahead walk-forward on test (fixed parameters; state updated with realized y_t).",
            "Jump metrics treat ARIMA prediction as a score (clipped to [0,1]) for PR-AUC and thresholding.",
        ],
    }

    # Save outputs
    os.makedirs(OUTDIR, exist_ok=True)
    base = os.path.join(OUTDIR, run_name)
    out.to_csv(f"{base}_predictions.csv", index=False)
    with open(f"{base}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    _write_md_report(f"{base}_report.md", report)

    # Plot
    if _HAS_PLOTLY:
        _make_plot(out, run_name, target_col, EXOG_COLS, f"{base}_plot.html")

    return report


def main():
    train_raw = _prep_df(_load_table(TRAIN_PATH))
    test_raw = _prep_df(_load_table(TEST_PATH))

    # Quick validation print (won't crash if extra cols exist)
    print("[INFO] Train columns:", list(train_raw.columns))
    print("[INFO] Test columns :", list(test_raw.columns))

    summary_rows = []
    for run_name, target_col, task in TARGETS:
        print(f"\n[RUN] {run_name} | target={target_col} | task={task}")
        rep = run_one(train_raw, test_raw, run_name, target_col, task)

        row = {
            "run_name": run_name,
            "target_col": target_col,
            "task": task,
            "order": str(rep["model"]["order_p_d_q"]),
            "aic": rep["model"]["aic"],
            "bic": rep["model"]["bic"],
            "test_rows_used": rep["data"]["test_rows_used"],
        }
        # Flatten main metrics into summary
        if task == "regression":
            m = rep["metrics"]["regression"]
            row.update({f"reg_{k}": v for k, v in m.items()})
        else:
            m = rep["metrics"]["jump"]
            row.update({f"jump_{k}": v for k, v in m.items() if k != "confusion_matrix_rows_true_0_1_cols_pred_0_1"})
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTDIR, "summary_metrics.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\n[OK] Saved summary: {summary_path}")
    print(f"[OK] Outputs folder: {OUTDIR}")


if __name__ == "__main__":
    main()
