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


# ============================================================
# ✅✅✅ EDIT ONLY THIS BLOCK ✅✅✅
# ============================================================

TRAIN_PATH = r"D:\ACADS\4-2\FIN SOP\Data\Code\Models\targets_out\train_return.csv"
TEST_PATH  = r"D:\ACADS\4-2\FIN SOP\Data\Code\Models\targets_out\test_return.csv"

OUTDIR = r"D:\ACADS\4-2\FIN SOP\Data\Code\Models\ARIMAX\arimax_return_outputs"

DATE_COL = "Date"

# ✅ Return only
RUN_NAME = "price_only_logret"
TARGET_COL = "Return"     # you said you computed this yourself already

# Price-only baseline now
EXOG_COLS: List[str] = []        # later: add sentiment/channel cols
SHIFT_EXOG_BY_1 = True           # later: use t-1 exog to predict return_t

# ARIMA/ARIMAX model selection
AUTO_ORDER = True
P_MAX = 3
Q_MAX = 3
D_VALUES = [0, 1]
TREND = "c"

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

    # keep raw for debugging
    raw = out[DATE_COL].astype(str).str.strip()

    # ✅ strict parse for your format: DD-MM-YYYY
    dt = pd.to_datetime(raw, format="%d-%m-%Y", errors="coerce")

    # fallback: try generic dayfirst parse (handles dd/mm/yyyy etc)
    if dt.isna().mean() > 0.01:
        dt2 = pd.to_datetime(raw, errors="coerce", dayfirst=True)
        dt = dt.fillna(dt2)

    out[DATE_COL] = dt

    bad = out[out[DATE_COL].isna()]
    if len(bad) > 0:
        print(f"[WARN] Unparsed dates count: {len(bad)}")
        print("[WARN] Examples of bad Date strings:")
        print(bad[DATE_COL].head(20).to_string(index=False))

    out = out.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
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

    # MAPE is not meaningful for returns, but we still compute safely
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
                    res = model.fit(disp=False, method="lbfgs", maxiter=500)

                    # ✅ skip non-converged fits
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
    preds = []
    cur = res
    for i in range(len(y_test)):
        x_t = None
        if exog_test is not None:
            x_t = np.asarray(exog_test[i:i+1], dtype=float)

        yhat = cur.forecast(steps=1, exog=x_t)
        preds.append(float(yhat[0]))
        cur = cur.extend(endog=np.asarray([y_test[i]], dtype=float), exog=x_t)

    return np.asarray(preds, dtype=float)


def _make_plot(df: pd.DataFrame, run_name: str, target_col: str, exog_cols: List[str], out_html: str) -> None:
    if not _HAS_PLOTLY:
        return

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x = df[DATE_COL] if DATE_COL in df.columns else np.arange(len(df))

    fig.add_trace(go.Scatter(x=x, y=df[target_col], name="Actual", mode="lines"), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=df["y_pred"], name="Predicted", mode="lines"), secondary_y=False)

    for c in exog_cols:
        if c in df.columns:
            fig.add_trace(go.Scatter(x=x, y=df[c], name=f"Exog: {c}", mode="lines"), secondary_y=True)

    fig.update_layout(
        title=f"{run_name} — {target_col}",
        hovermode="x unified",
        legend=dict(orientation="h"),
        margin=dict(l=50, r=50, t=60, b=40),
    )
    fig.update_yaxes(title_text="Return", secondary_y=False)
    fig.update_yaxes(title_text="Exog", secondary_y=True)

    fig.write_html(out_html)


def _param_rows(res) -> List[Dict[str, Any]]:
    names = getattr(res, "param_names", None)
    if names is None:
        names = getattr(res.model, "param_names", None)
    if names is None:
        names = [f"param_{i}" for i in range(len(np.asarray(res.params)))]

    params = np.asarray(res.params, dtype=float)
    bse = getattr(res, "bse", None)
    pvals = getattr(res, "pvalues", None)

    def _to_arr(x):
        if x is None:
            return None
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 0 or len(arr) != len(names):
            return None
        return arr

    bse_arr = _to_arr(bse)
    pval_arr = _to_arr(pvals)

    rows = []
    for i, nm in enumerate(names):
        row = {"param": str(nm), "value": float(params[i]) if i < len(params) else float("nan")}
        if bse_arr is not None:
            row["std_err"] = float(bse_arr[i])
        if pval_arr is not None:
            row["p_value"] = float(pval_arr[i])
        rows.append(row)
    return rows


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    train_raw = _prep_df(_load_table(TRAIN_PATH))
    test_raw  = _prep_df(_load_table(TEST_PATH))

    print("[INFO] Train rows (after dropping blank dates):", len(train_raw))
    print("[INFO] Test rows  (after dropping blank dates):", len(test_raw))

    # Shift exog if needed (currently none)
    train_use = _safe_shift_exog(train_raw, EXOG_COLS)
    test_use  = _safe_shift_exog(test_raw, EXOG_COLS)

    # Coerce numeric
    train_use = _coerce_numeric(train_use, [TARGET_COL] + EXOG_COLS)
    test_use  = _coerce_numeric(test_use, [TARGET_COL] + EXOG_COLS)

    # Drop NaNs for target (and exog if any)
    train_before = len(train_use)
    test_before = len(test_use)
    train_use = train_use.dropna(subset=[TARGET_COL] + EXOG_COLS).reset_index(drop=True)
    test_use  = test_use.dropna(subset=[TARGET_COL] + EXOG_COLS).reset_index(drop=True)

    print("[INFO] Train rows used (after NaN drop):", len(train_use), "from", train_before)
    print("[INFO] Test rows used  (after NaN drop):", len(test_use), "from", test_before)

    y_train = train_use[TARGET_COL].astype(float).to_numpy()
    y_test  = test_use[TARGET_COL].astype(float).to_numpy()

    exog_train = train_use[EXOG_COLS].astype(float).to_numpy() if EXOG_COLS else None
    exog_test  = test_use[EXOG_COLS].astype(float).to_numpy() if EXOG_COLS else None

    if AUTO_ORDER:
        order = _select_order_by_aic(y_train, exog_train, P_MAX, D_VALUES, Q_MAX, TREND)
    else:
        order = (1, 0, 1)

    cfg = FitConfig(order=order, trend=TREND)
    res = _fit_sarimax(y_train, exog_train, cfg)
    print("[INFO] Final model order:", order)
    print("[INFO] Final model converged:", getattr(res, "mle_retvals", {}).get("converged", None))

    y_pred = _walk_forward_predict(res, y_test, exog_test)

    out = test_use[[DATE_COL, TARGET_COL] + EXOG_COLS].copy()
    out["y_pred"] = y_pred
    out["y_err"] = out[TARGET_COL].astype(float) - out["y_pred"].astype(float)

    metrics = {"regression": _regression_metrics(out[TARGET_COL].to_numpy(), out["y_pred"].to_numpy())}

    diagnostics: Dict[str, Any] = {}
    if _HAS_DIAG and len(res.resid) > 20:
        try:
            lb = acorr_ljungbox(res.resid, lags=[10], return_df=True)
            diagnostics["ljung_box_pvalue_lag10"] = float(lb["lb_pvalue"].iloc[0])
        except Exception:
            pass

    report = {
        "run_name": RUN_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data": {
            "train_file": TRAIN_PATH,
            "test_file": TEST_PATH,
            "date_col": DATE_COL,
            "target_col": TARGET_COL,
            "exog_cols": EXOG_COLS,
            "shift_exog_by_1": bool(SHIFT_EXOG_BY_1),
            "train_rows_used": int(len(train_use)),
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
            "Return target is taken directly from the file (no recomputation).",
            "Blank Date rows are dropped before sorting to avoid corrupted ordering.",
            "NaN target rows are dropped safely.",
            "1-step-ahead walk-forward predictions on test.",
        ],
    }

    base = os.path.join(OUTDIR, RUN_NAME)
    out.to_csv(f"{base}_predictions.csv", index=False)

    with open(f"{base}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # simple MD report
    with open(f"{base}_report.md", "w", encoding="utf-8") as f:
        f.write(f"# {RUN_NAME}\n\n")
        f.write(f"- Timestamp (UTC): {report['timestamp_utc']}\n")
        f.write(f"- Order: {order}\n")
        f.write(f"- AIC: {report['model']['aic']}\n")
        f.write(f"- BIC: {report['model']['bic']}\n\n")
        f.write("## Metrics\n")
        for k, v in report["metrics"]["regression"].items():
            f.write(f"- {k}: {v}\n")

    if _HAS_PLOTLY:
        _make_plot(out, RUN_NAME, TARGET_COL, EXOG_COLS, f"{base}_plot.html")

    # summary metrics (single row)
    summary = pd.DataFrame([{
        "run_name": RUN_NAME,
        "target_col": TARGET_COL,
        "order": str(order),
        "aic": report["model"]["aic"],
        "bic": report["model"]["bic"],
        "test_rows_used": report["data"]["test_rows_used"],
        **{f"reg_{k}": v for k, v in report["metrics"]["regression"].items()}
    }])
    summary.to_csv(os.path.join(OUTDIR, "summary_metrics.csv"), index=False)

    print("\n[OK] Saved outputs to:", OUTDIR)
    print("[OK] Predictions:", f"{base}_predictions.csv")
    print("[OK] Plot (if enabled):", f"{base}_plot.html")


if __name__ == "__main__":
    main()
