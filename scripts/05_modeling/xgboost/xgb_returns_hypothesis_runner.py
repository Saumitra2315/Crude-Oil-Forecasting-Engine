# xgb_returns_hypothesis_runner.py
# ------------------------------------------------------------
# Trains & evaluates 3 XGBoost time-series regression models:
#   (1) Return-history only (lags)
#   (2) Return-history + global polarity (calendar t-1)
#   (3) Return-history + 6 channel indices (calendar t-1)
#
# Outputs (written to OUTDIR):
#   - metrics_summary.csv (MAE/MSE/RMSE for each scenario)
#   - predictions_<scenario>.csv (Date, y_true, y_pred + features)
#   - run_<scenario>_meta.json (params + sizes + best_iteration)
#   - model_<scenario>.json (xgboost Booster)
#   - SHAP (if installed):
#       * shap_global_importance_<scenario>.csv
#       * shap_local_<scenario>.csv
#       * shap_summary_bar_<scenario>.png
#       * shap_summary_beeswarm_<scenario>.png
#       * shap_dependence_<scenario>__<feature>.png
#
# Time-series constraints followed:
#   - No shuffling; strict sort by Date
#   - Train file = 2017–2023, Test file = 2024–2025 (kept independent)
#   - No scaling/normalization of FEATURES (trees don’t need it)
#   - Same lag structure across all 3 scenarios
#
# IMPORTANT NOTE ABOUT RETURNS SCALE:
#   Daily returns are tiny (~0.01). With your locked gamma=0.1, XGBoost may refuse
#   to make ANY splits (no tree can achieve >= 0.1 loss reduction), resulting in a
#   constant prediction equal to the mean return.
#   Fix (without changing your locked hyperparameters): we SCALE the TARGET during
#   training (e.g., *100), then scale predictions back.
# ------------------------------------------------------------

from __future__ import annotations

import os
import re
import json
import math
import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt

# SHAP (interpretability)
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False


# ============================================================
# ✅✅✅ EDIT ONLY THIS BLOCK ✅✅✅
# ============================================================

# Your FINAL train + test datasets (with lagged returns + global polarity + 6 channels)
TRAIN_PATH = "D:/ACADS/4-2/FIN SOP/Data/Code/Datasets/train_dataset_17_23.xlsx"
TEST_PATH  = "D:/ACADS/4-2/FIN SOP/Data/Code/Datasets/test_dataset_24_25(half).xlsx"  # placeholder
# ^^^ Replace above with your actual test file. Example:
# TEST_PATH  = "D:/ACADS/4-2/FIN SOP/Data/Code/Models/XGBoost/xgb_outputs_returns/../Datasets/test_returns_lagged4_24_25_with_channels_tminus1_plus_globalpol_tminus1.xlsx"

OUTDIR = "D:/ACADS/4-2/FIN SOP/Data/Code/Models/XGBoost/xgb_outputs_returns"

DATE_COL = "Date"
TARGET_COL = "Return"

# Lag columns (must exist as-is)
LAG_COLS = ["ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4"]

# Global polarity column (Column H in your file)
GLOBAL_POL_COL = "Global_Sentiment"

# 6 channel columns (must exist as-is)
CHANNEL_COLS = [
    "z_Supply Shock and availability risk",
    "z_Transport logistics and chokepoints",
    "z_Demand and macro economy",
    "z_Inventory SPR and refinery",
    "z_OPEC producer policy",
    "z_Geopolitical Normalisation and Peace",
]

# Development switch:
# True  -> quick run on laptop
# False -> full run on lab PC
DEV_MODE = False

# Validation split inside TRAIN (time-ordered; last fraction used for early stopping)
VAL_FRACTION = 0.20

# Early stopping (handled via xgboost.train for max compatibility)
USE_EARLY_STOPPING = True
EARLY_STOPPING_ROUNDS = 10 if DEV_MODE else 200

# SHAP sampling limits (time-ordered samples; no shuffle)
SHAP_MAX_ROWS_TEST = 300 if DEV_MODE else 5000
SHAP_MAX_ROWS_BG   = 300 if DEV_MODE else 2000

# GPU usage (auto-fallback if GPU build not available)
USE_GPU = True

# ✅ Key fix for your “constant prediction” problem:
# Scale the TARGET during training so gamma=0.1 makes sense.
# 100.0 means “train in % returns”; 10000.0 means “train in bps”.
TARGET_SCALE = 10000.0

# XGBoost hyperparameters (locked across all 3 scenarios)
# NOTE: in xgboost.train, use num_boost_round = N_ESTIMATORS.
XGB_PARAMS = {
    # core
    "objective": "reg:squarederror",
    "eval_metric": "rmse",

    # model capacity (keep conservative for returns)
    "max_depth": 3,
    "min_child_weight": 20,
    "gamma": 0.1,

    # randomness / robustness
    "subsample": 0.75,
    "colsample_bytree": 0.70,

    # regularization
    "lambda": 10.0,   # L2  (xgboost.train uses 'lambda')
    "alpha": 0.0,     # L1

    # boosting schedule
    "eta": 0.02,      # learning_rate

    # reproducibility
    "seed": 42,
}

# “epochs” analogue in XGBoost = boosting rounds (trees)
N_ESTIMATORS = 1000 if DEV_MODE else 3000

# Optional debugging prints
DEBUG_SHOW_FIRST_TREE = True if DEV_MODE else False

# ============================================================
# (Do not edit below unless changing logic)
# ============================================================


@dataclass
class Scenario:
    name: str
    feature_cols: List[str]


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_table(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".xlsx") or p.endswith(".xls"):
        return pd.read_excel(path)
    if p.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [("" if c is None else str(c)).strip() for c in out.columns]
    return out


def _parse_dates_robust(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Robust date parsing for Excel + CSV.

    Handles:
      - true datetime dtype
      - Excel serial numbers
      - strings like '2017-01-10 00:00:00'
      - strings like '10/01/2017' or '10-01-2017'

    Keeps only rows with parsed dates, then sorts.
    """
    out = df.copy()

    if pd.api.types.is_datetime64_any_dtype(out[date_col]):
        out = out.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
        return out

    s = out[date_col]

    def _excel_serial_to_dt(x: pd.Series) -> pd.Series:
        xnum = pd.to_numeric(x, errors="coerce")
        mask = xnum.between(20000, 60000)  # heuristic for modern Excel dates
        dt = pd.Series(pd.NaT, index=x.index)
        if mask.any():
            dt.loc[mask] = pd.to_datetime(xnum.loc[mask], unit="D", origin="1899-12-30", errors="coerce")
        return dt

    dt = pd.Series(pd.NaT, index=s.index)

    if pd.api.types.is_numeric_dtype(s):
        dt = _excel_serial_to_dt(s)
    else:
        dt = _excel_serial_to_dt(s.astype(str).str.strip())

    raw = s.astype(str).str.strip()

    dt_a = pd.to_datetime(raw, errors="coerce", dayfirst=True)
    dt_b = pd.to_datetime(raw, errors="coerce", dayfirst=False)

    remain = dt.isna()
    if remain.any():
        na_a = dt_a[remain].isna().mean()
        na_b = dt_b[remain].isna().mean()
        pick = dt_a if na_a <= na_b else dt_b
        dt.loc[remain] = pick.loc[remain]

    out[date_col] = dt

    bad_n = int(out[date_col].isna().sum())
    if bad_n:
        print(f"[WARN] Unparsed dates dropped: {bad_n}")
        examples = raw[out[date_col].isna()].head(8).tolist()
        print("[WARN] Example bad Date values:", examples)

    out = out.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    return out


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _time_split_train_val(df: pd.DataFrame, val_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    val_n = max(1, int(round(n * val_fraction)))
    train_n = n - val_n

    if train_n < 100:
        train_n = max(100, train_n)
        val_n = n - train_n

    train_df = df.iloc[:train_n].copy()
    val_df = df.iloc[train_n:].copy()
    return train_df, val_df


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(math.sqrt(mse))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}


def _debug_tree_if_needed(booster: xgb.Booster, scenario_name: str) -> None:
    if not DEBUG_SHOW_FIRST_TREE:
        return
    try:
        dump = booster.get_dump(with_stats=False)
        if not dump:
            print("[DEBUG] No trees in booster?")
            return
        first = dump[0]
        has_split = ("[" in first and "]" in first)
        print(f"[DEBUG] {scenario_name}: first tree has split? {has_split}")
        print("[DEBUG] First tree snippet:", "".join(first.splitlines()[:12]))
    except Exception as e:
        print(f"[DEBUG] Tree dump failed: {e}")


def _train_booster_with_gpu_fallback(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
) -> Tuple[xgb.Booster, bool, Dict[str, float]]:
    """Train via xgboost.train (works across old/new xgboost versions).

    Returns: booster, used_gpu, best_info
    """

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    evals = [(dtrain, "train"), (dval, "val")]

    def _fit(params: Dict[str, float]) -> xgb.Booster:
        return xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=N_ESTIMATORS,
            evals=evals,
            early_stopping_rounds=(EARLY_STOPPING_ROUNDS if USE_EARLY_STOPPING else None),
            verbose_eval=False,
        )

    if USE_GPU:
        # Newer XGBoost (>=2.x/3.x) prefers: device='cuda', tree_method='hist'
        try:
            params_modern = dict(XGB_PARAMS)
            params_modern["device"] = "cuda"
            params_modern["tree_method"] = "hist"
            booster = _fit(params_modern)
            best = {
                "best_iteration": int(getattr(booster, "best_iteration", -1)),
                "best_score": float(getattr(booster, "best_score", np.nan)),
            }
            return booster, True, best
        except Exception as e:
            print("[WARN] GPU training (device=cuda) failed, trying legacy gpu_hist. Reason:", str(e))

        # Older GPU style: tree_method='gpu_hist'
        try:
            params_gpu = dict(XGB_PARAMS)
            params_gpu["tree_method"] = "gpu_hist"
            params_gpu["predictor"] = "gpu_predictor"
            booster = _fit(params_gpu)
            best = {
                "best_iteration": int(getattr(booster, "best_iteration", -1)),
                "best_score": float(getattr(booster, "best_score", np.nan)),
            }
            return booster, True, best
        except Exception as e:
            print("[WARN] GPU training failed, falling back to CPU. Reason:", str(e))

    params_cpu = dict(XGB_PARAMS)
    params_cpu["tree_method"] = "hist"
    params_cpu["predictor"] = "auto"
    booster = _fit(params_cpu)
    best = {
        "best_iteration": int(getattr(booster, "best_iteration", -1)),
        "best_score": float(getattr(booster, "best_score", np.nan)),
    }
    return booster, False, best


def _predict_booster(booster: xgb.Booster, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
    dtest = xgb.DMatrix(X, feature_names=feature_names)

    try:
        best_iter = getattr(booster, "best_iteration", None)
        if best_iter is not None and isinstance(best_iter, int) and best_iter >= 0:
            return booster.predict(dtest, iteration_range=(0, best_iter + 1))
    except Exception:
        pass

    try:
        best_ntree_limit = getattr(booster, "best_ntree_limit", None)
        if best_ntree_limit is not None and int(best_ntree_limit) > 0:
            return booster.predict(dtest, ntree_limit=int(best_ntree_limit))
    except Exception:
        pass

    return booster.predict(dtest)


def _shap_outputs(
    scenario_name: str,
    booster: xgb.Booster,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    outdir: str,
) -> None:
    if not _HAS_SHAP:
        print("[WARN] SHAP not installed. Install with: pip install shap")
        return

    bg = train_df[feature_cols].iloc[:min(SHAP_MAX_ROWS_BG, len(train_df))].copy()
    te = test_df[feature_cols].iloc[:min(SHAP_MAX_ROWS_TEST, len(test_df))].copy()

    bg = bg.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    te = te.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    explainer = shap.TreeExplainer(booster)
    shap_vals = explainer.shap_values(te)

    expected = explainer.expected_value
    base_val_scaled = float(expected) if np.isscalar(expected) else float(expected[0])

    shap_vals = np.asarray(shap_vals, dtype=float)

    # Convert SHAP values back to ORIGINAL return units
    shap_vals = shap_vals / TARGET_SCALE
    base_val = base_val_scaled / TARGET_SCALE

    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    gi = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
    gi.to_csv(os.path.join(outdir, f"shap_global_importance_{scenario_name}.csv"), index=False)

    local = pd.DataFrame(shap_vals, columns=[f"shap_{c}" for c in feature_cols])
    local.insert(0, "base_value", float(base_val))

    meta = test_df.iloc[:len(te)][[DATE_COL, TARGET_COL]].copy().reset_index(drop=True)
    y_pred_scaled = _predict_booster(booster, te.to_numpy(), feature_cols)
    meta["y_pred"] = y_pred_scaled / TARGET_SCALE

    local_out = pd.concat([meta, local, te.reset_index(drop=True)], axis=1)
    local_out.to_csv(os.path.join(outdir, f"shap_local_{scenario_name}.csv"), index=False)

    plt.figure()
    shap.summary_plot(shap_vals, te, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"shap_summary_bar_{scenario_name}.png"), dpi=200)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_vals, te, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"shap_summary_beeswarm_{scenario_name}.png"), dpi=200)
    plt.close()

    for feat in feature_cols:
        try:
            plt.figure()
            shap.dependence_plot(feat, shap_vals, te, show=False, interaction_index=None)
            plt.tight_layout()
            safe = re.sub("[^A-Za-z0-9_-]+", "_", feat)[:80]
            plt.savefig(os.path.join(outdir, f"shap_dependence_{scenario_name}__{safe}.png"), dpi=200)
            plt.close()
        except Exception as e:
            print(f"[WARN] Dependence plot failed for {feat}: {e}")


def main():
    _safe_mkdir(OUTDIR)

    train_raw = _normalize_columns(_read_table(TRAIN_PATH))
    test_raw = _normalize_columns(_read_table(TEST_PATH))

    if DATE_COL not in train_raw.columns or DATE_COL not in test_raw.columns:
        raise ValueError(f"DATE_COL='{DATE_COL}' must exist in both train and test files.")

    train = _parse_dates_robust(train_raw, DATE_COL)
    test = _parse_dates_robust(test_raw, DATE_COL)

    for c in [TARGET_COL] + LAG_COLS + [GLOBAL_POL_COL] + CHANNEL_COLS:
        if c not in train.columns:
            raise ValueError(f"Missing column in TRAIN: {c}")
        if c not in test.columns:
            raise ValueError(f"Missing column in TEST: {c}")

    numeric_cols = list(set([TARGET_COL] + LAG_COLS + [GLOBAL_POL_COL] + CHANNEL_COLS))
    train = _coerce_numeric(train, numeric_cols)
    test = _coerce_numeric(test, numeric_cols)

    # Fill missing sentiment/channel values with 0.0
    train[GLOBAL_POL_COL] = train[GLOBAL_POL_COL].fillna(0.0)
    test[GLOBAL_POL_COL] = test[GLOBAL_POL_COL].fillna(0.0)
    for c in CHANNEL_COLS:
        train[c] = train[c].fillna(0.0)
        test[c] = test[c].fillna(0.0)

    # Drop only rows with missing target/lags
    train_before = len(train)
    test_before = len(test)
    train = train.dropna(subset=[TARGET_COL] + LAG_COLS).reset_index(drop=True)
    test = test.dropna(subset=[TARGET_COL] + LAG_COLS).reset_index(drop=True)

    print(f"[INFO] Train rows after date-parse: {train_before}")
    print(f"[INFO] Train rows used (after NaN drop in target/lags): {len(train)} (dropped {train_before - len(train)})")
    print(f"[INFO] Test rows after date-parse : {test_before}")
    print(f"[INFO] Test rows used  (after NaN drop in target/lags): {len(test)} (dropped {test_before - len(test)})")

    train_fit, val_fit = _time_split_train_val(train, VAL_FRACTION)

    scenarios = [
        Scenario("baseline_returns_only", LAG_COLS),
        Scenario("returns_plus_global_polarity", LAG_COLS + [GLOBAL_POL_COL]),
        Scenario("returns_plus_6channels", LAG_COLS + CHANNEL_COLS),
    ]

    metrics_rows: List[Dict[str, object]] = []

    for sc in scenarios:
        print("" + "=" * 70)
        print("[RUN]", sc.name)
        print("[FEATURES]", sc.feature_cols)

        X_train = train_fit[sc.feature_cols].astype(float).to_numpy()
        y_train = train_fit[TARGET_COL].astype(float).to_numpy() * TARGET_SCALE

        X_val = val_fit[sc.feature_cols].astype(float).to_numpy()
        y_val = val_fit[TARGET_COL].astype(float).to_numpy() * TARGET_SCALE

        X_test = test[sc.feature_cols].astype(float).to_numpy()
        y_test = test[TARGET_COL].astype(float).to_numpy()  # original scale

        booster, used_gpu, best = _train_booster_with_gpu_fallback(
            X_train, y_train, X_val, y_val, feature_names=sc.feature_cols
        )

        _debug_tree_if_needed(booster, sc.name)

        print("[INFO] Used GPU:", used_gpu)
        if best.get("best_iteration", -1) >= 0:
            best_rmse_scaled = best.get("best_score")
            best_rmse = (float(best_rmse_scaled) / TARGET_SCALE) if best_rmse_scaled is not None else None
            print("[INFO] Best iteration:", best["best_iteration"], "| Best val RMSE (scaled):", best_rmse_scaled,
                  "| Best val RMSE (original):", best_rmse)

        y_pred_scaled = _predict_booster(booster, X_test, sc.feature_cols)
        y_pred = y_pred_scaled / TARGET_SCALE

        m = _metrics(y_test, y_pred)
        print("[METRICS]", m)

        # Save predictions (Date as string to avoid Excel #######)
        pred_df = test[[DATE_COL, TARGET_COL]].copy()
        pred_df.rename(columns={TARGET_COL: "y_true"}, inplace=True)
        pred_df[DATE_COL] = pd.to_datetime(pred_df[DATE_COL]).dt.strftime("%d-%m-%Y")
        pred_df["y_pred"] = y_pred
        for c in sc.feature_cols:
            pred_df[c] = test[c].values
        pred_df.to_csv(os.path.join(OUTDIR, f"predictions_{sc.name}.csv"), index=False)

        # Save model weights per scenario
        model_path = os.path.join(OUTDIR, f"model_{sc.name}.json")
        booster.save_model(model_path)

        run_meta = {
            "scenario": sc.name,
            "train_path": TRAIN_PATH,
            "test_path": TEST_PATH,
            "date_col": DATE_COL,
            "target_col": TARGET_COL,
            "feature_cols": sc.feature_cols,
            "target_scale": TARGET_SCALE,
            "n_train_total": int(len(train)),
            "n_train_fit": int(len(train_fit)),
            "n_val": int(len(val_fit)),
            "n_test": int(len(test)),
            "used_gpu": bool(used_gpu),
            "dev_mode": bool(DEV_MODE),
            "params_locked": {
                "XGB_PARAMS": XGB_PARAMS,
                "N_ESTIMATORS": N_ESTIMATORS,
                "EARLY_STOPPING_ROUNDS": (EARLY_STOPPING_ROUNDS if USE_EARLY_STOPPING else None),
            },
            "best": best,
            "metrics": m,
            "model_path": model_path,
        }
        with open(os.path.join(OUTDIR, f"run_{sc.name}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2)

        _shap_outputs(
            scenario_name=sc.name,
            booster=booster,
            train_df=train_fit[[DATE_COL, TARGET_COL] + sc.feature_cols].copy(),
            test_df=test[[DATE_COL, TARGET_COL] + sc.feature_cols].copy(),
            feature_cols=sc.feature_cols,
            outdir=OUTDIR,
        )

        metrics_rows.append({
            "scenario": sc.name,
            **m,
            "used_gpu": used_gpu,
            "n_train_fit": len(train_fit),
            "n_val": len(val_fit),
            "n_test": len(test),
        })

    summary = pd.DataFrame(metrics_rows)
    summary_path = os.path.join(OUTDIR, "metrics_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("[OK] Saved outputs to:", OUTDIR)
    print("[OK] Metrics summary:", summary_path)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
