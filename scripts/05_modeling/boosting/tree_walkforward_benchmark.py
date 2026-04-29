from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODEL_INPUTS_DIR = ARTIFACTS_DIR / "model_inputs"
DERIVED_FEATURES_DIR = ARTIFACTS_DIR / "derived_features"
RESULTS_DIR = ARTIFACTS_DIR / "results" / "tree_model_walkforward"
MPLCONFIG_DIR = REPO_ROOT / ".cache" / "matplotlib"

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    import shap

    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

try:
    import xgboost as xgb

    _HAS_XGBOOST = True
except Exception:
    xgb = None
    _HAS_XGBOOST = False

try:
    import lightgbm as lgb

    _HAS_LIGHTGBM = True
except Exception:
    lgb = None
    _HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor

    _HAS_CATBOOST = True
except Exception:
    CatBoostClassifier = None
    CatBoostRegressor = None
    _HAS_CATBOOST = False

DATE_COL = "Date"
CHANNEL_DATE_COL = "date"
TARGET_RETURN_COL = "target_return"
TARGET_VOL_COL = "target_vol_parkinson"
TARGET_JUMP_COL = "target_jump_flag"
GLOBAL_SENTIMENT_COL = "Global_Sentiment"

CHANNEL_COLS = [
    "z_Supply Shock and availability risk",
    "z_Transport logistics and chokepoints",
    "z_Demand and macro economy",
    "z_Inventory SPR and refinery",
    "z_OPEC producer policy",
    "z_Geopolitical Normalisation and Peace",
]

TOP3_CHANNELS = [
    "z_Demand and macro economy",
    "z_Inventory SPR and refinery",
    "z_Supply Shock and availability risk",
]

MARKET_CONTEXT_COLS = ["CO1 Comdty - Open Interest", "SMAVG (15)"]
TRAIN_VAL_FRACTION = 0.20
N_LAGS = 4
NEWS_LAG_DAYS = 1
TEST_STEP_ROWS = 63
MIN_TRAIN_ROWS = 252
IMPUTE_CAP_DAYS = 5
DECAY_WINDOW = 10
DECAY_LAMBDA = 1.5
SHAP_MAX_ROWS_TEST = 5000
SHAP_MAX_ROWS_BG = 2000
XGB_EARLY_STOPPING_ROUNDS = 200
LGB_EARLY_STOPPING_ROUNDS = 200
CATBOOST_EARLY_STOPPING_ROUNDS = 200
USE_GPU = True


def _xgb_supports_device_param() -> bool:
    if not _HAS_XGBOOST:
        return False
    version = getattr(xgb, "__version__", "")
    match = re.match(r"^(\d+)\.(\d+)", version)
    if not match:
        return False
    return int(match.group(1)) >= 2

XGB_REGRESSION_PARAMS = {
    "max_depth": 3,
    "min_child_weight": 20,
    "gamma": 0.1,
    "subsample": 0.75,
    "colsample_bytree": 0.70,
    "lambda": 10.0,
    "alpha": 0.0,
    "eta": 0.02,
    "seed": 42,
}

XGB_CLASSIFICATION_PARAMS = {
    "max_depth": 3,
    "min_child_weight": 10,
    "gamma": 0.0,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "lambda": 5.0,
    "alpha": 0.0,
    "eta": 0.03,
    "seed": 42,
}

LGB_REGRESSION_PARAMS = {
    "learning_rate": 0.02,
    "n_estimators": 3000,
    "num_leaves": 31,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "random_state": 42,
    "verbosity": -1,
}

LGB_CLASSIFICATION_PARAMS = {
    "learning_rate": 0.03,
    "n_estimators": 2500,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 3.0,
    "random_state": 42,
    "verbosity": -1,
}

CATBOOST_REGRESSION_PARAMS = {
    "loss_function": "RMSE",
    "learning_rate": 0.03,
    "depth": 6,
    "iterations": 3000,
    "l2_leaf_reg": 5.0,
    "random_seed": 42,
}

CATBOOST_CLASSIFICATION_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "PRAUC",
    "learning_rate": 0.03,
    "depth": 6,
    "iterations": 2500,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
}


@dataclass
class TaskConfig:
    name: str
    target_col: str
    task_type: str
    scale: float
    lag_feature_cols: List[str]
    threshold: float = 0.5


@dataclass
class Scenario:
    name: str
    channel_variant: str
    feature_cols: List[str]
    notes: str


@dataclass
class ModelSpec:
    name: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark XGBoost, LightGBM, and CatBoost with the same walk-forward tasks, "
            "channel ablations, and preprocessing variants."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["xgboost", "lightgbm", "catboost"],
        default=["xgboost", "lightgbm", "catboost"],
        help="Subset of tree models to benchmark.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["return", "volatility", "jump"],
        default=["return", "volatility", "jump"],
        help="Subset of tasks to run.",
    )
    parser.add_argument(
        "--scenario-set",
        choices=["core", "ablation", "full"],
        default="full",
        help="core=market/global/top3/all6, ablation=single-channel runs, full=both.",
    )
    parser.add_argument(
        "--outdir",
        default=str(RESULTS_DIR),
        help="Output directory for metrics, predictions, SHAP files, and metadata.",
    )
    parser.add_argument(
        "--scenario-names",
        nargs="+",
        default=None,
        help="Optional explicit scenario names to run instead of the whole scenario-set.",
    )
    parser.add_argument(
        "--shap-scenarios",
        nargs="+",
        default=None,
        help="Optional explicit scenario names for SHAP export. Defaults to --scenario-names when provided.",
    )
    parser.add_argument(
        "--test-step-rows",
        type=int,
        default=TEST_STEP_ROWS,
        help="Number of rows per walk-forward prediction block.",
    )
    parser.add_argument(
        "--news-lag-days",
        type=int,
        default=NEWS_LAG_DAYS,
        help="Lag to apply when mapping news features to market dates.",
    )
    parser.add_argument(
        "--disable-shap",
        action="store_true",
        help="Skip SHAP exports even if shap is installed.",
    )
    return parser.parse_args()


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).replace("\ufeff", "").strip() for c in out.columns]
    return out


def _read_table(path: Path) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith(".csv"):
        return pd.read_csv(path)
    if p.endswith(".xlsx") or p.endswith(".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


def _parse_dates_robust(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()

    if pd.api.types.is_datetime64_any_dtype(out[date_col]):
        out = out.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
        return out

    s = out[date_col]

    def _excel_serial_to_dt(x: pd.Series) -> pd.Series:
        xnum = pd.to_numeric(x, errors="coerce")
        mask = xnum.between(20000, 60000)
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
    out = out.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    return out


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _load_targets_with_split() -> pd.DataFrame:
    train = _normalize_columns(_read_table(MODEL_INPUTS_DIR / "train_with_targets.csv"))
    test = _normalize_columns(_read_table(MODEL_INPUTS_DIR / "test_with_targets.csv"))

    train = _parse_dates_robust(train, DATE_COL)
    test = _parse_dates_robust(test, DATE_COL)

    train["split"] = "train"
    test["split"] = "test"

    full = pd.concat([train, test], ignore_index=True)
    full = _coerce_numeric(full, MARKET_CONTEXT_COLS + [TARGET_RETURN_COL, TARGET_VOL_COL, TARGET_JUMP_COL])
    full = full.sort_values(DATE_COL).reset_index(drop=True)
    return full


def _load_sentiment_full() -> pd.DataFrame:
    train = _normalize_columns(_read_table(MODEL_INPUTS_DIR / "GSent_train_17-23.csv"))
    test = _normalize_columns(_read_table(MODEL_INPUTS_DIR / "GSent_test_24-25.csv"))
    full = pd.concat([train, test], ignore_index=True)
    full = _parse_dates_robust(full, CHANNEL_DATE_COL)
    full = _coerce_numeric(full, ["mean_polarity"])
    full = full[[CHANNEL_DATE_COL, "mean_polarity"]].drop_duplicates(CHANNEL_DATE_COL, keep="last")
    full = full.rename(columns={"mean_polarity": GLOBAL_SENTIMENT_COL})
    return full


def _load_channels_full() -> pd.DataFrame:
    train = _normalize_columns(_read_table(DERIVED_FEATURES_DIR / "train_2017_2023_daily_channel_index.csv"))
    test = _normalize_columns(_read_table(DERIVED_FEATURES_DIR / "test_2024_2025_daily_channel_index.csv"))
    full = pd.concat([train, test], ignore_index=True)
    full = _parse_dates_robust(full, CHANNEL_DATE_COL)
    numeric_cols = [
        c for c in full.columns
        if c.startswith("mass_") or c.startswith("num_") or c.startswith("z_") or c.startswith("low_evidence_")
    ]
    full = _coerce_numeric(full, numeric_cols + ["n_rows"])
    return full


def _locf_with_cap(series: pd.Series, cap: int) -> pd.Series:
    out = series.copy()
    last_val = np.nan
    run_len = 0

    for i in range(len(out)):
        val = out.iat[i]
        if pd.isna(val):
            run_len += 1
            if not pd.isna(last_val) and run_len <= cap:
                out.iat[i] = last_val
            else:
                out.iat[i] = 0.0
        else:
            last_val = val
            run_len = 0

    return out


def _exp_decay(series: pd.Series, window: int, decay_lambda: float, fill_value: float) -> pd.Series:
    base = pd.to_numeric(series, errors="coerce").fillna(fill_value)
    w_raw = np.exp(-np.arange(window) / decay_lambda)

    def _roll(vals: np.ndarray) -> float:
        n = len(vals)
        w = w_raw[:n]
        w = w / w.sum()
        return float(np.dot(vals, w[::-1]))

    return base.rolling(window=window, min_periods=1).apply(_roll, raw=True)


def _build_channel_feature_frame(channels: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]]]:
    work = channels[[CHANNEL_DATE_COL] + [c for c in channels.columns if c.startswith("z_") or c.startswith("low_evidence_")]].copy()
    work = work.sort_values(CHANNEL_DATE_COL).reset_index(drop=True)

    variant_maps: Dict[str, Dict[str, str]] = {
        "raw": {},
        "raw_decay": {},
        "imputed_locf5": {},
        "imputed_locf5_decay": {},
    }
    out = work[[CHANNEL_DATE_COL]].copy()

    for zcol in CHANNEL_COLS:
        lecol = zcol.replace("z_", "low_evidence_")
        raw_col = f"raw__{zcol}"
        raw_decay_col = f"raw_decay__{zcol}"
        imp_col = f"imputed_locf5__{zcol}"
        imp_decay_col = f"imputed_locf5_decay__{zcol}"

        series = pd.to_numeric(work.get(zcol), errors="coerce")
        low_evidence = (
            pd.to_numeric(work.get(lecol), errors="coerce").fillna(0).astype(int)
            if lecol in work.columns else pd.Series(0, index=work.index)
        )

        imputed = series.copy()
        imputed.loc[(low_evidence == 1) & (imputed.isna())] = 0.0
        imputed = _locf_with_cap(imputed, IMPUTE_CAP_DAYS)

        out[raw_col] = series
        out[raw_decay_col] = _exp_decay(series, DECAY_WINDOW, DECAY_LAMBDA, fill_value=0.0)
        out[imp_col] = imputed
        out[imp_decay_col] = _exp_decay(imputed, DECAY_WINDOW, DECAY_LAMBDA, fill_value=0.0)

        variant_maps["raw"][zcol] = raw_col
        variant_maps["raw_decay"][zcol] = raw_decay_col
        variant_maps["imputed_locf5"][zcol] = imp_col
        variant_maps["imputed_locf5_decay"][zcol] = imp_decay_col

    return out, variant_maps


def _merge_news_features(base: pd.DataFrame, sentiment: pd.DataFrame, channel_features: pd.DataFrame, news_lag_days: int) -> pd.DataFrame:
    out = base.copy()

    sent = sentiment.copy()
    sent[DATE_COL] = sent[CHANNEL_DATE_COL] + pd.Timedelta(days=news_lag_days)
    sent = sent[[DATE_COL, GLOBAL_SENTIMENT_COL]].drop_duplicates(DATE_COL, keep="last")

    ch = channel_features.copy()
    ch[DATE_COL] = ch[CHANNEL_DATE_COL] + pd.Timedelta(days=news_lag_days)
    ch = ch.drop(columns=[CHANNEL_DATE_COL])

    out = out.merge(sent, on=DATE_COL, how="left")
    out = out.merge(ch, on=DATE_COL, how="left")
    return out


def _add_market_lags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(DATE_COL).reset_index(drop=True)
    source_map = {
        "ret": TARGET_RETURN_COL,
        "vol": TARGET_VOL_COL,
        "jump": TARGET_JUMP_COL,
    }

    for prefix, src in source_map.items():
        for lag in range(1, N_LAGS + 1):
            out[f"{prefix}_lag{lag}"] = pd.to_numeric(out[src], errors="coerce").shift(lag)

    return out


def _build_modeling_frame(news_lag_days: int) -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]]]:
    base = _load_targets_with_split()
    sentiment = _load_sentiment_full()
    channels = _load_channels_full()
    channel_features, variant_maps = _build_channel_feature_frame(channels)

    out = _merge_news_features(base, sentiment, channel_features, news_lag_days=news_lag_days)
    out = _add_market_lags(out)
    return out, variant_maps


def _task_configs() -> Dict[str, TaskConfig]:
    return {
        "return": TaskConfig(
            name="return",
            target_col=TARGET_RETURN_COL,
            task_type="regression",
            scale=10000.0,
            lag_feature_cols=[f"ret_lag{i}" for i in range(1, N_LAGS + 1)],
        ),
        "volatility": TaskConfig(
            name="volatility",
            target_col=TARGET_VOL_COL,
            task_type="regression",
            scale=10000.0,
            lag_feature_cols=[f"vol_lag{i}" for i in range(1, N_LAGS + 1)] + [f"ret_lag{i}" for i in range(1, N_LAGS + 1)],
        ),
        "jump": TaskConfig(
            name="jump",
            target_col=TARGET_JUMP_COL,
            task_type="classification",
            scale=1.0,
            lag_feature_cols=[f"jump_lag{i}" for i in range(1, N_LAGS + 1)] + [f"vol_lag{i}" for i in range(1, N_LAGS + 1)] + [f"ret_lag{i}" for i in range(1, N_LAGS + 1)],
            threshold=0.5,
        ),
    }


def _scenario_definitions(scenario_set: str, variant_maps: Dict[str, Dict[str, str]]) -> List[Scenario]:
    scenarios: List[Scenario] = []

    core_channels = {
        "market_only": [],
        "market_plus_global_sentiment": [],
        "market_plus_top3_channels": TOP3_CHANNELS,
        "market_plus_all6_channels": CHANNEL_COLS,
    }
    single_channel_runs = {
        f"single__{re.sub('[^A-Za-z0-9]+', '_', c.replace('z_', '')).strip('_').lower()}": [c]
        for c in CHANNEL_COLS
    }

    include_core = scenario_set in {"core", "full"}
    include_ablation = scenario_set in {"ablation", "full"}

    for variant_name, mapping in variant_maps.items():
        if include_core:
            for name, base_channels in core_channels.items():
                features = list(MARKET_CONTEXT_COLS)
                notes = [f"channel_variant={variant_name}"]

                if name == "market_plus_global_sentiment":
                    features += [GLOBAL_SENTIMENT_COL]
                    notes.append("global_sentiment_only")
                elif name in {"market_plus_top3_channels", "market_plus_all6_channels"}:
                    features += [mapping[c] for c in base_channels]
                    notes.append(f"channels={len(base_channels)}")
                else:
                    notes.append("no_news_features")

                scenarios.append(
                    Scenario(
                        name=f"{variant_name}__{name}",
                        channel_variant=variant_name,
                        feature_cols=features,
                        notes=", ".join(notes),
                    )
                )

        if include_ablation:
            for name, base_channels in single_channel_runs.items():
                features = list(MARKET_CONTEXT_COLS) + [mapping[c] for c in base_channels]
                scenarios.append(
                    Scenario(
                        name=f"{variant_name}__{name}",
                        channel_variant=variant_name,
                        feature_cols=features,
                        notes=f"channel_variant={variant_name}, single_channel={base_channels[0]}",
                    )
                )

    deduped: List[Scenario] = []
    seen: set[str] = set()
    for sc in scenarios:
        key = f"{sc.name}::{','.join(sc.feature_cols)}"
        if key not in seen:
            seen.add(key)
            deduped.append(sc)
    return deduped


def _filter_scenarios(scenarios: List[Scenario], names: Optional[List[str]]) -> List[Scenario]:
    if not names:
        return scenarios

    wanted = set(names)
    filtered = [sc for sc in scenarios if sc.name in wanted]
    found = {sc.name for sc in filtered}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"Unknown scenario names: {missing}")
    return filtered


def _time_split_train_val(df: pd.DataFrame, val_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    val_n = max(1, int(round(n * val_fraction)))
    train_n = n - val_n
    if train_n < max(100, N_LAGS + 5):
        train_n = max(100, train_n)
        val_n = n - train_n
    return df.iloc[:train_n].copy(), df.iloc[train_n:].copy()


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(math.sqrt(mse))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}


def _classification_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true_int = np.asarray(y_true, dtype=int)
    y_hat = (np.asarray(y_proba, dtype=float) >= threshold).astype(int)

    out: Dict[str, float] = {
        "Accuracy": float(accuracy_score(y_true_int, y_hat)),
        "Precision": float(precision_score(y_true_int, y_hat, zero_division=0)),
        "Recall": float(recall_score(y_true_int, y_hat, zero_division=0)),
        "F1": float(f1_score(y_true_int, y_hat, zero_division=0)),
        "PositiveRate": float(np.mean(y_true_int)),
        "PredictedPositiveRate": float(np.mean(y_hat)),
    }

    try:
        out["ROC_AUC"] = float(roc_auc_score(y_true_int, y_proba))
    except Exception:
        out["ROC_AUC"] = float("nan")

    try:
        out["PR_AUC"] = float(average_precision_score(y_true_int, y_proba))
    except Exception:
        out["PR_AUC"] = float("nan")

    return out


def _available_models(requested: List[str]) -> List[ModelSpec]:
    availability = {
        "xgboost": _HAS_XGBOOST,
        "lightgbm": _HAS_LIGHTGBM,
        "catboost": _HAS_CATBOOST,
    }
    missing = [name for name in requested if not availability.get(name, False)]
    if missing:
        raise ImportError(
            "Missing model packages for: " + ", ".join(missing) + ". Install them before running this benchmark."
        )
    return [ModelSpec(name=name) for name in requested]


def _xgb_params(task: TaskConfig, y_train: np.ndarray) -> Tuple[Dict[str, Any], int]:
    if task.task_type == "regression":
        params = dict(XGB_REGRESSION_PARAMS)
        params["objective"] = "reg:squarederror"
        params["eval_metric"] = "rmse"
        return params, 3000

    params = dict(XGB_CLASSIFICATION_PARAMS)
    params["objective"] = "binary:logistic"
    params["eval_metric"] = "aucpr"
    pos = float(np.sum(y_train == 1.0))
    neg = float(np.sum(y_train == 0.0))
    if pos > 0 and neg > 0:
        params["scale_pos_weight"] = neg / pos
    return params, 2500


def _fit_xgboost(task: TaskConfig, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, feature_names: List[str]) -> Tuple[Any, bool, Dict[str, float]]:
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    params, n_estimators = _xgb_params(task, y_train)

    def _fit(params_local: Dict[str, Any]) -> Any:
        return xgb.train(
            params=params_local,
            dtrain=dtrain,
            num_boost_round=n_estimators,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )

    if USE_GPU:
        if _xgb_supports_device_param():
            try:
                params_modern = dict(params)
                params_modern["device"] = "cuda"
                params_modern["tree_method"] = "hist"
                model = _fit(params_modern)
                return model, True, {
                    "best_iteration": int(getattr(model, "best_iteration", -1)),
                    "best_score": float(getattr(model, "best_score", np.nan)),
                }
            except Exception:
                pass

        try:
            params_gpu = dict(params)
            params_gpu["tree_method"] = "gpu_hist"
            params_gpu["predictor"] = "gpu_predictor"
            model = _fit(params_gpu)
            return model, True, {
                "best_iteration": int(getattr(model, "best_iteration", -1)),
                "best_score": float(getattr(model, "best_score", np.nan)),
            }
        except Exception:
            pass

    params_cpu = dict(params)
    params_cpu["tree_method"] = "hist"
    params_cpu["predictor"] = "auto"
    model = _fit(params_cpu)
    return model, False, {
        "best_iteration": int(getattr(model, "best_iteration", -1)),
        "best_score": float(getattr(model, "best_score", np.nan)),
    }


def _predict_xgboost(model: Any, X: np.ndarray, feature_names: List[str], task: TaskConfig) -> np.ndarray:
    dmat = xgb.DMatrix(X, feature_names=feature_names)
    try:
        best_iter = getattr(model, "best_iteration", None)
        if best_iter is not None and isinstance(best_iter, int) and best_iter >= 0:
            return model.predict(dmat, iteration_range=(0, best_iter + 1))
    except Exception:
        pass
    return model.predict(dmat)


def _save_xgboost(model: Any, path: Path) -> None:
    model.save_model(path)


def _fit_lightgbm(task: TaskConfig, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, feature_names: List[str]) -> Tuple[Any, bool, Dict[str, float]]:
    params = dict(LGB_REGRESSION_PARAMS if task.task_type == "regression" else LGB_CLASSIFICATION_PARAMS)
    callbacks = [lgb.early_stopping(LGB_EARLY_STOPPING_ROUNDS, verbose=False)]

    if task.task_type == "regression":
        model = lgb.LGBMRegressor(objective="regression", **params)
        fit_kwargs = {
            "eval_set": [(X_val, y_val)],
            "eval_metric": "rmse",
            "callbacks": callbacks,
        }
    else:
        pos = float(np.sum(y_train == 1.0))
        neg = float(np.sum(y_train == 0.0))
        if pos > 0 and neg > 0:
            params["scale_pos_weight"] = neg / pos
        model = lgb.LGBMClassifier(objective="binary", **params)
        fit_kwargs = {
            "eval_set": [(X_val, y_val)],
            "eval_metric": "average_precision",
            "callbacks": callbacks,
        }

    used_gpu = False
    if USE_GPU:
        try:
            model.set_params(device_type="gpu")
            model.fit(X_train, y_train, **fit_kwargs)
            used_gpu = True
        except Exception:
            model = model.__class__(**{k: v for k, v in model.get_params().items() if k != "device_type"})
            model.fit(X_train, y_train, **fit_kwargs)
    else:
        model.fit(X_train, y_train, **fit_kwargs)

    best_iter = getattr(model, "best_iteration_", None)
    best_score = np.nan
    try:
        best_score = float(model.best_score_["valid_0"][model.evals_result_["valid_0"].keys().__iter__().__next__()])
    except Exception:
        pass

    return model, used_gpu, {
        "best_iteration": int(best_iter) if best_iter is not None else -1,
        "best_score": best_score,
    }


def _predict_lightgbm(model: Any, X: np.ndarray, feature_names: List[str], task: TaskConfig) -> np.ndarray:
    if task.task_type == "regression":
        return model.predict(X)
    return model.predict_proba(X)[:, 1]


def _save_lightgbm(model: Any, path: Path) -> None:
    model.booster_.save_model(str(path))


def _fit_catboost(task: TaskConfig, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, feature_names: List[str]) -> Tuple[Any, bool, Dict[str, float]]:
    params = dict(CATBOOST_REGRESSION_PARAMS if task.task_type == "regression" else CATBOOST_CLASSIFICATION_PARAMS)
    params["verbose"] = False
    params["early_stopping_rounds"] = CATBOOST_EARLY_STOPPING_ROUNDS

    if task.task_type == "regression":
        model = CatBoostRegressor(**params)
    else:
        pos = float(np.sum(y_train == 1.0))
        neg = float(np.sum(y_train == 0.0))
        if pos > 0 and neg > 0:
            params["auto_class_weights"] = "Balanced"
        model = CatBoostClassifier(**params)

    used_gpu = False
    if USE_GPU:
        try:
            model.set_params(task_type="GPU")
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            used_gpu = True
        except Exception:
            model = model.__class__(**{k: v for k, v in params.items() if k != "task_type"})
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    else:
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

    try:
        best_iteration = int(model.get_best_iteration())
    except Exception:
        best_iteration = -1

    try:
        best_score_map = model.get_best_score().get("validation", {})
        best_score = float(next(iter(best_score_map.values()))) if best_score_map else float("nan")
    except Exception:
        best_score = float("nan")

    return model, used_gpu, {
        "best_iteration": best_iteration,
        "best_score": best_score,
    }


def _predict_catboost(model: Any, X: np.ndarray, feature_names: List[str], task: TaskConfig) -> np.ndarray:
    if task.task_type == "regression":
        return np.asarray(model.predict(X), dtype=float)
    proba = model.predict_proba(X)
    return np.asarray(proba[:, 1], dtype=float)


def _save_catboost(model: Any, path: Path) -> None:
    model.save_model(str(path))


def _fit_model(model_name: str, task: TaskConfig, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, feature_names: List[str]) -> Tuple[Any, bool, Dict[str, float]]:
    if model_name == "xgboost":
        return _fit_xgboost(task, X_train, y_train, X_val, y_val, feature_names)
    if model_name == "lightgbm":
        return _fit_lightgbm(task, X_train, y_train, X_val, y_val, feature_names)
    if model_name == "catboost":
        return _fit_catboost(task, X_train, y_train, X_val, y_val, feature_names)
    raise ValueError(f"Unsupported model: {model_name}")


def _predict_model(model_name: str, model: Any, X: np.ndarray, feature_names: List[str], task: TaskConfig) -> np.ndarray:
    if model_name == "xgboost":
        return _predict_xgboost(model, X, feature_names, task)
    if model_name == "lightgbm":
        return _predict_lightgbm(model, X, feature_names, task)
    if model_name == "catboost":
        return _predict_catboost(model, X, feature_names, task)
    raise ValueError(f"Unsupported model: {model_name}")


def _save_model(model_name: str, model: Any, path: Path) -> None:
    if model_name == "xgboost":
        _save_xgboost(model, path)
        return
    if model_name == "lightgbm":
        _save_lightgbm(model, path)
        return
    if model_name == "catboost":
        _save_catboost(model, path)
        return
    with path.open("wb") as handle:
        pickle.dump(model, handle)


def _prepare_task_frames(df: pd.DataFrame, task: TaskConfig, scenario: Scenario) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = [task.target_col] + task.lag_feature_cols + MARKET_CONTEXT_COLS
    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].copy()

    train = train.dropna(subset=required + [task.target_col]).reset_index(drop=True)
    test = test.dropna(subset=required + [task.target_col]).reset_index(drop=True)

    if task.task_type == "classification":
        train[task.target_col] = train[task.target_col].astype(float).round().astype(int)
        test[task.target_col] = test[task.target_col].astype(float).round().astype(int)

    for col in [c for c in scenario.feature_cols if c == GLOBAL_SENTIMENT_COL]:
        train[col] = pd.to_numeric(train[col], errors="coerce").fillna(0.0)
        test[col] = pd.to_numeric(test[col], errors="coerce").fillna(0.0)

    return train, test


def _walk_forward_backtest(model_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, task: TaskConfig, scenario: Scenario, test_step_rows: int) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if len(train_df) < MIN_TRAIN_ROWS:
        raise ValueError(f"Not enough training rows for task={task.name}: {len(train_df)}")

    history = train_df.copy().reset_index(drop=True)
    fold_rows: List[Dict[str, Any]] = []
    pred_blocks: List[pd.DataFrame] = []
    used_gpu_any = False
    feature_cols = task.lag_feature_cols + scenario.feature_cols

    for fold_idx, start in enumerate(range(0, len(test_df), test_step_rows), start=1):
        block = test_df.iloc[start:start + test_step_rows].copy().reset_index(drop=True)
        fit_df, val_df = _time_split_train_val(history, TRAIN_VAL_FRACTION)

        X_train = fit_df[feature_cols].astype(float).to_numpy()
        X_val = val_df[feature_cols].astype(float).to_numpy()
        X_block = block[feature_cols].astype(float).to_numpy()

        if task.task_type == "regression":
            y_train = fit_df[task.target_col].astype(float).to_numpy() * task.scale
            y_val = val_df[task.target_col].astype(float).to_numpy() * task.scale
            y_true = block[task.target_col].astype(float).to_numpy()
        else:
            y_train = fit_df[task.target_col].astype(float).to_numpy()
            y_val = val_df[task.target_col].astype(float).to_numpy()
            y_true = block[task.target_col].astype(float).to_numpy()

        model, used_gpu, best = _fit_model(model_name, task, X_train, y_train, X_val, y_val, feature_cols)
        used_gpu_any = used_gpu_any or used_gpu
        raw_pred = _predict_model(model_name, model, X_block, feature_cols, task)

        if task.task_type == "regression":
            y_pred = raw_pred / task.scale
            metrics = _regression_metrics(y_true, y_pred)
            block_pred = block[[DATE_COL]].copy()
            block_pred["y_true"] = y_true
            block_pred["y_pred"] = y_pred
        else:
            y_proba = raw_pred.astype(float)
            metrics = _classification_metrics(y_true, y_proba, task.threshold)
            block_pred = block[[DATE_COL]].copy()
            block_pred["y_true"] = y_true
            block_pred["y_proba"] = y_proba
            block_pred["y_pred"] = (y_proba >= task.threshold).astype(int)

        block_pred["fold_idx"] = fold_idx
        pred_blocks.append(block_pred)

        fold_rows.append(
            {
                "model": model_name,
                "task": task.name,
                "scenario": scenario.name,
                "fold_idx": fold_idx,
                "train_rows": len(fit_df),
                "val_rows": len(val_df),
                "block_rows": len(block),
                "block_start": block[DATE_COL].min(),
                "block_end": block[DATE_COL].max(),
                "used_gpu": used_gpu,
                **best,
                **metrics,
            }
        )

        history = pd.concat([history, block], ignore_index=True)

    pred_df = pd.concat(pred_blocks, ignore_index=True)
    if task.task_type == "regression":
        summary_metrics = _regression_metrics(pred_df["y_true"].to_numpy(), pred_df["y_pred"].to_numpy())
    else:
        summary_metrics = _classification_metrics(pred_df["y_true"].to_numpy(), pred_df["y_proba"].to_numpy(), task.threshold)

    meta = {
        "used_gpu_any": used_gpu_any,
        "n_train_total": int(len(train_df)),
        "n_test_total": int(len(test_df)),
        "n_folds": int(len(fold_rows)),
    }
    return pd.DataFrame(fold_rows), pred_df, {**meta, **summary_metrics}


def _fit_reference_model(model_name: str, train_df: pd.DataFrame, task: TaskConfig, feature_cols: List[str]) -> Tuple[Any, bool, Dict[str, float], pd.DataFrame, pd.DataFrame]:
    fit_df, val_df = _time_split_train_val(train_df, TRAIN_VAL_FRACTION)
    X_train = fit_df[feature_cols].astype(float).to_numpy()
    X_val = val_df[feature_cols].astype(float).to_numpy()

    if task.task_type == "regression":
        y_train = fit_df[task.target_col].astype(float).to_numpy() * task.scale
        y_val = val_df[task.target_col].astype(float).to_numpy() * task.scale
    else:
        y_train = fit_df[task.target_col].astype(float).to_numpy()
        y_val = val_df[task.target_col].astype(float).to_numpy()

    model, used_gpu, best = _fit_model(model_name, task, X_train, y_train, X_val, y_val, feature_cols)
    return model, used_gpu, best, fit_df, val_df


def _shap_model_obj(model_name: str, model: Any) -> Any:
    if model_name == "xgboost":
        return model
    if model_name == "lightgbm":
        return model.booster_
    if model_name == "catboost":
        return model
    return model


def _normalize_shap_output(task: TaskConfig, shap_raw: Any, expected_raw: Any) -> Tuple[np.ndarray, float]:
    if task.task_type == "regression":
        shap_vals = np.asarray(shap_raw, dtype=float) / task.scale
        expected = expected_raw
        base_value = float(expected) if np.isscalar(expected) else float(np.asarray(expected, dtype=float).reshape(-1)[0])
        return shap_vals, base_value / task.scale

    if isinstance(shap_raw, list):
        idx = 1 if len(shap_raw) > 1 else 0
        shap_vals = np.asarray(shap_raw[idx], dtype=float)
    else:
        shap_arr = np.asarray(shap_raw, dtype=float)
        if shap_arr.ndim == 3:
            idx = 1 if shap_arr.shape[-1] > 1 else 0
            shap_vals = shap_arr[:, :, idx]
        else:
            shap_vals = shap_arr

    if isinstance(expected_raw, (list, tuple, np.ndarray)):
        expected_arr = np.asarray(expected_raw, dtype=float).reshape(-1)
        idx = 1 if len(expected_arr) > 1 else 0
        base_value = float(expected_arr[idx])
    else:
        base_value = float(expected_raw)

    return shap_vals, base_value


def _shap_outputs(outdir: Path, model_name: str, task: TaskConfig, scenario: Scenario, model: Any, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]) -> None:
    if not _HAS_SHAP:
        return

    bg = train_df[feature_cols].iloc[: min(SHAP_MAX_ROWS_BG, len(train_df))].copy()
    te = test_df[feature_cols].iloc[: min(SHAP_MAX_ROWS_TEST, len(test_df))].copy()
    bg = bg.apply(pd.to_numeric, errors="coerce")
    te = te.apply(pd.to_numeric, errors="coerce")

    shap_model = _shap_model_obj(model_name, model)
    explainer = shap.TreeExplainer(shap_model)
    shap_vals, base_value = _normalize_shap_output(task, explainer.shap_values(te), explainer.expected_value)

    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", f"{model_name}__{task.name}__{scenario.name}").strip("_")
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    gi = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
    gi.to_csv(outdir / f"shap_global_importance__{safe_name}.csv", index=False)

    pred_raw = _predict_model(model_name, model, te.to_numpy(), feature_cols, task)
    local = pd.DataFrame(shap_vals, columns=[f"shap_{c}" for c in feature_cols])
    local.insert(0, "base_value", base_value)
    meta = test_df.iloc[: len(te)][[DATE_COL, task.target_col]].copy().reset_index(drop=True)
    meta.rename(columns={task.target_col: "y_true"}, inplace=True)
    if task.task_type == "regression":
        meta["y_pred"] = pred_raw / task.scale
    else:
        meta["y_proba"] = pred_raw.astype(float)
        meta["y_pred"] = (meta["y_proba"] >= task.threshold).astype(int)
    pd.concat([meta, local, te.reset_index(drop=True)], axis=1).to_csv(outdir / f"shap_local__{safe_name}.csv", index=False)

    plt.figure()
    shap.summary_plot(shap_vals, te, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(outdir / f"shap_summary_bar__{safe_name}.png", dpi=200)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_vals, te, show=False)
    plt.tight_layout()
    plt.savefig(outdir / f"shap_summary_beeswarm__{safe_name}.png", dpi=200)
    plt.close()


def _select_shap_scenarios(scenarios: List[Scenario], explicit_names: Optional[List[str]] = None) -> set[str]:
    if explicit_names:
        return {sc.name for sc in scenarios if sc.name in set(explicit_names)}

    preferred = {
        "imputed_locf5__market_plus_all6_channels",
        "imputed_locf5_decay__market_plus_all6_channels",
    }
    available = {sc.name for sc in scenarios}
    return preferred.intersection(available)


def main() -> None:
    args = _parse_args()
    outdir = Path(args.outdir)
    _safe_mkdir(outdir)

    models = _available_models(args.models)
    full_df, variant_maps = _build_modeling_frame(news_lag_days=args.news_lag_days)
    task_map = _task_configs()
    scenarios = _scenario_definitions(args.scenario_set, variant_maps)
    scenarios = _filter_scenarios(scenarios, args.scenario_names)
    shap_seed_names = args.shap_scenarios if args.shap_scenarios is not None else args.scenario_names
    shap_targets = _select_shap_scenarios(scenarios, shap_seed_names)

    model_input_path = outdir / "modeling_frame_snapshot.csv"
    full_df.to_csv(model_input_path, index=False)
    summary_rows: List[Dict[str, Any]] = []

    for model_spec in models:
        for task_name in args.tasks:
            task = task_map[task_name]
            for scenario in scenarios:
                print("=" * 80)
                print(f"[RUN] model={model_spec.name} task={task.name} scenario={scenario.name}")
                train_df, test_df = _prepare_task_frames(full_df, task, scenario)
                feature_cols = task.lag_feature_cols + scenario.feature_cols

                if len(test_df) == 0:
                    print(f"[WARN] Skipping model={model_spec.name} task={task.name} scenario={scenario.name}: no usable test rows")
                    continue

                fold_df, pred_df, summary = _walk_forward_backtest(
                    model_name=model_spec.name,
                    train_df=train_df,
                    test_df=test_df,
                    task=task,
                    scenario=scenario,
                    test_step_rows=args.test_step_rows,
                )

                safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", f"{model_spec.name}__{task.name}__{scenario.name}").strip("_")
                fold_df.to_csv(outdir / f"fold_metrics__{safe_name}.csv", index=False)
                pred_df.to_csv(outdir / f"predictions__{safe_name}.csv", index=False)

                ref_model, ref_used_gpu, ref_best, fit_df, val_df = _fit_reference_model(model_spec.name, train_df, task, feature_cols)
                suffix = ".json" if model_spec.name in {"xgboost", "catboost", "lightgbm"} else ".pkl"
                ref_path = outdir / f"reference_model__{safe_name}{suffix}"
                _save_model(model_spec.name, ref_model, ref_path)

                if (not args.disable_shap) and (scenario.name in shap_targets):
                    try:
                        _shap_outputs(outdir, model_spec.name, task, scenario, ref_model, fit_df, test_df, feature_cols)
                    except Exception as exc:
                        print(f"[WARN] SHAP export failed for {model_spec.name}/{task.name}/{scenario.name}: {exc}")

                summary_row = {
                    "model": model_spec.name,
                    "task": task.name,
                    "task_type": task.task_type,
                    "target_col": task.target_col,
                    "scenario": scenario.name,
                    "channel_variant": scenario.channel_variant,
                    "feature_count": len(feature_cols),
                    "features": " | ".join(feature_cols),
                    "notes": scenario.notes,
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                    "walkforward_folds": summary.pop("n_folds"),
                    "walkforward_used_gpu_any": summary.pop("used_gpu_any"),
                    "walkforward_n_train_total": summary.pop("n_train_total"),
                    "walkforward_n_test_total": summary.pop("n_test_total"),
                    "reference_used_gpu": ref_used_gpu,
                    "reference_best_iteration": ref_best.get("best_iteration"),
                    "reference_best_score": ref_best.get("best_score"),
                    "reference_model_path": str(ref_path),
                    **summary,
                }
                summary_rows.append(summary_row)

                with open(outdir / f"run_meta__{safe_name}.json", "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "model": asdict(model_spec),
                            "task": asdict(task),
                            "scenario": asdict(scenario),
                            "feature_cols": feature_cols,
                            "test_step_rows": args.test_step_rows,
                            "news_lag_days": args.news_lag_days,
                            "summary": summary_row,
                        },
                        handle,
                        indent=2,
                        default=str,
                    )

    if not summary_rows:
        raise RuntimeError("No experiments completed. Check input artifacts, installed model packages, and scenario/task filters.")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "metrics_summary.csv", index=False)

    with open(outdir / "run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "repo_root": str(REPO_ROOT),
                "outdir": str(outdir),
                "models": [m.name for m in models],
                "tasks": args.tasks,
                "scenario_set": args.scenario_set,
                "test_step_rows": args.test_step_rows,
                "news_lag_days": args.news_lag_days,
                "modeling_frame_snapshot": str(model_input_path),
            },
            handle,
            indent=2,
        )

    print("[OK] Saved multi-model walk-forward outputs to:", outdir)
    print("[OK] Metrics summary:", outdir / "metrics_summary.csv")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
