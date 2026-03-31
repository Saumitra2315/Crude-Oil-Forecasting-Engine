from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd


# ============================================================
# ✅✅✅ EDIT ONLY THIS SECTION ✅✅✅
# ============================================================

# Root folder containing daily files (year/month/dayfile)
# Example structures:
#   .../2017/01_Jan/2017-01-01_nli_probs.csv
#   .../2017/01_Jan/2017-01-01_nli_probs.xlsx
INPUT_ROOT = Path(r"D:\ACADS\4-2\FIN SOP\Data\Code\Relevance\Final_Polarity_Relevance\Final_Polarity_Relevance")

# Where to write the 3 aggregated outputs
OUTPUT_DIR = Path(r"D:\ACADS\4-2\FIN SOP\Data\Code\Relevance\Final_Polarity_Relevance\Final_Polarity_Relevance\Daily Channel Index")

# Your 5th-percentile mass thresholds (from your sheet, training years 2017-2023)
# These are used as FIXED thresholds for train/test/aux (no leakage).
MASS_THRESHOLDS: Dict[str, float] = {
    "Supply Shock and availability risk": 7.195134,
    "Transport logistics and chokepoints": 1.445090,
    "Demand and macro economy": 8.378556,
    "Inventory SPR and refinery": 3.497590,
    "OPEC producer policy": 1.054716,
    "Geopolitical Normalisation and Peace": 0.596221,
}

# If a daily file is missing OR too small, treat it as missing
MIN_FILE_SIZE_BYTES = 10 * 1024  # 10 KB

# Output filenames
OUT_TRAIN = "train_2017_2023_daily_channel_index.csv"
OUT_TEST = "test_2024_2025_daily_channel_index.csv"
OUT_AUX = "aux_2026_daily_channel_index.csv"

# ============================================================


DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def robust_read_any(path: Path) -> pd.DataFrame:
    """
    Reads CSV/XLSX robustly.
    - CSV: tries delimiter sniffing (sep=None) + skips bad lines
    - XLSX: standard read_excel
    """
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    # CSV path
    try:
        return pd.read_csv(path, engine="python", sep=None, on_bad_lines="skip")
    except UnicodeDecodeError:
        return pd.read_csv(path, engine="python", sep=None, on_bad_lines="skip", encoding="latin-1")


def extract_date_from_filename(path: Path) -> Optional[pd.Timestamp]:
    """
    Extracts YYYY-MM-DD from filename.
    Example: 2017-09-04_nli_probs.csv -> 2017-09-04
    """
    m = DATE_RE.search(path.name)
    if not m:
        return None
    try:
        return pd.to_datetime(m.group(1), format="%Y-%m-%d")
    except Exception:
        return None


def find_polarity_column(df: pd.DataFrame) -> Optional[str]:
    """
    Finds the polarity column robustly.
    Preferred: 'polarity'
    Fallbacks: 'Polarity', 'polarity_score'
    """
    candidates = ["polarity", "Polarity", "polarity_score", "PolarityScore"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_numeric(series: pd.Series) -> pd.Series:
    """Convert to numeric safely; NaNs -> 0."""
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def aggregate_one_day_file(path: Path, channels: List[str]) -> Optional[Dict]:
    """
    Reads one daily file and returns a dict of:
      - date
      - n_rows (headlines in file)
      - mass_<ch>
      - num_<ch>
      - z_<ch>
      - low_evidence_<ch>  (1 if mass<threshold else 0)

    Returns None if file is unusable (no date, missing polarity, missing p_ cols).
    """
    day = extract_date_from_filename(path)
    if day is None:
        return None

    # size check
    try:
        if path.stat().st_size < MIN_FILE_SIZE_BYTES:
            # treat as missing
            return {"date": day, "_missing_reason": "<10KB"}
    except FileNotFoundError:
        return {"date": day, "_missing_reason": "not_found"}

    df = robust_read_any(path)

    pol_col = find_polarity_column(df)
    if pol_col is None:
        # if polarity isn't present, we cannot compute the index yet
        return {"date": day, "_missing_reason": "no_polarity_column"}

    # Keep only needed cols if present; but be robust
    # Ensure polarity numeric
    polarity = safe_numeric(df[pol_col])

    # Optional: if Title exists, you *can* dedupe here too, but your pipeline already did.
    n_rows = len(df)

    out: Dict = {"date": day, "n_rows": n_rows}

    # ----------------------------
    # ⭐⭐ THIS IS THE AGGREGATION LOGIC ⭐⭐
    #
    # mass_day(c) = Σ_h relevance(h,c)
    # numerator_day(c) = Σ_h relevance(h,c) * polarity(h)
    # Z_day(c) = numerator_day(c) / mass_day(c)      (weighted avg)
    #
    # Low evidence rule:
    # if mass_day(c) < threshold(c)  => Z_day(c) = NaN
    # ----------------------------
    for ch in channels:
        pcol = f"p_{ch}"
        if pcol not in df.columns:
            # If a channel column is missing, treat as missing for that day
            out[f"mass_{ch}"] = np.nan
            out[f"num_{ch}"] = np.nan
            out[f"z_{ch}"] = np.nan
            out[f"low_evidence_{ch}"] = 1
            continue

        relevance = safe_numeric(df[pcol])
        mass = float(relevance.sum())
        num = float((relevance * polarity).sum())

        out[f"mass_{ch}"] = mass
        out[f"num_{ch}"] = num

        thr = MASS_THRESHOLDS[ch]
        if mass < thr or mass <= 0.0:
            out[f"z_{ch}"] = np.nan
            out[f"low_evidence_{ch}"] = 1
        else:
            out[f"z_{ch}"] = num / mass
            out[f"low_evidence_{ch}"] = 0

    return out


def build_full_calendar(df_days: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Ensures every calendar day exists as a row (missing days remain NaN).
    """
    cal = pd.DataFrame({"date": pd.date_range(start=start, end=end, freq="D")})
    merged = cal.merge(df_days, on="date", how="left")
    return merged


def main():
    channels = list(MASS_THRESHOLDS.keys())

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"INPUT_ROOT not found: {INPUT_ROOT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Scan for both csv and excel
    files = sorted([p for p in INPUT_ROOT.rglob("*") if p.suffix.lower() in [".csv", ".xlsx", ".xls"]])

    rows: List[Dict] = []
    for p in files:
        d = aggregate_one_day_file(p, channels)
        if d is not None:
            rows.append(d)

    if not rows:
        raise RuntimeError("No usable daily files found (no dates could be parsed).")

    df = pd.DataFrame(rows)

    # Some dates may appear multiple times if duplicates exist; keep the one with the most data.
    # Heuristic: prefer rows with larger n_rows and not missing.
    df["_missing"] = df.get("_missing_reason").notna()
    df["n_rows"] = pd.to_numeric(df.get("n_rows"), errors="coerce").fillna(0)

    df = (
        df.sort_values(["date", "_missing", "n_rows"], ascending=[True, True, False])
          .drop_duplicates(subset=["date"], keep="first")
          .sort_values("date")
          .reset_index(drop=True)
    )
    df = df.drop(columns=["_missing"], errors="ignore")

    # Build splits (calendar-complete)
    train = build_full_calendar(df, "2017-01-01", "2023-12-31")
    test = build_full_calendar(df, "2024-01-01", "2025-12-31")
    aux  = build_full_calendar(df, "2026-01-01", "2026-12-31")

    # Write
    train_path = OUTPUT_DIR / OUT_TRAIN
    test_path  = OUTPUT_DIR / OUT_TEST
    aux_path   = OUTPUT_DIR / OUT_AUX

    train.to_csv(train_path, index=False, encoding="utf-8-sig")
    test.to_csv(test_path, index=False, encoding="utf-8-sig")
    aux.to_csv(aux_path, index=False, encoding="utf-8-sig")

    print("\n=== DONE ===")
    print(f"Scanned files: {len(files)}")
    print(f"Unique dates aggregated: {df['date'].nunique()}")
    print(f"Wrote:\n  {train_path}\n  {test_path}\n  {aux_path}\n")
    print("Note: z_<channel> is NaN if low evidence (mass<threshold) or missing file/columns.")
    print("You can impute missing/NaN Z values AFTER this step (as you planned).")


if __name__ == "__main__":
    main()
