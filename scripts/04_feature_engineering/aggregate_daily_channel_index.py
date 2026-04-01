from __future__ import annotations


import re

from pathlib import Path

from typing import Dict, Optional, List


import numpy as np

import pandas as pd


INPUT_ROOT = Path(r"D:\ACADS\4-2\FIN SOP\Data\Code\Relevance\Final_Polarity_Relevance\Final_Polarity_Relevance")


OUTPUT_DIR = Path(r"D:\ACADS\4-2\FIN SOP\Data\Code\Relevance\Final_Polarity_Relevance\Final_Polarity_Relevance\Daily Channel Index")


MASS_THRESHOLDS: Dict[str, float] = {
    "Supply Shock and availability risk": 7.195134,
    "Transport logistics and chokepoints": 1.445090,
    "Demand and macro economy": 8.378556,
    "Inventory SPR and refinery": 3.497590,
    "OPEC producer policy": 1.054716,
    "Geopolitical Normalisation and Peace": 0.596221,
}


MIN_FILE_SIZE_BYTES = 10 * 1024


OUT_TRAIN = "train_2017_2023_daily_channel_index.csv"

OUT_TEST = "test_2024_2025_daily_channel_index.csv"

OUT_AUX = "aux_2026_daily_channel_index.csv"


DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def robust_read_any(path: Path) -> pd.DataFrame:

    """
    Reads CSV/XLSX robustly.
    - CSV: tries delimiter sniffing (sep=None) + skips bad lines
    - XLSX: standard read_excel
    """

    if path.suffix.lower() in [".xlsx", ".xls"]:

        return pd.read_excel(path)


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


    try:

        if path.stat().st_size < MIN_FILE_SIZE_BYTES:

            return {"date": day, "_missing_reason": "<10KB"}

    except FileNotFoundError:

        return {"date": day, "_missing_reason": "not_found"}


    df = robust_read_any(path)


    pol_col = find_polarity_column(df)

    if pol_col is None:

        return {"date": day, "_missing_reason": "no_polarity_column"}


    polarity = safe_numeric(df[pol_col])


    n_rows = len(df)


    out: Dict = {"date": day, "n_rows": n_rows}


    for ch in channels:

        pcol = f"p_{ch}"

        if pcol not in df.columns:

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


    files = sorted([p for p in INPUT_ROOT.rglob("*") if p.suffix.lower() in [".csv", ".xlsx", ".xls"]])


    rows: List[Dict] = []

    for p in files:

        d = aggregate_one_day_file(p, channels)

        if d is not None:

            rows.append(d)


    if not rows:

        raise RuntimeError("No usable daily files found (no dates could be parsed).")


    df = pd.DataFrame(rows)


    df["_missing"] = df.get("_missing_reason").notna()

    df["n_rows"] = pd.to_numeric(df.get("n_rows"), errors="coerce").fillna(0)


    df = (
        df.sort_values(["date", "_missing", "n_rows"], ascending=[True, True, False])
          .drop_duplicates(subset=["date"], keep="first")
          .sort_values("date")
          .reset_index(drop=True)
    )

    df = df.drop(columns=["_missing"], errors="ignore")


    train = build_full_calendar(df, "2017-01-01", "2023-12-31")

    test = build_full_calendar(df, "2024-01-01", "2025-12-31")

    aux  = build_full_calendar(df, "2026-01-01", "2026-12-31")


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
