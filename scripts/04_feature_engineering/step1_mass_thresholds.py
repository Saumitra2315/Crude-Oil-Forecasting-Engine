from __future__ import annotations


import re

import time

from pathlib import Path

from typing import Dict, List, Tuple, Optional


import pandas as pd


OUTPUT_ROOT = Path(r"D:\ACADS\4-2\FIN SOP\Data\Code\Relevance\Final_Polarity_Relevance\Final_Polarity_Relevance")


MIN_FILE_SIZE_KB = 10


TRAIN_YEARS = list(range(2017, 2024))

ALL_YEARS = list(range(2017, 2026))


PERCENTILE = 0.05


CHANNELS = [
    "Supply Shock and availability risk",
    "Transport logistics and chokepoints",
    "Demand and macro economy",
    "Inventory SPR and refinery",
    "OPEC producer policy",
    "Geopolitical Normalisation and Peace",
]


OUT_DAILY_MASS_CSV = OUTPUT_ROOT / "step1_daily_masses.csv"

OUT_THRESHOLDS_CSV = OUTPUT_ROOT / "step1_thresholds_summary.csv"


DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_nli_probs", re.IGNORECASE)


def parse_date_from_filename(p: Path) -> Optional[pd.Timestamp]:

    """
    Extracts date from filenames like: 2017-01-01_nli_probs.xlsx / .csv
    Returns pandas Timestamp or None if not matched.
    """

    m = DATE_RE.search(p.name)

    if not m:

        return None

    try:

        return pd.to_datetime(m.group(1), format="%Y-%m-%d")

    except Exception:

        return None


def is_too_small(p: Path, min_kb: int) -> bool:

    return p.exists() and (p.stat().st_size < min_kb * 1024)


def read_daily_file(p: Path) -> pd.DataFrame:

    """
    Reads either .xlsx or .csv robustly.
    We only need the p_<channel> columns, so we keep it simple and safe.
    """

    suf = p.suffix.lower()

    if suf in [".xlsx", ".xls"]:

        return pd.read_excel(p, engine="openpyxl")

    elif suf == ".csv":

        try:

            return pd.read_csv(p, encoding="utf-8")

        except UnicodeDecodeError:

            return pd.read_csv(p, encoding="latin-1")

    else:

        raise ValueError(f"Unsupported file type: {p.suffix}")


def month_key_from_path(p: Path) -> str:

    """
    Returns something like "2017/01_Jan" from path:
      .../2017/01_Jan/2017-01-01_nli_probs.xlsx
    Useful for progress reporting.
    """

    try:

        rel = p.relative_to(OUTPUT_ROOT)

        if len(rel.parts) >= 2:

            return f"{rel.parts[0]}/{rel.parts[1]}"

    except Exception:

        pass

    return "UNKNOWN_MONTH"


def compute_mass_for_day(df: pd.DataFrame) -> Dict[str, float]:

    """
    mass_day(c) = sum over headlines of relevance(h,c)
    In your output files, relevance(h,c) is p_<channel>.
    """

    out: Dict[str, float] = {}

    for ch in CHANNELS:

        col = f"p_{ch}"

        if col in df.columns:

            out[ch] = float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())

        else:

            out[ch] = float("nan")

    return out


def safe_quantile(series: pd.Series, q: float) -> float:

    """
    Quantile that ignores NaNs and handles empty series.
    """

    s = series.dropna()

    if s.empty:

        return float("nan")

    return float(s.quantile(q))


def main():

    if not OUTPUT_ROOT.exists():

        raise FileNotFoundError(f"OUTPUT_ROOT not found: {OUTPUT_ROOT}")


    candidates: List[Path] = []

    for y in ALL_YEARS:

        yd = OUTPUT_ROOT / str(y)

        if yd.exists():

            candidates.extend(sorted(yd.rglob("*_nli_probs.xlsx")))

            candidates.extend(sorted(yd.rglob("*_nli_probs.csv")))


    if not candidates:

        raise RuntimeError(f"No daily *_nli_probs.(xlsx/csv) files found under: {OUTPUT_ROOT}")


    print(f"Found {len(candidates)} candidate daily files under OUTPUT_ROOT.")


    records: List[Dict] = []

    bad_files: List[Tuple[str, str]] = []


    current_month = None

    month_start_time = None


    for p in candidates:

        day = parse_date_from_filename(p)

        if day is None:

            continue


        mk = month_key_from_path(p)

        if mk != current_month:

            if current_month is not None and month_start_time is not None:

                elapsed = time.time() - month_start_time

                print(f"[DONE] Month {current_month} finished in {elapsed:.1f} sec")

            current_month = mk

            month_start_time = time.time()

            print(f"\n[START] Processing Month {current_month} ...")


        if is_too_small(p, MIN_FILE_SIZE_KB):

            bad_files.append((day.strftime("%Y-%m-%d"), f"<{MIN_FILE_SIZE_KB}KB: {p}"))

            continue


        try:

            df = read_daily_file(p)

        except Exception as e:

            bad_files.append((day.strftime("%Y-%m-%d"), f"read error: {p} -> {e}"))

            continue


        masses = compute_mass_for_day(df)


        rec = {
            "date": day,
            "year": int(day.year),
            "month": int(day.month),
            "file": str(p),
        }

        for ch in CHANNELS:

            rec[f"mass_{ch}"] = masses[ch]

        records.append(rec)


    if current_month is not None and month_start_time is not None:

        elapsed = time.time() - month_start_time

        print(f"[DONE] Month {current_month} finished in {elapsed:.1f} sec")


    if not records:

        raise RuntimeError("No usable daily files were processed (all missing/too small/read errors).")


    mass_df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)


    mass_df.to_csv(OUT_DAILY_MASS_CSV, index=False)

    print(f"\nSaved daily mass table: {OUT_DAILY_MASS_CSV}")

    print("Daily mass table columns explaination:")

    print("  mass_<channel> = Σ p_<channel> across all headlines in that day-file")


    train_df = mass_df[mass_df["year"].isin(TRAIN_YEARS)].copy()

    all_df = mass_df[mass_df["year"].isin(ALL_YEARS)].copy()


    summary_rows: List[Dict] = []

    for ch in CHANNELS:

        col = f"mass_{ch}"


        train_series = train_df[col]

        train_q05 = safe_quantile(train_series, PERCENTILE)

        train_n = int(train_series.dropna().shape[0])

        train_min = float(train_series.dropna().min()) if train_n > 0 else float("nan")

        train_med = float(train_series.dropna().median()) if train_n > 0 else float("nan")

        train_mean = float(train_series.dropna().mean()) if train_n > 0 else float("nan")


        all_series = all_df[col]

        all_q05 = safe_quantile(all_series, PERCENTILE)

        all_n = int(all_series.dropna().shape[0])


        summary_rows.append({
            "channel": ch,
            "definition": "mass_day(c) = Σ Relevance(h,c) = Σ p_<channel>",
            "train_years": "2017-2023",
            "train_days_used": train_n,
            "train_mass_min": train_min,
            "train_mass_median": train_med,
            "train_mass_mean": train_mean,
            "train_mass_5th_percentile_threshold": train_q05,
            "all_years": f"{min(ALL_YEARS)}-{max(ALL_YEARS)}",
            "all_days_used": all_n,
            "all_mass_5th_percentile": all_q05,
            "recommended_threshold_to_use": train_q05,
        })


    thresholds_df = pd.DataFrame(summary_rows)


    thresholds_df.to_csv(OUT_THRESHOLDS_CSV, index=False)

    print(f"\nSaved threshold summary: {OUT_THRESHOLDS_CSV}")


    print("\n===== Channel-wise 5th percentile mass thresholds (recommended: TRAIN 2017–2023) =====")

    for _, r in thresholds_df.iterrows():

        print(f"- {r['channel']}: threshold = {r['train_mass_5th_percentile_threshold']:.6f} "
              f"(train_days_used={r['train_days_used']})")


    if bad_files:

        bad_files_sorted = sorted(bad_files, key=lambda x: x[0])

        print(f"\n[WARN] {len(bad_files_sorted)} files treated as missing/bad (too small or unreadable).")

        print("First 20 examples:")

        for d, reason in bad_files_sorted[:20]:

            print(f"  {d} -> {reason}")


    print("\n=== STEP 1 DONE ===")

    print("Next Step (Step 2) will use the threshold to set Z_day(c)=0 or missing when mass_day(c) is too small.")


if __name__ == "__main__":

    main()
