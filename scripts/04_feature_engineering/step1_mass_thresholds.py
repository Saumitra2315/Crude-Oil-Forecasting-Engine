from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd


# ============================================================
# ✅✅✅ EDIT ONLY THIS SECTION ✅✅✅
# ============================================================

# Root folder that contains your daily output files (year -> month -> day files)
# Example (Windows):
# OUTPUT_ROOT = Path(r"D:\ACADS\4-2\FIN SOP\Data\Code\Relevance\Final_Polarity_Relevance\Final_Polarity_Relevance")
#
# Example (Linux):
# OUTPUT_ROOT = Path(r"/home/user/crudeoil-data-mnli/Relevance Scores/Final Relevance Scores")

OUTPUT_ROOT = Path(r"D:\ACADS\4-2\FIN SOP\Data\Code\Relevance\Final_Polarity_Relevance\Final_Polarity_Relevance")

# Consider a file "missing/bad" if it exists but is smaller than this (KB)
MIN_FILE_SIZE_KB = 10

# Train/Test split (used only for threshold computation summary)
TRAIN_YEARS = list(range(2017, 2024))  # 2017..2023
ALL_YEARS = list(range(2017, 2026))    # adjust if you want 2026 included

# 5th percentile threshold
PERCENTILE = 0.05

# Your EXACT channel names (must match how you wrote p_<channel> in the pipeline output)
CHANNELS = [
    "Supply Shock and availability risk",
    "Transport logistics and chokepoints",
    "Demand and macro economy",
    "Inventory SPR and refinery",
    "OPEC producer policy",
    "Geopolitical Normalisation and Peace",
]

# Output report files (will be created inside OUTPUT_ROOT)
OUT_DAILY_MASS_CSV = OUTPUT_ROOT / "step1_daily_masses.csv"
OUT_THRESHOLDS_CSV = OUTPUT_ROOT / "step1_thresholds_summary.csv"

# ============================================================


# ---------- Helpers ----------

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
        # openpyxl is the standard engine for xlsx
        return pd.read_excel(p, engine="openpyxl")
    elif suf == ".csv":
        # try UTF-8 first, fallback latin-1
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
    # Expect .../<year>/<month>/<file>
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
            # sum ignoring NaNs
            out[ch] = float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())
        else:
            # column missing => mark as NaN (so you can detect issues)
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


# ---------- Main Step-1 Logic ----------

def main():
    if not OUTPUT_ROOT.exists():
        raise FileNotFoundError(f"OUTPUT_ROOT not found: {OUTPUT_ROOT}")

    # Collect all daily output files across years/months
    # We match both xlsx and csv.
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
    bad_files: List[Tuple[str, str]] = []  # (date_str, reason)

    # Progress timer per month (so you’re “intimated” when each month is done)
    current_month = None
    month_start_time = None

    for p in candidates:
        day = parse_date_from_filename(p)
        if day is None:
            # If file doesn't match expected pattern, skip
            continue

        # Month progress handling
        mk = month_key_from_path(p)
        if mk != current_month:
            # Close previous month timer
            if current_month is not None and month_start_time is not None:
                elapsed = time.time() - month_start_time
                print(f"[DONE] Month {current_month} finished in {elapsed:.1f} sec")
            # Start new month timer
            current_month = mk
            month_start_time = time.time()
            print(f"\n[START] Processing Month {current_month} ...")

        # Skip tiny files (treat them as "missing/bad" like your missing report)
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
        # Add mass columns
        for ch in CHANNELS:
            rec[f"mass_{ch}"] = masses[ch]
        records.append(rec)

    # Close last month timer
    if current_month is not None and month_start_time is not None:
        elapsed = time.time() - month_start_time
        print(f"[DONE] Month {current_month} finished in {elapsed:.1f} sec")

    if not records:
        raise RuntimeError("No usable daily files were processed (all missing/too small/read errors).")

    # Create daily mass DataFrame
    mass_df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

    # Save daily masses
    mass_df.to_csv(OUT_DAILY_MASS_CSV, index=False)
    print(f"\nSaved daily mass table: {OUT_DAILY_MASS_CSV}")
    print("Daily mass table columns explaination:")
    print("  mass_<channel> = Σ p_<channel> across all headlines in that day-file")

    # ---- Threshold computation (5th percentile) ----
    # We compute it two ways:
    # 1) Train-only (2017–2023) => recommended to avoid leakage
    # 2) All years => just for comparison/debug
    train_df = mass_df[mass_df["year"].isin(TRAIN_YEARS)].copy()
    all_df = mass_df[mass_df["year"].isin(ALL_YEARS)].copy()

    summary_rows: List[Dict] = []
    for ch in CHANNELS:
        col = f"mass_{ch}"

        # Train stats
        train_series = train_df[col]
        train_q05 = safe_quantile(train_series, PERCENTILE)
        train_n = int(train_series.dropna().shape[0])
        train_min = float(train_series.dropna().min()) if train_n > 0 else float("nan")
        train_med = float(train_series.dropna().median()) if train_n > 0 else float("nan")
        train_mean = float(train_series.dropna().mean()) if train_n > 0 else float("nan")

        # All stats
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
            "recommended_threshold_to_use": train_q05,  # recommend train threshold
        })

    thresholds_df = pd.DataFrame(summary_rows)

    # Save threshold summary
    thresholds_df.to_csv(OUT_THRESHOLDS_CSV, index=False)
    print(f"\nSaved threshold summary: {OUT_THRESHOLDS_CSV}")

    # Print thresholds nicely
    print("\n===== Channel-wise 5th percentile mass thresholds (recommended: TRAIN 2017–2023) =====")
    for _, r in thresholds_df.iterrows():
        print(f"- {r['channel']}: threshold = {r['train_mass_5th_percentile_threshold']:.6f} "
              f"(train_days_used={r['train_days_used']})")

    # Show bad files list (if any)
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
