from __future__ import annotations

import pandas as pd
import numpy as np

# ============================================================
# ✅✅✅ EDIT ONLY THIS BLOCK ✅✅✅
# ============================================================

PRICE_PATH = r"D:\ACADS\4-2\FIN SOP\Data\Code\Models\targets_out\test_return.csv"   # price/returns file
SENT_PATH  = r"D:\ACADS\4-2\FIN SOP\Data\Code\Models\GSent_test_24-25.csv"  # sentiment file
OUT_PATH   = r"D:\ACADS\4-2\FIN SOP\Data\Code\Models\targets_out\test_return_with_sent_tminus1.csv"

# Column names
PRICE_DATE_COL = "Date"     # in price file
SENT_DATE_COL  = "date"     # in sentiment file
SENT_VALUE_COL = "mean_polarity" # in sentiment file (your daily sentiment score)

# New column name to create in price df
NEW_COL_NAME = "sentiment_t_minus_1"

# ============================================================
# (Do not edit below unless changing logic)
# ============================================================

def _read_any(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".csv"):
        return pd.read_csv(path)
    if p.endswith(".xlsx") or p.endswith(".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")

def _parse_date_series(s: pd.Series) -> pd.Series:
    """
    Robust date parsing:
    - strips whitespace
    - tries dayfirst parsing
    - falls back to generic parsing
    """
    s = s.astype(str).str.strip()

    # first try: dayfirst (handles 24-11-2023)
    dt1 = pd.to_datetime(s, errors="coerce", dayfirst=True)

    # fallback: generic parse for any leftovers
    if dt1.isna().any():
        dt2 = pd.to_datetime(s, errors="coerce")
        dt1 = dt1.fillna(dt2)

    return dt1

def main():
    price = _read_any(PRICE_PATH)
    sent = _read_any(SENT_PATH)

    # strip column names (common Excel/csv issue)
    price.columns = price.columns.astype(str).str.strip()
    sent.columns  = sent.columns.astype(str).str.strip()

    # validate columns exist
    for col in [PRICE_DATE_COL]:
        if col not in price.columns:
            raise ValueError(f"PRICE file missing column: {col}. Found: {price.columns.tolist()}")

    for col in [SENT_DATE_COL, SENT_VALUE_COL]:
        if col not in sent.columns:
            raise ValueError(f"SENT file missing column: {col}. Found: {sent.columns.tolist()}")

    # parse dates robustly
    price["_dt"] = _parse_date_series(price[PRICE_DATE_COL])
    sent["_dt"]  = _parse_date_series(sent[SENT_DATE_COL])

    # drop rows with unparseable dates (print examples)
    bad_price = price[price["_dt"].isna()]
    bad_sent = sent[sent["_dt"].isna()]

    if len(bad_price) > 0:
        print(f"[WARN] Unparsed PRICE dates: {len(bad_price)}")
        print(bad_price[[PRICE_DATE_COL]].head(10).to_string(index=False))

    if len(bad_sent) > 0:
        print(f"[WARN] Unparsed SENT dates: {len(bad_sent)}")
        print(bad_sent[[SENT_DATE_COL]].head(10).to_string(index=False))

    price = price.dropna(subset=["_dt"]).copy()
    sent  = sent.dropna(subset=["_dt"]).copy()

    # ensure sentiment numeric
    sent[SENT_VALUE_COL] = pd.to_numeric(sent[SENT_VALUE_COL], errors="coerce")

    # create lookup dict: date -> sentiment
    # (if duplicates exist, take last non-NaN)
    sent = sent.sort_values("_dt")
    sent_map = sent.dropna(subset=[SENT_VALUE_COL]).drop_duplicates("_dt", keep="last").set_index("_dt")[SENT_VALUE_COL]

    # compute t-1 calendar day
    price["_dt_tminus1"] = price["_dt"] - pd.Timedelta(days=1)

    # map sentiment to price dates (t-1); missing -> 0
    price[NEW_COL_NAME] = price["_dt_tminus1"].map(sent_map).fillna(0.0).astype(float)

    # cleanup helper cols
    price = price.drop(columns=["_dt", "_dt_tminus1"])

    # write output
    if OUT_PATH.lower().endswith(".csv"):
        price.to_csv(OUT_PATH, index=False)
    else:
        price.to_excel(OUT_PATH, index=False)

    print(f"[OK] Wrote: {OUT_PATH}")
    print(f"[OK] Added column: {NEW_COL_NAME}")

if __name__ == "__main__":
    main()
