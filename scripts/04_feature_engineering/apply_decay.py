import numpy as np

import pandas as pd

from pathlib import Path


INPUT_CSV = Path(r"D:\ACADS\4-2\FIN SOP\Data\Code\Relevance\Final_Polarity_Relevance\Final_Polarity_Relevance\Daily Channel Index\train_2017_2023_raw_index.csv")


OUTPUT_CSV = Path(r"D:\ACADS\4-2\FIN SOP\Data\Code\Relevance\Final_Polarity_Relevance\Final_Polarity_Relevance\Daily Channel Index\train_2017_2023_decayed_L10_lambda1_5.csv")


DATE_COL = "date"


L = 10

LAMBDA = 1.5


CHANNEL_COLS_EXACT = [
    "z_Supply Shock and availability risk",
    "z_Transport logistics and chokepoints",
    "z_Demand and macro economy",
    "z_Inventory SPR and refinery",
    "z_OPEC producer policy",
    "z_Geopolitical Normalisation and Peace",
]


def main():

    if not INPUT_CSV.exists():

        raise FileNotFoundError(f"Input not found: {INPUT_CSV}")


    df = pd.read_csv(INPUT_CSV)


    if DATE_COL not in df.columns:

        raise ValueError(f"Missing '{DATE_COL}' column in input file.")


    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")

    df = df.dropna(subset=[DATE_COL]).copy()


    df = df.sort_values(DATE_COL).set_index(DATE_COL)


    if all(c in df.columns for c in CHANNEL_COLS_EXACT):

        target_cols = CHANNEL_COLS_EXACT

    else:

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:

            raise ValueError("No numeric columns found to apply decay on.")

        target_cols = numeric_cols


    for c in target_cols:

        df[c] = pd.to_numeric(df[c], errors="coerce")


    df[target_cols] = df[target_cols].fillna(0.0)


    full_index = pd.date_range(df.index.min(), df.index.max(), freq="D")

    df = df.reindex(full_index)

    df[target_cols] = df[target_cols].fillna(0.0)


    w_raw = np.exp(-np.arange(L) / LAMBDA)


    def decay_roll(window_vals: np.ndarray) -> float:

        n = len(window_vals)

        w = w_raw[:n]

        w = w / w.sum()

        return float(np.dot(window_vals, w[::-1]))


    decayed = df[target_cols].rolling(window=L, min_periods=1).apply(decay_roll, raw=True)


    out = df[target_cols].copy()

    for c in target_cols:

        out[f"{c}__decayed_L{L}_lambda{int(LAMBDA)}"] = decayed[c]


    out = out.reset_index().rename(columns={"index": DATE_COL})

    out[DATE_COL] = out[DATE_COL].dt.strftime("%d-%m-%Y")


    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


    w_norm = w_raw / w_raw.sum()

    print("=== Exponential decay settings ===")

    print(f"L={L}, lambda={LAMBDA}")

    print("First 5 normalized weights (today, yesterday, ...):", np.round(w_norm[:5], 6).tolist())

    print("Last weight (oldest in window):", float(np.round(w_norm[-1], 8)))

    print(f"\nDone. Wrote: {OUTPUT_CSV}")


if __name__ == "__main__":

    main()
