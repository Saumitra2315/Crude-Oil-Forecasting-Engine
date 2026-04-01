from __future__ import annotations


import os

import numpy as np

import pandas as pd


TRAIN_PATH = r"D:\ACADS\4-2\FIN SOP\Data\Code\Models\Price_train_17-23.csv"

TEST_PATH  = r"D:\ACADS\4-2\FIN SOP\Data\Code\Models\Price_test_24-25.csv"


OUTDIR = r"D:\ACADS\4-2\FIN SOP\Data\Code\Models\targets_out"

OUTPUT_FORMAT = "csv"


DATE_COL  = "Date"

OPEN_COL  = "Open"

HIGH_COL  = "High"

LOW_COL   = "Low"

CLOSE_COL = "Close"


M_LOOKBACK = 60

K_THRESHOLD = 3.0


def load_table(path: str) -> pd.DataFrame:

    p = path.lower()

    if p.endswith(".csv"):

        return pd.read_csv(path)

    if p.endswith(".xlsx") or p.endswith(".xls"):

        return pd.read_excel(path)

    raise ValueError(f"Unsupported file type: {path} (use .csv/.xlsx/.xls)")


def save_table(df: pd.DataFrame, path: str) -> None:

    p = path.lower()

    if p.endswith(".csv"):

        df.to_csv(path, index=False)

    elif p.endswith(".xlsx") or p.endswith(".xls"):

        df.to_excel(path, index=False)

    else:

        raise ValueError(f"Unsupported output type: {path} (use .csv/.xlsx/.xls)")


def add_targets_independent(df: pd.DataFrame) -> pd.DataFrame:

    out = df.copy()


    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")

    out = out.sort_values(DATE_COL).reset_index(drop=True)


    for c in [OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]:

        out[c] = pd.to_numeric(out[c], errors="coerce")


    close_pos = out[CLOSE_COL].where(out[CLOSE_COL] > 0)

    out["target_return"] = np.log(close_pos) - np.log(close_pos.shift(1))


    high_pos = out[HIGH_COL].where(out[HIGH_COL] > 0)

    low_pos  = out[LOW_COL].where(out[LOW_COL] > 0)


    ln_hl = np.log(high_pos / low_pos)

    sigma2_p = (1.0 / (4.0 * np.log(2.0))) * (ln_hl ** 2)

    out["parkinson_var"] = sigma2_p

    out["target_vol_parkinson"] = np.sqrt(sigma2_p)


    sigmahat_pre = np.sqrt(
        out["parkinson_var"]
        .rolling(window=M_LOOKBACK, min_periods=M_LOOKBACK)
        .mean()
        .shift(1)
    )

    out["sigmahat_pre_m"] = sigmahat_pre


    abs_logret = np.abs(out["target_return"])


    out["target_jump_flag"] = np.where(
        (abs_logret.notna()) & (sigmahat_pre.notna()) & (abs_logret > (K_THRESHOLD * sigmahat_pre)),
        1,
        np.where((abs_logret.notna()) & (sigmahat_pre.notna()), 0, np.nan)
    )


    return out


def main():

    os.makedirs(OUTDIR, exist_ok=True)


    train_df = load_table(TRAIN_PATH)

    test_df  = load_table(TEST_PATH)


    train_out = add_targets_independent(train_df)

    test_out  = add_targets_independent(test_df)


    train_out_path = os.path.join(OUTDIR, f"train_with_targets.{OUTPUT_FORMAT}")

    test_out_path  = os.path.join(OUTDIR, f"test_with_targets.{OUTPUT_FORMAT}")


    save_table(train_out, train_out_path)

    save_table(test_out, test_out_path)


    print("[OK] Saved:", train_out_path)

    print("[OK] Saved:", test_out_path)


    print("\nTrain NaNs in targets:")

    print(train_out[["target_return", "target_vol_parkinson", "target_jump_flag"]].isna().sum())


    print("\nTest NaNs in targets:")

    print(test_out[["target_return", "target_vol_parkinson", "target_jump_flag"]].isna().sum())


if __name__ == "__main__":

    main()
