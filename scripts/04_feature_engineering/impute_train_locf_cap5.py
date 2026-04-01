import pandas as pd

import numpy as np

from pathlib import Path


INPUT_PATH = Path(r"D:\ACADS\4-2\FIN SOP\Data\Code\Relevance\Final_Polarity_Relevance\Final_Polarity_Relevance\Daily Channel Index\train_2017_2023_daily_channel_index.csv")


OUTPUT_PATH = INPUT_PATH.with_name("train_2017_2023_daily_channel_index_imputed_locf5.csv")


CAP_DAYS = 5


def locf_with_cap(series: pd.Series, cap: int) -> pd.Series:

    """
    Forward Fill (LOCF) with a strict cap.
    - Fill NaNs with last valid value up to 'cap' consecutive days.
    - If NaN streak exceeds cap, fill the remaining NaNs in that streak with 0.0.

    Assumes series is ordered by date at daily frequency.
    """

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


def main():

    df = pd.read_csv(INPUT_PATH)


    if "date" not in df.columns:

        raise ValueError("Expected a 'date' column in the input file.")


    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)


    low_cols = [c for c in df.columns if c.startswith("low_evidence_")]

    if not low_cols:

        raise ValueError("No low_evidence_* columns found. Cannot apply threshold->0 rule.")


    channels = [c.replace("low_evidence_", "") for c in low_cols]


    for ch in channels:

        zcol = f"z_{ch}"

        lecol = f"low_evidence_{ch}"


        if zcol not in df.columns:

            continue


        df[zcol] = pd.to_numeric(df[zcol], errors="coerce")

        df[lecol] = pd.to_numeric(df[lecol], errors="coerce").fillna(0).astype(int)


        mask = (df[lecol] == 1) & (df[zcol].isna())

        df.loc[mask, zcol] = 0.0


    for ch in channels:

        zcol = f"z_{ch}"

        if zcol not in df.columns:

            continue


        df[zcol] = locf_with_cap(df[zcol], CAP_DAYS)


    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Done.\nInput:  {INPUT_PATH}\nOutput: {OUTPUT_PATH}")


if __name__ == "__main__":

    main()
