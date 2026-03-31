from __future__ import annotations

import re
import csv
from time import perf_counter
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from datasets import Dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset


# ============================================================
# ✅✅✅ EDIT ONLY THIS SECTION ✅✅✅
# ============================================================

INPUT_ROOT = Path(r"/home/user/crudeoil-data-mnli/data final headlines")
OUTPUT_ROOT = Path(r"/home/user/crudeoil-data-mnli/Relevance Scores/Final Relevance Scores")

YEARS = list(range(2017, 2027))  # 2017..2026 inclusive
OUTPUT_SUFFIX = "_nli_probs"     # output filename: <stem>_nli_probs.csv

MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-zeroshot-v2.0"
BATCH_SIZE = 32  # Lab GPU: 32/64 usually fine (RTX 4000 Ada). Start with 32.

# If False: keep irrelevant rows but set all channel probs = 0
# If True: drop irrelevant rows entirely
DROP_IRRELEVANT = False

# Relevance labels (binary softmax)  ✅ (UNCHANGED)
RELEVANCE_LABELS = [
    "Commodity/Energy markets and macroeconomy (oil, crude, energy, inflation, rates, inventories, OPEC, shipping, sanctions)",
    "Unrelated/general news (sports, entertainment, local crime, lifestyle, real estate, etc.)",
]

# Stage-2 Channel hypotheses ✅ (UNCHANGED)
CHANNEL_HYPOTHESES: Dict[str, str] = {
    "Supply Shock and availability risk":
        "This headline indicates a disruption or reduction or increase in physical crude oil supply (production or exports).",

    "Transport logistics and chokepoints":
        "This headline indicates shipping, logistics, or chokepoint related disruption or recovery that affects oil flows (routes, ports, canals).",

    "Demand and macro economy":
        "This headline indicates demand-side or macroeconomic forces that affect oil consumption or prices (growth, inflation, rates, tariff).",

    "Inventory SPR and refinery":
        "This headline indicates inventories, storage, SPR actions, or refinery operations affecting crude/products availability.",

    "OPEC producer policy":
        "This headline indicates OPEC+ or producer policy decisions affecting oil supply (quotas, cuts, compliance, meetings).",

    "Geopolitical Normalisation and Peace":
        "This headline indicates de-escalation, ceasefire, sanctions relief, or normalization that reduces oil risk premium.",
}

CLEAN_CHANNELS = list(CHANNEL_HYPOTHESES.keys())
STAGE2_LABELS = list(CHANNEL_HYPOTHESES.values())
LABEL_TO_CHANNEL = {v: k for k, v in CHANNEL_HYPOTHESES.items()}

RELEVANCE_CUTOFF = 0.50

ENABLE_SALIENCE_WEIGHTS = False  # as per your final setting

# Word filter: keep only headlines with >= 5 word-like tokens
MIN_WORDS_TO_KEEP = 5

# Force GPU-only (no CPU fallback)
REQUIRE_CUDA = True

# ============================================================


def setup_model():
    if REQUIRE_CUDA and (not torch.cuda.is_available()):
        raise RuntimeError(
            "CUDA is not available but REQUIRE_CUDA=True. "
            "Make sure you're running on the lab GPU node and using CUDA_VISIBLE_DEVICES=0."
        )

    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading model: {MODEL_NAME}")
    print("GPU:" if device == 0 else "CPU:", torch.cuda.get_device_name(0) if device == 0 else "No CUDA")

    return pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        device=device,
        torch_dtype=torch.float16 if device == 0 else torch.float32,
        batch_size=BATCH_SIZE,
    )


def robust_read_csv(path: Path) -> pd.DataFrame:
    """Reads CSV even when delimiter/quoting is messy."""
    try:
        return pd.read_csv(path, engine="python", sep=None, on_bad_lines="skip")
    except UnicodeDecodeError:
        return pd.read_csv(path, engine="python", sep=None, on_bad_lines="skip", encoding="latin-1")


def normalize_title(s: str) -> str:
    s = str(s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def count_words(s: str) -> int:
    """
    Counts word-like tokens (letters/digits). Ignores punctuation-only tokens like '-', '.', etc.
    """
    tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(s))
    return len(tokens)


def scores_to_dict(out: Dict) -> Dict[str, float]:
    """HF pipeline returns labels sorted desc; convert to dict label->score."""
    return {lbl: float(sc) for lbl, sc in zip(out["labels"], out["scores"])}


def write_csv_safe(df: pd.DataFrame, output_csv: Path) -> None:
    """
    ✅ Fix for ';' splitting:
    Force quoting on ALL fields so delimiters inside Title never break columns.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL,     # <-- key fix
        escapechar="\\",
        doublequote=True,
        lineterminator="\n",
    )


def process_one_file(input_csv: Path, output_csv: Path, clf) -> Optional[str]:
    """
    Returns None if success, else error string.
    """
    try:
        df = robust_read_csv(input_csv)

        if "Title" not in df.columns:
            return "Missing 'Title' column"

        # Basic cleaning
        df = df.dropna(subset=["Title"]).copy()
        df["Title"] = df["Title"].astype(str)

        # Word-count filter (robust)
        df["_word_count"] = df["Title"].map(count_words)
        df = df[df["_word_count"] >= MIN_WORDS_TO_KEEP].copy()

        # Deduplicate (robust)
        df["_title_norm"] = df["Title"].map(normalize_title)
        df = df.drop_duplicates(subset=["_title_norm"], keep="first").copy()

        if df.empty:
            # Write empty file with headers so day is still "done"
            df = df.drop(columns=["_title_norm", "_word_count"], errors="ignore")
            write_csv_safe(df, output_csv)
            return None

        hf = Dataset.from_pandas(df)

        # ----------------------------
        # Stage 1: relevance probability (softmax)
        # ----------------------------
        relevance_probs: List[float] = []
        for out in clf(
            KeyDataset(hf, "Title"),
            candidate_labels=RELEVANCE_LABELS,
            multi_label=False,
            batch_size=BATCH_SIZE,
        ):
            d = scores_to_dict(out)
            relevance_probs.append(d.get(RELEVANCE_LABELS[0], 0.0))

        df["prob_relevant"] = relevance_probs

        # Decide which rows are relevant (optional)
        is_relevant = df["prob_relevant"] >= RELEVANCE_CUTOFF
        if DROP_IRRELEVANT:
            df = df[is_relevant].copy()
            if df.empty:
                df = df.drop(columns=["_title_norm", "_word_count"], errors="ignore")
                write_csv_safe(df, output_csv)
                return None
            hf2 = Dataset.from_pandas(df)
        else:
            hf2 = hf  # keep all; later we’ll zero-out channel probs for irrelevant rows

        # ----------------------------
        # Stage 2: channel probabilities (multi-label, raw scores)
        # ----------------------------
        for ch in CLEAN_CHANNELS:
            df[f"p_{ch}"] = 0.0

        idx_map = df.index.to_list()

        channel_outs = clf(
            KeyDataset(hf2, "Title"),
            candidate_labels=STAGE2_LABELS,
            multi_label=True,
            batch_size=BATCH_SIZE,
        )

        for i, out in enumerate(channel_outs):
            score_map = scores_to_dict(out)
            row_ix = idx_map[i]

            # If keeping irrelevant: zero out probabilities when relevance is low
            if (not DROP_IRRELEVANT) and (df.loc[row_ix, "prob_relevant"] < RELEVANCE_CUTOFF):
                continue

            raw_by_channel: Dict[str, float] = {ch: 0.0 for ch in CLEAN_CHANNELS}
            for lbl, sc in score_map.items():
                ch = LABEL_TO_CHANNEL.get(lbl)
                if ch is not None:
                    raw_by_channel[ch] = float(sc)

            for ch in CLEAN_CHANNELS:
                df.loc[row_ix, f"p_{ch}"] = raw_by_channel.get(ch, 0.0)

        # Cleanup helper columns
        df = df.drop(columns=["_title_norm", "_word_count"], errors="ignore")

        # Save (✅ safe writer)
        write_csv_safe(df, output_csv)
        return None

    except Exception as e:
        return str(e)


def main():
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Input root not found: {INPUT_ROOT}")

    clf = setup_model()

    total = 0
    skipped = 0
    ok = 0
    failed = 0

    for year in YEARS:
        year_dir = INPUT_ROOT / str(year)
        if not year_dir.exists():
            print(f"[WARN] Missing year folder: {year_dir}")
            continue

        month_dirs = sorted([p for p in year_dir.iterdir() if p.is_dir()])
        if not month_dirs:
            print(f"[WARN] No month folders under: {year_dir}")
            continue

        print(f"\n=== Year {year}: found {len(month_dirs)} month folders ===")

        for month_dir in month_dirs:
            month_start = perf_counter()

            csv_files = sorted(month_dir.glob("*.csv"))
            if not csv_files:
                print(f"[WARN] No daily CSVs under: {month_dir}")
                continue

            for input_csv in csv_files:
                total += 1

                rel = input_csv.relative_to(INPUT_ROOT)  # <year>/<month>/<file>.csv
                out_dir = OUTPUT_ROOT / rel.parent
                out_name = f"{input_csv.stem}{OUTPUT_SUFFIX}.csv"
                output_csv = out_dir / out_name

                # Resume-safe
                if output_csv.exists():
                    skipped += 1
                    continue

                err = process_one_file(input_csv, output_csv, clf)
                if err is None:
                    ok += 1
                else:
                    failed += 1
                    print(f"[FAIL] {rel} -> {err}")

            elapsed = perf_counter() - month_start
            print(f"[MONTH DONE] {year}/{month_dir.name}  |  elapsed: {elapsed:.1f}s")

        print(f"Year {year} done. OK={ok}, Failed={failed}, Skipped={skipped}, TotalSeen={total}")

    print("\n=== ALL DONE ===")
    print(f"Total seen:   {total}")
    print(f"Processed OK: {ok}")
    print(f"Skipped:      {skipped}")
    print(f"Failed:       {failed}")
    print(f"Output root:  {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
