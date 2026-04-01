from __future__ import annotations


import re

from pathlib import Path

from typing import Dict, List


import pandas as pd

import torch

from datasets import Dataset

from transformers import pipeline

from transformers.pipelines.pt_utils import KeyDataset


INPUT_CSV_PATH = Path(r"D:\ACADS\4-2\FIN SOP\Data\headlines data\2025\01_Jan\2025-01-31.csv")


OUTPUT_CSV_PATH = Path(r"D:\ACADS\4-2\FIN SOP\Data\Code\Relevance2025-01-31_nli_probs.csv")


MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-zeroshot-v2.0"

BATCH_SIZE = 16


DROP_IRRELEVANT = False


RELEVANCE_LABELS = [
    "Commodity/Energy markets and macroeconomy (oil, crude, energy, inflation, rates, inventories, OPEC, shipping, sanctions)",
    "Unrelated/general news (sports, entertainment, local crime, lifestyle, real estate, etc.)",
]


channel_mapping = {
    "Supply Shock and availability risk (export curbs, export bans, embargoes, sanctions on energy, oil field shut-ins, upstream outages, production disruption, force majeure, pipeline rupture, pipeline sabotage, drone/missile attack on facilities, terminal closure, port shutdown, refinery fire, unplanned outage, hurricanes in Gulf of Mexico, freeze-offs, wildfires, power outages, civil unrest, war disrupting pipelines, insurgent attacks, supply tightness, capacity constraints, spare capacity decline, decline rates, OSP changes, grade differentials widening)":
        "Supply Shock and availability risk",

    "Transport logistics and chokepoints (Suez Canal, SUMED pipeline, Bab el-Mandeb, Red Sea attacks, Strait of Hormuz, Strait of Malacca, Panama Canal drought restrictions, Turkish Straits/Bosphorus, Danish Straits, shipping route disruption, rerouting via Cape of Good Hope, tanker traffic congestion, port closures, canal closure, vessel grounding, shipping delays, freight rates spike, VLCC/Aframax/Suezmax rates, war-risk premium, piracy, maritime security, insurance costs, sanctions shipping, shadow fleet, tanker seizures, naval escorts, chokepoint blockade)":
        "Transport logistics and chokepoints",

    "Demand and macro economy (Global GDP growth, recession fears, growth slowdown, soft landing, inflation/CPI, PPI, PMI/ISM, interest rate decisions, central banks, Fed/FOMC, ECB, BoE, rate cuts, rate hikes, dollar strength/DXY, risk-off sentiment, equities selloff, credit stress, banking stress, China growth concerns, stimulus, industrial output, manufacturing contraction, consumer demand, aviation fuel consumption, jet demand, gasoline demand, diesel demand, refinery demand, travel season, demand destruction, macro uncertainty)":
        "Demand and macro economy",

    "Inventory SPR and refinery (crude oil inventories, stock build, stock draw, EIA weekly report, API data, Cushing stocks, storage hub, floating storage, SPR release, SPR refill, SPR purchase, emergency release, refinery maintenance, turnaround, refinery outages, refinery utilization, refinery runs, throughput, distillate inventories, gasoline inventories, jet fuel inventories, product stocks, crack spreads, refinery margins, run cuts, import surge, export surge, inventory overhang, tight inventories, days of supply)":
        "Inventory SPR and refinery",

    "OPEC producer policy (OPEC+, OPEC meeting, ministerial meeting, JMMC, JTC, production quotas, voluntary cuts, output targets, compliance with cuts, overproduction, compensation cuts, baseline revision, capacity assessment, maximum sustainable capacity, spare capacity policy, unwind cuts, phased return, production hike, supply restraint, quota rollout, UAE/Iraq/Kazakhstan compliance, Saudi voluntary cut, market management, quota extension, policy unchanged decision)":
        "OPEC producer policy",

    "Geopolitical Normalisation and Peace (ceasefire, truce, de-escalation, peace talks, diplomatic breakthrough, normalization, conflict eases, hostilities end, sanctions relief, lifting of sanctions, easing restrictions, nuclear deal, JCPOA talks, diplomatic agreement, reopening of ports, reopening of pipelines, restoration of exports, return of supply, security improves, shipping resumes safely, risk premium fades, troops withdrawal, détente, reconciliation, stabilization, settlement agreement)":
        "Geopolitical Normalisation and Peace",
}


RICH_CHANNEL_LABELS = list(channel_mapping.keys())

CLEAN_CHANNELS = list(dict.fromkeys(channel_mapping.values()))


def setup_model():

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


def scores_to_dict(out: Dict) -> Dict[str, float]:

    """HF pipeline returns labels sorted desc; convert to dict label->score."""

    return {lbl: float(sc) for lbl, sc in zip(out["labels"], out["scores"])}


def main():

    if not INPUT_CSV_PATH.exists():

        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV_PATH}")


    df = robust_read_csv(INPUT_CSV_PATH)


    if "Title" not in df.columns:

        raise ValueError("Input CSV must contain a 'Title' column.")


    df = df.dropna(subset=["Title"]).copy()

    df["Title"] = df["Title"].astype(str)

    df = df[df["Title"].str.len() > 15].copy()


    df["_title_norm"] = df["Title"].map(normalize_title)

    df = df.drop_duplicates(subset=["_title_norm"], keep="first").copy()


    if df.empty:

        print("No usable headlines after cleaning/dedup.")

        return


    clf = setup_model()


    hf = Dataset.from_pandas(df)


    relevance_probs: List[float] = []

    for out in clf(KeyDataset(hf, "Title"), candidate_labels=RELEVANCE_LABELS, multi_label=False, batch_size=BATCH_SIZE):

        d = scores_to_dict(out)

        relevance_probs.append(d.get(RELEVANCE_LABELS[0], 0.0))


    df["prob_relevant"] = relevance_probs


    is_relevant = df["prob_relevant"] >= 0.5

    if DROP_IRRELEVANT:

        df = df[is_relevant].copy()

        if df.empty:

            print("All rows marked irrelevant; nothing to write.")

            return

        hf2 = Dataset.from_pandas(df)

    else:

        hf2 = hf


    for ch in CLEAN_CHANNELS:

        df[f"p_{ch}"] = 0.0


    idx_map = df.index.to_list()

    channel_outs = clf(KeyDataset(hf2, "Title"), candidate_labels=RICH_CHANNEL_LABELS, multi_label=True, batch_size=BATCH_SIZE)


    for i, out in enumerate(channel_outs):

        score_map = scores_to_dict(out)

        row_ix = idx_map[i]


        if (not DROP_IRRELEVANT) and (df.loc[row_ix, "prob_relevant"] < 0.5):

            continue


        for rich_label, clean_name in channel_mapping.items():

            df.loc[row_ix, f"p_{clean_name}"] = float(score_map.get(rich_label, 0.0))


    df = df.drop(columns=["_title_norm"], errors="ignore")


    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)


    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print(f"Done. Wrote: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":

    main()
