from __future__ import annotations


import time

from pathlib import Path

from typing import List, Dict


import pandas as pd

import torch

from tqdm import tqdm

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


MODEL_FOLDER = Path(r"D:\college\4-2 mine\manogna\CrudeBert\CrudeBert")


INPUT_ROOT  = Path(r"D:\college\4-2 mine\manogna\data final headlines")

OUTPUT_ROOT = Path(r"D:\college\4-2 mine\manogna\Crude_Bert_Headlines")


TITLE_COL   = "Title"


DEVICE      = "cpu"

BATCH_SIZE  = 64

MAX_LENGTH  = 64


SKIP_EXISTING_OUTPUTS = True


SUPPORTED_EXT = {".xlsx", ".xls", ".csv"}


def load_crudebert(model_folder: Path, device: str = "cpu"):

    config_path = model_folder / "crude_bert_config.json"

    model_path  = model_folder / "crude_bert_model.bin"


    print("Config exists:", config_path.exists(), config_path)

    print("Model exists :", model_path.exists(), model_path)


    if not config_path.exists():

        raise FileNotFoundError(f"crude_bert_config.json not found at: {config_path}")

    if not model_path.exists():

        raise FileNotFoundError(f"crude_bert_model.bin not found at: {model_path}")


    config = AutoConfig.from_pretrained(str(config_path))

    model = AutoModelForSequenceClassification.from_config(config)


    state_dict = torch.load(str(model_path), map_location="cpu")

    state_dict.pop("bert.embeddings.position_ids", None)

    model.load_state_dict(state_dict, strict=False)


    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


    model.config.id2label = {0: "positive", 1: "negative", 2: "neutral"}

    model.config.label2id = {"positive": 0, "negative": 1, "neutral": 2}

    model.config.num_labels = 3


    model.to(device)

    model.eval()


    print("Loaded model + tokenizer ✅")

    print("Device:", device)

    print("num_labels:", model.config.num_labels)

    print("id2label:", model.config.id2label)


    return model, tokenizer


@torch.no_grad()

def crudebert_batch_probs(
    titles: List[str],
    model,
    tokenizer,
    device: str,
    batch_size: int,
    max_length: int,
) -> List[Dict[str, float]]:

    results: List[Dict[str, float]] = []


    for i in range(0, len(titles), batch_size):

        batch = titles[i:i + batch_size]


        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}


        logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)


        for row in probs:

            p_pos = float(row[0].item())

            p_neg = float(row[1].item())

            p_neu = float(row[2].item())


            dominant = max(
                [("positive", p_pos), ("negative", p_neg), ("neutral", p_neu)],
                key=lambda x: x[1]
            )[0]


            results.append({
                "positive": p_pos,
                "negative": p_neg,
                "neutral": p_neu,
                "dominant": dominant
            })


    return results


def read_file(path: Path) -> pd.DataFrame:

    ext = path.suffix.lower()

    if ext in [".xlsx", ".xls"]:

        return pd.read_excel(path)

    elif ext == ".csv":

        return pd.read_csv(path)

    else:

        raise ValueError(f"Unsupported file type: {ext} ({path})")


def write_excel(df: pd.DataFrame, out_path: Path):

    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_excel(out_path, index=False, engine="openpyxl")


def collect_files(input_root: Path) -> List[Path]:

    files = []

    for p in input_root.rglob("*"):

        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:

            files.append(p)

    return sorted(files)


def main():

    if not INPUT_ROOT.exists():

        raise FileNotFoundError(f"INPUT_ROOT not found: {INPUT_ROOT}")


    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


    model, tokenizer = load_crudebert(MODEL_FOLDER, device=DEVICE)


    files = collect_files(INPUT_ROOT)

    print(f"\nFound {len(files)} files under: {INPUT_ROOT}\n")


    ok = 0

    skipped = 0

    failed = 0


    start_time = time.time()


    for fpath in tqdm(files, desc="Processing files", unit="file"):

        try:

            rel = fpath.relative_to(INPUT_ROOT)

            out_path = (OUTPUT_ROOT / rel).with_suffix(".xlsx")


            if SKIP_EXISTING_OUTPUTS and out_path.exists():

                skipped += 1

                continue


            df = read_file(fpath)


            if TITLE_COL not in df.columns:

                raise ValueError(f"Missing column '{TITLE_COL}'. Columns: {list(df.columns)}")


            titles = df[TITLE_COL].fillna("").astype(str).str.strip().tolist()


            df["negative"] = pd.NA

            df["neutral"] = pd.NA

            df["positive"] = pd.NA

            df["dominant"] = pd.NA


            idx = [i for i, t in enumerate(titles) if t]

            texts = [titles[i] for i in idx]


            if texts:

                scored = crudebert_batch_probs(
                    texts,
                    model=model,
                    tokenizer=tokenizer,
                    device=DEVICE,
                    batch_size=BATCH_SIZE,
                    max_length=MAX_LENGTH
                )


                for row_i, s in zip(idx, scored):

                    df.at[row_i, "positive"] = s["positive"]

                    df.at[row_i, "negative"] = s["negative"]

                    df.at[row_i, "neutral"] = s["neutral"]

                    df.at[row_i, "dominant"] = s["dominant"]


            write_excel(df, out_path)

            ok += 1


        except Exception as e:

            failed += 1

            tqdm.write(f"❌ Failed: {fpath} | {e}")


    elapsed = time.time() - start_time


    print("\nDONE ✅")

    print(f"Success : {ok}")

    print(f"Skipped : {skipped}")

    print(f"Failed  : {failed}")

    print(f"Time    : {elapsed/60:.2f} minutes")

    print(f"Output  : {OUTPUT_ROOT}")


if __name__ == "__main__":

    main()
