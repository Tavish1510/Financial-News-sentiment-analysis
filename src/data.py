"""
Financial PhraseBank dataset loader.

We load from HuggingFace's `datasets-server` rows API rather than the
`datasets` library, because:
  - `datasets>=3.0` blocks loading scripts (which the original
    `takala/financial_phrasebank` dataset uses)
  - The auto-Parquet conversion branch for this dataset is broken (the
    script downloads from researchgate.net which the conversion worker
    can't reach reliably)
  - The rows API works for any indexed dataset, no script execution, no
    `trust_remote_code`, no version pinning
"""

from __future__ import annotations

import time

import pandas as pd
import requests
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from src.config import LABEL_LIST, SEED


PHRASEBANK_CONFIG = "sentences_75agree"
ROWS_API = "https://datasets-server.huggingface.co/rows"
BATCH_SIZE = 100
MAX_RETRIES = 3


def _fetch_batch(dataset_id: str, config: str, split: str, offset: int) -> dict:
    params = {
        "dataset": dataset_id,
        "config": config,
        "split": split,
        "offset": offset,
        "length": BATCH_SIZE,
    }
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(ROWS_API, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"datasets-server fetch failed: {last_err}")


def load_phrasebank(config: str = PHRASEBANK_CONFIG) -> pd.DataFrame:
    """Load Financial PhraseBank via HF datasets-server rows API."""
    rows = []
    offset = 0
    total = None

    while True:
        data = _fetch_batch("takala/financial_phrasebank", config, "train", offset)

        batch = data.get("rows", [])
        if not batch:
            break

        for item in batch:
            rows.append(item["row"])

        total = total or data.get("num_rows_total")
        offset += len(batch)

        if total is not None and offset >= total:
            break

    if not rows:
        raise RuntimeError(
            "Could not fetch any rows from HuggingFace datasets-server. "
            "If this persists, manually upload Sentences_75Agree.txt into "
            "data/raw/ and use _load_from_local()."
        )

    df = pd.DataFrame(rows)
    df = df.rename(columns={"sentence": "text"})
    df["label_name"] = df["label"].map({i: l for i, l in enumerate(LABEL_LIST)})
    print(f"Loaded {len(df):,} sentences from datasets-server")
    return df


def basic_clean(text: str) -> str:
    """Lightweight cleaning — preserve case and most punctuation for the
    transformer; just strip whitespace and collapse internal whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = " ".join(text.split())
    return text


def build_splits(df: pd.DataFrame, test_size: float = 0.10, val_size: float = 0.10) -> DatasetDict:
    """Split into train/validation/test with stratification on label."""
    df = df.copy()
    df["text"] = df["text"].apply(basic_clean)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=SEED
    )
    train, val = train_test_split(
        train_val, test_size=val_size / (1 - test_size), stratify=train_val["label"], random_state=SEED
    )

    return DatasetDict(
        {
            "train": Dataset.from_pandas(train.reset_index(drop=True)),
            "validation": Dataset.from_pandas(val.reset_index(drop=True)),
            "test": Dataset.from_pandas(test.reset_index(drop=True)),
        }
    )


def class_distribution(df: pd.DataFrame) -> pd.Series:
    return df["label_name"].value_counts(normalize=True).sort_index()
