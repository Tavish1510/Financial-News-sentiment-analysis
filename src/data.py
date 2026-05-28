"""
Financial PhraseBank dataset loader.

HuggingFace auto-converts every dataset to Parquet on a side branch called
`refs/convert/parquet`. We load directly from there to avoid invoking the
loading script (which is no longer supported in `datasets >= 3.0`).

Works with any `datasets` version, no `trust_remote_code` needed.
"""

from __future__ import annotations

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

from src.config import LABEL_LIST, SEED


PHRASEBANK_CONFIG = "sentences_75agree"


def load_phrasebank(config: str = PHRASEBANK_CONFIG) -> pd.DataFrame:
    """Load Financial PhraseBank from HF's auto-converted Parquet branch."""
    path = hf_hub_download(
        repo_id="takala/financial_phrasebank",
        filename=f"{config}/train/0000.parquet",
        repo_type="dataset",
        revision="refs/convert/parquet",
    )
    df = pd.read_parquet(path)
    df = df.rename(columns={"sentence": "text"})
    df["label_name"] = df["label"].map({i: l for i, l in enumerate(LABEL_LIST)})
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
