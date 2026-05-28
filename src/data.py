"""
Financial PhraseBank dataset loader.

The dataset is hosted on HuggingFace as `takala/financial_phrasebank` with
four agreement-level configurations:
  - sentences_50agree    (5,842 sentences)
  - sentences_66agree    (4,217 sentences)
  - sentences_75agree    (3,453 sentences)
  - sentences_allagree   (2,264 sentences)

We use `sentences_75agree` as a good balance between size and label quality.
"""

from __future__ import annotations

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split

from src.config import LABEL_LIST, SEED


PHRASEBANK_CONFIG = "sentences_75agree"


def load_phrasebank(config: str = PHRASEBANK_CONFIG) -> pd.DataFrame:
    """Load the Financial PhraseBank dataset from HuggingFace as a DataFrame."""
    ds = load_dataset("takala/financial_phrasebank", config, trust_remote_code=True)
    df = ds["train"].to_pandas()
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
