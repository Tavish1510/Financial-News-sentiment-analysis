"""Financial PhraseBank dataset loader.

Downloads the original ZIP from HuggingFace Hub, extracts the text file,
parses the `sentence@label` format, and returns a tidy DataFrame.
The ZIP is cached locally after first download.
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from src.config import LABEL_LIST, SEED


PHRASEBANK_CONFIG = "sentences_75agree"

# Map our config name → the filename inside the ZIP
CONFIG_TO_FILENAME = {
    "sentences_50agree": "Sentences_50Agree.txt",
    "sentences_66agree": "Sentences_66Agree.txt",
    "sentences_75agree": "Sentences_75Agree.txt",
    "sentences_allagree": "Sentences_AllAgree.txt",
}

ZIP_URLS = [
    "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip",
    "https://www.researchgate.net/profile/Pekka-Malo/publication/251231364_"
    "FinancialPhraseBank-v10/data/0c96051eee4fb1d56e000000/FinancialPhraseBank-v10.zip",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept": "*/*",
}

CACHE_DIR = Path.home() / ".cache" / "financial_phrasebank"
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}


def _download_zip() -> bytes:
    """Download the FinancialPhraseBank ZIP, trying mirrors in order."""
    last_err = None
    for url in ZIP_URLS:
        try:
            print(f"Downloading: {url[:80]}...")
            r = requests.get(url, headers=HEADERS, timeout=120, allow_redirects=True)
            r.raise_for_status()
            content = r.content
            if len(content) < 10_000:
                raise RuntimeError(f"Downloaded file too small ({len(content)} bytes) — likely not the ZIP")
            # Verify it's actually a ZIP
            if content[:4] != b"PK\x03\x04":
                raise RuntimeError("Downloaded content is not a ZIP file")
            print(f"Downloaded {len(content):,} bytes")
            return content
        except Exception as e:
            print(f"  -> failed: {e}")
            last_err = e
    raise RuntimeError(f"All download sources failed. Last error: {last_err}")


def _get_zip_bytes() -> bytes:
    """Return ZIP bytes, using local cache if available."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "FinancialPhraseBank-v1.0.zip"

    if cache_path.exists() and cache_path.stat().st_size > 10_000:
        return cache_path.read_bytes()

    content = _download_zip()
    cache_path.write_bytes(content)
    return content


def _parse_text(text: str) -> list[dict]:
    """Parse the dataset's text format — each line is `sentence@label`."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "@" not in line:
            continue
        sentence, _, label = line.rpartition("@")
        sentence = sentence.strip()
        label = label.strip().lower()
        if label not in LABEL_MAP or not sentence:
            continue
        rows.append({"text": sentence, "label": LABEL_MAP[label]})
    return rows


def load_phrasebank(config: str = PHRASEBANK_CONFIG) -> pd.DataFrame:
    """Load Financial PhraseBank by downloading + parsing the original ZIP."""
    filename = CONFIG_TO_FILENAME.get(config)
    if filename is None:
        raise ValueError(f"Unknown config: {config}. Options: {list(CONFIG_TO_FILENAME)}")

    zip_bytes = _get_zip_bytes()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        # Find the file (it lives inside a FinancialPhraseBank-v1.0/ folder)
        candidates = [n for n in z.namelist() if n.endswith(filename)]
        if not candidates:
            raise RuntimeError(
                f"{filename} not found in ZIP. Contents: {z.namelist()[:10]}..."
            )
        with z.open(candidates[0]) as f:
            # The file is in latin-1 encoding (it contains €, £, etc.)
            text = f.read().decode("latin-1")

    rows = _parse_text(text)
    if not rows:
        raise RuntimeError(f"No rows parsed from {filename}")

    df = pd.DataFrame(rows)
    df["label_name"] = df["label"].map({i: l for i, l in enumerate(LABEL_LIST)})
    print(f"Loaded {len(df):,} sentences from {filename}")
    return df


def basic_clean(text: str) -> str:
    """Lightweight cleaning — preserve case and most punctuation."""
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
