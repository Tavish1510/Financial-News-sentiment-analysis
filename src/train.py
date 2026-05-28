"""
Fine-tuning pipeline for DistilBERT on the Financial PhraseBank.

Usage:
    python -m src.train

This script is also driven from `notebooks/02_fine_tuning.ipynb` (Colab-ready).
"""

from __future__ import annotations

import os

import evaluate
import numpy as np
import torch
from datasets import DatasetDict
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src import config
from src.data import build_splits, load_phrasebank
from src.model import load_base_model_for_training


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        "f1_weighted": f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
        "precision_macro": precision.compute(
            predictions=preds, references=labels, average="macro", zero_division=0
        )["precision"],
        "recall_macro": recall.compute(
            predictions=preds, references=labels, average="macro", zero_division=0
        )["recall"],
    }


def tokenize_dataset(dataset_dict: DatasetDict, tokenizer):
    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=config.MAX_LENGTH)

    return dataset_dict.map(_tok, batched=True)


def train(push_to_hub: bool = False, hub_model_id: str | None = None) -> Trainer:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Financial PhraseBank...")
    df = load_phrasebank()
    print(f"  -> {len(df):,} sentences, classes: {df['label_name'].value_counts().to_dict()}")

    splits = build_splits(df)
    print(f"Splits: train={len(splits['train']):,}, val={len(splits['validation']):,}, test={len(splits['test']):,}")

    tokenizer, model = load_base_model_for_training()
    encoded = tokenize_dataset(splits, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(config.MODEL_DIR / "checkpoints"),
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        seed=config.SEED,
        report_to="none",
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        fp16=torch.cuda.is_available(),
    )

    # `tokenizer=` was renamed to `processing_class=` in transformers>=4.46.
    # Use whichever the installed version accepts.
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    import inspect
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    print("\nTraining...")
    trainer.train()

    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(encoded["test"])
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k:25s} {v:.4f}")

    final_dir = config.MODEL_DIR / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)

    if push_to_hub:
        trainer.push_to_hub("Fine-tuned DistilBERT on Financial PhraseBank")
        print(f"\nModel pushed to: https://huggingface.co/{hub_model_id}")

    return trainer


if __name__ == "__main__":
    train()
