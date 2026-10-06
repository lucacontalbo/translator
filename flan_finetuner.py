#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
from typing import Any, Dict, List

from datasets import DatasetDict, load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="google/flan-t5-large")
    parser.add_argument("--source_col", type=str, default="sources_translated")
    parser.add_argument("--target_col", type=str, default="targets_translated")

    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=512)

    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)  # global batch = 64

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)

    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_validation_split(dataset: DatasetDict) -> DatasetDict:
    if "validation" in dataset:
        return dataset
    split = dataset["train"].train_test_split(test_size=0.02, seed=42)
    return DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })


def parse_targets(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value]

    if not isinstance(value, str):
        return [str(value)]

    stripped = value.strip()

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(stripped)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass

    return [stripped]


def main() -> None:
    args = parse_args()

    try:
        dataset = load_from_disk(args.dataset_path)
        if not isinstance(dataset, DatasetDict):
            raise ValueError("Expected a DatasetDict from save_to_disk().")
    except:
        dataset = load_dataset(args.dataset_path)

    dataset = ensure_validation_split(dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    def preprocess(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        sources = [str(x) for x in batch[args.source_col]]
        targets = [
            json.dumps(parse_targets(x), ensure_ascii=False)
            for x in batch[args.target_col]
        ]

        model_inputs = tokenizer(
            sources,
            max_length=args.max_source_length,
            truncation=True,
            padding=False,
        )

        labels = tokenizer(
            text_target=targets,
            max_length=args.max_target_length,
            truncation=True,
            padding=False,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=8 if (args.fp16 or args.bf16) else None,
    )

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        #overwrite_output_dir=True,
        do_train=True,
        do_eval=True,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,

        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,

        optim="adamw_torch",
        lr_scheduler_type="linear",
        warmup_ratio=0.0,

        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
        report_to="none",
        logging_nan_inf_filter=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
