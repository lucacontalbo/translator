#!/usr/bin/env python3
"""
Translate chentong00/propositionizer-wiki-data into a target language using
a free, local Hugging Face MarianMT model.

Example:
    python translate_propositionizer.py --language italian --split train --output_dir data/propositionizer_it

Requirements:
    pip install datasets transformers sentencepiece torch tqdm
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict, load_dataset
from transformers import MarianMTModel, MarianTokenizer
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")

# Expand this map as needed.
LANGUAGE_TO_MARIAN_MODEL = {
    "italian": "Helsinki-NLP/opus-mt-en-it",
    "it": "Helsinki-NLP/opus-mt-en-it",
    "french": "Helsinki-NLP/opus-mt-en-fr",
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "spanish": "Helsinki-NLP/opus-mt-en-es",
    "es": "Helsinki-NLP/opus-mt-en-es",
    "german": "Helsinki-NLP/opus-mt-en-de",
    "de": "Helsinki-NLP/opus-mt-en-de",
    "portuguese": "Helsinki-NLP/opus-mt-en-pt",
    "pt": "Helsinki-NLP/opus-mt-en-pt",
    "dutch": "Helsinki-NLP/opus-mt-en-nl",
    "nl": "Helsinki-NLP/opus-mt-en-nl",
    "russian": "Helsinki-NLP/opus-mt-en-ru",
    "ru": "Helsinki-NLP/opus-mt-en-ru",
}

# Conservative default for Marian models.
DEFAULT_MAX_INPUT_TOKENS = 448


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        default="chentong00/propositionizer-wiki-data",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--split",
        default=None,
        choices=[None, "train", "validation", "test"],
        help="Optional split to translate; if omitted, translate all splits",
    )
    parser.add_argument(
        "--language",
        required=True,
        help="Target language, e.g. italian",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Override translation model name",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Translation batch size",
    )
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=DEFAULT_MAX_INPUT_TOKENS,
        help="Max source token length fed to the translation model",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where the translated dataset will be saved",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output_dir if it exists",
    )
    return parser.parse_args()


def resolve_model_name(language: str, explicit_model_name: str | None) -> str:
    if explicit_model_name:
        return explicit_model_name

    key = language.strip().lower()
    if key not in LANGUAGE_TO_MARIAN_MODEL:
        raise ValueError(
            f"No default Marian model configured for language='{language}'. "
            f"Pass --model_name explicitly or extend LANGUAGE_TO_MARIAN_MODEL."
        )
    return LANGUAGE_TO_MARIAN_MODEL[key]


def safe_parse_targets(value: Any) -> List[str]:
    """
    The dataset viewer shows `targets` as a serialized list of strings.
    We parse robustly in case the field comes as a string or already as a list.
    """
    if isinstance(value, list):
        return [str(x) for x in value]

    if not isinstance(value, str):
        raise TypeError(f"Unsupported targets type: {type(value)}")

    # Try JSON first, then Python literal format.
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass

    raise ValueError(f"Could not parse targets field: {value[:200]!r}")


class Translator:
    def __init__(self, model_name: str, max_input_tokens: int):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.max_input_tokens = max_input_tokens
        self.model.generation_config.max_length = None

        # Use GPU if available.
        try:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            self.device = "cpu"

        self.model.to(self.device)

    def translate_texts(self, texts: List[str]) -> List[str]:
        import torch

        if not texts:
            return []

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=512,
                num_beams=4,
            )

        """print(texts)
        print()
        print()
        print("***************************************")
        print()
        print()
        print(self.tokenizer.batch_decode(generated, skip_special_tokens=True))"""

        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def translate_dataset_split(self, ds: Dataset, batch_size: int) -> Dataset:
        def _batched_translate(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            source_texts = batch["sources"]

            print(batch)
            print(a)
            parsed_targets_batch = [safe_parse_targets(x) for x in batch["targets"]]
            flat_targets = [item for items in parsed_targets_batch for item in items]

            translated_sources = self.translate_texts(source_texts)

            translated_flat_targets: List[str] = []
            for start in range(0, len(flat_targets), batch_size):
                translated_flat_targets.extend(
                    self.translate_texts(flat_targets[start : start + batch_size])
                )

            rebuilt_targets: List[List[str]] = []
            cursor = 0
            for items in parsed_targets_batch:
                rebuilt_targets.append(
                    translated_flat_targets[cursor : cursor + len(items)]
                )
                cursor += len(items)

            return {
                "sources_original": source_texts,
                "targets_original": batch["targets"],
                "sources_translated": translated_sources,
                "targets_translated": [json.dumps(x, ensure_ascii=False) for x in rebuilt_targets],
            }

        return ds.map(
            _batched_translate,
            batched=True,
            batch_size=batch_size,
            desc="Translating dataset",
        )


def main() -> None:
    args = parse_args()

    if os.path.exists(args.output_dir):
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} already exists. Use --overwrite or change --output_dir."
            )

    model_name = resolve_model_name(args.language, args.model_name)
    print(f"Using model: {model_name}")

    dataset = load_dataset(args.dataset_name)

    if args.split is not None:
        dataset = DatasetDict({args.split: dataset[args.split]})

    translator = Translator(model_name=model_name, max_input_tokens=args.max_input_tokens)

    translated = DatasetDict()
    for split_name, split_ds in dataset.items():
        print(f"Translating split: {split_name} ({len(split_ds)} rows)")
        translated[split_name] = translator.translate_dataset_split(
            split_ds, batch_size=args.batch_size
        )

    translated.save_to_disk(args.output_dir)
    print(f"Saved translated dataset to: {args.output_dir}")


if __name__ == "__main__":
    main()
