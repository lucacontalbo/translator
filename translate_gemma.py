#!/usr/bin/env python3
"""
Translate chentong00/propositionizer-wiki-data into one or more languages
with google/translategemma-12b-it (supports ~55 languages, given as ISO codes),
using vLLM for fast offline batched inference.

Example:
    python translate_gemma.py --languages it fr de ja zh --split train --output_dir data/propositionizer

Each language is saved to <output_dir>/<lang>.
Requirements: a CUDA GPU, access to the gated model (`huggingface-cli login`) and `pip install vllm`.
"""

import argparse
import ast
import json
import os

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

MODEL_NAME = "google/translategemma-12b-it"
CACHE_DIR = "./hf_cache/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="chentong00/propositionizer-wiki-data")
    parser.add_argument("--split", default=None, choices=["train", "validation", "test"],
                        help="Split to translate; if omitted, translate all splits")
    parser.add_argument("--languages", nargs="+", required=True,
                        help="Target language codes, e.g. it fr de-DE pt-BR ja zh")
    parser.add_argument("--source_language", default="en")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Hard cap on generated tokens per text")
    parser.add_argument("--max_length_ratio", type=float, default=3.0,
                        help="Also cap generated tokens at this multiple of the input length, "
                             "so short texts stuck in a repetition loop stop early")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="Max prompt + output tokens per request (bounds KV-cache size)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs to shard the model across")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--quantization", default="fp8", choices=["fp8", "none"],
                        help="fp8: quantize weights on load (faster on Ada/Hopper GPUs); none: full bf16")
    parser.add_argument("--kv_cache_dtype", default="auto", choices=["auto", "fp8"],
                        help="fp8 doubles KV-cache capacity (more concurrent requests) at a small quality cost")
    parser.add_argument("--max_num_seqs", type=int, default=512,
                        help="Max requests processed concurrently")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_targets(value) -> list[str]:
    """`targets` is a serialized list of strings (JSON or Python literal)."""
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


SENTINEL = ""  # private-use character, never present in real text


class Translator:
    def __init__(self, source_lang: str, args: argparse.Namespace):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        self.llm = LLM(
            model=MODEL_NAME,
            dtype="bfloat16",
            download_dir=CACHE_DIR,
            quantization=None if args.quantization == "none" else args.quantization,
            kv_cache_dtype=args.kv_cache_dtype,
            max_num_seqs=args.max_num_seqs,
            enable_prefix_caching=True,  # the instruction prefix is identical across a language's requests
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            limit_mm_per_prompt={"image": 0},  # text-only: skip vision encoder memory profiling
        )
        self.source_lang = source_lang
        self.max_new_tokens = args.max_new_tokens
        self.max_length_ratio = args.max_length_ratio
        self.max_model_len = args.max_model_len
        self._templates: dict[str, tuple[str, str]] = {}

    def _template(self, target_lang: str) -> tuple[str, str]:
        """Render the chat template once per language and split it around the text slot.

        The template only trims the text, so prefix + text.strip() + suffix equals a full render,
        and this avoids re-rendering the large Jinja template for every request.
        """
        if target_lang not in self._templates:
            messages = [{"role": "user", "content": [{
                "type": "text",
                "source_lang_code": self.source_lang,
                "target_lang_code": target_lang,
                "text": SENTINEL,
            }]}]
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prefix, suffix = rendered.split(SENTINEL)
            self._templates[target_lang] = (prefix, suffix)
        return self._templates[target_lang]

    def translate(self, texts: list[str], target_lang: str) -> list[str]:
        """Translate all texts in a single vLLM call; vLLM handles batching and scheduling.

        Duplicate texts are translated once, and empty texts are returned as "" without a request.
        """
        unique = [t for t in dict.fromkeys(t.strip() for t in texts) if t]
        if not unique:
            return ["" for _ in texts]
        prefix, suffix = self._template(target_lang)
        # The chat template already emits <bos>, so tokenize without adding special tokens again.
        overhead = len(self.tokenizer(prefix + suffix, add_special_tokens=False)["input_ids"])
        all_ids = self.tokenizer([prefix + t + suffix for t in unique], add_special_tokens=False)["input_ids"]

        prompts, params = [], []
        for ids in all_ids:
            text_len = len(ids) - overhead
            max_tokens = min(self.max_new_tokens,
                             int(self.max_length_ratio * text_len) + 32,
                             self.max_model_len - len(ids))
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            params.append(SamplingParams(temperature=0.0, max_tokens=max(max_tokens, 1)))

        outputs = self.llm.generate(prompts, params)
        translations = {t: o.outputs[0].text.strip() for t, o in zip(unique, outputs)}
        return [translations.get(t.strip(), "") for t in texts]


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset_name)
    if args.split:
        dataset = {args.split: dataset[args.split]}

    translator = Translator(args.source_language, args)

    for lang in args.languages:
        out_dir = os.path.join(args.output_dir, lang)
        if os.path.exists(out_dir) and not args.overwrite:
            print(f"Skipping {lang}: {out_dir} exists (use --overwrite)")
            continue
        for split_name, split_ds in dataset.items():
            print(f"[{lang}] translating {split_name} ({len(split_ds)} rows)")
            sources = list(split_ds["sources"])  # datasets>=4 returns a lazy Column, not a list
            targets = [parse_targets(t) for t in split_ds["targets"]]
            flat = [p for props in targets for p in props]

            # One request list for the whole split so vLLM can keep the GPU saturated.
            translated_all = translator.translate(sources + flat, lang)
            sources_tr, flat_tr = translated_all[:len(sources)], translated_all[len(sources):]

            rebuilt, cursor = [], 0
            for props in targets:
                rebuilt.append(json.dumps(flat_tr[cursor:cursor + len(props)], ensure_ascii=False))
                cursor += len(props)

            translated = (split_ds
                          .add_column("sources_translated", sources_tr)
                          .add_column("targets_translated", rebuilt))
            translated.save_to_disk(os.path.join(out_dir, split_name))
        print(f"Saved {lang} to {out_dir}")


if __name__ == "__main__":
    main()
