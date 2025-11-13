"""Supervised fine-tuning of SmolLM on the CoLA portion of GLUE."""
from __future__ import annotations

import argparse
import os
from typing import Dict

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

LABEL_WORDS: Dict[int, str] = {
    0: "unacceptable",
    1: "acceptable",
}

DEFAULT_PROMPT = (
    "Decide if the following sentence is linguistically acceptable. "
    "Answer with 'acceptable' or 'unacceptable'.\n"
    "Sentence: {sentence}\nAnswer:"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised fine-tuning on CoLA for SmolLM.")
    parser.add_argument("--model-name", default="HuggingFaceTB/SmolLM-135M", help="HF model repo id.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join("sft_checkpoints", "SmolLM-135M-cola"),
        help="Where to store checkpoints and final model.",
    )
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT, help="Template used to build prompts.")
    parser.add_argument("--max-length", type=int, default=256, help="Max token length for each example.")
    parser.add_argument("--train-batch-size", type=int, default=4, help="Per-device train batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=4, help="Per-device eval batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="AdamW learning rate.")
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 training when available.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional train set cap.")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Optional eval set cap.")
    parser.add_argument("--run-name", default=None, help="Optional tracking run name.")
    parser.add_argument(
        "--checkpoint-name",
        default="sft_final.pt",
        help="Filename for the torch checkpoint compatible with evaluate_glue.py.",
    )
    return parser.parse_args()


def build_supervised_dataset(tokenizer, raw_dataset, max_length: int, prompt_template: str):
    def preprocess(example):
        prompt = prompt_template.format(sentence=example["sentence"]).strip()
        label_text = LABEL_WORDS[int(example["label"])]
        answer_ids = tokenizer(
            " " + label_text,
            add_special_tokens=False,
        )["input_ids"]
        prompt_budget = max(max_length - len(answer_ids) - 1, 1)
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=prompt_budget,
        )["input_ids"]
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("Tokenizer must define an EOS token.")
        input_ids = prompt_ids + answer_ids + [eos_id]
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + answer_ids + [eos_id]
        pad_len = max_length - len(input_ids)
        if pad_len < 0:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]
            pad_len = 0
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must define a pad token.")
        if pad_len:
            input_ids = input_ids + [pad_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [-100] * pad_len
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    columns_to_remove = raw_dataset.column_names
    return raw_dataset.map(
        preprocess,
        remove_columns=columns_to_remove,
        desc="Tokenizing CoLA",
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, padding_side="right", padding=True)
    if tokenizer.eos_token is None or tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must expose an EOS token for causal LM training.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset("glue", "cola")
    train_dataset = raw["train"]
    eval_dataset = raw["validation"]

    if args.max_train_samples:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    if args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    train_dataset = build_supervised_dataset(tokenizer, train_dataset, args.max_length, args.prompt_template)
    eval_dataset = build_supervised_dataset(tokenizer, eval_dataset, args.max_length, args.prompt_template)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    data_collator = DataCollatorWithPadding(
        tokenizer,
        padding="longest",
        return_tensors="pt",
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        report_to="tensorboard",
        bf16=args.bf16,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    checkpoint_path = os.path.join(args.output_dir, args.checkpoint_name)
    torch.save({"model": model.state_dict()}, checkpoint_path)
    print(f"Saved fine-tuned checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
