"""Evaluate a GRPO-trained causal LM on GLUE tasks via lightweight prompting."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TaskConfig:
    name: str
    template: str
    task_type: str  # "classification" or "regression"
    default_split: str
    label_words: Optional[Dict[int, List[str]]] = None


TASK_CONFIGS: Dict[str, TaskConfig] = {
    "stsb": TaskConfig(
        name="STS-B",
        template=(
            "Rate the semantic similarity of the two sentences on a scale from 0 (no overlap) "
            "to 5 (semantically equivalent). Respond with a single number.\n"
            "Sentence 1: {sentence1}\nSentence 2: {sentence2}\nScore:"
        ),
        task_type="regression",
        default_split="validation",
    ),
    "cola": TaskConfig(
        name="CoLA",
        template=(
            "Decide if the following sentence is linguistically acceptable. "
            "Answer with 'acceptable' or 'unacceptable'.\n"
            "Sentence: {sentence}\nAnswer:"
        ),
        task_type="classification",
        default_split="validation",
        label_words={
            0: ["unacceptable", "not acceptable", "incorrect", "no", "0"],
            1: ["acceptable", "correct", "yes", "1"],
        },
    ),
    "mrpc": TaskConfig(
        name="MRPC",
        template=(
            "Do the two sentences have the same meaning? Answer 'paraphrase' or 'not paraphrase'.\n"
            "Sentence 1: {sentence1}\nSentence 2: {sentence2}\nAnswer:"
        ),
        task_type="classification",
        default_split="validation",
        label_words={
            0: ["not paraphrase", "different", "no", "0", "not equivalent"],
            1: ["paraphrase", "same meaning", "yes", "duplicate", "equivalent", "1"],
        },
    ),
    "rte": TaskConfig(
        name="RTE",
        template=(
            "Does the first sentence entail the second sentence? Answer 'entailment' or 'not entailment'.\n"
            "Sentence 1: {sentence1}\nSentence 2: {sentence2}\nAnswer:"
        ),
        task_type="classification",
        default_split="validation",
        label_words={
            0: ["entailment", "yes"],
            1: ["not entailment", "no", "contradiction"],
        },
    ),
    "wnli": TaskConfig(
        name="WNLI",
        template=(
            "Does the hypothesis follow from the premise? Answer 'entailment' or 'not entailment'.\n"
            "Premise: {sentence1}\nHypothesis: {sentence2}\nAnswer:"
        ),
        task_type="classification",
        default_split="validation",
        label_words={
            0: ["not entailment", "no", "contradiction"],
            1: ["entailment", "yes"],
        },
    ),
}


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def resolve_task_list(tasks: List[str]) -> List[str]:
    if "all" in tasks:
        return list(TASK_CONFIGS.keys())
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a GRPO-trained SmolLM checkpoint on GLUE.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to the GRPO checkpoint .pt file (produced by grpo_pretrain.py). Required unless --use-base-model is set.",
    )
    parser.add_argument("--model-name", default="HuggingFaceTB/SmolLM-135M", help="Base model repo id.")
    parser.add_argument(
        "--use-base-model",
        action="store_true",
        help="Skip loading a GRPO checkpoint and evaluate the base model weights.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list(TASK_CONFIGS.keys()) + ["all"],
        default=["all"],
        help=(
            "Space-separated list of supported GLUE tasks (sts-b cola mrpc rte wnli), "
            "or 'all' to run the full subset."
        ),
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split override (only applied when evaluating a single task).",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of evaluation examples.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for prompt generation.")
    parser.add_argument("--max-new-tokens", type=int, default=6, help="Max tokens to generate for answers.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 for greedy).")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p for sampling (if temperature > 0).")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device. 'auto' prefers CUDA, then MPS, then CPU.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument(
        "--save-predictions",
        default=None,
        help="Directory path to dump per-task prediction JSON files.",
    )
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_name: str, checkpoint: str, device: torch.device, vocab_size: int) -> AutoModelForCausalLM:
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=dtype)
    model.resize_token_embeddings(vocab_size)
    if checkpoint:
        checkpoint_obj = torch.load(checkpoint, map_location=device)
        state_dict = checkpoint_obj.get("model", checkpoint_obj)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys when loading checkpoint: {missing}")
        if unexpected:
            print(f"[WARN] Unexpected keys when loading checkpoint: {unexpected}")
    model.to(device)
    model.eval()
    return model


def prepare_dataset(task: str, split: Optional[str], max_samples: Optional[int]) -> Dataset:
    config = TASK_CONFIGS[task]
    ds = load_dataset("glue", task)
    chosen_split = split or config.default_split
    data = ds[chosen_split]
    if max_samples is not None:
        max_samples = min(max_samples, len(data))
        data = data.select(list(range(max_samples)))
    return data


def iterate_batches(dataset: Dataset, batch_size: int) -> Iterable[Tuple[int, List[Dict[str, Any]]]]:
    batch: List[Dict[str, Any]] = []
    start_index = 0
    for idx, example in enumerate(dataset):
        if not batch:
            start_index = idx
        batch.append(example)
        if len(batch) == batch_size:
            yield start_index, batch
            batch = []
    if batch:
        yield start_index, batch


def build_prompt(task: str, example: Dict[str, Any]) -> str:
    config = TASK_CONFIGS[task]
    return config.template.format(**example)


def parse_classification_answer(task: str, text: str) -> Optional[int]:
    text = text.strip().lower()
    if not text:
        return None
    config = TASK_CONFIGS[task]
    if config.label_words is None:
        return None
    for label_id, keywords in config.label_words.items():
        for keyword in keywords:
            if keyword in text:
                return label_id
    tokens = text.split()
    if tokens:
        token = tokens[0]
        for label_id, keywords in config.label_words.items():
            if token in keywords:
                return label_id
    return None


FLOAT_MATCH = re.compile(r"-?\d+(?:\.\d+)?")


def parse_regression_answer(text: str) -> Optional[float]:
    match = FLOAT_MATCH.search(text.replace(",", "."))
    if not match:
        return None
    value = float(match.group())
    return max(0.0, min(5.0, value))


def accuracy_score(labels: Sequence[int], preds: Sequence[Optional[int]]) -> float:
    correct = sum(int(p is not None and p == t) for t, p in zip(labels, preds))
    return correct / len(labels) if labels else 0.0


def binary_f1(labels: Sequence[int], preds: Sequence[Optional[int]], positive_label: int = 1) -> float:
    tp = fp = fn = 0
    for truth, pred in zip(labels, preds):
        pred_positive = pred == positive_label
        truth_positive = truth == positive_label
        if pred_positive and truth_positive:
            tp += 1
        elif pred_positive and not truth_positive:
            fp += 1
        elif not pred_positive and truth_positive:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def matthews_corrcoef(labels: Sequence[int], preds: Sequence[Optional[int]]) -> float:
    tp = tn = fp = fn = 0
    for truth, pred in zip(labels, preds):
        predicted = pred if pred is not None else 1 - truth
        if truth == 1 and predicted == 1:
            tp += 1
        elif truth == 0 and predicted == 0:
            tn += 1
        elif truth == 0 and predicted == 1:
            fp += 1
        else:
            fn += 1
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return (tp * tn - fp * fn) / denom


def compute_metrics(task: str, labels: List[int], preds: List[Optional[int]], scores: List[Optional[float]]) -> Dict[str, float]:
    if task == "cola":
        return {"matthews": matthews_corrcoef(labels, preds)}
    if task in {"rte", "wnli"}:
        return {"accuracy": accuracy_score(labels, preds)}
    if task == "mrpc":
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": binary_f1(labels, preds, positive_label=1),
        }
    if task == "stsb":
        labels_array = np.asarray(labels, dtype=np.float32)
        valid_scores = np.asarray([s if s is not None else 0.0 for s in scores], dtype=np.float32)
        pearson = pearsonr(labels_array, valid_scores)[0]
        spearman = spearmanr(labels_array, valid_scores)[0]
        mse = np.mean((labels_array - valid_scores) ** 2)
        return {"pearson": pearson, "spearman": spearman, "rmse": math.sqrt(mse)}
    return {"accuracy": accuracy_score(labels, preds)}


def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    task: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> Tuple[List[Optional[int]], List[Optional[float]], List[str], List[str]]:
    preds: List[Optional[int]] = []
    scores: List[Optional[float]] = []
    prompts_store: List[str] = []
    answers_store: List[str] = []

    do_sample = temperature > 0.0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs.update({"temperature": temperature, "top_p": top_p})

    total_batches = math.ceil(len(dataset) / batch_size)
    for start, batch in tqdm(
        iterate_batches(dataset, batch_size),
        total=total_batches,
        desc="Evaluating",
        ncols=100,
    ):
        batch_prompts = [build_prompt(task, example) for example in batch]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        input_lengths = attention_mask.sum(dim=1).tolist()

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict_in_generate=True,
                **gen_kwargs,
            )

        sequences = outputs.sequences
        completions: List[str] = []
        for seq, input_len in zip(sequences, input_lengths):
            completion_ids = seq[int(input_len) :]
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            completions.append(completion_text)

        for idx_in_batch, completion in enumerate(completions):
            example_idx = start + idx_in_batch
            prompts_store.append(batch_prompts[idx_in_batch])
            answers_store.append(completion)
            if TASK_CONFIGS[task].task_type == "classification":
                pred_label = parse_classification_answer(task, completion)
                preds.append(pred_label)
                scores.append(None)
            else:
                pred_score = parse_regression_answer(completion)
                preds.append(None)
                scores.append(pred_score)

    return preds, scores, prompts_store, answers_store


def save_predictions_for_task(
    task: str,
    output_dir: str,
    dataset: Dataset,
    preds: List[Optional[int]],
    scores: List[Optional[float]],
    prompts: List[str],
    completions: List[str],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    task_path = os.path.join(output_dir, f"{task}_predictions.json")
    records: List[Dict[str, Any]] = []
    for idx in range(len(prompts)):
        record: Dict[str, Any] = {
            "index": idx,
            "prompt": prompts[idx],
            "model_completion": completions[idx],
            "label": _to_python_scalar(dataset[idx]["label"]),
        }
        if TASK_CONFIGS[task].task_type == "classification":
            record["pred_label"] = preds[idx]
        else:
            record["pred_score"] = scores[idx]
        records.append(record)
    with open(task_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[info] Saved {len(records)} predictions to {task_path}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not args.use_base_model and not args.checkpoint:
        raise ValueError("Either provide --checkpoint or pass --use-base-model to evaluate the original model.")

    model = load_model(args.model_name, args.checkpoint if not args.use_base_model else None, device, len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id

    tasks_to_run = resolve_task_list(args.tasks)
    if len(tasks_to_run) > 1 and args.split:
        print("[warn] Ignoring --split override because multiple tasks were requested.")

    prediction_dir = args.save_predictions
    if prediction_dir:
        os.makedirs(prediction_dir, exist_ok=True)

    summary: List[Tuple[str, Dict[str, float]]] = []

    for task in tasks_to_run:
        split_override = args.split if (args.split and len(tasks_to_run) == 1) else None
        dataset = prepare_dataset(task, split_override, args.max_samples)

        preds, scores, prompts, completions = evaluate(
            model,
            tokenizer,
            dataset,
            task,
            args.batch_size,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            device,
        )

        labels = dataset["label"]
        metrics = compute_metrics(task, labels, preds, scores)
        summary.append((task, metrics))

        metric_readout = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"\n[{task}] {TASK_CONFIGS[task].name}: {metric_readout}")

        if prediction_dir:
            save_predictions_for_task(task, prediction_dir, dataset, preds, scores, prompts, completions)

    print("\n==== GLUE evaluation summary ====")
    for task, metrics in summary:
        metric_readout = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"{task}: {metric_readout}")


if __name__ == "__main__":
    main()
