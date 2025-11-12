"""PoC GRPO pre-training script for SmolLM on CoLA sentences with custom reward."""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO pre-training PoC for SmolLM")
    parser.add_argument("--csv", default="cola_with_scores.csv", help="CSV file with text and score columns")
    parser.add_argument("--embeddings", default="text_embeddings.npy", help="NumPy file with sentence embeddings")
    parser.add_argument("--model-name", default="HuggingFaceTB/SmolLM-135M", help="HF model repo")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4, help="Number of completions per prompt for GRPO")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200, help="Stop after this many GRPO updates")
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--clip-range", type=float, default=0.2, help="Clamp applied to log-prob ratio (RPO style)")
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--adv-std-eps", type=float, default=1e-4, help="Std clamp for group advantages")
    parser.add_argument("--save-dir", default="grpo_checkpoints")
    return parser.parse_args()


class ColaDataset(Dataset):
    def __init__(self, csv_path: str, embedding_path: str):
        df = pd.read_csv(csv_path)
        if not {"text", "scores"}.issubset(df.columns):
            raise ValueError("CSV must contain 'text' and 'scores' columns")
        embeddings = np.load(embedding_path)
        if len(df) != len(embeddings):
            raise ValueError("CSV rows and embedding rows differ")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
        embeddings = embeddings / norms
        self.texts: List[str] = df["text"].tolist()
        self.scores = torch.tensor(df["scores"].values, dtype=torch.float32)
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "text": self.texts[idx],
            "score": self.scores[idx],
            "embedding": self.embeddings[idx],
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    texts = [item["text"] for item in batch]
    scores = torch.stack([item["score"] for item in batch])
    embeddings = torch.stack([item["embedding"] for item in batch])
    return {"text": texts, "score": scores, "embedding": embeddings}


def compute_logprobs(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)
    target_ids = input_ids[:, 1:].contiguous()
    target_mask = attention_mask[:, 1:].contiguous()
    log_probs = log_probs[:, :-1, :]
    gathered = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    gathered = gathered * target_mask
    token_counts = target_mask.sum(dim=1).clamp(min=1).to(gathered.dtype)
    seq_logprob = gathered.sum(dim=1)
    ppl = torch.exp(-gathered.sum(dim=1) / token_counts)
    return seq_logprob, ppl, gathered, token_counts, outputs


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ColaDataset(args.csv, args.embeddings)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    st_device = "mps" if torch.backends.mps.is_available() else "cpu"
    sentence_model = SentenceTransformer(args.embedding_model, device=st_device)

    os.makedirs(args.save_dir, exist_ok=True)

    step = 0
    for epoch in range(args.epochs):
        for batch in dataloader:
            model.train()

            scores = batch["score"].to(device)
            orig_embeddings = batch["embedding"].to(device)
            batch_size = scores.size(0)
            group_size = args.group_size

            bos = torch.full((batch_size, 1), tokenizer.bos_token_id, dtype=torch.long, device=device)
            bos = bos.repeat_interleave(group_size, dim=0)
            scores_rep = scores.repeat_interleave(group_size, dim=0)
            orig_embeddings_rep = orig_embeddings.repeat_interleave(group_size, dim=0)

            with torch.no_grad():
                generated = model.generate(
                    input_ids=bos,
                    attention_mask=torch.ones_like(bos),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            attention_mask = (generated != tokenizer.pad_token_id).long()

            with torch.no_grad():
                old_logprob, perplexity, _, _, _ = compute_logprobs(model, generated, attention_mask)
                generated_texts = tokenizer.batch_decode(generated[:, 1:], skip_special_tokens=True)
                gen_embeddings = sentence_model.encode(
                    generated_texts, convert_to_tensor=True, normalize_embeddings=True
                ).to(device)
                similarity = F.cosine_similarity(gen_embeddings, orig_embeddings_rep, dim=-1)
                reward = scores_rep * perplexity + (1 - scores_rep) * similarity

            new_logprob, _, _, token_counts, outputs = compute_logprobs(model, generated, attention_mask)
            logits = outputs.logits[:, :-1, :]
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
            entropy = (entropy * attention_mask[:, 1:]).sum(dim=1) / token_counts
            entropy = entropy.mean()

            reward_groups = reward.view(batch_size, group_size)
            reward_mean = reward_groups.mean(dim=1, keepdim=True)
            reward_std = reward_groups.std(dim=1, keepdim=True)
            reward_std = torch.clamp(reward_std, min=args.adv_std_eps)
            advantages = ((reward_groups - reward_mean) / reward_std).view(-1).detach()

            log_ratio = new_logprob - old_logprob
            clipped_log_ratio = torch.clamp(log_ratio, -args.clip_range, args.clip_range)
            policy_loss = -(advantages * clipped_log_ratio).mean()

            loss = policy_loss - args.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            step += 1
            if step % args.log_every == 0:
                avg_reward = reward.mean().item()
                avg_ppl = perplexity.mean().item()
                avg_sim = similarity.mean().item()
                print(
                    f"step={step} epoch={epoch} loss={loss.item():.4f} reward={avg_reward:.4f} "
                    f"ppl={avg_ppl:.2f} sim={avg_sim:.3f}"
                )

            if step % (args.log_every * 5) == 0:
                ckpt_path = os.path.join(args.save_dir, f"grpo_step_{step}.pt")
                torch.save({
                    "model": model.state_dict(),
                    "step": step,
                }, ckpt_path)

            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    final_ckpt = os.path.join(args.save_dir, "grpo_final.pt")
    torch.save({"model": model.state_dict(), "step": step}, final_ckpt)
    print(f"Saved final checkpoint to {final_ckpt}")


if __name__ == "__main__":
    main()
