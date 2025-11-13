# RLPT: SmolLM GRPO on CoLA

End-to-end playground for experimenting with reinforcement learning from policy optimization (GRPO-style) on the GLUE CoLA task using the 135M-parameter SmolLM model. The repository covers dataset construction, supervised fine-tuning, GRPO reward shaping based on sentence embeddings, and lightweight GLUE evaluation utilities.

## Project Layout

| Path | Description |
| --- | --- |
| `create_data.ipynb` | Notebook that builds `data/cola_with_scores.csv`, pre-computes normalized sentence embeddings in `text_embeddings.npy`, and surfaces exploratory stats. |
| `sft_cola.py` | Supervised fine-tuning script for SmolLM on CoLA with prompt-based labeling. Writes checkpoints under `sft_checkpoints/`. |
| `grpo_pretrain.py` | Proof-of-concept GRPO loop that samples multi-completion groups, scores them with cosine similarity rewards, and logs to TensorBoard. Outputs checkpoints to timestamped folders inside `grpo_checkpoints/`. |
| `evaluate_glue.py` | Prompt-based evaluation harness for CoLA, STS-B, MRPC, RTE, and WNLI. Optionally loads GRPO checkpoints and emits prediction JSON files into `glue_preds/`. |
| `cola_eval.json`, `glue_preds/` | Example prediction dumps for debugging or qualitative analysis. |
| `in_domain_train.tsv`, `more_sentences.txt` | Extra raw text sources you can feed into the notebook to diversify prompts. |

## Environment Setup

1. Install Python 3.10+ and ensure you have a CUDA- or MPS-capable PyTorch build if you want accelerator support.
2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Log into the Hugging Face Hub if you need gated model access:

```bash
huggingface-cli login
```

## Data Preparation

1. Launch `create_data.ipynb` (Jupyter or VS Code). The notebook:
   - Pulls the GLUE CoLA split.
   - Generates acceptability scores and optional heuristics (stored in the `scores` column).
   - Saves the processed rows to `data/cola_with_scores.csv`.
   - Encodes every sentence with `SentenceTransformer` to produce `text_embeddings.npy` (already L2-normalized for cosine similarity).
2. Inspect the CSV to confirm it has `text` and `scores` columns. The GRPO script enforces these names and length-checks each example.

You can swap in your own TSV/CSV files as long as you keep the column names consistent and regenerate embeddings to match the number of rows.

## Supervised Fine-Tuning (SFT)

`sft_cola.py` offers a baseline before RL:

```bash
python sft_cola.py \
  --model-name HuggingFaceTB/SmolLM-135M \
  --output-dir sft_checkpoints/SmolLM-135M-cola \
  --train-batch-size 4 \
  --eval-batch-size 4 \
  --num-train-epochs 3 \
  --bf16
```

Key options:

- `--prompt-template`: customize the task wording.
- `--max-length`: total token budget for prompt + label.
- `--max-train-samples` / `--max-eval-samples`: quick dry-runs.
- `--checkpoint-name`: filename for the serialized `torch.save` checkpoint (picked up by `evaluate_glue.py`).

Checkpoints and TensorBoard logs land in `sft_checkpoints/SmolLM-135M-cola/`.

## GRPO Pre-Training

`grpo_pretrain.py` implements a grouped RPO update:

```bash
python grpo_pretrain.py \
  --csv data/cola_with_scores.csv \
  --embeddings text_embeddings.npy \
  --model-name HuggingFaceTB/SmolLM-135M \
  --embedding-model all-MiniLM-L6-v2 \
  --batch-size 8 \
  --group-size 4 \
  --max-new-tokens 4 \
  --max-steps 2000 \
  --learning-rate 5e-6 \
  --temperature 0.9
```

How it works:

- Each CoLA sentence is randomly split into a prompt/completion pair that respects minimum token budgets.
- For every prompt, the model samples `group_size` completions; log-probs from the current policy are cached (`old_logprob`).
- Rewards combine z-scored perplexity (prefers fluent text) with cosine similarity between generated text embeddings and the original target embedding, scaled by the human-provided `scores`.
- Advantages are normalized within a group, following the GRPO recipe, and the loss clamps the log-prob ratio (`clip-range`) similarly to PPO/RPO.
- TensorBoard scalars and periodic checkpoints (`grpo_step_<N>.pt` plus `grpo_final.pt`) are stored under `grpo_checkpoints/<timestamp>/`.

You can resume training by pointing `--save-dir` to an existing folder and loading the desired checkpoint manually before launching the script.

## GLUE Evaluation

`evaluate_glue.py` measures zero-/few-shot accuracy using simple prompts:

```bash
python evaluate_glue.py \
  --checkpoint grpo_checkpoints/20251113_131457/grpo_final.pt \
  --tasks cola stsb mrpc rte wnli \
  --batch-size 4 \
  --max-new-tokens 6 \
  --temperature 0.0 \
  --save-predictions glue_preds
```

Highlights:

- Supports `--use-base-model` to benchmark the untouched SmolLM weights.
- Each task has an in-repo template (see `TASK_CONFIGS`) with label-word expansions, so qualitative outputs can differ from exact label strings.
- Regression metrics (STS-B) use Pearson/Spearman correlations; classification tasks compute accuracy.
- Setting `--max-samples` lets you shorten runs while iterating.
- When `--save-predictions` is set, per-task JSON files resembling `glue_preds/*.json` are written for error analysis. An example CoLA dump also lives at `cola_eval.json`.

## Logging & Artifacts

- `grpo_checkpoints/<timestamp>/events.out.tfevents...` can be inspected with TensorBoard (`tensorboard --logdir grpo_checkpoints`).
- `text_embeddings.npy` should line up exactly with the CSV row count; regenerate both if you change the dataset.
- `glue_preds/` and other JSON artifacts are ignored by training scripts but help track experiments.

## Tips & Troubleshooting

- **Hardware**: The scripts auto-detect CUDA, fall back to Metal (MPS) on macOS, then CPU. Override via `evaluate_glue.py --device`.
- **Tokenizer padding**: Both training scripts ensure `pad_token` equals `eos_token` when missing; if you try different models, verify they expose these tokens.
- **Sentence Transformer caching**: The GRPO reward model runs on the same device as the policy by default. If GPU memory is tight, move it to CPU by overriding `SentenceTransformer(..., device='cpu')`.
- **Longer completions**: Increase `--max-new-tokens` and adjust `group_size` and batch size to fit memory; the grouped sampling cost scales linearly with both knobs.

With the pieces above you can reproduce the baseline SFT run, experiment with custom reward formulations, and benchmark progress on selected GLUE tasks.
