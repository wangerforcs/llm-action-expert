# KV Action Expert prototype

This is the first PyTorch prototype for the idea in `Llm_action_expert.md`.
It freezes an instruction-tuned causal-LM backbone during **prefill**, extracts
its actual per-layer key/value cache, and trains a separate, small causal
Transformer to emit the next tool call.  The expert never receives the prompt
text or tool schema as tokens.  Its only context input is the backbone KV
memory.  Input/output token embeddings are borrowed from the frozen backbone,
so the expert does not acquire a misleadingly large vocabulary embedding/head.

## Recommended first experiment

Download the gated `Salesforce/APIGen-MT-5k` dataset (accept its CC-BY-NC-4.0
terms) and use `Qwen/Qwen3-4B` as the default backbone. It is a current Qwen3
dense model with 36 layers and GQA (32 attention heads / 8 KV heads), making it
a much more meaningful semantic backbone for this hypothesis. APIGen-MT
provides 5,000 verified multi-turn tool trajectories in the ShareGPT-like
format consumed by this repository. If VRAM is tight, use `Qwen/Qwen3-1.7B`
for debugging only; it is a 4.08 GB download.

```bash
conda activate vlash
pip install -r DUAL_LLM/requirements.txt
huggingface-cli login                         # needed for APIGen-MT terms
python DUAL_LLM/scripts/download_data.py \
  --dataset Salesforce/APIGen-MT-5k --out DUAL_LLM/data/apigen_mt_5k.json
huggingface-cli download Qwen/Qwen3-4B \
  --local-dir DUAL_LLM/models/Qwen3-4B
```

For the paper-scale setting, retain this exact implementation and switch the
backbone to a larger current Qwen3 model; use APIGen-MT for controlled
ablations, then add a normalized ToolBench trajectory dump. Do not train or evaluate on
APIGen-MT for commercial use: its stated license is non-commercial.

## Run

```bash
# Default formal run (H200-oriented; batch size 16).
bash DUAL_LLM/run.sh

# Fast structural check, no model or dataset download required.
python DUAL_LLM/scripts/smoke_test.py

# 90/10 deterministic split; first run is intentionally modest.
python DUAL_LLM/scripts/train.py \
  --data DUAL_LLM/data/apigen_mt_5k.json \
  --backbone Qwen/Qwen3-4B \
  --output DUAL_LLM/runs/qwen3_4b_kv \
  --max-context-tokens 4096 --max-action-tokens 192 \
  --expert-width 512 --expert-layers 4 --expert-heads 8 \
  --kv-layers auto --kv-tokens 256 --epochs 3 --batch-size 4 \
  --wandb-project dual-llm-kv-action --wandb-run-name qwen3-4b-kv-ar

python DUAL_LLM/scripts/evaluate.py \
  --data DUAL_LLM/data/apigen_mt_5k.json \
  --checkpoint DUAL_LLM/runs/qwen3_4b_kv/best.pt \
  --backbone Qwen/Qwen3-4B
```

`run.sh` accepts environment-variable overrides without editing it, for example
`BATCH_SIZE=8 RUN_NAME=qwen3-4b-kv-bs8 bash DUAL_LLM/run.sh`.

The training log reports loss; `evaluate.py` reports tool-name accuracy,
canonical JSON exact match, and valid-JSON rate with greedy decoding on the
same held-out split. The important interface ablation is a second training run
with `--representation last_hidden`; it gives the same expert only final hidden
states rather than attention KV.

Training writes step loss/learning rate and epoch train/validation loss to
Weights & Biases. Run `wandb login` once before an online run. Add
`--wandb-mode offline` on a machine without W&B network access, or
`--wandb-mode disabled` to turn tracking off. Step metrics are logged every 50
optimizer steps by default; tune this with `--wandb-log-interval`.

## Data contract

The loader accepts the official APIGen-MT records directly. Each
`function_call` becomes one supervised decision; its prompt contains the
system policy, tools, all preceding turns, and an assistant generation cue.
The target is a compact canonical JSON string, e.g.

```json
{"name":"get_reservation_details","arguments":{"reservation_id":"C6X779"}}
```

For another source, supply JSON/JSONL records with `prompt` and either
`action` (a JSON object/string) or `target` (string). This intentionally makes
ToolBench preprocessing an explicit, auditable conversion step rather than
silently assuming one of its many released formats.
