# KV Action Expert prototype

This is the first PyTorch prototype for the idea in `idea_notes/Llm_action_expert.md`.
It freezes an instruction-tuned causal-LM backbone during **prefill**, extracts
its actual per-layer key/value cache, and trains a separate, small causal
Transformer to emit the next tool call. Context is rendered with Qwen3's native
tool template and ends exactly at `<tool_call>`; the expert predicts only the
JSON body and literal `</tool_call>` terminator, never `<think>` content. The
expert never receives prompt text or tool schema as tokens—only the backbone KV
memory. Input/output token embeddings are borrowed from the frozen backbone.

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
# Default formal run: two-GPU DDP (H200-oriented; batch size 16 per GPU).
bash DUAL_LLM/run.sh

# Fast structural check, no model or dataset download required.
python DUAL_LLM/scripts/smoke_test.py

# 90/10 deterministic split; first run is intentionally modest.
python DUAL_LLM/scripts/train.py \
  --data DUAL_LLM/data/apigen_mt_5k.json \
  --backbone Qwen/Qwen3-4B \
  --output DUAL_LLM/runs/qwen3_4b_kv \
  --max-context-tokens all --max-action-tokens 192 \
  --expert-width 1024 --expert-layers 36 --expert-heads 8 \
  --kv-layers all --kv-tokens all --epochs 3 --batch-size 16 \
  --wandb-project dual-llm-kv-action --wandb-run-name qwen3-4b-kv-ar

python DUAL_LLM/scripts/evaluate.py \
  --data DUAL_LLM/data/apigen_mt_5k.json \
  --checkpoint DUAL_LLM/runs/qwen3_4b_kv/best.pt \
  --backbone Qwen/Qwen3-4B
```

`run.sh` launches PyTorch DDP with two GPUs by default. `BATCH_SIZE` is per GPU,
so the default effective global batch is 32. It accepts environment-variable
overrides without editing it, for example
`CUDA_VISIBLE_DEVICES=0,1 BATCH_SIZE=8 RUN_NAME=qwen3-4b-kv-bs8 bash DUAL_LLM/run.sh`.
Use `NUM_GPUS=1` for a single-card debug run.

The default optimizer uses 100 linear warmup updates followed by cosine decay
to 10% of the initial learning rate. To initialize a new run from a previous
Action Expert checkpoint while restarting optimizer and scheduler state, use
`RESUME_FROM=runs/old/best.pt RUN_NAME=continued bash run.sh` from `DUAL_LLM/`.
At every epoch, rank 0 writes `epoch_001.pt`, `epoch_002.pt`, and so on; it also
maintains `last.pt` and validation-selected `best.pt`.

The default is a Pi05-inspired, layer-synchronous pair: Pi05 has a full-depth
but narrower Gemma-300M Action Expert beside its Gemma-2B backbone. We preserve
that principle for Qwen3-4B: the Action Expert has a Qwen-aligned 36 layers,
but width 1024 rather than Qwen's 2560. Its roughly 600M Transformer trunk is
therefore close to Pi05's backbone/expert capacity ratio. It has eight 128-d
heads, exactly matching Qwen3-4B's eight KV heads and 128-d head size, so each
expert layer directly consumes its corresponding layer's unprojected KV cache
as Pi05 does. `--max-context-tokens all --kv-tokens all` retains every native
Qwen prompt token and every corresponding KV cache entry; this is the default.
Length-aware batching groups similarly sized trajectories to avoid padding
short inputs up to a long one. The frozen prefill calls Qwen's decoder directly
and never materializes vocabulary logits for prefix tokens; only the KV cache is
kept. The training log reports loss; `evaluate.py` reports tool-name accuracy,
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

The loader accepts the official APIGen-MT records directly. It first performs a
deterministic 90/10 split over the 5,000 original trajectories, then expands
each split separately into decisions—so no trajectory crosses train/validation.
By default APIGen's synthetic `think` function is excluded from both tool schema
and supervision; it is not an external tool action. Every remaining decision's
native-Qwen prompt contains system policy, tools, all preceding external calls
and observations, and ends at `<tool_call>`. Its target JSON body is, e.g.

```json
{"name":"get_reservation_details","arguments":{"reservation_id":"C6X779"}}
```

The decoder training sequence appends `</tool_call>` after that JSON; evaluation
stops at this literal tag rather than relying on EOS.

For another source, supply JSON/JSONL records with `prompt` and either
`action` (a JSON object/string) or `target` (string). This intentionally makes
ToolBench preprocessing an explicit, auditable conversion step rather than
silently assuming one of its many released formats.
