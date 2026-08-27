#!/usr/bin/env python
"""Measure tool-span decode speed: frozen Qwen3-4B vs a configured AE.

This is deliberately a compute microbenchmark, not a quality evaluation. The
default Action Expert is the random Qwen3-like tower. Passing a checkpoint
instead constructs and loads exactly the checkpoint's expert architecture.
Both paths are forced to decode the same fixed number of tokens, so neither
JSON validity nor early stopping changes the timing comparison. Both paths
start from an identical native-Qwen prefix ending in ``<tool_call>``; prefix
prefill is reported separately because it is common to both systems.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kv_action.data import build_examples, load_records
from kv_action.model import KVConditionedPolicy


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def prefill(backbone, ids, device):
    mask = torch.ones_like(ids)
    positions = torch.arange(ids.size(1), device=device).unsqueeze(0)
    sync(device)
    start = time.perf_counter()
    out = backbone(
        input_ids=ids, attention_mask=mask, position_ids=positions,
        use_cache=True, logits_to_keep=1, return_dict=True,
    )
    sync(device)
    return out, mask, time.perf_counter() - start


@torch.inference_mode()
def qwen_decode(backbone, first_out, prompt_length, steps, device):
    """Generate exactly ``steps`` tokens; exclude the common prefix prefill."""
    out, cache = first_out, first_out.past_key_values
    sync(device)
    start = time.perf_counter()
    for step in range(steps):
        nxt = out.logits[:, -1].argmax(-1)
        if step + 1 < steps:  # no need for a final, unused next-token forward
            full_length = prompt_length + step + 1
            out = backbone(
                input_ids=nxt[:, None],
                attention_mask=torch.ones((nxt.size(0), full_length), device=device, dtype=torch.long),
                past_key_values=cache,
                cache_position=torch.tensor([full_length - 1], device=device),
                logits_to_keep=1, use_cache=True, return_dict=True,
            )
            cache = out.past_key_values
    sync(device)
    return time.perf_counter() - start


def backbone_kvs(policy, cache):
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    return [(cache[layer][0].contiguous(), cache[layer][1].contiguous()) for layer in policy.selected_layers]


@torch.inference_mode()
def ae_decode(policy, cache, prefix_mask, steps, tokenizer, device):
    """Generate exactly ``steps`` random-AE tokens; exclude prefix prefill."""
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    current = torch.full((prefix_mask.size(0), 1), bos, device=device, dtype=torch.long)
    kvs, action_cache = backbone_kvs(policy, cache), None
    sync(device)
    start = time.perf_counter()
    for _ in range(steps):
        logits, action_cache = policy.decode_step_from_prefill(current, kvs, prefix_mask, action_cache)
        current = logits.argmax(-1)[:, None]
    sync(device)
    return time.perf_counter() - start


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--checkpoint", help="Optional AE checkpoint. Its saved architecture/configuration is used exactly.")
    p.add_argument("--decode-tokens", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1, help="Decode batch size. Each measured prompt is repeated to isolate batch compute from padding effects.")
    p.add_argument("--num-examples", type=int, default=4)
    p.add_argument("--warmup-examples", type=int, default=1)
    p.add_argument("--max-context-tokens", type=int, default=0, help="0 keeps every native prompt token.")
    p.add_argument("--output", help="Optional JSON report path.")
    args = p.parse_args()
    if args.decode_tokens < 2 or args.num_examples < 1 or args.batch_size < 1:
        raise ValueError("--decode-tokens, --num-examples, and --batch-size must all be positive (decode tokens >=2)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    backbone = AutoModelForCausalLM.from_pretrained(args.backbone, torch_dtype=dtype).to(device).eval()
    cfg = backbone.config
    if getattr(cfg, "model_type", None) != "qwen3":
        raise ValueError(f"This benchmark currently requires Qwen3; got {getattr(cfg, 'model_type', None)!r}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False) if args.checkpoint else None
    if checkpoint is None:
        # Exact architecture selected for the current formal AE: Qwen3 blocks,
        # width 1024, and one expert layer per Qwen3-4B KV layer.
        expert_cfg = {
            "layer_ids": list(range(cfg.num_hidden_layers)), "kv_tokens": None,
            "expert_width": 1024, "expert_layers": cfg.num_hidden_layers,
            "expert_heads": cfg.num_key_value_heads, "representation": "kv", "expert_arch": "qwen3",
        }
    else:
        saved = checkpoint["config"]
        expert_cfg = {
            "layer_ids": saved["layer_ids"], "kv_tokens": saved["kv_tokens"],
            "expert_width": saved["expert_width"], "expert_layers": saved["expert_layers"],
            "expert_heads": saved["expert_heads"], "representation": saved.get("representation", "kv"),
            # Checkpoints made before this field existed are the legacy custom AE.
            "expert_arch": saved.get("expert_arch", "custom"),
        }
    policy = KVConditionedPolicy(
        backbone, expert_cfg["layer_ids"], expert_cfg["kv_tokens"],
        expert_cfg["expert_width"], expert_cfg["expert_layers"], expert_cfg["expert_heads"],
        expert_cfg["representation"], expert_cfg["expert_arch"],
    ).to(device).eval()
    if checkpoint is not None:
        policy.expert.load_state_dict(checkpoint["expert"])
    records = load_records(args.data)
    examples = build_examples(records, tokenizer)
    take = args.warmup_examples + args.num_examples
    if len(examples) < take:
        raise ValueError(f"Need {take} tool-call examples, found {len(examples)}")
    print(
        f"AE benchmark; arch={expert_cfg['expert_arch']}; checkpoint={args.checkpoint or 'random initialization'}; "
        f"device={device}; decode_tokens={args.decode_tokens}; batch_size={args.batch_size}; "
        f"measured_examples={args.num_examples}; warmup={args.warmup_examples}."
    )

    results = []
    for index, example in enumerate(examples[:take]):
        token_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
        if args.max_context_tokens and len(token_ids) > args.max_context_tokens:
            token_ids = token_ids[-args.max_context_tokens :]
        # Repeating an identical native-Qwen prefix gives a controlled decode
        # batch: no sequence is padded up to another one's length. This is a
        # throughput measurement, not an estimate of an arbitrary serving mix.
        ids = torch.tensor([token_ids], device=device, dtype=torch.long).repeat(args.batch_size, 1)
        qwen_prefill, _, qwen_prefill_s = prefill(backbone, ids, device)
        qwen_s = qwen_decode(backbone, qwen_prefill, ids.size(1), args.decode_tokens, device)
        ae_prefill, ae_mask, ae_prefill_s = prefill(backbone, ids, device)
        ae_s = ae_decode(policy, ae_prefill.past_key_values, ae_mask, args.decode_tokens, tokenizer, device)
        if index >= args.warmup_examples:
            results.append({"prompt_tokens": ids.size(1), "qwen_prefill_s": qwen_prefill_s, "ae_prefill_s": ae_prefill_s, "qwen_decode_s": qwen_s, "ae_decode_s": ae_s})

    qwen_times, ae_times = [r["qwen_decode_s"] for r in results], [r["ae_decode_s"] for r in results]
    mean_qwen, mean_ae = statistics.mean(qwen_times), statistics.mean(ae_times)
    report = {
        "random_expert": checkpoint is None,
        "checkpoint": args.checkpoint,
        "expert": {"architecture": expert_cfg["expert_arch"], "layers": expert_cfg["expert_layers"], "width": expert_cfg["expert_width"], "kv_heads": expert_cfg["expert_heads"]},
        "decode_tokens": args.decode_tokens,
        "batch_size": args.batch_size,
        "examples": len(results),
        "mean_prompt_tokens": statistics.mean(r["prompt_tokens"] for r in results),
        "mean_qwen_prefill_s": statistics.mean(r["qwen_prefill_s"] for r in results),
        "mean_ae_prefill_s": statistics.mean(r["ae_prefill_s"] for r in results),
        "qwen_decode_s": mean_qwen,
        "ae_decode_s": mean_ae,
        "qwen_tokens_per_s": args.decode_tokens * args.batch_size / mean_qwen,
        "ae_tokens_per_s": args.decode_tokens * args.batch_size / mean_ae,
        "ae_over_qwen_decode_speedup": mean_qwen / mean_ae,
        "per_example": results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved report to {output}")


if __name__ == "__main__":
    main()
