#!/usr/bin/env python
import argparse
import json
import random
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kv_action.data import TOOL_CALL_CLOSE, ActionDataset, build_examples, canonical_action, load_records, split_trajectories
from kv_action.model import KVConditionedPolicy


def tool_name(text):
    try:
        obj = json.loads(text)
        return obj.get("name") or obj.get("function", {}).get("name")
    except (ValueError, AttributeError):
        return None


def first_json_object(text):
    """Return a canonical first JSON object from a raw AR continuation."""
    try:
        value, _ = json.JSONDecoder().raw_decode(text.lstrip())
        if isinstance(value, dict):
            return canonical_action(value)
    except json.JSONDecodeError:
        pass
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--checkpoint", required=True); p.add_argument("--backbone", required=True)
    p.add_argument("--batch-size", type=int, default=None, help="Evaluation batch size; defaults to the training batch size stored in checkpoint.")
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--trajectory-index", type=int, default=None, help="Evaluate all tool-call decisions from one original APIGen trajectory; skips the 90/10 split.")
    p.add_argument("--print-predictions", action="store_true", help="Print raw and parsed continuations; useful with --trajectory-index.")
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    batch_size = args.batch_size or cfg.get("batch_size", 1)
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    records = load_records(args.data)
    if args.trajectory_index is not None:
        if not 0 <= args.trajectory_index < len(records):
            raise ValueError(f"--trajectory-index must be in [0, {len(records) - 1}]")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    if args.trajectory_index is not None:
        examples = build_examples([records[args.trajectory_index]], tokenizer, cfg.get("include_think_actions", False))
        print(f"Evaluating trajectory {args.trajectory_index}: {len(examples)} external tool-call decisions")
    else:
        if cfg.get("split_strategy") != "trajectory":
            raise ValueError("This checkpoint used the legacy decision-level split; use a newly trained trajectory-split checkpoint for valid evaluation.")
        _, valid_records = split_trajectories(records, cfg.get("seed", args.seed), cfg.get("val_fraction", 0.1))
        examples = build_examples(valid_records, tokenizer, cfg.get("include_think_actions", False))
        if args.limit:
            examples = examples[:args.limit]
    max_context_tokens = None if cfg["max_context_tokens"] == "all" else cfg["max_context_tokens"]
    ds = ActionDataset(examples, tokenizer, max_context_tokens, cfg["max_action_tokens"])
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=ds.collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    backbone = AutoModelForCausalLM.from_pretrained(args.backbone, torch_dtype=dtype).to(device)
    model = KVConditionedPolicy(
        backbone, cfg["layer_ids"], cfg["kv_tokens"], cfg["expert_width"], int(cfg["expert_layers"]),
        cfg["expert_heads"], cfg.get("representation", "kv"), cfg.get("expert_arch", "custom"),
    ).to(device)
    first = next(iter(loader))
    model(first.context_ids.to(device), first.context_mask.to(device), first.decoder_ids[:, :1].to(device))
    model.expert.load_state_dict(checkpoint["expert"]); model.eval()
    eos = tokenizer.eos_token_id
    close_ids = tokenizer(TOOL_CALL_CLOSE.strip(), add_special_tokens=False)["input_ids"]
    if not close_ids:
        raise RuntimeError("Tokenizer did not encode </tool_call> terminator")
    predictions = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluating"):
            context_ids, context_mask = batch.context_ids.to(device), batch.context_mask.to(device)
            # Qwen prefill is the costly stage. It is invariant across all
            # action tokens for this decision, so compute it exactly once.
            kvs, prefix_mask = model.prefill(context_ids, context_mask)
            generated = torch.full((context_ids.size(0), 1), tokenizer.bos_token_id or eos, device=device, dtype=torch.long)
            finished = torch.zeros(context_ids.size(0), device=device, dtype=torch.bool)
            action_kvs = None
            for _ in range(args.max_new_tokens):
                logits, action_kvs = model.decode_step_from_prefill(generated[:, -1:], kvs, prefix_mask, action_kvs)
                # Keep completed rows at EOS while the other rows finish.
                # This permits batched AR decoding without altering completed
                # predictions; decode below explicitly truncates at first EOS.
                nxt = torch.where(finished, torch.full_like(logits.argmax(-1), eos), logits.argmax(-1))
                generated = torch.cat([generated, nxt[:, None]], 1)
                closes_tool_call = torch.zeros_like(finished)
                if generated.size(1) - 1 >= len(close_ids):
                    suffix = generated[:, -len(close_ids) :]
                    closes_tool_call = suffix.eq(torch.tensor(close_ids, device=device)).all(dim=-1)
                finished |= nxt.eq(eos) | closes_tool_call
                if finished.all(): break
            for ids, gold in zip(generated[:, 1:], batch.targets):
                token_ids = ids.tolist()
                if eos in token_ids:
                    token_ids = token_ids[: token_ids.index(eos)]
                raw = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                pred = first_json_object(raw)
                predictions.append({"raw_prediction": raw, "prediction": pred, "target": gold})
    valid = [tool_name(x["prediction"]) is not None for x in predictions]
    exact = [canonical_action(x["prediction"]) == canonical_action(x["target"]) for x in predictions]
    tool_acc = [tool_name(x["prediction"]) == tool_name(x["target"]) for x in predictions]
    result = {"n": len(predictions), "valid_json_rate": sum(valid) / len(valid), "tool_name_accuracy": sum(tool_acc) / len(tool_acc), "canonical_exact_match": sum(exact) / len(exact)}
    print(json.dumps(result, indent=2))
    if args.print_predictions:
        for index, item in enumerate(predictions):
            print(f"\n--- prediction {index} raw ---\n{item['raw_prediction']}")
            print(f"--- prediction {index} parsed ---\n{item['prediction']}")
            print(f"--- prediction {index} target ---\n{item['target']}")
    suffix = f"eval_trajectory_{args.trajectory_index}.json" if args.trajectory_index is not None else "eval_predictions.json"
    out = Path(args.checkpoint).with_name(suffix)
    out.write_text(json.dumps({"metrics": result, "predictions": predictions}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
