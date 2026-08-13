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
from kv_action.data import ActionDataset, build_examples, canonical_action, load_records
from kv_action.model import KVConditionedPolicy


def tool_name(text):
    try:
        obj = json.loads(text)
        return obj.get("name") or obj.get("function", {}).get("name")
    except (ValueError, AttributeError):
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--checkpoint", required=True); p.add_argument("--backbone", required=True)
    p.add_argument("--batch-size", type=int, default=1); p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    examples = build_examples(load_records(args.data)); random.Random(args.seed).shuffle(examples)
    # A checkpoint trained with --limit must be evaluated on its corresponding
    # deterministic held-out subset, not on the full dataset's final 10%.
    effective_limit = args.limit or cfg.get("limit", 0)
    if effective_limit:
        examples = examples[:effective_limit]
    examples = examples[max(1, int(len(examples) * .9)):]
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    ds = ActionDataset(examples, tokenizer, cfg["max_context_tokens"], cfg["max_action_tokens"])
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=ds.collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    backbone = AutoModelForCausalLM.from_pretrained(args.backbone, torch_dtype=dtype).to(device)
    model = KVConditionedPolicy(backbone, cfg["layer_ids"], cfg["kv_tokens"], cfg["expert_width"], cfg["expert_layers"], cfg["expert_heads"], cfg.get("representation", "kv")).to(device)
    first = next(iter(loader))
    model(first.context_ids.to(device), first.context_mask.to(device), first.decoder_ids[:, :1].to(device))
    model.expert.load_state_dict(checkpoint["expert"]); model.eval()
    eos = tokenizer.eos_token_id
    predictions = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluating"):
            context_ids, context_mask = batch.context_ids.to(device), batch.context_mask.to(device)
            generated = torch.full((context_ids.size(0), 1), tokenizer.bos_token_id or eos, device=device, dtype=torch.long)
            finished = torch.zeros(context_ids.size(0), device=device, dtype=torch.bool)
            for _ in range(args.max_new_tokens):
                logits = model(context_ids, context_mask, generated).logits[:, -1]
                nxt = logits.argmax(-1)
                generated = torch.cat([generated, nxt[:, None]], 1)
                finished |= nxt.eq(eos)
                if finished.all(): break
            for ids, gold in zip(generated[:, 1:], batch.targets):
                pred = tokenizer.decode(ids.tolist(), skip_special_tokens=True).strip()
                predictions.append({"prediction": pred, "target": gold})
    valid = [tool_name(x["prediction"]) is not None for x in predictions]
    exact = [canonical_action(x["prediction"]) == canonical_action(x["target"]) for x in predictions]
    tool_acc = [tool_name(x["prediction"]) == tool_name(x["target"]) for x in predictions]
    result = {"n": len(predictions), "valid_json_rate": sum(valid) / len(valid), "tool_name_accuracy": sum(tool_acc) / len(tool_acc), "canonical_exact_match": sum(exact) / len(exact)}
    print(json.dumps(result, indent=2))
    out = Path(args.checkpoint).with_name("eval_predictions.json")
    out.write_text(json.dumps({"metrics": result, "predictions": predictions}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
