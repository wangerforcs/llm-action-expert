#!/usr/bin/env python
import argparse
import json
import random
from pathlib import Path
import sys

import torch
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kv_action.data import ActionDataset, build_examples, load_records
from kv_action.model import KVConditionedPolicy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-context-tokens", type=int, default=1536)
    p.add_argument("--max-action-tokens", type=int, default=192)
    p.add_argument("--expert-width", type=int, default=384)
    p.add_argument("--expert-layers", type=int, default=4)
    p.add_argument("--expert-heads", type=int, default=6)
    p.add_argument("--kv-layers", default="auto", help="Comma-separated backbone layers, or auto for five evenly spaced layers.")
    p.add_argument("--kv-tokens", type=int, default=128)
    p.add_argument("--representation", choices=["kv", "last_hidden"], default="kv")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--wandb-project", default="dual-llm-kv-action")
    p.add_argument("--wandb-entity", default=None, help="W&B team/user; omit to use your default entity.")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    p.add_argument("--wandb-log-interval", type=int, default=50, help="Log step metrics to W&B every N optimizer steps.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.wandb_log_interval < 1:
        raise ValueError("--wandb-log-interval must be at least 1")
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    examples = build_examples(load_records(args.data))
    random.Random(args.seed).shuffle(examples)
    if args.limit:
        examples = examples[: args.limit]
    cut = max(1, int(len(examples) * 0.9))
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=False)
    train_ds = ActionDataset(examples[:cut], tokenizer, args.max_context_tokens, args.max_action_tokens)
    valid_ds = ActionDataset(examples[cut:], tokenizer, args.max_context_tokens, args.max_action_tokens)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=train_ds.collate)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, collate_fn=valid_ds.collate)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16 if device.type == "cuda" else torch.float32
    backbone = AutoModelForCausalLM.from_pretrained(args.backbone, torch_dtype=dtype, trust_remote_code=False).to(device)
    n_layers = backbone.config.num_hidden_layers
    layer_ids = (
        sorted(set(round(i * (n_layers - 1) / 4) for i in range(5)))
        if args.kv_layers == "auto"
        else [int(x) for x in args.kv_layers.split(",")]
    )
    if any(x < 0 or x >= n_layers for x in layer_ids):
        raise ValueError(f"--kv-layers must be in [0, {n_layers - 1}]")
    model = KVConditionedPolicy(backbone, layer_ids, args.kv_tokens, args.expert_width, args.expert_layers, args.expert_heads, args.representation).to(device)
    # Materialize per-layer adapters before constructing optimizer/checkpoint state.
    bootstrap = next(iter(train_loader))
    model(bootstrap.context_ids.to(device), bootstrap.context_mask.to(device), bootstrap.decoder_ids.to(device))
    optimizer = AdamW(model.expert.parameters(), lr=args.lr, weight_decay=0.01)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"layer_ids": layer_ids, "num_examples": len(examples)}
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        dir=str(out),
        config=config,
        mode=args.wandb_mode,
    )
    best = float("inf")
    global_step = 0
    for epoch in range(args.epochs):
        model.train(); total = 0.0
        for batch in tqdm(train_loader, desc=f"train {epoch + 1}"):
            optimizer.zero_grad(set_to_none=True)
            result = model(batch.context_ids.to(device), batch.context_mask.to(device), batch.decoder_ids.to(device), batch.labels.to(device))
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.expert.parameters(), 1.0)
            optimizer.step(); total += result.loss.item(); global_step += 1
            if global_step % args.wandb_log_interval == 0:
                wandb.log({"train/loss_step": result.loss.item(), "train/lr": optimizer.param_groups[0]["lr"], "epoch": epoch + 1}, step=global_step)
        model.eval(); valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                result = model(batch.context_ids.to(device), batch.context_mask.to(device), batch.decoder_ids.to(device), batch.labels.to(device))
                valid_loss += result.loss.item()
        valid_loss /= max(1, len(valid_loader))
        metrics = {"epoch": epoch + 1, "train/loss_epoch": total / max(1, len(train_loader)), "valid/loss": valid_loss}
        print(json.dumps(metrics))
        wandb.log(metrics, step=global_step)
        if valid_loss < best:
            best = valid_loss
            torch.save({"expert": model.expert.state_dict(), "config": config}, out / "best.pt")
            run.summary["best_valid_loss"] = best
    run.finish()


if __name__ == "__main__":
    main()
