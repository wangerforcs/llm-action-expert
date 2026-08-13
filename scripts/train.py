#!/usr/bin/env python
import argparse
import json
import math
import os
import random
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kv_action.data import ActionDataset, build_examples, load_records, split_trajectories
from kv_action.model import KVConditionedPolicy


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Shuffle batches while keeping similarly long native prompts together."""

    def __init__(self, examples, batch_size: int, seed: int):
        self.lengths = [len(item["prompt"]) for item in examples]
        self.batch_size, self.seed, self.epoch = batch_size, seed, 0

    def __iter__(self):
        order = sorted(range(len(self.lengths)), key=self.lengths.__getitem__)
        batches = [order[i : i + self.batch_size] for i in range(0, len(order), self.batch_size)]
        rng = random.Random(self.seed + self.epoch)
        for batch in batches:
            rng.shuffle(batch)
        rng.shuffle(batches)
        self.epoch += 1
        yield from batches

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


class DistributedLengthBucketBatchSampler(LengthBucketBatchSampler):
    """Length buckets sharded by DDP rank, without duplicate training batches."""

    def __init__(self, examples, batch_size: int, seed: int, rank: int, world_size: int):
        super().__init__(examples, batch_size, seed)
        self.rank, self.world_size = rank, world_size

    def __iter__(self):
        order = sorted(range(len(self.lengths)), key=self.lengths.__getitem__)
        batches = [order[i : i + self.batch_size] for i in range(0, len(order), self.batch_size)]
        # Each rank must execute the same count of backward/all-reduce calls.
        usable = len(batches) - (len(batches) % self.world_size)
        batches = batches[:usable]
        rng = random.Random(self.seed + self.epoch)
        for batch in batches:
            rng.shuffle(batch)
        rng.shuffle(batches)
        self.epoch += 1
        yield from batches[self.rank :: self.world_size]

    def __len__(self):
        total_batches = (len(self.lengths) + self.batch_size - 1) // self.batch_size
        return (total_batches // self.world_size)


def distributed_context() -> tuple[int, int, int]:
    """Initialize torchrun/DDP if launched with WORLD_SIZE > 1."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return 0, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank, dist.get_rank(), world_size


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-context-tokens", default="all", help="Native prompt tokens retained: positive integer or all (default).")
    p.add_argument("--max-action-tokens", type=int, default=192)
    p.add_argument("--expert-width", type=int, default=1024)
    p.add_argument("--expert-layers", default="36", help="Action-tower depth; 36 aligns one-to-one with Qwen3-4B's layers.")
    p.add_argument("--expert-heads", type=int, default=8)
    p.add_argument("--kv-layers", default="all", help="Comma-separated layers, auto for five probes, all for one-to-one Qwen3 alignment, or paired for downsampled mapping.")
    p.add_argument("--kv-tokens", default="all", help="Backbone KV tokens retained per layer; an integer suffix length or all input context tokens.")
    p.add_argument("--representation", choices=["kv", "last_hidden"], default="kv")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=100, help="Linear LR warmup updates before cosine decay.")
    p.add_argument("--min-lr-ratio", type=float, default=0.1, help="Final cosine-decay LR as a fraction of --lr.")
    p.add_argument("--resume-from", default=None, help="Checkpoint whose Action Expert weights initialize this run; optimizer/scheduler restart.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--length-bucketing", action=argparse.BooleanOptionalAction, default=True, help="Group similar prompt lengths to avoid padding waste.")
    p.add_argument("--include-think-actions", action="store_true", help="Also train on APIGen's synthetic `think` function; off by default.")
    p.add_argument("--wandb-project", default="dual-llm-kv-action")
    p.add_argument("--wandb-entity", default=None, help="W&B team/user; omit to use your default entity.")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    p.add_argument("--wandb-log-interval", type=int, default=50, help="Log step metrics to W&B every N optimizer steps.")
    return p.parse_args()


def main():
    args = parse_args()
    local_rank, rank, world_size = distributed_context()
    is_main = rank == 0
    if args.wandb_log_interval < 1:
        raise ValueError("--wandb-log-interval must be at least 1")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("--min-lr-ratio must lie in [0, 1]")
    if args.kv_tokens != "all":
        try:
            args.kv_tokens = int(args.kv_tokens)
        except ValueError as exc:
            raise ValueError("--kv-tokens must be a positive integer or 'all'") from exc
        if args.kv_tokens < 1:
            raise ValueError("--kv-tokens must be a positive integer or 'all'")
    if args.max_context_tokens != "all":
        try:
            args.max_context_tokens = int(args.max_context_tokens)
        except ValueError as exc:
            raise ValueError("--max-context-tokens must be a positive integer or 'all'") from exc
        if args.max_context_tokens < 1:
            raise ValueError("--max-context-tokens must be a positive integer or 'all'")
    random.seed(args.seed + rank); torch.manual_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if world_size > 1 else "cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=False)
    records = load_records(args.data)
    train_records, valid_records = split_trajectories(records, args.seed, args.val_fraction)
    if args.limit:
        train_records = train_records[:args.limit]
        # Keep quick sanity runs genuinely quick while preserving a separate
        # trajectory-level validation set.
        valid_records = valid_records[: max(1, round(args.limit * args.val_fraction / (1 - args.val_fraction)))]
    train_examples = build_examples(train_records, tokenizer, args.include_think_actions)
    valid_examples = build_examples(valid_records, tokenizer, args.include_think_actions)
    max_context_tokens = None if args.max_context_tokens == "all" else args.max_context_tokens
    train_ds = ActionDataset(train_examples, tokenizer, max_context_tokens, args.max_action_tokens)
    valid_ds = ActionDataset(valid_examples, tokenizer, max_context_tokens, args.max_action_tokens)
    if args.length_bucketing:
        train_sampler = (
            DistributedLengthBucketBatchSampler(train_examples, args.batch_size, args.seed, rank, world_size)
            if world_size > 1 else LengthBucketBatchSampler(train_examples, args.batch_size, args.seed)
        )
        valid_sampler = (
            DistributedLengthBucketBatchSampler(valid_examples, args.batch_size, args.seed + 10_000, rank, world_size)
            if world_size > 1 else LengthBucketBatchSampler(valid_examples, args.batch_size, args.seed + 10_000)
        )
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=train_ds.collate)
        valid_loader = DataLoader(valid_ds, batch_sampler=valid_sampler, collate_fn=valid_ds.collate)
    else:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if world_size > 1 else None
        valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True) if world_size > 1 else None
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, shuffle=train_sampler is None, collate_fn=train_ds.collate)
        valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, sampler=valid_sampler, shuffle=False, collate_fn=valid_ds.collate)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16 if device.type == "cuda" else torch.float32
    backbone = AutoModelForCausalLM.from_pretrained(args.backbone, torch_dtype=dtype, trust_remote_code=False).to(device)
    n_layers = backbone.config.num_hidden_layers
    if args.kv_layers == "paired":
        pair_depth = len(range(18)) if args.expert_layers == "auto" else int(args.expert_layers)
        if pair_depth < 1 or pair_depth > n_layers:
            raise ValueError(f"paired KV mapping needs expert depth in [1, {n_layers}]")
        layer_ids = sorted(set(round(i * (n_layers - 1) / max(1, pair_depth - 1)) for i in range(pair_depth)))
    else:
        layer_ids = (
        list(range(n_layers)) if args.kv_layers == "all" else
        sorted(set(round(i * (n_layers - 1) / 4) for i in range(5)))
        if args.kv_layers == "auto"
        else [int(x) for x in args.kv_layers.split(",")]
        )
    if any(x < 0 or x >= n_layers for x in layer_ids):
        raise ValueError(f"--kv-layers must be in [0, {n_layers - 1}]")
    if args.representation == "last_hidden":
        # The hidden-state interface has a single memory source by definition.
        layer_ids, expert_layers = [n_layers - 1], 1
    else:
        expert_layers = len(layer_ids) if args.expert_layers == "auto" else int(args.expert_layers)
    if args.representation == "kv" and expert_layers != len(layer_ids):
        raise ValueError("For layer-synchronous KV mode, --expert-layers must equal selected KV layers (use auto).")
    kv_tokens = None if args.kv_tokens == "all" else args.kv_tokens
    policy = KVConditionedPolicy(backbone, layer_ids, kv_tokens, args.expert_width, expert_layers, args.expert_heads, args.representation).to(device)
    if args.resume_from is not None:
        resume_path = Path(args.resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError(f"--resume-from checkpoint not found: {resume_path}")
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        if "expert" not in resume:
            raise ValueError(f"Checkpoint has no Action Expert state: {resume_path}")
        policy.expert.load_state_dict(resume["expert"], strict=True)
        if is_main:
            print(f"Initialized Action Expert weights from {resume_path}; optimizer and scheduler start fresh.")
    model = DDP(policy, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False) if world_size > 1 else policy
    optimizer = AdamW(policy.expert.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(args.warmup_steps, max(0, total_steps - 1))

    def lr_multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_multiplier)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "kv_tokens": kv_tokens,
        "layer_ids": layer_ids,
        "expert_layers": expert_layers,
        "split_strategy": "trajectory",
        "num_train_trajectories": len(train_records),
        "num_valid_trajectories": len(valid_records),
        "num_train_examples": len(train_examples),
        "num_valid_examples": len(valid_examples),
    }
    if is_main:
        (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            dir=str(out),
            config=config | {"world_size": world_size, "global_batch_size": args.batch_size * world_size},
            mode=args.wandb_mode,
        )
    else:
        run = None
    best = float("inf")
    global_step = 0
    for epoch in range(args.epochs):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        model.train(); total = 0.0; train_batches = 0
        for batch in tqdm(train_loader, desc=f"train {epoch + 1}", disable=not is_main):
            optimizer.zero_grad(set_to_none=True)
            result = model(batch.context_ids.to(device), batch.context_mask.to(device), batch.decoder_ids.to(device), batch.labels.to(device))
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.expert.parameters(), 1.0)
            optimizer.step(); scheduler.step(); total += result.loss.item(); train_batches += 1; global_step += 1
            if is_main and global_step % args.wandb_log_interval == 0:
                wandb.log({"train/loss_step": result.loss.item(), "train/lr": optimizer.param_groups[0]["lr"], "epoch": epoch + 1}, step=global_step)
        if hasattr(valid_sampler, "set_epoch"):
            valid_sampler.set_epoch(epoch)
        model.eval(); valid_loss = 0.0; valid_batches = 0
        with torch.no_grad():
            for batch in valid_loader:
                result = model(batch.context_ids.to(device), batch.context_mask.to(device), batch.decoder_ids.to(device), batch.labels.to(device))
                valid_loss += result.loss.item(); valid_batches += 1
        totals = torch.tensor([total, train_batches, valid_loss, valid_batches], device=device, dtype=torch.float64)
        if world_size > 1:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        train_epoch_loss = (totals[0] / totals[1].clamp_min(1)).item()
        valid_loss = (totals[2] / totals[3].clamp_min(1)).item()
        metrics = {"epoch": epoch + 1, "train/loss_epoch": train_epoch_loss, "valid/loss": valid_loss}
        if is_main:
            print(json.dumps(metrics))
            wandb.log(metrics, step=global_step)
        if is_main and valid_loss < best:
            best = valid_loss
            torch.save({"expert": policy.expert.state_dict(), "config": config}, out / "best.pt")
            run.summary["best_valid_loss"] = best
        if is_main:
            torch.save({"expert": policy.expert.state_dict(), "config": config}, out / "last.pt")
            torch.save({"expert": policy.expert.state_dict(), "config": config}, out / f"epoch_{epoch + 1:03d}.pt")
    if is_main:
        run.finish()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
