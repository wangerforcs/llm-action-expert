import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


TOOL_CALL_OPEN = "<tool_call>\n"
TOOL_CALL_CLOSE = "\n</tool_call>"


def canonical_action(value: Any) -> str:
    """Serialize the JSON body of a tool call deterministically."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_records(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty data file: {path}")
    return json.loads(text) if text[0] == "[" else [json.loads(line) for line in text.splitlines() if line.strip()]


def split_trajectories(records: list[dict[str, Any]], seed: int, val_fraction: float = 0.1) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic trajectory-level split; no trajectory crosses train/valid."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    n_valid = max(1, round(len(records) * val_fraction))
    valid_ids = set(indices[-n_valid:])
    return ([r for i, r in enumerate(records) if i not in valid_ids], [r for i, r in enumerate(records) if i in valid_ids])


def _tool_spec(record: dict[str, Any], include_think_actions: bool) -> list[dict[str, Any]]:
    tools = record.get("tools", [])
    tools = json.loads(tools) if isinstance(tools, str) else tools
    return tools if include_think_actions else [tool for tool in tools if tool.get("name") != "think"]


def _tool_message(raw: Any) -> dict[str, Any]:
    action = json.loads(raw) if isinstance(raw, str) else raw
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"type": "function", "function": {"name": action["name"], "arguments": action.get("arguments", {})}}
        ],
    }


def render_qwen_tool_prefix(tokenizer, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    """Exact Qwen native template, stopped immediately after `<tool_call>`."""
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )
    return prompt + TOOL_CALL_OPEN


def build_examples(
    records: list[dict[str, Any]],
    tokenizer=None,
    include_think_actions: bool = False,
) -> list[dict[str, str]]:
    """Build native-Qwen, tool-call-only action decisions from APIGen trajectories.

    Context ends with Qwen's literal `<tool_call>` opener. The target is only
    the JSON body plus literal `</tool_call>` terminator: no `<think>` tokens
    are predicted by the Action Expert. Earlier function calls and observations
    remain in the native Qwen message history for later decisions.
    """
    examples: list[dict[str, str]] = []
    for trajectory_id, record in enumerate(records):
        if "prompt" in record and ("action" in record or "target" in record):
            if tokenizer is None:
                raise ValueError("Native Qwen data construction requires a tokenizer")
            action = canonical_action(record.get("action", record.get("target")))
            examples.append({"prompt": record["prompt"], "target": action, "completion": action + TOOL_CALL_CLOSE, "trajectory_id": str(trajectory_id)})
            continue
        if tokenizer is None:
            raise ValueError("APIGen conversion requires the backbone tokenizer for its native tool template")
        turns = record.get("conversations") or record.get("messages")
        if not turns:
            continue
        messages: list[dict[str, Any]] = [{"role": "system", "content": record.get("system", "")}]
        tools = _tool_spec(record, include_think_actions)
        skip_next_observation = False
        for turn in turns:
            role = turn.get("from", turn.get("role", ""))
            value = turn.get("value", turn.get("content", ""))
            if role in {"function_call", "tool", "assistant_tool_call"}:
                action = json.loads(value) if isinstance(value, str) else value
                name = action.get("name", action.get("function", {}).get("name"))
                canonical = canonical_action(action)
                is_think = name == "think"
                if include_think_actions or not is_think:
                    examples.append(
                        {
                            "prompt": render_qwen_tool_prefix(tokenizer, messages, tools),
                            "target": canonical,
                            "completion": canonical + TOOL_CALL_CLOSE,
                            "trajectory_id": str(trajectory_id),
                        }
                    )
                # APIGen's `think` is an internal reasoning trace, not an
                # external tool action in this experiment. Excluding it avoids
                # leaking textual rationale into the next action's context.
                if include_think_actions or not is_think:
                    messages.append(_tool_message(action))
                skip_next_observation = is_think and not include_think_actions
            elif role == "observation":
                if not skip_next_observation:
                    messages.append({"role": "tool", "content": value})
                skip_next_observation = False
            elif role == "human":
                messages.append({"role": "user", "content": value})
            elif role == "gpt":
                messages.append({"role": "assistant", "content": value})
    if not examples:
        raise ValueError("No supervised external tool actions found.")
    return examples


@dataclass
class Batch:
    context_ids: torch.Tensor
    context_mask: torch.Tensor
    decoder_ids: torch.Tensor
    labels: torch.Tensor
    targets: list[str]


class ActionDataset(Dataset):
    def __init__(self, examples: list[dict[str, str]], tokenizer, max_context_tokens: int | None, max_action_tokens: int):
        self.examples, self.tokenizer = examples, tokenizer
        self.max_context_tokens, self.max_action_tokens = max_context_tokens, max_action_tokens
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        item = self.examples[index]
        ids = self.tokenizer(item["prompt"], truncation=False, add_special_tokens=False)["input_ids"]
        if self.max_context_tokens is not None and len(ids) > self.max_context_tokens:
            # This is intentionally a visible hard context limit. It retains
            # template/tool definitions at the beginning and latest state at
            # the end; `kv-tokens all` then retains every KV that survived it.
            head = self.max_context_tokens // 2
            ids = ids[:head] + ids[-(self.max_context_tokens - head) :]
        target = self.tokenizer(item["completion"], truncation=True, max_length=self.max_action_tokens - 1, add_special_tokens=False)["input_ids"]
        bos = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.eos_token_id
        return {"context": ids, "decoder": [bos] + target, "target": item["target"]}

    def collate(self, items: list[dict[str, Any]]) -> Batch:
        pad = self.tokenizer.pad_token_id
        max_ctx, max_dec = max(len(x["context"]) for x in items), max(len(x["decoder"]) for x in items)
        context_ids = torch.full((len(items), max_ctx), pad, dtype=torch.long)
        context_mask = torch.zeros((len(items), max_ctx), dtype=torch.long)
        decoder_ids = torch.full((len(items), max_dec), pad, dtype=torch.long)
        labels = torch.full((len(items), max_dec), -100, dtype=torch.long)
        for row, item in enumerate(items):
            nc, nd = len(item["context"]), len(item["decoder"])
            context_ids[row, max_ctx - nc :] = torch.tensor(item["context"])
            context_mask[row, max_ctx - nc :] = 1
            decoder_ids[row, :nd] = torch.tensor(item["decoder"])
            labels[row, :nd] = decoder_ids[row, :nd]
        return Batch(context_ids, context_mask, decoder_ids, labels, [x["target"] for x in items])
