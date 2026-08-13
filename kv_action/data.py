import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def canonical_action(value: Any) -> str:
    """Serialize actions deterministically, making exact-match meaningful."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _prompt(system: str, tools: Any, turns: list[dict[str, Any]]) -> str:
    tools_text = tools if isinstance(tools, str) else json.dumps(tools, ensure_ascii=False)
    lines = [f"<SYSTEM>\n{system}\n</SYSTEM>", f"<TOOLS>\n{tools_text}\n</TOOLS>"]
    role_map = {"human": "USER", "gpt": "ASSISTANT", "observation": "TOOL_RESPONSE"}
    for turn in turns:
        role = role_map.get(turn.get("from", ""), turn.get("from", "UNKNOWN").upper())
        lines.append(f"<{role}>\n{turn.get('value', '')}\n</{role}>")
    lines.append("<ASSISTANT_TOOL_CALL>\n")
    return "\n".join(lines)


def build_examples(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Convert APIGen-MT or the minimal prompt/action contract to decisions."""
    examples: list[dict[str, str]] = []
    for record in records:
        if "prompt" in record and ("action" in record or "target" in record):
            examples.append({"prompt": record["prompt"], "target": canonical_action(record.get("action", record.get("target")))})
            continue
        turns = record.get("conversations") or record.get("messages")
        if not turns:
            continue
        history: list[dict[str, Any]] = []
        for turn in turns:
            role = turn.get("from", turn.get("role", ""))
            if role in {"function_call", "tool", "assistant_tool_call"}:
                raw = turn.get("value", turn.get("content", turn.get("arguments", "")))
                examples.append({
                    "prompt": _prompt(record.get("system", ""), record.get("tools", []), history),
                    "target": canonical_action(raw),
                })
            else:
                history.append({"from": role, "value": turn.get("value", turn.get("content", ""))})
    if not examples:
        raise ValueError("No actions found. Expected APIGen `function_call` turns or prompt/action records.")
    return examples


def load_records(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty data file: {path}")
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


@dataclass
class Batch:
    context_ids: torch.Tensor
    context_mask: torch.Tensor
    decoder_ids: torch.Tensor
    labels: torch.Tensor
    targets: list[str]


class ActionDataset(Dataset):
    def __init__(self, examples: list[dict[str, str]], tokenizer, max_context_tokens: int, max_action_tokens: int):
        self.examples, self.tokenizer = examples, tokenizer
        self.max_context_tokens, self.max_action_tokens = max_context_tokens, max_action_tokens
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        item = self.examples[index]
        # Tool schemas sit at the beginning, while the latest observation/user
        # request sits at the end. Keeping only either side breaks a tool agent;
        # retain both when a long APIGen trajectory exceeds the context budget.
        context = self.tokenizer(item["prompt"], truncation=False, add_special_tokens=True)
        ids = context["input_ids"]
        if len(ids) > self.max_context_tokens:
            head = self.max_context_tokens // 2
            ids = ids[:head] + ids[-(self.max_context_tokens - head) :]
        target = self.tokenizer(item["target"], truncation=True, max_length=self.max_action_tokens - 1, add_special_tokens=False)["input_ids"]
        # decoder starts with BOS; labels are shifted internally by the model.
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
            # Left padding keeps the most recent real tokens aligned with the
            # retained KV suffix in the frozen backbone cache.
            context_ids[row, max_ctx - nc :] = torch.tensor(item["context"])
            context_mask[row, max_ctx - nc :] = 1
            decoder_ids[row, :nd] = torch.tensor(item["decoder"])
            labels[row, :nd] = decoder_ids[row, :nd]
        return Batch(context_ids, context_mask, decoder_ids, labels, [x["target"] for x in items])
