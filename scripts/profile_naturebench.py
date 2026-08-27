#!/usr/bin/env python
"""Profile generated text, reasoning, and tool-call payloads in NatureBench.

Token counts use the supplied tokenizer (Qwen3-4B by default), so they answer
the deployment question "how many tokens would this content occupy in our
Qwen+AE protocol?" They are not a reconstruction of another provider's bill.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer


class TokenAccumulator:
    """Bounded-memory batched tokenizer accounting."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.buffers = defaultdict(list)
        self.chars = Counter()
        self.tokens = Counter()

    def add(self, family, kind, text):
        if not text:
            return
        text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, sort_keys=True)
        key = (family, kind)
        # A few traces embed multi-megabyte source files in one tool argument.
        # Fast tokenizers can transiently allocate far more than the source
        # length on such one-item batches. Chunk only those outliers; boundary
        # error is negligible for aggregate proportions.
        if len(text) > 200_000:
            self.flush(key)
            for start in range(0, len(text), 65_536):
                piece = text[start : start + 65_536]
                self.tokens[key] += len(self.tokenizer(piece, add_special_tokens=False)["input_ids"])
            return
        self.buffers[key].append(text)
        self.chars[key] += len(text)
        if self.chars[key] >= 250_000:
            self.flush(key)

    def flush(self, key):
        values = self.buffers[key]
        if values:
            self.tokens[key] += sum(len(ids) for ids in self.tokenizer(values, add_special_tokens=False)["input_ids"])
        self.buffers[key].clear()
        self.chars[key] = 0

    def flush_all(self):
        for key in list(self.buffers):
            self.flush(key)


def parse_claude(path, family, tokens, counters):
    traces = 1
    for line in path.open(encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            counters[(family, "malformed_lines")] += 1
            continue
        if row.get("type") != "assistant":
            continue
        message = row.get("message", {})
        for part in message.get("content", []):
            kind = part.get("type")
            if kind == "thinking":
                tokens.add(family, "reasoning", part.get("thinking", ""))
            elif kind == "text":
                tokens.add(family, "text", part.get("text", ""))
            elif kind == "tool_use":
                payload = {"name": part.get("name"), "input": part.get("input", {})}
                tokens.add(family, "tool", payload)
                tokens.add(family, f"tool::{part.get('name', 'unknown')}", payload)
                counters[(family, "tool_calls")] += 1
    counters[(family, "traces")] += traces


def parse_codex(path, family, tokens, counters):
    counters[(family, "traces")] += 1
    for line in path.open(encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            counters[(family, "malformed_lines")] += 1
            continue
        if row.get("type") == "response_item":
            item = row.get("payload", {})
            item_type = item.get("type")
            if item_type in {"function_call", "custom_tool_call"}:
                # Both spellings are direct model output. Include the tool name
                # and arguments/input, but not its result.
                payload = item.get("arguments") if item_type == "function_call" else item.get("input")
                action = {"name": item.get("name"), "arguments": payload}
                tokens.add(family, "tool", action)
                tokens.add(family, f"tool::{item.get('name', 'unknown')}", action)
                counters[(family, "tool_calls")] += 1
            elif item_type == "message" and item.get("role") == "assistant":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        tokens.add(family, "text", part.get("text", ""))
        elif row.get("type") == "event_msg" and row.get("payload", {}).get("type") == "token_count":
            # Codex records provider usage; hidden reasoning is not readable in
            # the transcript, so retain it as a separate accounting field.
            info = row["payload"].get("info") or {}
            usage = info.get("last_token_usage") or {}
            counters[(family, "provider_output_tokens")] += usage.get("output_tokens", 0)
            counters[(family, "provider_reasoning_tokens")] += usage.get("reasoning_output_tokens", 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="NatureBench trajectories directory")
    p.add_argument("--tokenizer", default="/data/datasets/models-hf/Qwen3-4B")
    p.add_argument("--families", nargs="*", default=None, help="Optional directory names, e.g. codex__gpt-5.5")
    p.add_argument("--max-traces-per-family", type=int, default=0, help="Deterministic random cap; 0 uses every JSONL transcript.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    root = Path(args.root)
    families = [p for p in sorted(root.iterdir()) if p.is_dir() and (args.families is None or p.name in args.families)]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokens, counters = TokenAccumulator(tokenizer), Counter()
    skipped = Counter()
    sampled = {}
    for family_dir in families:
        family = family_dir.name
        transcripts = sorted(family_dir.rglob("transcript.jsonl"))
        if args.max_traces_per_family and len(transcripts) > args.max_traces_per_family:
            transcripts = random.Random(args.seed).sample(transcripts, args.max_traces_per_family)
        sampled[family] = len(transcripts)
        for transcript in transcripts:
            if family.startswith("claude-code__"):
                parse_claude(transcript, family, tokens, counters)
            elif family.startswith("codex__"):
                parse_codex(transcript, family, tokens, counters)
            else:
                skipped[family] += 1
        # Gemini CLI uses a different JSON schema and is intentionally not
        # mixed into this first Claude/Codex action-format comparison.
        if family.startswith("gemini-cli__"):
            skipped[family] += len(list(family_dir.rglob("transcript.json")))

    tokens.flush_all()
    summary, aggregate, aggregate_tool_breakdown = {}, Counter(), Counter()
    for family_dir in families:
        family = family_dir.name
        if family in skipped:
            continue
        token_counts = {}
        for kind in ("reasoning", "text", "tool"):
            token_counts[kind] = tokens.tokens[(family, kind)]
        visible_total = sum(token_counts.values())
        tool_share_visible = token_counts["tool"] / visible_total if visible_total else 0.0
        non_reasoning = token_counts["text"] + token_counts["tool"]
        tool_breakdown = {
            key.removeprefix("tool::"): value
            for (token_family, key), value in tokens.tokens.items()
            if token_family == family and key.startswith("tool::")
        }
        tool_share_non_reasoning = token_counts["tool"] / non_reasoning if non_reasoning else 0.0
        row = {
            "traces": counters[(family, "traces")],
            "tool_calls": counters[(family, "tool_calls")],
            "qwen_tokens": token_counts | {"visible_total": visible_total},
            "tool_input_tokens_by_name": dict(sorted(tool_breakdown.items(), key=lambda x: -x[1])),
            "tool_share_of_visible_output": tool_share_visible,
            "tool_share_excluding_reasoning": tool_share_non_reasoning,
            "malformed_lines_skipped": counters[(family, "malformed_lines")],
            "codex_provider_output_tokens": counters[(family, "provider_output_tokens")],
            "codex_provider_reasoning_tokens": counters[(family, "provider_reasoning_tokens")],
        }
        summary[family] = row
        for k, v in token_counts.items():
            aggregate[k] += v
        aggregate_tool_breakdown.update(tool_breakdown)
        aggregate["traces"] += row["traces"]
        aggregate["tool_calls"] += row["tool_calls"]
    visible_total = aggregate["reasoning"] + aggregate["text"] + aggregate["tool"]
    aggregate_result = {
        "traces": aggregate["traces"], "tool_calls": aggregate["tool_calls"],
        "qwen_tokens": {"reasoning": aggregate["reasoning"], "text": aggregate["text"], "tool": aggregate["tool"], "visible_total": visible_total},
        "tool_share_of_visible_output": aggregate["tool"] / visible_total if visible_total else 0.0,
        "tool_share_excluding_reasoning": aggregate["tool"] / (aggregate["tool"] + aggregate["text"]) if aggregate["tool"] + aggregate["text"] else 0.0,
        "tool_input_tokens_by_name": dict(sorted(aggregate_tool_breakdown.items(), key=lambda x: -x[1])),
    }
    report = {
        "tokenizer": args.tokenizer,
        "definition": "tool = serialized model-emitted tool name plus arguments/input; tool results are excluded. Qwen-token counts are a common-tokenizer proxy, not provider billing.",
        "aggregate": aggregate_result,
        "by_family": summary,
        "skipped": dict(skipped),
        "sampled_transcripts": sampled,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
