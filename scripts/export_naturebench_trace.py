#!/usr/bin/env python
"""Render a Claude-Code NatureBench JSONL trajectory into a readable Markdown log."""
import argparse
import json
from pathlib import Path


def compact(value, limit):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    value = " ".join(value.strip().split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def fence(text):
    return "```text\n" + text + "\n```\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Claude Code transcript.jsonl")
    p.add_argument("--output", required=True, help="Markdown output path")
    p.add_argument("--max-events", type=int, default=40, help="Number of model/tool events to retain")
    p.add_argument("--max-chars", type=int, default=800, help="Characters retained per field")
    args = p.parse_args()

    source = Path(args.input)
    events, title = [], None
    for line in source.open(encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("type") not in {"assistant", "user"}:
            continue
        message = row.get("message", {})
        if not isinstance(message, dict):
            continue
        role, parts = message.get("role"), message.get("content", [])
        # Claude Code's initial user prompt is a string; later tool-return
        # messages use the normal list-of-content-blocks representation.
        if role == "user" and isinstance(parts, str):
            if title is None:
                title = parts
            continue
        if not isinstance(parts, list):
            continue
        for part in parts:
            kind = part.get("type") if isinstance(part, dict) else None
            if role == "assistant" and kind == "thinking":
                events.append(("模型思考", part.get("thinking", "")))
            elif role == "assistant" and kind == "text":
                events.append(("模型文本", part.get("text", "")))
            elif role == "assistant" and kind == "tool_use":
                name, tool_input = part.get("name", "unknown"), part.get("input", {})
                events.append((f"模型工具调用：{name}", tool_input))
            elif role == "user" and kind == "tool_result":
                content = part.get("content", "")
                events.append(("工具返回（回填到模型上下文）", content))
            elif role == "user" and kind == "text" and title is None:
                title = part.get("text", "")

    lines = [
        "# NatureBench 真实轨迹节选",
        "",
        f"- 来源：`{source}`",
        "- 模型：`claude-code__qwen-3.7-max`",
        f"- 展示前 {min(args.max_events, len(events))} 个模型/工具事件；每个字段截断为 {args.max_chars} 字符。",
        "",
        "## 初始用户任务",
        "",
        fence(compact(title or "（未找到初始用户文本）", args.max_chars)),
        "## 时间顺序",
        "",
    ]
    for index, (label, payload) in enumerate(events[: args.max_events], 1):
        lines.extend([f"### {index}. {label}", "", fence(compact(payload, args.max_chars))])
    if len(events) > args.max_events:
        lines.extend([f"其余 {len(events) - args.max_events} 个事件未展开。", ""])
    lines.extend([
        "## 读取方式",
        "",
        "初始用户任务、每次工具返回和先前模型输出都会继续留在下一次 Qwen 推理的上下文中。"
        "此处的 `工具返回` 在原始 Claude Code JSONL 中以 `user/tool_result` 形式编码，"
        "但它不是新的真人发言。",
        "",
    ])
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
