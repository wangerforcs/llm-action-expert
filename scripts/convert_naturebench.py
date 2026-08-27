#!/usr/bin/env python
"""Convert one NatureBench transcript into the rollout trajectory schema.

The source transcript contains model-generated tool calls and tool-result
messages, but no APIGen-style JSON schema. We retain the initial human task,
replayable result pairs, and infer permissive JSON schemas from observed tool
arguments so the native Qwen chat template can render the trajectory.
"""
import argparse
import json
from pathlib import Path


def result_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item.get("content", ""))))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return json.dumps(value, ensure_ascii=False)


def json_type(value):
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-tool-calls", type=int, default=8)
    args = p.parse_args()

    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines()]
    initial_user = ""
    calls, results = [], {}
    schemas = {}
    for row in rows:
        payload = row.get("payload", {})
        message = row.get("message", {}) or (payload if payload.get("type") == "message" else {})
        row_type = row.get("type")
        if row_type == "response_item" and payload.get("type") == "message":
            row_type = "assistant" if payload.get("role") == "assistant" else "user"
        if row_type == "user":
            content = message.get("content")
            if isinstance(content, str) and not initial_user:
                initial_user = content
            elif isinstance(content, list):
                user_text = "\n".join(
                    str(block.get("text", block.get("content", "")))
                    for block in content
                    if isinstance(block, dict) and block.get("type") not in {"tool_result", "thinking"}
                ).strip()
                if user_text and (not initial_user or "# Role & Objective" in user_text or "# Task Definition" in user_text):
                    initial_user = user_text
                for block in content:
                    if block.get("type") != "tool_result":
                        continue
                    results[block.get("tool_use_id")] = result_text(block.get("content", ""))
        if row.get("type") == "response_item" and payload.get("type") in {"function_call", "custom_tool_call"}:
            name = payload.get("name", "unknown")
            arguments = payload.get("arguments", payload.get("input", {}))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"input": arguments}
            if not isinstance(arguments, dict):
                arguments = {"input": arguments}
            calls.append((payload.get("call_id", payload.get("id")), name, arguments))
            props = schemas.setdefault(name, {})
            for key, value in arguments.items():
                props.setdefault(key, {"type": json_type(value)})
        if row.get("type") == "response_item" and payload.get("type") == "function_call_output":
            results[payload.get("call_id")] = result_text(payload.get("output", payload.get("content", "")))
        if row_type != "assistant" or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if block.get("type") != "tool_use":
                continue
            name, arguments = block.get("name", "unknown"), block.get("input", {})
            if not isinstance(arguments, dict):
                arguments = {"input": arguments}
            calls.append((block.get("id"), name, arguments))
            props = schemas.setdefault(name, {})
            for key, value in arguments.items():
                props.setdefault(key, {"type": json_type(value)})

    calls = [call for call in calls if call[0] in results][: args.max_tool_calls]
    conversations = [{"from": "human", "value": initial_user}]
    for call_id, name, arguments in calls:
        conversations.append({"from": "function_call", "value": json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)})
        conversations.append({"from": "observation", "value": results[call_id]})
    tools = [
        {
            "name": name,
            "description": f"NatureBench tool `{name}`.",
            "parameters": {"type": "object", "properties": props, "additionalProperties": True},
        }
        for name, props in sorted(schemas.items())
    ]
    record = {
        "system": "You are a coding agent. Solve the user's task using the available tools. Keep the conversation history and inspect tool results before deciding the next action.",
        "tools": tools,
        "conversations": conversations,
        "source": str(args.input),
        "source_tool_calls": len(calls),
    }
    # NatureBench stores the evaluation harness/developer prompt inside the
    # first user message. Keep the task instructions but remove its injected
    # platform preamble, which otherwise makes a small Qwen model discuss
    # skills and permissions instead of using the tools.
    marker = "# Role & Objective"
    if marker in initial_user:
        initial_user = initial_user[initial_user.index(marker):]
        record["conversations"][0]["value"] = initial_user
    Path(args.output).write_text(json.dumps([record], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": args.output, "tool_calls": len(calls), "tools": sorted(schemas), "initial_user_chars": len(initial_user)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
