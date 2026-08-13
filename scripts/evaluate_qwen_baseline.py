#!/usr/bin/env python
"""Direct native-Qwen tool-calling baseline on one APIGen trajectory.

This does not use the Action Expert or its manual prompt serialization. It
renders Qwen3's own chat/tool template and asks the frozen Qwen model to
generate each ground-truth tool action directly.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kv_action.data import canonical_action, load_records


def messages_before_call(record, call_index, include_think_actions):
    messages = [{"role": "system", "content": record.get("system", "")}]
    calls_seen = 0
    skip_next_observation = False
    for turn in record["conversations"]:
        role, value = turn.get("from"), turn.get("value", "")
        if role == "function_call":
            action = json.loads(value)
            is_think = action.get("name") == "think"
            if is_think and not include_think_actions:
                skip_next_observation = True
                continue
            if calls_seen == call_index:
                return messages, canonical_action(value)
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"type": "function", "function": {"name": action["name"], "arguments": action.get("arguments", {})}}],
                }
            )
            calls_seen += 1
        elif role == "observation":
            if not skip_next_observation:
                messages.append({"role": "tool", "content": value})
            skip_next_observation = False
        elif role == "human":
            messages.append({"role": "user", "content": value})
        elif role == "gpt":
            messages.append({"role": "assistant", "content": value})
    raise IndexError(f"trajectory has only {calls_seen} function calls")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--trajectory-index", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--show-prompt-tail", action="store_true")
    p.add_argument("--include-think-actions", action="store_true", help="Match the optional training mode that supervises APIGen's synthetic think function.")
    args = p.parse_args()
    record = load_records(args.data)[args.trajectory_index]
    tools = json.loads(record["tools"]) if isinstance(record["tools"], str) else record["tools"]
    if not args.include_think_actions:
        tools = [tool for tool in tools if tool.get("name") != "think"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    model = AutoModelForCausalLM.from_pretrained(args.backbone, torch_dtype=dtype).to(device).eval()
    calls = sum(
        t.get("from") == "function_call"
        and (args.include_think_actions or json.loads(t["value"]).get("name") != "think")
        for t in record["conversations"]
    )
    terminators = [tokenizer.eos_token_id]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        terminators.append(im_end)
    with torch.no_grad():
        for call_idx in range(calls):
            messages, target = messages_before_call(record, call_idx, args.include_think_actions)
            rendered = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=False,
            )
            inputs = tokenizer(rendered, return_tensors="pt").to(device)
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
            )
            raw = tokenizer.decode(output[0, inputs.input_ids.size(1) :], skip_special_tokens=False)
            print(f"\n===== trajectory {args.trajectory_index}, action {call_idx} =====")
            if args.show_prompt_tail:
                print(f"--- native Qwen prompt tail ---\n{rendered[-2500:]}")
            print(f"--- Qwen3 raw output ---\n{raw}")
            print(f"--- APIGen target ---\n{target}")


if __name__ == "__main__":
    main()
