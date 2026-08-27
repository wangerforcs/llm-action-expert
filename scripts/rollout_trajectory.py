#!/usr/bin/env python
"""Autoregressive multi-turn Qwen/AE agent rollout and latency benchmark.

Only logged *user* messages are supplied from APIGen.  Qwen generates every
assistant turn.  In ``hybrid`` mode generation is intercepted at Qwen's actual
``<tool_call>`` token; the Action Expert completes its JSON body from the live
Qwen KV cache.  In ``qwen`` mode Qwen also generates that body, providing the
direct baseline under exactly the same conversation protocol.

Tool execution is necessarily replayed. In the default strict protocol, a
logged observation is inserted only if the generated call exactly matches the
next logged call; otherwise a local tool error is inserted and the expected
logged call is not consumed. It is not included in timing.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kv_action.data import TOOL_CALL_CLOSE, _tool_message, _tool_spec, canonical_action, load_records
from kv_action.model import KVConditionedPolicy


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def first_json_object(text):
    try:
        value, _ = json.JSONDecoder().raw_decode(text.lstrip())
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def strip_think(text):
    """Remove Qwen's internal think spans before storing assistant history."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def get_tool_pairs(record, include_think):
    """Return logged external call/result pairs, excluding synthetic think."""
    pairs, pending = [], None
    for turn in record.get("conversations", record.get("messages", [])):
        role = turn.get("from", turn.get("role", ""))
        value = turn.get("value", turn.get("content", ""))
        if role in {"function_call", "tool", "assistant_tool_call"}:
            action = json.loads(value) if isinstance(value, str) else value
            if include_think or action.get("name") != "think":
                pending = action
            else:
                pending = None
        elif role == "observation" and pending is not None:
            pairs.append((pending, value))
            pending = None
    return pairs


def user_messages(record):
    """The benchmark receives no dataset assistant messages: Qwen writes them."""
    return [t.get("value", t.get("content", "")) for t in record.get("conversations", record.get("messages", [])) if t.get("from", t.get("role", "")) == "human"]


def prompt_suffix(tokenizer, prompt_text, marker):
    """Tokenize only the newly added native-template turn.

    We deliberately locate the last user/assistant marker in text rather than
    diffing tokenized full histories. BPE boundaries and tool-argument key
    ordering make a full-history token diff invalid even when the conversation
    is semantically identical.
    """
    pos = prompt_text.rfind(marker)
    if pos < 0:
        raise RuntimeError(f"Could not find expected chat-template marker {marker!r}.")
    return tokenizer(prompt_text[pos:], add_special_tokens=False)["input_ids"]


@torch.no_grad()
def append_qwen_cache(backbone, cache, cached_ids, append_ids, device):
    """Append a whole native-template block to Qwen KV in one forward pass."""
    if not append_ids:
        raise RuntimeError("No new tokens to append to the live Qwen cache.")
    old_length, new_length = len(cached_ids), len(cached_ids) + len(append_ids)
    ids = torch.tensor([append_ids], device=device, dtype=torch.long)
    kwargs = {
        "input_ids": ids,
        "attention_mask": torch.ones((1, new_length), device=device, dtype=torch.long),
        "position_ids": torch.arange(old_length, new_length, device=device).unsqueeze(0),
        "use_cache": True,
        "logits_to_keep": 1,
        "return_dict": True,
    }
    if cache is not None:
        kwargs["past_key_values"] = cache
        kwargs["cache_position"] = torch.arange(old_length, new_length, device=device)
    sync(device)
    start = time.perf_counter()
    out = backbone(**kwargs)
    sync(device)
    return out, out.past_key_values, cached_ids + append_ids, time.perf_counter() - start


@torch.no_grad()
def qwen_generate(backbone, tokenizer, cache, cached_ids, append_ids, desired_prompt_ids, max_new_tokens, device, stop_at_tool_open):
    """Continue greedy Qwen decoding from its *persistent* cross-turn cache.

    The first call appends the complete prompt. Later calls append only the
    native-template delta (new user/tool messages and the generation header).
    Each decoded token is inserted into the same cache before it is returned.
    """
    out, cache, cached_ids, append_s = append_qwen_cache(backbone, cache, cached_ids, append_ids, device)
    tool_open = tokenizer.convert_tokens_to_ids("<tool_call>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos = tokenizer.eos_token_id
    generated = []
    sync(device)
    decode_start = time.perf_counter()
    for _ in range(max_new_tokens):
        nxt = out.logits[:, -1].argmax(-1)
        token = int(nxt.item())
        generated.append(token)
        out, cache, cached_ids, _ = append_qwen_cache(backbone, cache, cached_ids, [token], device)
        if stop_at_tool_open and token == tool_open:
            break
        if token in {eos, im_end}:
            break
    sync(device)
    decode_s = time.perf_counter() - decode_start
    return generated, cache, cached_ids, append_s, decode_s, len(desired_prompt_ids), len(append_ids)


def cache_for_expert(policy, cache, full_mask):
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    kvs = []
    for layer in policy.selected_layers:
        if hasattr(cache, "layers"):
            # Transformers 5.x DynamicCache no longer implements tuple
            # indexing; Qwen3's attention layers expose the tensors here.
            key, value = cache.layers[layer].keys, cache.layers[layer].values
        else:
            key, value = cache[layer][:2]
        if policy.kv_tokens is not None:
            key, value = key[:, :, -policy.kv_tokens :], value[:, :, -policy.kv_tokens :]
        kvs.append((key.contiguous(), value.contiguous()))
    return kvs, full_mask if policy.kv_tokens is None else full_mask[:, -policy.kv_tokens :]


@torch.no_grad()
def ae_complete(policy, tokenizer, cache, full_mask, max_new_tokens, device):
    """Complete JSON from a Qwen cache whose final token is <tool_call>."""
    eos, bos = tokenizer.eos_token_id, tokenizer.bos_token_id or tokenizer.eos_token_id
    close_ids = tokenizer(TOOL_CALL_CLOSE.strip(), add_special_tokens=False)["input_ids"]
    kvs, prefix_mask = cache_for_expert(policy, cache, full_mask)
    current = torch.tensor([[bos]], device=device, dtype=torch.long)
    generated, action_kvs = [], None
    sync(device)
    start = time.perf_counter()
    for _ in range(max_new_tokens):
        logits, action_kvs = policy.decode_step_from_prefill(current, kvs, prefix_mask, action_kvs)
        nxt = logits.argmax(-1)
        generated.append(int(nxt.item()))
        token_ids = generated
        if token_ids[-1] == eos or (len(token_ids) >= len(close_ids) and token_ids[-len(close_ids):] == close_ids):
            break
        current = nxt[:, None]
    sync(device)
    decode_s = time.perf_counter() - start
    token_ids = generated
    if eos in token_ids:
        token_ids = token_ids[:token_ids.index(eos)]
    terminated = len(token_ids) >= len(close_ids) and token_ids[-len(close_ids):] == close_ids
    return token_ids, tokenizer.decode(token_ids, skip_special_tokens=True).strip(), terminated, decode_s


def split_qwen_turn(tokenizer, token_ids):
    """Separate natural-language assistant prefix and its first tool JSON."""
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    text = text.replace("<|im_end|>", "").strip()
    if "<tool_call>" not in text:
        return strip_think(text), None, text
    prefix, body = text.split("<tool_call>", 1)
    body = body.split("</tool_call>", 1)[0].strip()
    return strip_think(prefix.rstrip()), first_json_object(body), text


def render_generation_prompt(tokenizer, messages, tools):
    return tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, enable_thinking=False, tokenize=False)


def append_tool_history(backbone, tokenizer, cache, cached_ids, messages, tools, im_end_id, device):
    """Append tool-call closure and tool result after the actual model output."""
    history = tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=False,
        enable_thinking=False, tokenize=False,
    )
    close_end = history.rfind("</tool_call>") + len("</tool_call>")
    if close_end == len("</tool_call>") - 1:
        raise RuntimeError("Native Qwen history did not contain </tool_call> after a predicted action.")
    suffix = history[close_end:]
    # Qwen baseline already generated the close tag and <|im_end|>; hybrid AE
    # stops at the close tag and needs Qwen to consume the structural suffix.
    if cached_ids and cached_ids[-1] == im_end_id and suffix.startswith("<|im_end|>"):
        suffix = suffix[len("<|im_end|>"):]
    append_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
    if not append_ids:
        raise RuntimeError("Tool history produced no structural/result tokens to append.")
    _, cache, cached_ids, elapsed = append_qwen_cache(backbone, cache, cached_ids, append_ids, device)
    return cache, cached_ids, elapsed


def rollout_one(record, trajectory_index, backbone, policy, tokenizer, args, device):
    messages = [{"role": "system", "content": record.get("system", "")}]
    tools = _tool_spec(record, args.include_think_actions)
    tool_pairs, pair_index, actions = get_tool_pairs(record, args.include_think_actions), 0, []
    # `cached_ids` is the exact token sequence in `cache`. It grows across
    # users, generated assistant messages, and replayed tool observations.
    # Never replace it with a re-prefilled full prompt.
    cache, cached_ids = None, []
    next_prompt_append_ids = None
    sync(device)
    start = time.perf_counter()
    for user_index, text in enumerate(user_messages(record)):
        messages.append({"role": "user", "content": text})
        # A tool response creates another assistant turn; keep asking Qwen
        # until it returns ordinary text, then advance to the next user turn.
        for assistant_turn in range(args.max_agent_turns):
            desired_prompt_text = render_generation_prompt(tokenizer, messages, tools)
            desired_prompt_ids = tokenizer(desired_prompt_text, add_special_tokens=False)["input_ids"]
            if cache is None:
                append_ids = desired_prompt_ids
            elif next_prompt_append_ids is not None:
                append_ids = next_prompt_append_ids
                next_prompt_append_ids = None
            else:
                # A new human turn is the only non-tool transition reaching
                # this branch without an explicit suffix prepared below.
                append_ids = prompt_suffix(tokenizer, desired_prompt_text, "<|im_start|>user\n")
            cache_tokens_before = len(cached_ids)
            qwen_ids, cache, cached_ids, append_s, decode_s, prompt_tokens, appended_context_tokens = qwen_generate(
                backbone, tokenizer, cache, cached_ids, append_ids, desired_prompt_ids,
                args.max_new_tokens, device, args.mode == "hybrid"
            )
            has_open = tokenizer.convert_tokens_to_ids("<tool_call>") in qwen_ids
            action_sync_s = ae_s = tool_append_s = 0.0
            if args.mode == "hybrid" and has_open:
                open_at = qwen_ids.index(tokenizer.convert_tokens_to_ids("<tool_call>"))
                assistant_prefix = strip_think(tokenizer.decode(qwen_ids[:open_at], skip_special_tokens=True).strip())
                ae_ids, raw_body, terminated, ae_s = ae_complete(
                    policy, tokenizer, cache,
                    torch.ones((1, len(cached_ids)), device=device, dtype=torch.long),
                    args.max_new_tokens, device,
                )
                # The Action Expert's KV is only for generating the action.
                # Qwen must consume the same completed action once (as a block)
                # so future assistant turns see an exact full Qwen KV history.
                if not terminated:
                    raise RuntimeError("AE reached --max-new-tokens before emitting </tool_call>; cannot continue a cache-consistent rollout.")
                _, cache, cached_ids, action_sync_s = append_qwen_cache(backbone, cache, cached_ids, ae_ids, device)
                predicted = first_json_object(raw_body)
                raw = tokenizer.decode(qwen_ids[:open_at], skip_special_tokens=True) + "<tool_call>\n" + raw_body
                generated_tokens = len(qwen_ids) + len(ae_ids)
            else:
                assistant_prefix, predicted, raw = split_qwen_turn(tokenizer, qwen_ids)
                generated_tokens = len(qwen_ids)
            if predicted is None:
                messages.append({"role": "assistant", "content": assistant_prefix})
                event_timing = {
                    "qwen_prefill_s": append_s if cache_tokens_before == 0 else 0.0,
                    "qwen_history_append_s": append_s if cache_tokens_before else 0.0,
                    "qwen_decode_s": decode_s,
                    "ae_decode_s": ae_s,
                    "qwen_action_sync_s": action_sync_s,
                    "qwen_tool_append_s": 0.0,
                    "cache_tokens_before": cache_tokens_before,
                    "prompt_tokens": prompt_tokens,
                    "appended_context_tokens": appended_context_tokens,
                }
                actions.append({"user_turn": user_index, "assistant_turn": assistant_turn, "kind": "text", "generated_tokens": generated_tokens, "raw": raw, **event_timing})
                break
            target, observation = tool_pairs[pair_index] if pair_index < len(tool_pairs) else (None, "")
            exact_match = target is not None and canonical_action(predicted) == canonical_action(target)
            if target is not None and (exact_match or args.tool_result_mode == "oracle-next"):
                # `oracle-next` is useful as a controlled latency upper bound,
                # but it must never be confused with a valid agent rollout.
                result_source = "recorded_match" if exact_match else "recorded_oracle_next"
                pair_index += 1
            else:
                observation = json.dumps(
                    {"error": "No recorded result is available for this generated call. The next expected call differs.",
                     "next_expected": canonical_action(target) if target is not None else None},
                    ensure_ascii=False,
                )
                result_source = "local_error"
            messages.append({"role": "assistant", "content": assistant_prefix, "tool_calls": [{"type": "function", "function": {"name": predicted["name"], "arguments": predicted.get("arguments", {})}}]})
            messages.append({"role": "tool", "content": observation})
            cache, cached_ids, tool_append_s = append_tool_history(
                backbone, tokenizer, cache, cached_ids, messages, tools,
                tokenizer.convert_tokens_to_ids("<|im_end|>"), device,
            )
            next_prompt_append_ids = prompt_suffix(
                tokenizer, render_generation_prompt(tokenizer, messages, tools),
                "<|im_start|>assistant\n",
            )
            event_timing = {
                "qwen_prefill_s": append_s if cache_tokens_before == 0 else 0.0,
                "qwen_history_append_s": append_s if cache_tokens_before else 0.0,
                "qwen_decode_s": decode_s,
                "ae_decode_s": ae_s,
                "qwen_action_sync_s": action_sync_s,
                "qwen_tool_append_s": tool_append_s,
                "cache_tokens_before": cache_tokens_before,
                "prompt_tokens": prompt_tokens,
                "appended_context_tokens": appended_context_tokens,
            }
            actions.append({
                "user_turn": user_index, "assistant_turn": assistant_turn, "kind": "tool_call",
                "generated_tokens": generated_tokens, "raw": raw, "prediction": canonical_action(predicted),
                "target": canonical_action(target) if target is not None else None,
                "exact_match": exact_match,
                "tool_name_correct": target is not None and predicted.get("name") == target.get("name"),
                "tool_result_source": result_source,
                **event_timing,
            })
        else:
            raise RuntimeError(f"Exceeded --max-agent-turns={args.max_agent_turns} after user turn {user_index}")
    sync(device)
    return {"trajectory_index": trajectory_index, "wall_s": time.perf_counter() - start, "actions": actions, "unused_logged_tool_pairs": len(tool_pairs) - pair_index, "final_messages": messages}


def main():
    p = argparse.ArgumentParser(description="User-driven multi-turn Qwen / Qwen+AE rollout benchmark")
    p.add_argument("--data", required=True); p.add_argument("--backbone", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--mode", choices=["hybrid", "qwen"], default="hybrid", help="hybrid: switch to AE at Qwen's <tool_call>; qwen: pure-Qwen baseline.")
    p.add_argument("--trajectory-index", type=int, action="append", default=[])
    p.add_argument("--num-trajectories", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=192); p.add_argument("--max-agent-turns", type=int, default=8)
    p.add_argument("--tool-result-mode", choices=["strict", "oracle-next"], default="strict", help="strict: replay a result only for an exact next-call match; oracle-next: always replay the next logged result (latency upper bound only).")
    p.add_argument("--include-think-actions", action="store_true"); p.add_argument("--print-predictions", action="store_true"); p.add_argument("--output")
    args = p.parse_args()
    checkpoint, cfg = torch.load(args.checkpoint, map_location="cpu", weights_only=False), None
    cfg = checkpoint["config"]
    if args.include_think_actions != cfg.get("include_think_actions", False): raise ValueError("--include-think-actions must match checkpoint training")
    records, indices = load_records(args.data), (args.trajectory_index or list(range(args.num_trajectories)))
    if not indices or any(i < 0 or i >= len(records) for i in indices): raise ValueError(f"trajectory indexes must be in [0, {len(records) - 1}]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    load_start = time.perf_counter(); tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    backbone = AutoModelForCausalLM.from_pretrained(args.backbone, torch_dtype=dtype).to(device)
    policy = KVConditionedPolicy(
        backbone, cfg["layer_ids"], cfg["kv_tokens"], cfg["expert_width"], int(cfg["expert_layers"]),
        cfg["expert_heads"], cfg.get("representation", "kv"), cfg.get("expert_arch", "custom"),
    ).to(device)
    policy.expert.load_state_dict(checkpoint["expert"]); policy.eval(); sync(device); load_s = time.perf_counter() - load_start
    print(f"Loaded in {load_s:.2f}s. mode={args.mode}; user-driven trajectories={indices}; device={device}.")
    rollouts = [rollout_one(records[i], i, backbone, policy, tokenizer, args, device) for i in indices]
    events = [a for r in rollouts for a in r["actions"]]; calls = [a for a in events if a["kind"] == "tool_call"]
    wall_s = sum(r["wall_s"] for r in rollouts)
    q_pre = sum(a["qwen_prefill_s"] for a in events)
    q_append = sum(a["qwen_history_append_s"] for a in events)
    q_dec = sum(a["qwen_decode_s"] for a in events)
    ae_dec = sum(a["ae_decode_s"] for a in events)
    q_sync = sum(a["qwen_action_sync_s"] for a in events)
    q_tool = sum(a["qwen_tool_append_s"] for a in events)
    summary = {"mode": args.mode, "tool_result_mode": args.tool_result_mode, "trajectories": len(rollouts), "assistant_turns": len(events), "tool_calls": len(calls), "model_load_s": load_s, "rollout_wall_s": wall_s, "mean_trajectory_wall_s": wall_s / len(rollouts), "qwen_initial_prefill_s": q_pre, "qwen_history_append_s": q_append, "qwen_decode_s": q_dec, "ae_decode_s": ae_dec, "qwen_action_sync_s": q_sync, "qwen_tool_append_s": q_tool, "other_host_s": wall_s - q_pre - q_append - q_dec - ae_dec - q_sync - q_tool, "generated_tokens": sum(a["generated_tokens"] for a in events), "tool_name_accuracy": sum(a["tool_name_correct"] for a in calls) / len(calls) if calls else 0.0, "canonical_exact_match": sum(a["exact_match"] for a in calls) / len(calls) if calls else 0.0, "cache_protocol": "Qwen KV is initialized once. Later user/tool-template suffixes are block-appended to the live cache. In hybrid mode, AE action tokens are block-forwarded through Qwen once to synchronize its future-history KV.", "replay_protocol": "Only logged user messages are inputs. Model-generated assistant messages form history. In strict mode, a recorded observation is replayed only for an exact next-call match; otherwise a local tool error is inserted. Observation/API time is excluded."}
    report = {"summary": summary, "trajectories": rollouts}; print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.print_predictions:
        for r in rollouts:
            for a in r["actions"]: print(f"\n--- trajectory {r['trajectory_index']} user {a['user_turn']} assistant {a['assistant_turn']} ({a['kind']}) ---\n{a['raw']}")
    output = Path(args.output) if args.output else Path(args.checkpoint).with_name(f"rollout_{args.mode}_" + "_".join(map(str, indices)) + ".json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"); print(f"Saved report to {output}")


if __name__ == "__main__": main()
