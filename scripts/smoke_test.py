#!/usr/bin/env python
"""CPU-only structural test of layer-wise KV conditioning."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kv_action.model import KVActionExpert, Qwen3ActionExpert


def main():
    torch.manual_seed(0)
    b, action_tokens, context_tokens, heads, head_dim, backbone_width = 2, 5, 7, 2, 8, 32
    expert = KVActionExpert(backbone_width, width=16, layers=2, heads=2)
    embeddings = torch.randn(b, action_tokens, backbone_width)
    # Two layers with actual [B, H_kv, S, D] cache tensors, not hidden states.
    kvs = [(torch.randn(b, heads, context_tokens, head_dim), torch.randn(b, heads, context_tokens, head_dim)) for _ in range(2)]
    mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1]])
    out = expert(embeddings, kvs, mask)
    assert out.shape == (b, action_tokens, backbone_width)
    # Cached one-token decoding must be numerically identical to the parallel
    # causal pass used by training. This is the inference invariant that lets
    # the rollout avoid recomputing every earlier action token.
    cached_kvs, step_outputs = None, []
    for token_index in range(action_tokens):
        step, cached_kvs = expert.forward_step(embeddings[:, token_index : token_index + 1], kvs, mask, cached_kvs)
        step_outputs.append(step)
    torch.testing.assert_close(torch.cat(step_outputs, dim=1), out, rtol=1e-5, atol=1e-6)
    assert all(kv[0].shape[2] == action_tokens for kv in cached_kvs)
    out.square().mean().backward()
    assert all(p.grad is not None for p in expert.parameters() if p.requires_grad)

    # Pi05-style version: official Qwen3 decoder layers, initialized small but
    # with raw prefix KV shape compatible with its 2-head/8-dim cache.
    qwen_expert = Qwen3ActionExpert(
        backbone_width, width=16, layers=2, heads=2,
        backbone_config=SimpleNamespace(max_position_embeddings=64, rope_theta=1_000_000.0, rms_norm_eps=1e-6),
    ).eval()
    qwen_full = qwen_expert(embeddings, kvs, mask)
    qwen_cache, qwen_steps = None, []
    for token_index in range(action_tokens):
        step, qwen_cache = qwen_expert.forward_step(embeddings[:, token_index : token_index + 1], kvs, mask, qwen_cache)
        qwen_steps.append(step)
    torch.testing.assert_close(torch.cat(qwen_steps, dim=1), qwen_full, rtol=1e-5, atol=1e-6)
    print("OK: custom and official-Qwen3 experts both matched full and cached AR decoding.")


if __name__ == "__main__": main()
