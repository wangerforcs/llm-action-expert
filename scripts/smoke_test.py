#!/usr/bin/env python
"""CPU-only structural test of layer-wise KV conditioning."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kv_action.model import KVActionExpert


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
    out.square().mean().backward()
    assert all(p.grad is not None for p in expert.parameters() if p.requires_grad)
    print("OK: action decoder read two layers of [K,V] cache and backpropagated through expert only.")


if __name__ == "__main__": main()
