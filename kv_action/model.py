from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class ExpertBlock(nn.Module):
    """Causal action decoding plus cross-attention over frozen backbone KV."""

    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        if width % heads:
            raise ValueError("expert width must be divisible by number of heads")
        self.self_norm = nn.LayerNorm(width)
        self.self_attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(width)
        self.cross_attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, memory: torch.Tensor, memory_padding: torch.Tensor | None) -> torch.Tensor:
        n = x.size(1)
        causal = torch.ones(n, n, device=x.device, dtype=torch.bool).triu(1)
        y, _ = self.self_attn(self.self_norm(x), self.self_norm(x), self.self_norm(x), attn_mask=causal, need_weights=False)
        x = x + y
        y, _ = self.cross_attn(self.cross_norm(x), memory, memory, key_padding_mask=memory_padding, need_weights=False)
        x = x + y
        return x + self.mlp(self.mlp_norm(x))


class KVActionExpert(nn.Module):
    """Small token decoder conditioned exclusively on projected layer-wise K/V."""

    def __init__(self, backbone_width: int, width: int = 384, layers: int = 4, heads: int = 6, dropout: float = 0.0):
        super().__init__()
        self.width = width
        self.token_in = nn.Linear(backbone_width, width, bias=False)
        # A separate adapter per selected backbone layer preserves layer identity.
        self.kv_adapters = nn.ModuleList()
        self.layer_embedding = nn.ParameterList()
        self.blocks = nn.ModuleList([ExpertBlock(width, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.token_out = nn.Linear(width, backbone_width, bias=False)

    def _ensure_kv_adapters(self, kvs: list[tuple[torch.Tensor, torch.Tensor]]):
        if len(self.kv_adapters) == len(kvs):
            return
        if self.kv_adapters:
            raise RuntimeError("The selected KV-layer count changed after initialization")
        for key, value in kvs:
            in_dim = (key.shape[1] + value.shape[1]) * key.shape[-1]
            self.kv_adapters.append(nn.Linear(in_dim, self.width, bias=False).to(key.device, dtype=key.dtype))
            self.layer_embedding.append(nn.Parameter(torch.zeros(1, 1, self.width, device=key.device, dtype=key.dtype)))
        for p in self.layer_embedding:
            nn.init.normal_(p, std=0.02)

    def make_memory(self, kvs: list[tuple[torch.Tensor, torch.Tensor]], context_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_kv_adapters(kvs)
        chunks = []
        masks = []
        for adapter, layer_id, (key, value) in zip(self.kv_adapters, self.layer_embedding, kvs):
            # [B, kv_heads, S, head_dim] -> [B, S, (K,V)*kv_heads]
            pair = torch.cat([key, value], dim=1).transpose(1, 2).flatten(2)
            chunks.append(adapter(pair) + layer_id)
            masks.append(context_mask[:, : pair.size(1)])
        return torch.cat(chunks, dim=1), torch.cat(masks, dim=1)

    def forward(self, action_embeddings: torch.Tensor, kvs: list[tuple[torch.Tensor, torch.Tensor]], context_mask: torch.Tensor) -> torch.Tensor:
        memory, valid_memory = self.make_memory(kvs, context_mask)
        x = self.token_in(action_embeddings)
        for block in self.blocks:
            x = block(x, memory, ~valid_memory.bool())
        return self.token_out(self.norm(x))


@dataclass
class PolicyOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None


class KVConditionedPolicy(nn.Module):
    """Frozen HF backbone prefill + trainable KV action expert.

    The expert has no route to context token IDs; `context_ids` only enter the
    frozen backbone. `backbone.get_input_embeddings()` and `lm_head` are reused
    as frozen lexical interfaces for action tokens.
    """

    def __init__(self, backbone, selected_layers: list[int], kv_tokens: int, expert_width: int, expert_layers: int, expert_heads: int, representation: str = "kv"):
        super().__init__()
        self.backbone = backbone
        self.selected_layers = selected_layers
        self.kv_tokens = kv_tokens
        if representation not in {"kv", "last_hidden"}:
            raise ValueError("representation must be 'kv' or 'last_hidden'")
        self.representation = representation
        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone.eval()
        width = backbone.config.hidden_size
        backbone_dtype = next(backbone.parameters()).dtype
        self.expert = KVActionExpert(width, expert_width, expert_layers, expert_heads).to(dtype=backbone_dtype)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # frozen model must never apply dropout
        return self

    @torch.no_grad()
    def prefill(self, context_ids: torch.Tensor, context_mask: torch.Tensor):
        out = self.backbone(
            input_ids=context_ids,
            attention_mask=context_mask,
            use_cache=True,
            output_hidden_states=self.representation == "last_hidden",
            return_dict=True,
        )
        if self.representation == "last_hidden":
            # Deliberately does not use attention K/V: a controlled interface
            # ablation with the same expert capacity and lexical decoder.
            hidden = out.hidden_states[-1][:, -self.kv_tokens :]
            return [(hidden.unsqueeze(1), torch.zeros_like(hidden).unsqueeze(1))], context_mask[:, -self.kv_tokens :].contiguous()
        cache = out.past_key_values
        # transformers >=4.43 returns Cache; earlier releases return legacy tuples.
        if hasattr(cache, "to_legacy_cache"):
            cache = cache.to_legacy_cache()
        selected = []
        for layer in self.selected_layers:
            key, value = cache[layer][:2]
            selected.append((key[:, :, -self.kv_tokens :].contiguous(), value[:, :, -self.kv_tokens :].contiguous()))
        # The retained KV suffix corresponds to this suffix of the attention mask.
        return selected, context_mask[:, -self.kv_tokens :].contiguous()

    def forward(self, context_ids: torch.Tensor, context_mask: torch.Tensor, decoder_ids: torch.Tensor, labels: torch.Tensor | None = None) -> PolicyOutput:
        kvs, mask = self.prefill(context_ids, context_mask)
        embeddings = self.backbone.get_input_embeddings()(decoder_ids)
        hidden = self.expert(embeddings, kvs, mask)
        logits = self.backbone.get_output_embeddings()(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
        return PolicyOutput(logits, loss)
