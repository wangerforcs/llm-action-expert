from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class ExpertBlock(nn.Module):
    """Pi05-style suffix block: expert Q/K/V attend over frozen prefix K/V.

    At inference Pi05 runs the language prefix once, then passes its per-layer
    cache as ``past_key_values`` to the action Gemma. This block is that same
    operation for a narrow randomly-initialized action tower: its action K/V
    are concatenated with the *unprojected* backbone K/V before one attention
    softmax (rather than using a separate cross-attention module).
    """

    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        if width % heads:
            raise ValueError("expert width must be divisible by number of heads")
        self.width, self.heads, self.head_dim = width, heads, width // heads
        self.norm = nn.LayerNorm(width)
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width), nn.Dropout(dropout))

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def _rope(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (1_000_000.0 ** (torch.arange(0, self.head_dim, 2, device=q.device, dtype=torch.float32) / self.head_dim))
        angles = positions.float().unsqueeze(-1) * inv_freq
        angles = torch.cat([angles, angles], dim=-1).unsqueeze(1).to(dtype=q.dtype)
        cos, sin = angles.cos(), angles.sin()
        return q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(k) * sin

    def forward(self, x: torch.Tensor, prefix_k: torch.Tensor, prefix_v: torch.Tensor, prefix_valid: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        bsz, action_len, _ = x.shape
        if prefix_k.shape[1] != self.heads or prefix_k.shape[-1] != self.head_dim:
            raise ValueError(
                f"Raw KV bridge requires expert heads/head_dim ({self.heads}, {self.head_dim}) "
                f"to match backbone KV ({prefix_k.shape[1]}, {prefix_k.shape[-1]})."
            )
        h = self.norm(x)
        q = self.q_proj(h).view(bsz, action_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(bsz, action_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(bsz, action_len, self.heads, self.head_dim).transpose(1, 2)
        q, k = self._rope(q, k, positions)
        all_k, all_v = torch.cat([prefix_k, k], dim=2), torch.cat([prefix_v, v], dim=2)
        causal_actions = torch.ones(action_len, action_len, dtype=torch.bool, device=x.device).tril()
        allowed = torch.cat([prefix_valid[:, None, :].expand(-1, action_len, -1), causal_actions.expand(bsz, -1, -1)], dim=-1)
        y = F.scaled_dot_product_attention(q, all_k, all_v, attn_mask=allowed[:, None], dropout_p=0.0)
        y = self.o_proj(y.transpose(1, 2).reshape(bsz, action_len, self.width))
        x = x + y
        return x + self.mlp(self.mlp_norm(x))


class KVActionExpert(nn.Module):
    """Narrow, layer-synchronous decoder conditioned on backbone K/V.

    Block ``l`` reads only the real K/V cache emitted by backbone layer ``l``.
    This preserves the Pi05-style depth-wise two-tower correspondence while
    allowing a substantially smaller action tower width.
    """

    def __init__(self, backbone_width: int, width: int = 1024, layers: int = 4, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.width = width
        self.token_in = nn.Linear(backbone_width, width, bias=False)
        self.blocks = nn.ModuleList([ExpertBlock(width, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.token_out = nn.Linear(width, backbone_width, bias=False)

    def forward(self, action_embeddings: torch.Tensor, kvs: list[tuple[torch.Tensor, torch.Tensor]], context_mask: torch.Tensor) -> torch.Tensor:
        if len(kvs) != len(self.blocks):
            raise ValueError(
                f"Layer-synchronous expert requires equal tower depths, got "
                f"{len(self.blocks)} expert blocks and {len(kvs)} backbone KV layers."
            )
        x = self.token_in(action_embeddings)
        start = context_mask.sum(dim=-1, keepdim=True)
        positions = start + torch.arange(x.size(1), device=x.device)[None, :]
        for block, (prefix_k, prefix_v) in zip(self.blocks, kvs, strict=True):
            x = block(x, prefix_k, prefix_v, context_mask.bool(), positions)
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

    def __init__(self, backbone, selected_layers: list[int], kv_tokens: int | None, expert_width: int, expert_layers: int, expert_heads: int, representation: str = "kv"):
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
        if representation == "kv" and expert_layers != len(selected_layers):
            raise ValueError(
                "Pi05-style layer-synchronous mode requires --expert-layers to equal "
                "the number of selected backbone KV layers. The default `paired` mapping "
                "uses 18 uniformly-spaced Qwen layers for a Pi05-sized 18-layer expert."
            )
        width = backbone.config.hidden_size
        backbone_kv_heads = backbone.config.num_key_value_heads
        backbone_head_dim = getattr(backbone.config, "head_dim", width // backbone.config.num_attention_heads)
        if representation == "kv" and (expert_heads != backbone_kv_heads or expert_width // expert_heads != backbone_head_dim):
            raise ValueError(
                "Raw Pi05-style KV sharing requires expert_heads == backbone num_key_value_heads "
                "and expert_width / expert_heads == backbone head_dim. For Qwen3-4B use "
                "--expert-width 1024 --expert-heads 8."
            )
        backbone_dtype = next(backbone.parameters()).dtype
        self.expert = KVActionExpert(width, expert_width, expert_layers, expert_heads).to(dtype=backbone_dtype)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # frozen model must never apply dropout
        return self

    @torch.no_grad()
    def prefill(self, context_ids: torch.Tensor, context_mask: torch.Tensor):
        # Match Qwen's RoPE positions to actual (not left-padding) tokens.
        # The Action Expert starts immediately after the same effective prefix.
        position_ids = (context_mask.long().cumsum(dim=-1) - 1).clamp_min(0)
        # Call the decoder directly. Calling `ForCausalLM.forward` also applies
        # the 151k-vocabulary LM head to every prefix token; for a 16×long
        # native tool prompt that materializes tens of GB of useless logits.
        # The Action Expert needs only the decoder's per-layer KV cache.
        out = self.backbone.model(
            input_ids=context_ids,
            attention_mask=context_mask,
            position_ids=position_ids,
            use_cache=True,
            output_hidden_states=self.representation == "last_hidden",
            return_dict=True,
        )
        if self.representation == "last_hidden":
            # Deliberately does not use attention K/V: a controlled interface
            # ablation with the same expert capacity and lexical decoder.
            hidden = out.hidden_states[-1] if self.kv_tokens is None else out.hidden_states[-1][:, -self.kv_tokens :]
            mask = context_mask if self.kv_tokens is None else context_mask[:, -self.kv_tokens :]
            return [(hidden.unsqueeze(1), torch.zeros_like(hidden).unsqueeze(1))], mask.contiguous()
        cache = out.past_key_values
        # transformers >=4.43 returns Cache; earlier releases return legacy tuples.
        if hasattr(cache, "to_legacy_cache"):
            cache = cache.to_legacy_cache()
        selected = []
        for layer in self.selected_layers:
            key, value = cache[layer][:2]
            if self.kv_tokens is not None:
                key, value = key[:, :, -self.kv_tokens :], value[:, :, -self.kv_tokens :]
            selected.append((key.contiguous(), value.contiguous()))
        # The retained KV suffix corresponds to this suffix of the attention mask.
        mask = context_mask if self.kv_tokens is None else context_mask[:, -self.kv_tokens :]
        return selected, mask.contiguous()

    def forward(self, context_ids: torch.Tensor, context_mask: torch.Tensor, decoder_ids: torch.Tensor, labels: torch.Tensor | None = None) -> PolicyOutput:
        kvs, mask = self.prefill(context_ids, context_mask)
        return self.decode_from_prefill(decoder_ids, kvs, mask, labels)

    def decode_from_prefill(
        self,
        decoder_ids: torch.Tensor,
        kvs: list[tuple[torch.Tensor, torch.Tensor]],
        context_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> PolicyOutput:
        """Decode action tokens from a previously computed frozen prefix KV."""
        embeddings = self.backbone.get_input_embeddings()(decoder_ids)
        hidden = self.expert(embeddings, kvs, context_mask)
        logits = self.backbone.get_output_embeddings()(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
        return PolicyOutput(logits, loss)
