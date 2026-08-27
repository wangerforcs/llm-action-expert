from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model


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

    def _project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project an action suffix into this layer's unrepeated KV space."""
        bsz, action_len, _ = x.shape
        h = self.norm(x)
        q = self.q_proj(h).view(bsz, action_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(bsz, action_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(bsz, action_len, self.heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def _validate_prefix(self, prefix_k: torch.Tensor):
        if prefix_k.shape[1] != self.heads or prefix_k.shape[-1] != self.head_dim:
            raise ValueError(
                f"Raw KV bridge requires expert heads/head_dim ({self.heads}, {self.head_dim}) "
                f"to match backbone KV ({prefix_k.shape[1]}, {prefix_k.shape[-1]})."
            )

    def forward(self, x: torch.Tensor, prefix_k: torch.Tensor, prefix_v: torch.Tensor, prefix_valid: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        bsz, action_len, _ = x.shape
        self._validate_prefix(prefix_k)
        q, k, v = self._project_qkv(x)
        q, k = self._rope(q, k, positions)
        all_k, all_v = torch.cat([prefix_k, k], dim=2), torch.cat([prefix_v, v], dim=2)
        causal_actions = torch.ones(action_len, action_len, dtype=torch.bool, device=x.device).tril()
        allowed = torch.cat([prefix_valid[:, None, :].expand(-1, action_len, -1), causal_actions.expand(bsz, -1, -1)], dim=-1)
        y = F.scaled_dot_product_attention(q, all_k, all_v, attn_mask=allowed[:, None], dropout_p=0.0)
        y = self.o_proj(y.transpose(1, 2).reshape(bsz, action_len, self.width))
        x = x + y
        return x + self.mlp(self.mlp_norm(x))

    def forward_step(
        self,
        x: torch.Tensor,
        prefix_k: torch.Tensor,
        prefix_v: torch.Tensor,
        prefix_valid: torch.Tensor,
        positions: torch.Tensor,
        action_kv: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Decode one action token, reusing this layer's earlier action K/V."""
        if x.size(1) != 1:
            raise ValueError("forward_step accepts exactly one action token")
        self._validate_prefix(prefix_k)
        q, k, v = self._project_qkv(x)
        q, k = self._rope(q, k, positions)
        if action_kv is None:
            past_k = k[:, :, :0]
            past_v = v[:, :, :0]
        else:
            past_k, past_v = action_kv
            if past_k.shape[:2] != k.shape[:2] or past_k.shape[-1] != k.shape[-1]:
                raise ValueError("Action KV cache shape does not match expert block")
        all_k = torch.cat([prefix_k, past_k, k], dim=2)
        all_v = torch.cat([prefix_v, past_v, v], dim=2)
        # One query can attend to all earlier action tokens and itself.
        action_valid = torch.ones((x.size(0), 1, past_k.size(2) + 1), dtype=torch.bool, device=x.device)
        allowed = torch.cat([prefix_valid[:, None, :], action_valid], dim=-1)
        y = F.scaled_dot_product_attention(q, all_k, all_v, attn_mask=allowed[:, None], dropout_p=0.0)
        y = self.o_proj(y.transpose(1, 2).reshape(x.size(0), 1, self.width))
        x = x + y
        new_kv = (torch.cat([past_k, k], dim=2), torch.cat([past_v, v], dim=2))
        return x + self.mlp(self.mlp_norm(x)), new_kv


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

    def forward_step(
        self,
        action_embedding: torch.Tensor,
        kvs: list[tuple[torch.Tensor, torch.Tensor]],
        context_mask: torch.Tensor,
        action_kvs: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Incrementally decode one action token and update per-layer K/V.

        This is inference-only cache state. Training continues to use the full
        causal path above so all suffix positions receive one parallel pass.
        """
        if action_embedding.size(1) != 1:
            raise ValueError("forward_step accepts exactly one action token")
        if len(kvs) != len(self.blocks):
            raise ValueError("Layer-synchronous expert requires one KV pair per block")
        if action_kvs is None:
            action_kvs = [None] * len(self.blocks)
        if len(action_kvs) != len(self.blocks):
            raise ValueError("Action KV cache requires one entry per expert block")
        cache_lengths = {kv[0].size(2) for kv in action_kvs if kv is not None}
        if len(cache_lengths) > 1:
            raise ValueError("All expert-layer action KV caches must have the same length")
        cached_tokens = next(iter(cache_lengths), 0)
        x = self.token_in(action_embedding)
        positions = context_mask.sum(dim=-1, keepdim=True) + cached_tokens
        new_kvs = []
        for block, (prefix_k, prefix_v), action_kv in zip(self.blocks, kvs, action_kvs, strict=True):
            x, new_kv = block.forward_step(x, prefix_k, prefix_v, context_mask.bool(), positions, action_kv)
            new_kvs.append(new_kv)
        return self.token_out(self.norm(x)), new_kvs


@dataclass
class Qwen3ActionCache:
    """Runtime state for one Qwen3-expert autoregressive suffix."""

    cache: DynamicCache
    prefix_length: int


class Qwen3ActionExpert(nn.Module):
    """Pi05-style small expert built from the official Qwen3 decoder stack.

    This is deliberately a ``Qwen3Model`` rather than a hand-written
    Transformer. It retains Qwen3 RMSNorm, QK normalization, SwiGLU, RoPE and
    SDPA/cache handling. The model has no token embedding or LM head: frozen
    Qwen3-4B embeddings/LM head remain the shared lexical interface.
    """

    def __init__(self, backbone_width: int, width: int, layers: int, heads: int, backbone_config):
        super().__init__()
        if width % heads:
            raise ValueError("expert width must divide evenly into expert heads")
        head_dim = width // heads
        # ``vocab_size=1`` prevents allocation of an unused 151k-token table;
        # inputs_embeds are supplied through token_in below, as in Pi05.
        config = Qwen3Config(
            vocab_size=1,
            hidden_size=width,
            # Qwen3-0.6B uses 1024 -> 3072; retain that narrow-Qwen ratio
            # while extending depth to match all 36 backbone KV layers.
            intermediate_size=3 * width,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=heads,
            head_dim=head_dim,
            hidden_act="silu",
            max_position_embeddings=backbone_config.max_position_embeddings,
            rope_theta=getattr(backbone_config, "rope_theta", 1_000_000.0),
            rms_norm_eps=getattr(backbone_config, "rms_norm_eps", 1e-6),
            attention_bias=False,
            attention_dropout=0.0,
            tie_word_embeddings=False,
        )
        # SDPA dispatches to PyTorch's flash/efficient kernels where available
        # while preserving an installed-transformers implementation of Qwen3.
        config._attn_implementation = "sdpa"
        self.config = config
        self.width = width
        self.token_in = nn.Linear(backbone_width, width, bias=False)
        self.model = Qwen3Model(config)
        self.model.embed_tokens = None
        self.token_out = nn.Linear(width, backbone_width, bias=False)

    def _prefix_cache(self, kvs: list[tuple[torch.Tensor, torch.Tensor]]) -> DynamicCache:
        # transformers renamed the prefilled-cache constructor argument in
        # 4.57. The project environment currently pins 4.53, where converting
        # the legacy list is the supported equivalent.
        try:
            return DynamicCache(ddp_cache_data=kvs, config=self.config)
        except TypeError:
            return DynamicCache.from_legacy_cache(tuple(kvs))

    @staticmethod
    def _positions(context_mask: torch.Tensor, action_start: int, action_len: int) -> torch.Tensor:
        return context_mask.long().sum(dim=-1, keepdim=True) + action_start + torch.arange(action_len, device=context_mask.device)[None, :]

    def forward(self, action_embeddings: torch.Tensor, kvs: list[tuple[torch.Tensor, torch.Tensor]], context_mask: torch.Tensor) -> torch.Tensor:
        prefix_length = kvs[0][0].size(2)
        action_len = action_embeddings.size(1)
        cache = self._prefix_cache(kvs)
        inputs = self.token_in(action_embeddings)
        full_mask = torch.cat(
            [context_mask, torch.ones((context_mask.size(0), action_len), dtype=context_mask.dtype, device=context_mask.device)], dim=1
        )
        out = self.model(
            inputs_embeds=inputs,
            attention_mask=full_mask,
            position_ids=self._positions(context_mask, 0, action_len),
            past_key_values=cache,
            cache_position=torch.arange(prefix_length, prefix_length + action_len, device=inputs.device),
            use_cache=True,
            return_dict=True,
        )
        return self.token_out(out.last_hidden_state)

    def forward_step(
        self,
        action_embedding: torch.Tensor,
        kvs: list[tuple[torch.Tensor, torch.Tensor]],
        context_mask: torch.Tensor,
        action_cache: Qwen3ActionCache | None = None,
    ) -> tuple[torch.Tensor, Qwen3ActionCache]:
        """One-token official-Qwen3 decode with a cache seeded by backbone KV."""
        if action_embedding.size(1) != 1:
            raise ValueError("forward_step accepts exactly one action token")
        if action_cache is None:
            prefix_length = kvs[0][0].size(2)
            action_cache = Qwen3ActionCache(self._prefix_cache(kvs), prefix_length)
        action_start = action_cache.cache.get_seq_length() - action_cache.prefix_length
        inputs = self.token_in(action_embedding)
        full_mask = torch.cat(
            [context_mask, torch.ones((context_mask.size(0), action_start + 1), dtype=context_mask.dtype, device=context_mask.device)], dim=1
        )
        out = self.model(
            inputs_embeds=inputs,
            attention_mask=full_mask,
            position_ids=self._positions(context_mask, action_start, 1),
            past_key_values=action_cache.cache,
            cache_position=torch.tensor([action_cache.prefix_length + action_start], device=inputs.device),
            use_cache=True,
            return_dict=True,
        )
        return self.token_out(out.last_hidden_state), action_cache


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

    def __init__(
        self,
        backbone,
        selected_layers: list[int],
        kv_tokens: int | None,
        expert_width: int,
        expert_layers: int,
        expert_heads: int,
        representation: str = "kv",
        expert_arch: str = "custom",
    ):
        super().__init__()
        self.backbone = backbone
        self.selected_layers = selected_layers
        self.kv_tokens = kv_tokens
        self.expert_arch = expert_arch
        if representation not in {"kv", "last_hidden"}:
            raise ValueError("representation must be 'kv' or 'last_hidden'")
        if expert_arch not in {"custom", "qwen3"}:
            raise ValueError("expert_arch must be 'custom' or 'qwen3'")
        if expert_arch == "qwen3" and representation != "kv":
            raise ValueError("The Qwen3 expert currently supports only raw-KV conditioning")
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
        if expert_arch == "custom":
            self.expert = KVActionExpert(width, expert_width, expert_layers, expert_heads)
        else:
            self.expert = Qwen3ActionExpert(width, expert_width, expert_layers, expert_heads, backbone.config)
        self.expert.to(dtype=backbone_dtype)

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

    def decode_step_from_prefill(
        self,
        decoder_id: torch.Tensor,
        kvs: list[tuple[torch.Tensor, torch.Tensor]],
        context_mask: torch.Tensor,
        action_kvs=None,
    ) -> tuple[torch.Tensor, object]:
        """Return next-token logits from one decoder input token plus AE KV cache.

        ``decoder_id`` is analogous to the single new token passed to a normal
        CausalLM with ``past_key_values``. The cache holds only AE action K/V;
        frozen Qwen prefix K/V are supplied separately and remain shared.
        """
        if decoder_id.ndim == 1:
            decoder_id = decoder_id[:, None]
        if decoder_id.ndim != 2 or decoder_id.size(1) != 1:
            raise ValueError("decoder_id must have shape [batch] or [batch, 1]")
        embeddings = self.backbone.get_input_embeddings()(decoder_id)
        hidden, new_action_kvs = self.expert.forward_step(embeddings, kvs, context_mask, action_kvs)
        return self.backbone.get_output_embeddings()(hidden)[:, -1], new_action_kvs
