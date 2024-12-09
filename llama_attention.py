import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Tuple, overload

import torch
import transformers
from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODELS_DIR = "/share/u/can/models"

logger = logging.getLogger(__name__)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@dataclass(frozen=True)
class AttentionEdge:
    #! q_idx *attends* to the k_idx

    q_idx: int
    k_idx: int


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    cut_attn_edges: Optional[dict[int, list[AttentionEdge]]] = None,
    store_attn_matrices: Optional[dict[int, torch.Tensor]] = None,
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(attn_weight.dtype).to(attn_weight.device)

    # ---------------------------------------------------------------------
    if cut_attn_edges is not None:
        for head_idx, edges in cut_attn_edges.items():
            for edge in edges:
                attn_weight[:, head_idx, edge.q_idx, edge.k_idx] = float("-inf")
    # ---------------------------------------------------------------------

    attn_weight = torch.softmax(attn_weight, dim=-1)

    # ---------------------------------------------------------------------
    if store_attn_matrices is not None:
        for head_idx in store_attn_matrices:
            store_attn_matrices[head_idx] = attn_weight[:, head_idx, :, :]
    # ---------------------------------------------------------------------

    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    return attn_weight @ value


def attn_per_head(
    o_proj: torch.nn.modules.linear.Linear,
    attn_output: torch.Tensor,
):
    b, n_head, q_len, h_dim = attn_output.size()
    o_proj_weight_split = o_proj.weight.view(o_proj.out_features, n_head, h_dim)

    # print(f"{o_proj_weight_split.size()=}")
    # print(f"{attn_output.size()=}")

    per_head_contributions = []
    for i in range(n_head):
        attn_output_per_head = attn_output[:, i, :, :]  # shape: (b, q_len, h_dim)
        attn_output_per_head = attn_output_per_head.to(o_proj_weight_split[:, i, :].dtype).to(
            o_proj_weight_split[:, i, :].device
        )
        projected_per_head = (
            attn_output_per_head @ o_proj_weight_split[:, i, :].T
        )  # shape: (b, q_len, out_features)
        per_head_contributions.append(projected_per_head)

    per_head_contributions = torch.stack(
        per_head_contributions, dim=1
    )  # shape: (b, n_head, q_len, out_features)
    attn_output = per_head_contributions.sum(dim=1)  # shape: (b, q_len, out_features)

    return attn_output, per_head_contributions


@dataclass(frozen=True)
class KQV_patch:
    k: Optional[torch.Tensor] = None
    q: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None

    def __post_init__(self):
        assert (
            self.k is not None or self.q is not None or self.v is not None
        ), "At least one of k, q, or v must be provided."


def LlamaAttentionPatcher(
    block_name: Optional[str] = None,
    cut_attn_edges: Optional[dict[int, list[AttentionEdge]]] = None,
    save_attn_for: Optional[list[int]] = None,
    attn_matrices: Optional[dict[int, torch.Tensor]] = None,
    attn_contributions: Optional[dict[int, torch.Tensor]] = None,
    save_kqv_for: Optional[list[tuple[int, int]]] = None,  # (head_idx, token_idx)
    kqv_states: Optional[dict[tuple[int, int], torch.Tensor]] = None,
    kqv_patches: Optional[dict[tuple[int, int], KQV_patch]] = None,
) -> callable:
    """
    Wraps the forward method of the `LlamaSdpaAttention` class
    Provides extra arguments for intervention and grabbing attention weights for visualization

    Args:
        block_name: name of the block (mainly for logging and debugging purposes)
        cut_attn_edges: [head_idx, [AttentionEdge(q_idx, k_idx)]] to cut off attention enge q_idx --> k_idx via a specific head
        save_attn_weights: list of head indices to save attention weights for visualization
        attn_matrices: [head_idx, attn_matrix] to store the attention matrix for a specific head
    """

    if save_attn_for is not None:
        assert (
            attn_matrices is not None or attn_contributions is not None
        ), "with save_attn_weights = True you need to provide attn_matrices or attn_contribution"
        if attn_matrices is not None:
            assert isinstance(attn_matrices, dict) and len(attn_matrices) == 0
        if attn_contributions is not None:
            assert isinstance(attn_contributions, dict) and len(attn_contributions) == 0

    if attn_matrices is not None and attn_contributions is not None:
        assert save_attn_for is not None

    if save_kqv_for is not None:
        assert kqv_states is not None
        assert isinstance(kqv_states, dict) and len(kqv_states) == 0

    def forward_patched(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        logger.debug(f"LlamaAttentionPatcher <> {block_name}")

        if output_attentions:
            raise NotImplementedError(
                "LlamaAttentionPatcher does not support output_attentions=True. Use the `attention_matrices` instead, or the `attn_contributions` argument for per-head contributions."
            )
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        # ---------------------------------------------------------------------
        if save_attn_for is not None:
            for head_idx in save_attn_for:
                if attn_matrices is not None:
                    attn_matrices[head_idx] = torch.zeros(bsz, q_len, q_len) - 1
                if attn_contributions is not None:
                    attn_contributions[head_idx] = (
                        torch.zeros(bsz, q_len, hidden_states.size(-1)) - 1
                    )
        # ---------------------------------------------------------------------

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # ---------------------------------------------------------------------
        if save_kqv_for is not None:
            for head_idx, token_idx in save_kqv_for:
                kqv_states[(head_idx, token_idx)] = {
                    "query": query_states[:, head_idx, token_idx, :].clone().squeeze(),
                    "key": key_states[:, head_idx // self.num_key_value_heads, token_idx, :]
                    .clone()
                    .squeeze(),
                    "value": value_states[:, head_idx // self.num_key_value_heads, token_idx, :]
                    .clone()
                    .squeeze(),
                }
        if kqv_patches is not None:
            for head_idx, token_idx in kqv_patches:
                kqv_patch = kqv_patches[(head_idx, token_idx)]
                if kqv_patch.q is not None:
                    query_states[:, head_idx, token_idx, :] = kqv_patch.q
                if kqv_patch.k is not None:
                    key_states[:, head_idx // self.num_key_value_heads, token_idx, :] = kqv_patch.k
                if kqv_patch.v is not None:
                    value_states[:, head_idx // self.num_key_value_heads, token_idx, :] = (
                        kqv_patch.v
                    )
        # ---------------------------------------------------------------------

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        if cut_attn_edges is None and save_attn_for is None:
            # logger.info("defer to the default faster implementation")
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            # logger.warning(
            #     "need to use slower custom implementation, should give numerically identical results"
            # )
            attn_output = scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
                cut_attn_edges=cut_attn_edges,
                store_attn_matrices=attn_matrices,
            )

        # ---------------------------------------------------------------------
        if attn_contributions is not None:
            __attn_output, per_head_contribution = attn_per_head(self.o_proj, attn_output)
            for head_idx in attn_contributions:
                attn_contributions[head_idx] = per_head_contribution[:, head_idx, :, :]
        # ---------------------------------------------------------------------

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if attn_contributions is not None:
            if torch.allclose(attn_output, __attn_output, atol=1e-3) == False:
                logger.warning(
                    f"allclose(attn_output, __attn_output)=False | {attn_output.norm().item()=}, {__attn_output.norm().item()=}"
                )

        return attn_output, None, past_key_value

    return forward_patched
