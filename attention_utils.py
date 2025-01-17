import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class AttentionEdge:
    """Represents a directed attention connection from query token to key token."""
    q_idx: int  # query token index
    k_idx: int  # key token index

@dataclass(frozen=True)
class KQV_patch:
    """Container for patching key, query, or value vectors at specific positions."""
    k: Optional[torch.Tensor] = None
    q: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None

    def __post_init__(self):
        assert (
            self.k is not None or self.q is not None or self.v is not None
        ), "At least one of k, q, or v must be provided."

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats key/value heads to match the number of query heads.
    
    Args:
        hidden_states: Tensor of shape (batch, num_key_value_heads, seq_len, head_dim)
        n_rep: Number of times to repeat each head
    Returns:
        Tensor of shape (batch, num_key_value_heads * n_rep, seq_len, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q: The query tensor
        k: The key tensor
        cos: The cosine part of the rotary embedding
        sin: The sine part of the rotary embedding
        position_ids: Deprecated and unused
        unsqueeze_dim: Dimension along which to unsqueeze cos/sin for broadcasting
    Returns:
        Tuple of rotated query and key tensors
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    cut_attn_edges: Optional[Dict[int, List[AttentionEdge]]] = None,
    store_attn_matrices: Optional[Dict[int, torch.Tensor]] = None,
    softcap: Optional[float] = None,
) -> torch.Tensor:
    """Custom scaled dot-product attention with support for attention edge cutting.
    
    Args:
        query: Query tensor
        key: Key tensor
        value: Value tensor
        mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Whether to apply causal masking
        scale: Optional scaling factor (if None, uses 1/sqrt(d_k))
        cut_attn_edges: Dictionary mapping head indices to list of attention edges to cut
        store_attn_matrices: Dictionary to store attention matrices for specific heads
        softcap: Optional softcapping value for attention logits
    """
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    # Initialize attention bias tensor
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

    # Apply causal masking if requested
    if is_causal:
        assert mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    # Add external mask if provided
    if mask is not None:
        if mask.dtype == torch.bool:
            attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
        else:
            attn_bias += mask.squeeze()

    # Calculate attention scores
    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    attn_weight = attn_weight + attn_bias.to(attn_weight.dtype).to(attn_weight.device)

    # Apply softcap if provided
    if softcap is not None:
        attn_weight = attn_weight / softcap
        attn_weight = torch.tanh(attn_weight)
        attn_weight = attn_weight * softcap
    
    # Cut attention edges if specified
    if cut_attn_edges is not None:
        for head_idx, edges in cut_attn_edges.items():
            for edge in edges:
                attn_weight[:, head_idx, edge.q_idx, edge.k_idx] = float("-inf")

    # Apply softmax with explicit precision control
    attn_weight = F.softmax(attn_weight, dim=-1, dtype=torch.float32).to(query.dtype)

    # Apply dropout
    if dropout_p > 0.0:
        attn_weight = F.dropout(attn_weight, p=dropout_p)

    # Store attention matrices if requested
    if store_attn_matrices is not None:
        for head_idx in store_attn_matrices:
            store_attn_matrices[head_idx] = attn_weight[:, head_idx, :, :].clone()

    # Calculate output
    return torch.matmul(attn_weight, value)

def attn_per_head(
    o_proj: torch.nn.Linear,
    attn_output: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates per-head contributions to the output.
    
    Args:
        o_proj: Output projection layer
        attn_output: Attention output tensor of shape (batch, n_head, seq_len, head_dim)
    Returns:
        Tuple of:
            - Combined output tensor
            - Per-head contribution tensors
    """
    b, n_head, q_len, h_dim = attn_output.size()
    o_proj_weight_split = o_proj.weight.view(o_proj.out_features, n_head, h_dim)

    per_head_contributions   = []
    for i in range(n_head):
        attn_output_per_head = attn_output[:, i, :, :]
        attn_output_per_head = attn_output_per_head.to(o_proj_weight_split.dtype).to(o_proj_weight_split.device)
        projected_per_head = attn_output_per_head @ o_proj_weight_split[:, i, :].T
        per_head_contributions.append(projected_per_head)

    per_head_contributions = torch.stack(per_head_contributions, dim=1)
    attn_output = per_head_contributions.sum(dim=1)

    return attn_output, per_head_contributions 


def AttentionPatcher(
    block_name: Optional[str] = None,
    cut_attn_edges: Optional[Dict[int, List[AttentionEdge]]] = None,
    save_attn_for: Optional[List[int]] = None,
    attn_matrices: Optional[Dict[int, torch.Tensor]] = None,
    attn_contributions: Optional[Dict[int, torch.Tensor]] = None,
    save_kqv_for: Optional[List[Tuple[int, int]]] = None,  # (head_idx, token_idx)
    kqv_states: Optional[Dict[Tuple[int, int], Dict[str, torch.Tensor]]] = None,
    kqv_patches: Optional[Dict[Tuple[int, int], KQV_patch]] = None,
) -> callable:
    """
    Patches Llama3's and Gemma2's attention mechanism to support attention edge cutting and weight visualization.
    Important Note: Only works for Gemma2 with `attn_implementation='spda'`
    
    Args:
        block_name: Name of the attention block (for logging)
        cut_attn_edges: Dictionary mapping head indices to list of attention edges to cut
        save_attn_for: List of head indices to save attention weights for
        attn_matrices: Dictionary to store attention matrices
        attn_contributions: Dictionary to store per-head output contributions
        save_kqv_for: List of (head_idx, token_idx) pairs to save KQV states for
        kqv_states: Dictionary to store KQV states
        kqv_patches: Dictionary mapping (head_idx, token_idx) to KQV patches
    """
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
        logger.debug(f"GemmaAttentionPatcher <> {block_name}")

        if output_attentions:
            raise NotImplementedError(
                "GemmaAttentionPatcher does not support output_attentions=True. Use the `attention_matrices` instead, or the `attn_contributions` argument for per-head contributions."
            )

        batch_size, input_len, _ = hidden_states.shape
        num_heads = self.config.num_attention_heads
        num_key_value_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim
        num_key_value_groups = num_heads // num_key_value_heads

        # Initialize storage for attention matrices if requested
        if save_attn_for is not None:
            for head_idx in save_attn_for:
                if attn_matrices is not None:
                    attn_matrices[head_idx] = torch.zeros(batch_size, input_len, input_len) - 1
                if attn_contributions is not None:
                    attn_contributions[head_idx] = torch.zeros(batch_size, input_len, hidden_states.size(-1)) - 1

        # Project hidden states to query, key, value
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to separate heads
        query_states = query_states.view(batch_size, input_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, input_len, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, input_len, num_key_value_heads, head_dim).transpose(1, 2)

        # Save KQV states if requested
        if save_kqv_for is not None:
            for head_idx, token_idx in save_kqv_for:
                kqv_states[(head_idx, token_idx)] = {
                    "query": query_states[:, head_idx, token_idx, :].clone().squeeze(),
                    "key": key_states[:, head_idx // num_key_value_groups, token_idx, :].clone().squeeze(),
                    "value": value_states[:, head_idx // num_key_value_groups, token_idx, :].clone().squeeze(),
                }

        # Apply KQV patches if provided
        if kqv_patches is not None:
            for head_idx, token_idx in kqv_patches:
                kqv_patch = kqv_patches[(head_idx, token_idx)]
                if kqv_patch.q is not None:
                    query_states[:, head_idx, token_idx, :] = kqv_patch.q
                if kqv_patch.k is not None:
                    key_states[:, head_idx // num_key_value_groups, token_idx, :] = kqv_patch.k
                if kqv_patch.v is not None:
                    value_states[:, head_idx // num_key_value_groups, token_idx, :] = kqv_patch.v

        # Get position embeddings if not provided
        if position_embeddings is None:
            # NOTE: Rotary embeddings can be computed with complex numbers (output: freq_cis) or real numbers (output: cos, sin)
            # The used method depends on Llama and Gemma versions, complex is more stable, real is more readable.
            # Gemma2 and Llama3 compute once at the beginning of the forward pass. We recompute here for readability. 
            # This could be changed to be more efficient / true to the original implementation.
            cos, sin = self.rotary_emb(hidden_states, position_ids)
        else:
            cos, sin = position_embeddings

        # Apply rotary embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Repeat KV heads if necessary
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

        is_causal = True if causal_mask is None and input_len > 1 else False

        # Gemma2 has scaling, Llama3 does not
        # Gemma2 attn_implementation='spda' has no softcapping, but attn_implementation='eager' does
        spda_kwargs = {}
        if hasattr(self, 'scaling'):
            spda_kwargs['scale'] = self.scaling

        if cut_attn_edges is None and save_attn_for is None:
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
                **spda_kwargs,
            )
        else:
            attn_output = scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
                cut_attn_edges=cut_attn_edges,
                store_attn_matrices=attn_matrices,
                **spda_kwargs,
            )

        # Calculate per-head contributions if requested
        if attn_contributions is not None:
            __attn_output, per_head_contribution = attn_per_head(self.o_proj, attn_output)
            for head_idx in attn_contributions:
                attn_contributions[head_idx] = per_head_contribution[:, head_idx, :, :]

        # Project back to hidden dimension
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, input_len, -1)
        attn_output = self.o_proj(attn_output)

        # Verify output matches if per-head contributions were calculated
        if attn_contributions is not None:
            if not torch.allclose(attn_output, __attn_output, atol=1e-3):
                logger.warning(
                    f"allclose(attn_output, __attn_output)=False | {attn_output.norm().item()=}, {__attn_output.norm().item()=}"
                )

        # Return both attention output and weights (None) to match Llama3's interface
        return attn_output, None

    return forward_patched