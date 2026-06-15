from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import contextlib
import functools
import onnxruntime as ort
import numpy as np

class Config:
    pass

DEFAULT_FP16_ONNX_PATH = (
    "./onnx_export/vision.onnx"
)

VISION_EXPORT_OUTPUT_NAMES = [
    "inputs_embeds",
    "attention_mask",
    "position_embeddings",
    "hidden_states",
    "attn_probs",
    "attn_weights",
    "query_states_proj",
    "query_states_norm",
    "query_states_rope",
    "hidden_states_inputlayernorm",
    "output",
    "layer0_input_layernorm",
    "layer0_q_norm",
    "layer0_block_out",
    "layer1_input_layernorm",
    "layer4_input_layernorm",
    "layer8_input_layernorm",
]

class Gemma4VisionPatchEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.position_embedding_size = config.position_embedding_size

        self.input_proj = nn.Linear(3 * self.patch_size**2, self.hidden_size, bias=False)
        self.position_embedding_table = nn.Parameter(torch.ones(2, self.position_embedding_size, self.hidden_size))

    # def _position_embeddings(self, pixel_position_ids: torch.Tensor, padding_positions: torch.Tensor) -> torch.Tensor:
    #     """Prepare patch positions map for matmul with positon embedding table."""
    #     # Expanding and permute patch positions to (batch_size, num_patches, 2, position_embedding_size) for matmul.
    #     clamped_positions = pixel_position_ids.clamp(min=0)
    #     one_hot = F.one_hot(clamped_positions, num_classes=self.position_embedding_size)
    #     one_hot = one_hot.permute(0, 2, 1, 3).to(self.position_embedding_table)
    #     # Compute positional embeddings and sum across x and y.
    #     position_embeddings = one_hot @ self.position_embedding_table
    #     position_embeddings = position_embeddings.sum(dim=1)
    #     # Zero out embeddings for any padding patches.
    #     position_embeddings = torch.where(padding_positions.unsqueeze(-1), 0.0, position_embeddings)
    #     return position_embeddings

    def _position_embeddings(
        self,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        输入:
            pixel_position_ids:
                shape = (B, 2520, 2)

        padding_positions:
                shape = (B, 2520)
        """

        # x_position_ids = pixel_position_ids[..., 0].to(torch.int32)
        # y_position_ids = pixel_position_ids[..., 1].to(torch.int32)

        x_coord = pixel_position_ids[..., 0]
        y_coord = pixel_position_ids[..., 1]
        # int32 mask avoids ONNX Clip/Max fp32 casts on position indices
        x_position_ids = x_coord * (x_coord >= 0).to(x_coord.dtype)
        y_position_ids = y_coord * (y_coord >= 0).to(y_coord.dtype)

        # x_embeddings = self.position_embedding_table[
        #     0,
        #     x_position_ids,
        # ]   # (B, 2520, hidden_size)

        # y_embeddings = self.position_embedding_table[
        #     1,
        #     y_position_ids,
        # ]   # (B, 2520, hidden_size)

        x_table = self.position_embedding_table[0]
        y_table = self.position_embedding_table[1]

        x_embeddings = F.embedding(
            x_position_ids,
            x_table,
        )

        y_embeddings = F.embedding(
            y_position_ids,
            y_table,
        )

        position_embeddings = x_embeddings + y_embeddings

        # position_embeddings = torch.where(
        #     padding_positions.unsqueeze(-1),
        #     torch.zeros(
        #         1,
        #         dtype=position_embeddings.dtype,
        #         device=position_embeddings.device,
        #     ),
        #     position_embeddings,
        # )   # (B, 2520, hidden_size)
        position_embeddings = (
                position_embeddings
                * (~padding_positions).unsqueeze(-1).to(position_embeddings.dtype)
            )

        return position_embeddings

    def forward(
        self, pixel_values: torch.Tensor, pixel_position_ids: torch.Tensor, padding_positions: torch.Tensor
    ) -> torch.Tensor:
        # Gemma4 applies no normalization and instead scales in model code
        pixel_values = 2 * (pixel_values - 0.5)
        hidden_states = self.input_proj(pixel_values.to(self.input_proj.weight.dtype))
        position_embeddings = self._position_embeddings(pixel_position_ids, padding_positions)
        return hidden_states + position_embeddings

def create_bidirectional_mask(attention_mask,dtype):
    """
    attention_mask: [B, L]
        True = valid
        False = padding

    return:
        [B, 1, L, L] bool
    """

    attention_mask = attention_mask[:, None, None, :]  # (batch, 1, 1, seq_len)
    #attention_mask = torch.where(attention_mask, 0.0, torch.finfo(dtype).min)
    attention_mask = torch.where(attention_mask, 0.0, -10000.0)

    return attention_mask

class Gemma4VisionRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        inv_freq, self.attention_scaling = self.compute_default_rope_parameters(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    def compute_default_rope_parameters(
        self,
        config = None,
        device = None,
        seq_len = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_theta
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        # The reference implementation computes RoPE frequencies INDEPENDENTLY
        # for each spatial dimension using the partitioned head_dim (head_dim // ndim),
        # so both x and y dimensions get identical frequency ranges.
        # This is different from splitting the global inv_freq between dimensions.
        spatial_dim = dim // 2

        attention_factor = 1.0  # Unused in this type of RoPE
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, spatial_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / spatial_dim)
        )
        return inv_freq, attention_factor

    # def forward(self, x, position_ids):
    #     inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)  #[B,dim/2,1]
    #     device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"

    #     # Multidimensional positions: [batch, num_patches, ndim]. Apply rotations to each spatial dim separately
    #     all_cos, all_sin = [], []
    #     for i in range(2):
    #         dim_position_ids = position_ids[:, :, i]
    #         dim_position_ids_expanded = dim_position_ids[:, None, :].float()    #[B,1,L]

    #         with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
    #             freqs = (inv_freq_expanded.float() @ dim_position_ids_expanded.float()).transpose(1, 2) #[B,L,dim/2]
    #             emb = torch.cat((freqs, freqs), dim=-1)
    #             cos = emb.cos() * self.attention_scaling
    #             sin = emb.sin() * self.attention_scaling
    #         all_cos.append(cos)
    #         all_sin.append(sin)

    #     cos = torch.cat(all_cos, dim=-1).to(dtype=x.dtype)
    #     sin = torch.cat(all_sin, dim=-1).to(dtype=x.dtype)
    #     return cos, sin
    def forward(self, x, position_ids):
        dtype = x.dtype
        inv_freq_expanded = self.inv_freq[None, :, None].to(dtype=dtype).expand(
            position_ids.shape[0], -1, 1
        ).to(x.device)

        all_cos, all_sin = [], []
        for i in range(2):
            dim_position_ids = position_ids[:, :, i]
            dim_position_ids_expanded = dim_position_ids[:, None, :].to(dtype=dtype)
            freqs = (inv_freq_expanded @ dim_position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
            all_cos.append(cos)
            all_sin.append(sin)

        cos = torch.cat(all_cos, dim=-1).to(dtype=dtype)
        sin = torch.cat(all_sin, dim=-1).to(dtype=dtype)
        return cos, sin

# def rotate_half(x):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)
def rotate_half(x):
    D = x.shape[-1]

    x = x.reshape(*x.shape[:-1], 2, D // 2)

    x1 = x[..., 0, :]
    x2 = x[..., 1, :]

    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        x (`torch.Tensor`): The tensor to embed.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
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
    return (x * cos) + (rotate_half(x) * sin)

def apply_multidimensional_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 2,
) -> torch.Tensor:
    """Applies multidimensional RoPE to inputs.

    Args:
        x (`torch.Tensor`): The tensor to embed.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            If position_ids.ndim + 2 == x.ndim, then this function passes through to `apply_rotary_pos_emb()`.
            Otherwise, position_ids is used to split the inputs, x, into multiple pieces, where each piece is fed to
            `apply_rotary_pos_emb()`, and then concatenated back together.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.

    Returns:
      Tensor of shape [B, L, N, H] with RoPE applied.
    """
    # ndim = position_ids.shape[-1]
    # num_input_channels = x.shape[-1]
    # num_rotated_channels_per_dim = 2 * (num_input_channels // (2 * ndim))

    # # Correctly split the input tensor into ndim parts
    # split_sizes = [num_rotated_channels_per_dim] * ndim
    # x_parts = torch.split(x, split_sizes, dim=-1)
    # cos_parts = torch.split(cos, split_sizes, dim=-1)
    # sin_parts = torch.split(sin, split_sizes, dim=-1)
    # y_parts = [
    #     apply_rotary_pos_emb(
    #         x=x_parts[k],
    #         cos=cos_parts[k],
    #         sin=sin_parts[k],
    #         unsqueeze_dim=unsqueeze_dim,
    #     )
    #     for k in range(ndim)
    # ]
    # return torch.cat(y_parts, dim=-1)
    # ndim = 2
    # num_input_channels = 64
    # num_rotated_channels_per_dim = 2 * (num_input_channels // (2 * ndim))

    # # Correctly split the input tensor into ndim parts
    x1 = x[..., :32]
    x2 = x[..., 32:64]

    cos1 = cos[..., :32]
    cos2 = cos[..., 32:64]

    sin1 = sin[..., :32]
    sin2 = sin[..., 32:64]
    y1 = apply_rotary_pos_emb(
        x1,
        cos1,
        sin1,
        unsqueeze_dim=2,
    )

    y2 = apply_rotary_pos_emb(
        x2,
        cos2,
        sin2,
        unsqueeze_dim=2,
    )

    return torch.cat([y1, y2], dim=-1)

class Gemma4ClippableLinear(nn.Module):
    def __init__(
        self,
        config,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

        self.register_buffer("input_min", torch.tensor(-float("inf")))
        self.register_buffer("input_max", torch.tensor(float("inf")))
        self.register_buffer("output_min", torch.tensor(-float("inf")))
        self.register_buffer("output_max", torch.tensor(float("inf")))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = torch.clamp(hidden_states, self.input_min, self.input_max)

        hidden_states = self.linear(hidden_states)

        hidden_states = torch.clamp(hidden_states, self.output_min, self.output_max)

        return hidden_states


class Gemma4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale

        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim), requires_grad=True)

    def _norm(self, hidden_states: torch.Tensor):
        # fp16-native: scale by amax before squaring to avoid x² overflow (see proj.Fp16RMSNorm)
        eps = torch.full((), self.eps, dtype=hidden_states.dtype, device=hidden_states.device)
        amax = hidden_states.abs().amax(dim=-1, keepdim=True) + eps
        xs = hidden_states / amax
        mean_squared = xs.pow(2).mean(-1, keepdim=True) + eps
        return xs * torch.pow(mean_squared, -0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed_output = self._norm(hidden_states)
        if self.with_scale:
            normed_output = normed_output * self.weight
        return normed_output

# def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
#     """
#     This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
#     num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
#     """
#     batch, num_key_value_heads, slen, head_dim = hidden_states.shape
#     if n_rep == 1:
#         return hidden_states
#     hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
#     return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
def repeat_kv(hidden_states, n_rep):
    batch, kv_heads, slen, dim = hidden_states.shape
    hidden_states = hidden_states.unsqueeze(2)  # (B, kv, 1, S, D)
    hidden_states = hidden_states.repeat(1, 1, n_rep, 1, 1)
    return hidden_states.view(batch, kv_heads * n_rep, slen, dim)

def eager_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
):
    del dropout, is_causal, kwargs
    if scaling is None:
        scaling = module.head_dim ** -0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3))
    attn_weights = attn_weights * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask.to(dtype=query.dtype)
    attn_probs = torch.softmax(attn_weights, dim=-1).to(dtype=query.dtype)
    attn_output = torch.matmul(attn_probs, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_probs, attn_weights


class Gemma4VisionAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = self.config.attention_dropout

        self.scaling = 1.0
        self.is_causal = False
        self.k_proj = Gemma4ClippableLinear(config, config.hidden_size, config.num_key_value_heads * self.head_dim)
        self.q_proj = Gemma4ClippableLinear(config, config.hidden_size, config.num_attention_heads * self.head_dim)
        self.v_proj = Gemma4ClippableLinear(config, config.hidden_size, config.num_key_value_heads * self.head_dim)
        self.o_proj = Gemma4ClippableLinear(config, config.num_attention_heads * self.head_dim, config.hidden_size)

        self.q_norm = Gemma4RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        B, L, _ = hidden_states.shape
        #query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_proj(hidden_states)
        query_states_proj = query_states
        query_states = query_states.contiguous()
        query_states = query_states.reshape(B,L,self.config.num_attention_heads,self.head_dim)
        
        query_states = self.q_norm(query_states)
        query_states_norm = query_states
        query_states = apply_multidimensional_rope(query_states, cos, sin, position_ids)
        query_states_rope = query_states
        query_states = query_states.transpose(1, 2)

        key_states = self.k_proj(hidden_states).view(hidden_shape)
        key_states = self.k_norm(key_states)
        key_states = apply_multidimensional_rope(key_states, cos, sin, position_ids)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_proj(hidden_states).view(hidden_shape)
        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

        attn_output, attn_probs, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            **kwargs,
        )
        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_probs, attn_weights,query_states_proj,query_states_norm,query_states_rope

class GELUTanh(nn.Module):
    """
    A fast C implementation of the tanh approximation of the GeLU activation function. See
    https://huggingface.co/papers/1606.08415.

    This implementation is equivalent to NewGELU and FastGELU but much faster. However, it is not an exact numerical
    match due to rounding errors.
    """

    def __init__(self):
        super().__init__()
        self.act = functools.partial(nn.functional.gelu, approximate="tanh")

    def forward(self, input):
        return self.act(input)

class Gemma4VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = Gemma4ClippableLinear(config, self.hidden_size, self.intermediate_size)
        self.up_proj = Gemma4ClippableLinear(config, self.hidden_size, self.intermediate_size)
        self.down_proj = Gemma4ClippableLinear(config, self.intermediate_size, self.hidden_size)
        self.act_fn = GELUTanh()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class Gemma4VisionEncoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Gemma4VisionAttention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma4VisionMLP(config)
        self.input_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states_input_layernorm = hidden_states

        hidden_states, attn_probs, attn_weights,query_states_proj,query_states_norm,query_states_rope = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states,attn_probs, attn_weights,query_states_proj,query_states_norm,query_states_rope,hidden_states_input_layernorm
        

        # residual = hidden_states

        # hidden_states_input_layernorm = self.input_layernorm(hidden_states)

        # hidden_states_attn, _ = self.self_attn(
        #     hidden_states=hidden_states_input_layernorm,
        #     position_embeddings=position_embeddings,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     **kwargs,
        # )
        # return hidden_states_attn
        # hidden_states_post_attention_layernorm = self.post_attention_layernorm(hidden_states_attn)
        # hidden_states = residual + hidden_states_post_attention_layernorm
        
        # residual = hidden_states
        # hidden_states_pre_ffn = self.pre_feedforward_layernorm(hidden_states)
        # hidden_states_mlp = self.mlp(hidden_states)
        # hidden_states_post_ffn = self.post_feedforward_layernorm(hidden_states)
        # hidden_states = residual + hidden_states

        # return hidden_states_input_layernorm,hidden_states_attn,hidden_states_post_attention_layernorm,hidden_states_pre_ffn,hidden_states_mlp,hidden_states_post_ffn,hidden_states


class Gemma4VisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.rotary_emb = Gemma4VisionRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [Gemma4VisionEncoderLayer(config=config, layer_idx=i) for i in range(self.num_layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_position_ids: torch.LongTensor | None = None,
        **kwargs,
    ):
        r"""
        pixel_position_ids (torch.Tensor):
            Patch positions as (x, y) coordinates in the image as [batch, num_patches, 2].
        """
        attention_mask = create_bidirectional_mask(
            attention_mask=attention_mask,
            dtype=inputs_embeds.dtype
        )

        # embed positions
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, pixel_position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, *_ = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=pixel_position_ids,
                **kwargs,
            )

        return hidden_states

class Gemma4VisionPooler(nn.Module):
    """Scaling and optional spatial pooling for vision encodings"""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.root_hidden_size = self.hidden_size**0.5
    
        pool = torch.zeros(280, 2520)
        idx = 0
        for y in range(48):
            for x in range(48):
                out_idx = (x // 3) + 16 * (y // 3)
                pool[out_idx, idx] = 1.0 / 9.0
                idx += 1

        self.register_buffer("pool_matrix", pool)

    def _avg_pool_by_positions(self, hidden_states, pixel_position_ids, length):
        B, N, H = hidden_states.shape
        pool = self.pool_matrix.unsqueeze(0).expand(B, -1, -1)  # [B, 280, 2520]
        output = torch.bmm(pool, hidden_states)                # [B, 280, H]
        return output

    # def _avg_pool_by_positions(
    #     self, hidden_states: torch.Tensor, pixel_position_ids: torch.Tensor, length: int
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     2D spatial pooling according to patch positions.
    #     Pools the input tokens by averaging patches within a `k^2` grid, where `k` is determined by the ratio between
    #     input and output lengths
    #     """
    #     input_seq_len = hidden_states.shape[1]
    #     k = int((input_seq_len // length) ** 0.5)
    #     k_squared = k**2

    #     # Clamp padding positions (which are -1) to 0 so they don't break one_hot.
    #     # Padding patches have zero hidden states so they contribute nothing to the average.
    #     clamped_positions = pixel_position_ids.clamp(min=0)
    #     max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
    #     kernel_idxs = torch.div(clamped_positions, k, rounding_mode="floor")
    #     kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
    #     weights = F.one_hot(kernel_idxs.long(), length).float() / k_squared
    #     output = weights.transpose(1, 2) @ hidden_states.float()
    #     mask = torch.logical_not((weights == 0).all(dim=1))
    #     return output.to(hidden_states.dtype), mask

    # def _avg_pool_by_positions(self, hidden_states, pixel_position_ids, length):

    #     B, N, H = hidden_states.shape
    #     k = 3
    #     k_squared = k * k

    #     clamped = pixel_position_ids.clone()
    #     clamped[:, 2304:, :] = 0

    #     max_x = clamped[..., 0].amax(dim=-1, keepdim=True) + 1

    #     kernel_xy = torch.div(clamped, k, rounding_mode="floor")

    #     kernel_idxs = (
    #         kernel_xy[..., 0] + (max_x // k) * kernel_xy[..., 1]
    #     )

        # =========================
        # [B, N, K]
        # =========================
        # idx = torch.arange(length, dtype=torch.float32)

        # weights = (kernel_idxs.unsqueeze(-1) == idx).to(hidden_states.dtype)

        # output = torch.bmm(
        #     weights.transpose(1, 2),   # [B, K, N]
        #     hidden_states              # [B, N, H]
        # ) / k_squared

        # return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        output_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        #hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)
        hidden_states = (
                hidden_states
                * (~padding_positions).unsqueeze(-1).to(hidden_states.dtype)
            )

        hidden_states = self._avg_pool_by_positions(
                hidden_states, pixel_position_ids, output_length
            )

        hidden_states *= self.root_hidden_size
        hidden_states = hidden_states[:,:256,:]
        return hidden_states

class Gemma4VisionModel(nn.Module):
    """The Gemma 4 Vision Encoder."""

    def __init__(self, config):
        super().__init__()
        self.patch_embedder = Gemma4VisionPatchEmbedder(config)
        self.encoder = Gemma4VisionEncoder(config)
        self.pooler = Gemma4VisionPooler(config)
        self.config = config

    def load_from_pretrained(self, pretrained_vis_model):
        vision_state_dict = pretrained_vis_model.state_dict()
        missing, unexpected = self.load_state_dict(vision_state_dict, strict=False)

        print("missing:", missing)
        print("unexpected:", unexpected)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_position_ids: torch.LongTensor,
        **kwargs,
    ):
        r"""
        pixel_values (`torch.FloatTensor` or `list[torch.FloatTensor]`):
            The images to encode. Either a single `[batch, channels, height, width]` tensor
            (all images same size) or a list of `[1, channels, height, width]` tensors (different sizes).
        pixel_position_ids (`torch.LongTensor` of shape `(batch_size, max_patches, 2)`):
            The patch positions as (x, y) coordinates in the image. Padding patches are indicated by (-1, -1).
        """
        pooling_kernel_size = self.config.pooling_kernel_size
        output_length = pixel_values.shape[-2] // (pooling_kernel_size * pooling_kernel_size)   #280

        #padding_positions = (pixel_position_ids == -1).all(dim=-1)
        padding_positions = (
                (pixel_position_ids[..., 0] == -1)
                & (pixel_position_ids[..., 1] == -1)
            )
        inputs_embeds = self.patch_embedder(pixel_values, pixel_position_ids, padding_positions)    #[B,2520,768]
        hidden_states = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=~padding_positions,  # encoder expects True=valid, padding_positions is True=padding
            pixel_position_ids=pixel_position_ids,
            **kwargs,
        )
        return self.pooler(
            hidden_states=hidden_states,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            output_length=output_length,
        )


def build_vision_config() -> Config:
    config = Config()
    config.attention_bias = False
    config.attention_dropout = 0.0
    config.chunk_size_feed_forward = 0
    config.default_output_length = 280
    config.global_head_dim = 64
    config.head_dim = 64
    config.hidden_size = 768
    config.initializer_range = 0.02
    config.intermediate_size = 3072
    config.max_position_embeddings = 131072
    config.num_attention_heads = 12
    config.num_hidden_layers = 16
    config.num_key_value_heads = 12
    config.patch_size = 16
    config.pooling_kernel_size = 3
    config.position_embedding_size = 10240
    config.rms_norm_eps = 1e-06
    config.rope_theta = 100.0
    return config


def export_vision_encoder(
    model,
    path: str = DEFAULT_FP16_ONNX_PATH,
    model_path: str = "./gemma-4-E2B-it",
    opset_version: int = 11,
):
    config = build_vision_config()

    module = Gemma4VisionModel(config).to("cpu", torch.float16)
    module.load_from_pretrained(model)
    module.eval()

    processor = AutoProcessor.from_pretrained(model_path)
    image = Image.open(
        "path/to/image.jpg"
    ).convert("RGB").resize((768, 768))
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(torch.float16)

    pixel_values = inputs["pixel_values"]
    image_position_ids = inputs["image_position_ids"].to(torch.int32)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.onnx.export(
        module,
        (pixel_values, image_position_ids),
        path,
        input_names=["pixel_values", "image_position_ids"],
        output_names=["hidden_states"],
        opset_version=opset_version,
    )

    print(f"exported fp16 ONNX -> {path}")


def main():
    print("Loading model ...")
    MODEL_PATH = "./gemma-4-E2B-it"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cpu",
        attn_implementation="eager",
    ).eval()

    print("Exporting fp16 vision ONNX ...")
    export_vision_encoder(model.model.vision_tower, model_path=MODEL_PATH)

    print("\n✅ export done!")


if __name__ == "__main__":
    main()



# def export_vision_step_by_step(model, export_dir="./onnx_export"):
#     import os
#     os.makedirs(export_dir, exist_ok=True)
#     config = get_default_config()
#     device = torch.device("cpu")

#     # ===================== 完全沿用你的输入构造 =====================
#     processor = AutoProcessor.from_pretrained("./gemma-4-E2B-it")
#     image = Image.open('path/to/image.jpg').convert("RGB").resize((768, 768))
#     messages = [
#         {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "What is shown in this image?"}]}
#     ]
#     inputs = processor.apply_chat_template(
#         messages, tokenize=True, return_dict=True, return_tensors="pt"
#     ).to(device=device, dtype=torch.float32)

#     pixel_values = inputs["pixel_values"]
#     pixel_position_ids = inputs['image_position_ids'].to(torch.int32)
#     padding_positions = (pixel_position_ids[..., 0] == -1) & (pixel_position_ids[..., 1] == -1)
#     attention_mask = ~padding_positions
#     output_length = pixel_values.shape[-2] // (config.pooling_kernel_size ** 2)

#     # 加载你的完整模型
#     vis_model = Gemma4VisionModel(config).to(device).eval()
#     vis_model.load_from_pretrained(model)

#     print("\n===== 开始分步导出 =====")

#     # ==============================================
#     # 1. Patch Embedder
#     # ==============================================
#     print("\n[1/5] 导出 PatchEmbedder")
#     patch_embed = vis_model.patch_embedder
#     with torch.no_grad():
#         embed_out = patch_embed(pixel_values, pixel_position_ids, padding_positions)
#     torch.onnx.export(
#         patch_embed,
#         (pixel_values, pixel_position_ids, padding_positions),
#         f"{export_dir}/step1_patch_embedder.onnx",
#         input_names=["pixel_values", "pixel_position_ids", "padding_positions"],
#         output_names=["embed_out"],
#         opset_version=11,
#     )

#     # ==============================================
#     # 2. Rotary Position Embedding
#     # ==============================================
#     print("[2/5] 导出 RoPE")
#     rope = vis_model.encoder.rotary_emb
#     with torch.no_grad():
#         cos, sin = rope(embed_out, pixel_position_ids)
#     torch.onnx.export(
#         rope,
#         (embed_out, pixel_position_ids),
#         f"{export_dir}/step2_rope.onnx",
#         input_names=["hidden_states", "pixel_position_ids"],
#         output_names=["cos", "sin"],
#         opset_version=11,
#     )

#     # ==============================================
#     # 3. 第 0 层 Attention
#     # ==============================================
#     print("[3/5] 导出 Layer 0 Attention")
#     layer0 = vis_model.encoder.layers[0]
#     attn_mask_bidirectional = create_bidirectional_mask(attention_mask)
#     with torch.no_grad():
#         ln_out = layer0.input_layernorm(embed_out)
#     torch.onnx.export(
#         layer0.self_attn,
#         (ln_out, (cos, sin), attn_mask_bidirectional, pixel_position_ids),
#         f"{export_dir}/step3_attention.onnx",
#         input_names=["hidden_states", "cos", "sin", "attention_mask", "position_ids"],
#         output_names=["attn_out", "attn_weights"],
#         opset_version=11,
#     )

#     # ==============================================
#     # 4. 第 0 层 Encoder Layer
#     # ==============================================
#     print("[4/5] 导出 Layer 0 EncoderBlock")
#     torch.onnx.export(
#         layer0,
#         (embed_out, (cos, sin), attn_mask_bidirectional, pixel_position_ids),
#         f"{export_dir}/step4_encoder_layer.onnx",
#         input_names=["hidden_states", "cos", "sin", "attention_mask", "position_ids"],
#         output_names=["layer_out"],
#         opset_version=11,
#     )

#     # ==============================================
#     # 5. Pooler
#     # ==============================================
#     print("[5/5] 导出 Pooler")
#     with torch.no_grad():
#         encoder_out = vis_model.encoder(embed_out, attention_mask, pixel_position_ids)
#     torch.onnx.export(
#         vis_model.pooler,
#         (encoder_out, pixel_position_ids, padding_positions, output_length),
#         f"{export_dir}/step5_pooler.onnx",
#         input_names=["hidden_states", "pixel_position_ids", "padding_positions", "output_length"],
#         output_names=["pooled_out"],
#         opset_version=11,
#     )

#     print(f"\n✅ 全部导出完成！目录：{export_dir}")


# def get_default_config():
#     config = Config()
#     config.attention_bias = False
#     config.attention_dropout = 0.0
#     config.chunk_size_feed_forward = 0
#     config.default_output_length = 280
#     config.global_head_dim = 64
#     config.head_dim = 64
#     config.hidden_size = 768
#     config.initializer_range = 0.02
#     config.intermediate_size = 3072
#     config.max_position_embeddings = 131072
#     config.num_attention_heads = 12
#     config.num_hidden_layers = 16
#     config.num_key_value_heads = 12
#     config.patch_size = 16
#     config.pooling_kernel_size = 3
#     config.position_embedding_size = 10240
#     config.rms_norm_eps = 1e-06
#     config.rope_theta = 100.0
#     return config


# # ==========================
# # 运行入口
# # ==========================
# if __name__ == "__main__":
#     MODEL_PATH = "./gemma-4-E2B-it"
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_PATH, torch_dtype=torch.float32, device_map="cpu"
#     ).eval()

#     export_vision_step_by_step(model.model.vision_tower)

