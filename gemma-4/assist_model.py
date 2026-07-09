from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import functools
import onnxruntime as ort

def _compute_proportional_rope_parameters(
    config = None,
    device = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
    head_dim_key: str = "head_dim",
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with proportional RoPE.

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration. This function assumes that the config will provide at least the following
            properties:

            *   rope_theta (`float`, *optional*): The base wavelength from which the inverse frequencies will be derived. Defaults to `config.default_theta` if omitted.
            *   hidden_size (`int`): The numerator when deriving a head_dim, if not provided directly.
            *   num_attention_heads (`int`): The denominator when deriving a head_dim, if not provided directly.

            Additionally, this function will make use of the following properties if they are found in the config:

            *   head_dim (`int`, *optional*): The size of the key-value heads in the model. If None, this value will be
                derived as hidden_size // num_attention_heads.
            *   partial_rotary_factor (`float`, *optional*, defaults to 1.0): The proportion of the embedding dimension
                to apply rotary positional encoding, e.g., [0.0, 0.25, 0.5, 0.75, 1.0]. Unlike other RoPE functions
                that use this parameter, proportional RoPE will always return an encoding that is the size of
                `head_dim`.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.

    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
    #config.standardize_rope_params()
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters

    head_dim = getattr(config, head_dim_key, None) or config.hidden_size // config.num_attention_heads
    base = rope_parameters_dict["rope_theta"]
    factor = rope_parameters_dict.get("factor", 1.0)
    rope_proportion = rope_parameters_dict.get("partial_rotary_factor", 1.0)

    attention_factor = 1.0  # Unused in this type of RoPE

    rope_angles = int(rope_proportion * head_dim // 2)

    inv_freq_rotated = 1.0 / (
        base
        ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / head_dim)
    )

    nope_angles = head_dim // 2 - rope_angles
    if nope_angles > 0:
        inv_freq = torch.cat(
            (
                inv_freq_rotated,
                torch.zeros(nope_angles, dtype=torch.float32, device=device),
            ),
            dim=0,
        )
    else:
        inv_freq = inv_freq_rotated

    inv_freq /= factor
    return inv_freq, attention_factor

class Gemma4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale

        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim), requires_grad=True)

    def _norm(self, hidden_states: torch.Tensor):
        x = hidden_states
        eps = torch.full((), self.eps, dtype=x.dtype, device=x.device)
        amax = x.abs().amax(dim=-1, keepdim=True) + eps
        xs = x / amax
        mean_squared = xs.pow(2).mean(-1, keepdim=True) + eps
        return xs * torch.pow(mean_squared, -0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed_output = self._norm(hidden_states)
        if self.with_scale:
            normed_output = normed_output * self.weight
        return normed_output

class Gemma4TextMLP(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        use_double_wide_mlp = config.use_double_wide_mlp and is_kv_shared_layer
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size * (2 if use_double_wide_mlp else 1)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = GELUTanh()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

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

    return attn_output, attn_probs


class Gemma4TextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.head_dim = config.global_head_dim if not self.is_sliding and config.global_head_dim else config.head_dim
        self.use_alternative_attention = config.attention_k_eq_v and not self.is_sliding
        num_key_value_heads = (
            config.num_global_key_value_heads if self.use_alternative_attention else config.num_key_value_heads
        )
        self.num_key_value_groups = config.num_attention_heads // num_key_value_heads
        self.scaling = 1.0
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = config.use_bidirectional_attention != "all"

        # Shared kv cache
        first_kv_shared_layer_idx = self.config.num_hidden_layers - getattr(self.config, "num_kv_shared_layers", 0)
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx >= 0
        prev_layers = config.layer_types[:first_kv_shared_layer_idx]
        self.store_full_length_kv = not self.is_kv_shared_layer and layer_idx == len(prev_layers) - 1 - prev_layers[
            ::-1
        ].index(config.layer_types[layer_idx])

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.q_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)

        # Layers sharing kv states don't need any weight matrices
        if not self.is_kv_shared_layer:
            self.k_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
            self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)

            self.k_proj = nn.Linear(
                config.hidden_size, num_key_value_heads * self.head_dim, bias=config.attention_bias
            )
            self.v_proj = (
                nn.Linear(config.hidden_size, num_key_value_heads * self.head_dim, bias=config.attention_bias)
                if not self.use_alternative_attention
                else None
            )

        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
        past_key_values = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        # For layers with shared KV (from kv sharing point onwards), we reuse the same keys/values states as the last non-sharing layer.
        # We cannot simply reuse the cached state if we have a Cache, as sliding layers will not remember the full states in their Cache
        # once we are past the sliding window - so we always use `shared_kv_states` instead, even when past_key_values is not None
        if self.is_kv_shared_layer:
            key_states, value_states = shared_kv_states[self.layer_type]
            # Device of past layer may be different from current one
            key_states = key_states.to(query_states.device)
            value_states = value_states.to(query_states.device)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
            key_states = key_states.transpose(1, 2)

            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)

        if past_key_values is not None and not self.is_kv_shared_layer:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        if self.store_full_length_kv:
            shared_kv_states[self.layer_type] = key_states, value_states

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, shared_kv_states

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

class Gemma4TextDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Gemma4TextAttention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma4TextMLP(config, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.act_fn = GELUTanh()
            self.per_layer_input_gate = nn.Linear(self.hidden_size, self.hidden_size_per_layer_input, bias=False)
            self.per_layer_projection = nn.Linear(self.hidden_size_per_layer_input, self.hidden_size, bias=False)
            self.post_per_layer_input_norm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor = None,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ ,shared_kv_states= self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_kv_states=shared_kv_states,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        if self.hidden_size_per_layer_input:
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = self.act_fn(hidden_states)
            hidden_states = hidden_states * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states *= self.layer_scalar
        return hidden_states,shared_kv_states

class Gemma4TextRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None, layer_type=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.layer_types = ["full_attention","sliding_attention"]
        self.rope_init_fns = {}
        self.rope_type = {}

        for layer_type in self.layer_types:
            rope_params = self.config.rope_parameters[layer_type]
            if rope_params is None:
                continue
            rope_type = rope_params["rope_type"]
            if layer_type =='full_attention':
                rope_init_fn = _compute_proportional_rope_parameters
                self.rope_init_fns[layer_type] = rope_init_fn
                self.rope_type[layer_type] = rope_type
                curr_inv_freq, curr_attention_scaling = rope_init_fn(config=self.config, device=device, 
                                                                    layer_type=layer_type, head_dim_key="global_head_dim")
                self.register_buffer(f"{layer_type}_inv_freq", curr_inv_freq, persistent=False)
                self.register_buffer(f"{layer_type}_original_inv_freq", curr_inv_freq.clone(), persistent=False)
                setattr(self, f"{layer_type}_attention_scaling", curr_attention_scaling)
            else:
                rope_init_fn = self.compute_default_rope_parameters
                self.rope_init_fns[layer_type] = rope_init_fn
                self.rope_type[layer_type] = rope_type
                curr_inv_freq, curr_attention_scaling = rope_init_fn(self.config, device=device, 
                                                                    layer_type=layer_type)
                self.register_buffer(f"{layer_type}_inv_freq", curr_inv_freq, persistent=False)
                self.register_buffer(f"{layer_type}_original_inv_freq", curr_inv_freq.clone(), persistent=False)
                setattr(self, f"{layer_type}_attention_scaling", curr_attention_scaling)

    def compute_default_rope_parameters(
        self,
        config = None,
        device = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
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
            layer_type (`str`, *optional*):
                The current layer type if the model has different RoPE parameters per type.
                Should not be used unless `config.layer_types is not None`

        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
        base = config.rope_parameters[layer_type]["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    def forward(self, x, position_ids, layer_type=None):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")
        dtype = x.dtype

        inv_freq_expanded = inv_freq[None, :, None].to(dtype=dtype).expand(
            position_ids.shape[0], -1, 1
        ).to(x.device)
        position_ids_expanded = position_ids[:, None, :].to(dtype=dtype)

        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling

        return cos.to(dtype=dtype), sin.to(dtype=dtype)

class Gemma4TextScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.scalar_embed_scale = embed_scale
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)

class Gemma4TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
        )
        self.layers = nn.ModuleList(
            [Gemma4TextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma4TextRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.unique_layer_types = ['sliding_attention','full_attention']

        # Update `_keys_to_ignore_on_load_unexpected` to drop all k/v proj and norms for the shared layers
        self._keys_to_ignore_on_load_unexpected = ["full_attention", "sliding_attention"]
        for i, layer in enumerate(self.layers):
            if layer.self_attn.is_kv_shared_layer:
                self._keys_to_ignore_on_load_unexpected.extend(
                    [f"layers.{i}.self_attn.{name}" for name in ("k_proj", "v_proj", "k_norm", "v_norm")]
                )

    def load_from_pretrained(self, pretrained_model):
        vision_state_dict = pretrained_model.state_dict()
        missing, unexpected = self.load_state_dict(vision_state_dict, strict=False)

        print("missing:", missing)
        print("unexpected:", unexpected)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values = None,
        inputs_embeds: torch.FloatTensor | None = None,
        per_layer_inputs: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ):
        r"""
        per_layer_inputs (`torch.Tensor` of shape `(batch_size, sequence_length, num_hidden_layers, hidden_size_per_layer_input)`, *optional*):
            Pre-computed per-layer input embeddings. When provided, these are used directly instead of being
            computed from `input_ids` via `get_per_layer_inputs()`. This is primarily used by the multimodal
            model (`Gemma4Model`) which pre-computes per-layer inputs from the original `input_ids` *before*
            merging multimodal soft tokens into `inputs_embeds` — at which point the original token ids are
            no longer recoverable.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        causal_mask_mapping = attention_mask

        # embed positions
        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in self.unique_layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        # Initialize as empty dict - it will be filled in the right layers, or use passed ones
        shared_kv_states = kwargs.pop("shared_kv_states", {})

        # decoder layers
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):

            hidden_states,shared_kv_states = decoder_layer(
                hidden_states,
                per_layer_input,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings[self.config.layer_types[i]],
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states,shared_kv_states

 
class Gemma4AssistantMaskedEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        text_config = config.text_config
        self.config = config
        self.centroid_intermediate_top_k = self.config.centroid_intermediate_top_k
        self.hidden_size = text_config.hidden_size
        self.num_centroids = self.config.num_centroids
        self.vocab_size = text_config.vocab_size
        self.vocab_size_per_centroid = self.vocab_size // self.num_centroids

        self.centroids = nn.Linear(self.hidden_size, self.num_centroids, bias=False)
        self.register_buffer("token_ordering", torch.empty(self.vocab_size, dtype=torch.long))

    def forward(self, hidden_states: torch.Tensor, lm_head_weight: torch.Tensor) -> torch.Tensor:
        batch, seq_len = hidden_states.shape[:2]
        centroid_logits = self.centroids(hidden_states)

        _, top_k_indices = torch.topk(centroid_logits, k=self.centroid_intermediate_top_k, dim=-1)
        token_ordering = self.token_ordering.long()
        canonical_positions_per_cluster = token_ordering.view(self.num_centroids, self.vocab_size_per_centroid)

        # For selected top-K clusters, get canonical positions
        selected_canonical = canonical_positions_per_cluster[top_k_indices]  # [B, L, top_k, K]

        # Gather embeddings from lm_head at these canonical positions
        selected_flat = selected_canonical.reshape(-1)  # [B*L*top_k*K]
        selected_embeddings = lm_head_weight[selected_flat].view(
            batch, seq_len, self.centroid_intermediate_top_k * self.vocab_size_per_centroid, self.hidden_size
        )

        # Compute dot products: [B, L, 1, D] @ [B, L, D, top_k*K] -> [B, L, top_k*K]
        selected_logits = (hidden_states.unsqueeze(-2) @ selected_embeddings.transpose(-1, -2)).squeeze(-2)
        mask_value = selected_logits.min().item() - 1.0

        # Scatter logits directly to canonical positions in the output
        output = torch.full(
            (batch, seq_len, self.vocab_size),
            fill_value=mask_value,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        scatter_idx = selected_canonical.view(batch, seq_len, -1)  # [B, L, top_k*K]
        return output.scatter_(dim=-1, index=scatter_idx, src=selected_logits)

class Gemma4AssistantForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        text_config = config.text_config

        self.vocab_size = text_config.vocab_size
        self.hidden_size = text_config.hidden_size
        self.backbone_hidden_size = config.backbone_hidden_size

        self.model = Gemma4TextModel(text_config)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.pre_projection = nn.Linear(2 * self.backbone_hidden_size, self.hidden_size, bias=False)
        self.post_projection = nn.Linear(self.hidden_size, self.backbone_hidden_size, bias=False)

        self.masked_embedding = Gemma4AssistantMaskedEmbedder(config)

    def load_from_pretrained(self, pretrained_model):
        vision_state_dict = pretrained_model.state_dict()
        missing, unexpected = self.load_state_dict(vision_state_dict, strict=False)

        print("missing:", missing)
        print("unexpected:", unexpected)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,  # Not actually used, only kept in signature to be ignored
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        attention_mask: dict[str, torch.Tensor] | None = None,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool | None = None,  # Not actually used, only kept in signature to be ignored
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        shared_kv_states (`dict[str, torch.Tensor` of shape `(batch_size, 1, q_len, kv_len)`, *optional*):
            A dictionary containing the computed KV values for the last layer of each `layer_type` in this model.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Gemma4AssistantForCausalLM, Gemma4ForCausalLM

        >>> model = Gemma4ForCausalLM.from_pretrained("google/gemma-4-e2b-it")
        >>> assistant_model = Gemma4AssistantForCausalLM.from_pretrained("google/gemma-4-e2b-it-assistant")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-e2b-it")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, assistant_model=assistant_model, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        "What is your favorite condiment?"
        ```"""
        if inputs_embeds is None or shared_kv_states is None:
            raise ValueError("inputs_embeds and shared_kv_states cannot be None.")

        inputs_embeds = self.pre_projection(inputs_embeds)
        bidirectional_masks = self.create_attention_masks(inputs_embeds, attention_mask, shared_kv_states)

        outputs = self.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=bidirectional_masks,
            position_ids=position_ids,
            shared_kv_states=shared_kv_states,
            use_cache=False,
            **kwargs,
        )

        last_hidden_state = outputs.last_hidden_state
        projected_state = self.post_projection(last_hidden_state)

        logits = self.masked_embedding(last_hidden_state, self.lm_head.weight)

        return projected_state,logits,outputs.hidden_states

    def create_attention_masks(self, inputs_embeds, attention_mask, shared_kv_states):
        """
        Prepare the attention masks for the assisted model; the `shared_kv_states` acts as past cache in this instance.

        We use bidirectional masks to account for causality
            - There is no difference for the edge case of `q_len == 1` as it acts as full attention no matter what
            - SWA interprets the window as forward-looking (future) when `q_idx=1` and `kv>=1`
                - We switch from a future to a past perspective by flipping on the kv axis
                - To account for position invariant padding, we also flip the base attention mask before initial creation
        """
        config = self.config.text_config
        # (bsz, num_heads, seq_len, head_dim) -> (bsz, seq_len, head_dim)
        encoder_states_full_attn = shared_kv_states["full_attention"][0][:, 0]
        encoder_states_swa_attn = shared_kv_states["sliding_attention"][0][:, 0]

        sliding_attention_mask = attention_mask
        if attention_mask is not None:
            # Adjust for full mask --> cut mask only for valid kv states
            attention_mask = attention_mask[:, : encoder_states_full_attn.shape[1]]

            # 1. Take the last x entries to account for any potential SWA cutoff (from the main model)
            # 2. Flip the mask here to stay position invariant (along the original kv); see the flip at the end
            sliding_attention_mask = attention_mask[:, -encoder_states_swa_attn.shape[1] :].flip(dims=(1,))

        full_attention_mask = create_bidirectional_mask(
            config=config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_states_full_attn,
        )
        swa_mask = create_bidirectional_sliding_window_mask(
            config=config,
            inputs_embeds=inputs_embeds,
            attention_mask=sliding_attention_mask,
            encoder_hidden_states=encoder_states_swa_attn,
        )

        if swa_mask is not None:
            # Reverse the future token perspective to a past tokens perspective by flipping the construct (kv == -1)
            swa_mask = swa_mask.flip(dims=(-1,))

        return {"full_attention": full_attention_mask, "sliding_attention": swa_mask}

def create_bidirectional_mask(
    attention_mask: torch.Tensor | None,
    dtype: torch.dtype = torch.float16,
):
    """
    Create standard bidirectional mask

    Returns:
        shape = [B, 1, L, L]
        valid = 0
        masked = -inf
    """

    B, L = attention_mask.shape
    device = attention_mask.device

    mask = torch.ones(L, L, device=device, dtype=torch.bool)  # 全1 = 双向
    mask = mask.view(1, 1, L, L)

    # padding mask
    if attention_mask is not None:
        key_mask = attention_mask.bool()[:, None, None, :]
        mask = mask & key_mask

    min_dtype = torch.finfo(dtype).min

    mask = torch.where(
        mask,
        torch.zeros_like(mask, dtype=dtype),
        torch.full_like(mask, min_dtype, dtype=dtype)
    )

    return mask

def create_sliding_window_causal_mask(
    attention_mask: torch.Tensor | None,
    sliding_window: int,
    dtype: torch.dtype = torch.float16,
):
    """
    Create sliding-window causal mask.

    Example (window=3):

        0 ■ ⬚ ⬚ ⬚ ⬚
        1 ■ ■ ⬚ ⬚ ⬚
        2 ■ ■ ■ ⬚ ⬚
        3 ⬚ ■ ■ ■ ⬚
        4 ⬚ ⬚ ■ ■ ■
    """

    B, L = attention_mask.shape
    device = attention_mask.device

    # =========================
    # indices
    # =========================
    q_idx = torch.arange(L, device=device).view(L, 1)
    kv_idx = torch.arange(L, device=device).view(1, L)

    # =========================
    # causal
    # =========================
    causal = kv_idx <= q_idx

    # =========================
    # sliding window
    # =========================
    sliding = kv_idx > (q_idx - sliding_window)

    mask = causal & sliding

    mask = mask.view(1, 1, L, L)

    # =========================
    # padding mask
    # only mask key side
    # =========================
    if attention_mask is not None:
        key_mask = attention_mask.bool()[:, None, None, :]
        mask = mask & key_mask

    # =========================
    # bool -> float bias
    # =========================
    min_dtype = torch.finfo(dtype).min

    mask = torch.where(
        mask,
        torch.zeros_like(causal, dtype=dtype),
        torch.full_like(causal, min_dtype, dtype=dtype)
    )

    return mask


class LLMBlockWrapper(nn.Module):
    def __init__(self, model, config, embed=None, lm_head=None):
        super().__init__()
        self.model = model
        self.embed = embed
        self.config = config
        self.lm_head = lm_head

    def forward(
        self,
        last_token_id,
        last_hidden,
        attention_mask,
        position_ids,
        full_k,
        full_v,
        slide_k,
        slide_v
    ):
        shared_kv_states = {}
        shared_kv_states["full_attention"] = (full_k, full_v)
        shared_kv_states["sliding_attention"] = (slide_k, slide_v)

        last_token_embedding = self.embed(last_token_id)
        inputs_embeds = torch.cat([last_token_embedding, last_hidden], dim=-1)
        inputs_embeds = self.model.pre_projection(inputs_embeds)
        full_mask = create_bidirectional_mask(attention_mask, dtype=inputs_embeds.dtype)[:, :, -1:, :]
        sliding_mask = create_sliding_window_causal_mask(
            attention_mask, sliding_window=self.config.text_config.sliding_window, dtype=inputs_embeds.dtype
        )[:, :, -1:, :]

        bidirectional_masks = {
            "full_attention": full_mask,
            "sliding_attention": sliding_mask,
        }

        cos_full, sin_full = self.model.model.rotary_emb(inputs_embeds, position_ids, "full_attention")
        cos_slide, sin_slide = self.model.model.rotary_emb(inputs_embeds, position_ids, "sliding_attention")

        position_embeddings = {
            "full_attention": (cos_full, sin_full),
            "sliding_attention": (cos_slide, sin_slide)
        }

        hidden_states = inputs_embeds
        # decoder layers
        for i, decoder_layer in enumerate(self.model.model.layers):

            hidden_states, shared_kv_states = decoder_layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings[self.config.text_config.layer_types[i]],
                attention_mask=bidirectional_masks[self.config.text_config.layer_types[i]],
                shared_kv_states=shared_kv_states,
                use_cache=False,
            )
        hidden_states = self.model.model.norm(hidden_states)
        projected_state = self.model.post_projection(hidden_states)

        logits = self.model.masked_embedding(hidden_states, self.lm_head.weight)

        out_full = shared_kv_states.get("full_attention")
        out_slide = shared_kv_states.get("sliding_attention")

        out_full_k = out_full[0] if out_full is not None else torch.empty((), device=hidden_states.device)
        out_full_v = out_full[1] if out_full is not None else torch.empty((), device=hidden_states.device)
        out_slide_k = out_slide[0] if out_slide is not None else torch.empty((), device=hidden_states.device)
        out_slide_v = out_slide[1] if out_slide is not None else torch.empty((), device=hidden_states.device)

        return projected_state,logits,hidden_states

def export_block(model, config, inputs, path, device, embed=None, lm_head=None):

    last_token_id = torch.zeros(1, 1, dtype=torch.int32, device=device)
    last_hidden = torch.randn(1, 1, config.backbone_hidden_size, dtype=torch.float16, device=device)
    position_ids = torch.zeros(1, 1, dtype=torch.int32, device=device)
    full_k = torch.randn(1, 1, 512, 512, dtype=torch.float16, device=device)
    full_v = torch.randn(1, 1, 512, 512, dtype=torch.float16, device=device)
    slide_k = torch.randn(1, 1, 512, 256, dtype=torch.float16, device=device)
    slide_v = torch.randn(1, 1, 512, 256, dtype=torch.float16, device=device)
    attention_mask = inputs["attention_mask"]
    input_ids = inputs["input_ids"]
    B, L = input_ids.shape
    max_len = 512
    pad_len = max_len - L
    pad_mask = torch.zeros(
        (B, pad_len),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    attention_mask = torch.cat([attention_mask, pad_mask], dim=1).to(torch.int32)

    block_model = LLMBlockWrapper(model, config, embed, lm_head).to(device=device, dtype=torch.float16).eval()

    torch.onnx.export(
        block_model,
        (
            last_token_id, last_hidden, attention_mask,position_ids,full_k,full_v,slide_k,slide_v
        ),
        path,
        input_names=[
            "last_token_id", "last_hidden", "attention_mask","position_ids",
            "full_k","full_v","slide_k","slide_v",
        ],
        output_names=[
            "projected_state", "logits","hidden_states_out"
        ],
        opset_version=11,
        do_constant_folding=False
    )

class Config:
    pass

def get_default_config():
    text_config = Config()
    text_config.attention_bias=False
    text_config.attention_dropout=0.0
    text_config.attention_k_eq_v=False
    text_config.bos_token_id=2
    text_config.chunk_size_feed_forward=0
    text_config.eos_token_id=1
    text_config.final_logit_softcapping=None
    text_config.global_head_dim=512
    text_config.head_dim=256
    text_config.hidden_size=256
    text_config.hidden_size_per_layer_input=0
    text_config.id2label={
    "0": "LABEL_0",
    "1": "LABEL_1"
    }
    text_config.initializer_range=0.02
    text_config.intermediate_size=2048
    text_config.is_encoder_decoder=False
    text_config.label2id={
    "LABEL_0": 0,
    "LABEL_1": 1
    }
    text_config.layer_types=[
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention"
    ]
    text_config.max_position_embeddings=131072
    text_config.moe_intermediate_size=None
    text_config.num_attention_heads=4
    text_config.num_global_key_value_heads=None
    text_config.num_hidden_layers=4
    text_config.num_key_value_heads=1
    text_config.num_kv_shared_layers=4
    text_config.output_attentions=False
    text_config.output_hidden_states=False
    text_config.pad_token_id=0
    text_config.rms_norm_eps=1e-06
    text_config.rope_parameters={
    "full_attention": {
        "partial_rotary_factor": 0.25,
        "rope_theta": 1000000.0,
        "rope_type": "proportional"
    },
    "sliding_attention": {
        "rope_theta": 10000.0,
        "rope_type": "default"
    }
    }
    text_config.sliding_window=512
    text_config.tie_word_embeddings=True
    text_config.use_double_wide_mlp=False
    text_config.use_bidirectional_attention = None
    text_config.vocab_size=262144
    text_config.vocab_size_per_layer_input=0
    config = Config()
    config.backbone_hidden_size=1536
    config.boa_token_id=256000
    config.boi_token_id=255999
    config.centroid_intermediate_top_k=32
    config.eoa_token_id=258883
    config.eoi_token_id=258882
    config.image_token_id=258880
    config.num_centroids=2048
    config.text_config=text_config
    config.tie_word_embeddings=True
    config.use_ordered_embeddings=True
    return config

def main():
    print("Loading model ...")
    DEVICE = 'cpu'
    MODEL_PATH = "./gemma-4-E2B-it-assistant"
    target_model = AutoModelForCausalLM.from_pretrained(
        "./gemma-4-E2B-it",
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    ).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained("./gemma-4-E2B-it")
    image = Image.open('/e-vepfs-01/perception/wuhui/InternVL3_5-1B/InternVL3_5-1B-HF/examples/image1.jpg').convert("RGB").resize((768, 768))

    # Prompt - add image before text
    messages = [
        {
            "role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What is shown in this image?"}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(DEVICE).to(torch.float16)

    config = get_default_config()

    assistant_model = Gemma4AssistantForCausalLM(config)
    assistant_model.load_from_pretrained(model)
    assistant_model = assistant_model.to(device=DEVICE, dtype=torch.float16)
    assistant_model.eval()

    print("Exporting language modules...")
    export_block(
        assistant_model,
        config,
        inputs,
        path="/e-vepfs-01/ppdc/guanxj/ENetQuery/work_dirs/gemma4/onnx_export/assistant.onnx",
        device=DEVICE,
        embed=target_model.model.language_model.embed_tokens.to(device=DEVICE, dtype=torch.float16),
        lm_head=model.lm_head.to(device=DEVICE, dtype=torch.float16),
    )

    print("\n✅ All export done!")

if __name__ == "__main__":
    main()

