import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, InternVLForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np
import functools
from typing import Any

FLOAT_DTYPE = torch.float16
INT_DTYPE = torch.int32
EXPORT_DIR = "./onnx_export"

class Config:
    pass


def create_causal_mask(attention_mask, dtype=FLOAT_DTYPE):
    B, L = attention_mask.shape
    device = attention_mask.device

    # causal
    causal = torch.arange(L, device=device)
    causal = causal[None, :] <= causal[:, None]  # [L, L]

    # key padding mask（与 HF create_causal_mask 一致；不用 query_mask，
    # 否则 pad 行全 -inf → softmax 均匀 → pad hidden 虚高）
    key_mask = attention_mask[:, None, None, :] == 1  # [B,1,1,L]

    full_mask = causal[None, None, :, :] & key_mask

    min_val = torch.finfo(dtype).min
    zero = torch.zeros((), dtype=dtype, device=device)
    full_mask = torch.where(full_mask, zero, min_val)

    return full_mask

class SiLUActivation(nn.Module):
    """
    See Gaussian Error Linear Units (Hendrycks et al., https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid Linear
    Unit) was originally introduced and coined, and see Sigmoid-Weighted Linear Units for Neural Network Function
    Approximation in Reinforcement Learning (Elfwing et al., https://arxiv.org/abs/1702.03118) and Swish: a Self-Gated
    Activation Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1) where the SiLU was experimented with
    later.
    """

    def forward(self, input):
        return nn.functional.silu(input)


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        down_proj = self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen3RotaryEmbedding(nn.Module):

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

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


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

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
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
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


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
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=FLOAT_DTYPE))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 纯 fp16 RMSNorm；先 /amax 再平方，避免 x² 在 fp16 上溢（OM 无 fp32 Cast）
        # x * rsqrt(mean(x²)) == (x/amax) * rsqrt(mean((x/amax)²))
        x = hidden_states
        eps = torch.full((), self.variance_epsilon, dtype=x.dtype, device=x.device)
        # amax + eps 代替 clamp(min=eps)，避免 ONNX Clip/ClipByValue（Ascend OM 不支持）
        amax = x.abs().amax(dim=-1, keepdim=True) + eps
        xs = x / amax
        mean_squared = xs.pow(2).mean(-1, keepdim=True) + eps
        x = xs * torch.pow(mean_squared, -0.5)
        x = x * self.weight
        return x

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding_idx = 151643
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

    def load_from_pretrained(self, model):
        missing, unexpected = self.load_state_dict(
            model.state_dict(),
            strict=False
        )
        print("✅ TextModel 权重加载完成")
        print("missing keys:", missing)
        print("unexpected keys:", unexpected)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class LLMPreBlockWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embed = model.embed_tokens
        self.rotary_emb = model.rotary_emb
        self.model = model


    def forward(self, input_ids, image_embeds, attention_mask, position_ids):
        # embedding
        inputs_embeds = self.embed(input_ids)
        prefix = inputs_embeds[:, :4, :]
        suffix = inputs_embeds[:, 260:, :]

        inputs_embeds = torch.cat([
            prefix,
            image_embeds,
            suffix
        ], dim=1)

        attention_mask = create_causal_mask(attention_mask, inputs_embeds.dtype)
        cos, sin = self.rotary_emb(inputs_embeds, position_ids)

        return inputs_embeds, attention_mask, cos, sin

def export_preblock(model, inputs, path, device):

    model = LLMPreBlockWrapper(model).to(device=device, dtype=FLOAT_DTYPE).eval()

    input_ids = inputs["input_ids"].to(INT_DTYPE).to(device)
    attention_mask = inputs["attention_mask"].to(INT_DTYPE).to(device)
    image_embeds = torch.randn(1, 256, 1024, dtype=FLOAT_DTYPE, device=device)
    position_ids = torch.arange(512, device=device, dtype=INT_DTYPE).unsqueeze(0)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        (input_ids, image_embeds, attention_mask, position_ids),
        path,
        input_names=[
            "input_ids",
            "image_embeds",
            "attention_mask",
            "position_ids",
        ],
        output_names=[
            "inputs_embeds_out",
            "attention_mask_out",
            "cos",
            "sin"
        ],
        opset_version=11,
    )
    print(f"exported fp16 ONNX -> {path}")


class LLMHeadWrapper(nn.Module):
    """单 token lm_head，静态 [1,1,1024] -> [1,1,vocab]，decode 时切 last hidden[:, cur_len-1]。"""

    def __init__(self, lm_head):
        super().__init__()
        self.lm_head = lm_head

    def forward(self, hidden_states):
        return self.lm_head(hidden_states)


def export_lm_head(lm_head, path, device, hidden_size=1024):
    wrapper = LLMHeadWrapper(lm_head).to(device=device, dtype=FLOAT_DTYPE).eval()
    dummy = torch.randn(1, 1, hidden_size, dtype=FLOAT_DTYPE, device=device)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy,),
        path,
        input_names=["hidden_states"],
        output_names=["logits"],
        opset_version=11,
    )
    print(f"exported fp16 ONNX -> {path}")


class LLMBlockWrapper(nn.Module):
    def __init__(self, layers, start_idx, norm=None, lm_head=None):
        super().__init__()
        self.layers = layers
        self.start_idx = start_idx
        self.norm = norm
        self.lm_head = lm_head

    def forward(
        self,
        hidden_states,
        attention_mask,
        cos,
        sin
    ):

        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=(cos,sin),
            )

        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)   # [B, L, V]
            return hidden_states, logits

        return hidden_states

def export_block(layers, start_idx, path, inputs, device, norm=None, lm_head=None):

    model = LLMBlockWrapper(layers, start_idx, norm, lm_head).to(
        device=device, dtype=FLOAT_DTYPE
    ).eval()

    input_ids = inputs["input_ids"]
    attention_mask_raw = inputs["attention_mask"]

    B, L = input_ids.shape
    C = 1024

    hidden_states = torch.randn(B, L, C, dtype=FLOAT_DTYPE, device=device)

    attention_mask = create_causal_mask(
        attention_mask_raw,
        hidden_states.dtype
    )

    cos = torch.randn(B, L, 128, dtype=FLOAT_DTYPE, device=device)
    sin = torch.randn(B, L, 128, dtype=FLOAT_DTYPE, device=device)

    output_names = (
        ["hidden_states_out", "logits"]
        if lm_head is not None else ["hidden_states_out"]
    )

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        (hidden_states, attention_mask, cos, sin),
        path,
        input_names=[
            "hidden_states",
            "attention_mask",
            "cos",
            "sin"
        ],
        output_names=output_names,
        opset_version=11,
    )
    print(f"exported fp16 ONNX -> {path}")


def export_allchunk(module, model, inputs, device):
    os.makedirs(EXPORT_DIR, exist_ok=True)

    export_preblock(
        module,
        inputs,
        os.path.join(EXPORT_DIR, "llm_preblock.onnx"),
        device
    )

    export_block(
        module.layers[:10], 0,
        os.path.join(EXPORT_DIR, "llm_block1.onnx"),
        inputs, device
    )

    export_block(
        module.layers[10:20], 10,
        os.path.join(EXPORT_DIR, "llm_block2.onnx"),
        inputs, device
    )

    export_block(
        module.layers[20:], 20,
        os.path.join(EXPORT_DIR, "llm_block3.onnx"),
        inputs, device,
        norm=module.norm,
    )

    export_lm_head(
        model.lm_head,
        os.path.join(EXPORT_DIR, "lm_head.onnx"),
        device,
        hidden_size=1024,
    )

def pad(input_ids, attention_mask, pad_id, max_len=512):
    """
    input_ids: [B, L]
    attention_mask: [B, L]
    """
    B, L = input_ids.shape

    if L > max_len:
        raise ValueError(f"seq_len {L} > {max_len}")

    pad_len = max_len - L

    if pad_len > 0:
        pad_ids = torch.full(
            (B, pad_len),
            pad_id,
            dtype=input_ids.dtype,
            device=input_ids.device
        )

        pad_mask = torch.zeros(
            (B, pad_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )

        input_ids = torch.cat([input_ids, pad_ids], dim=1)
        attention_mask = torch.cat([attention_mask, pad_mask], dim=1)

    return input_ids, attention_mask

def main():
    print("Loading model ...")
    MODEL_PATH = "./InternVL3_5-1B-HF"
    IMAGE_PATH = "./InternVL3_5-1B-HF/examples/image1.jpg"
    DEVICE = 'cpu'
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    processor.image_processor.min_patches = 1
    processor.image_processor.max_patches = 1
    model = InternVLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=FLOAT_DTYPE,
        attn_implementation="eager",
        device_map=None
    ).eval()
    image = Image.open('./InternVL3_5-1B-HF/examples/image1.jpg').convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Please describe the image shortly."}
            ]
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors='pt'
    ).to(DEVICE)

    input_ids = inputs["input_ids"]      # [B, L]
    attention_mask = inputs["attention_mask"]
    input_ids, attention_mask = pad(
        input_ids,
        attention_mask,
        pad_id=151643,
        max_len=512
    )
    inputs["input_ids"] = input_ids.to(DEVICE, INT_DTYPE)
    inputs["attention_mask"] = attention_mask.to(DEVICE, INT_DTYPE)

    config = Config()
    config.attention_bias = False
    config.attention_dropout = 0.0
    config.bos_token_id = 151643
    config.eos_token_id = 151645
    config.ep_size = 1
    config.head_dim = 128
    config.hidden_size = 1024
    config.initializer_range = 0.02
    config.intermediate_size = 3072
    config.max_position_embeddings = 40960
    config.max_window_layers = 28
    config.micro_forward = False
    config.num_attention_heads = 16
    config.num_hidden_layers = 28
    config.num_key_value_heads = 8
    config.rms_norm_eps = 1e-06
    config.rope_scaling = None
    config.rope_theta = 1000000
    config.vocab_size = 151936
    config.dtype = "float16"

    text_model = Qwen3Model(config)
    text_model.load_from_pretrained(model.model.language_model)
    text_model = text_model.to(device=DEVICE, dtype=FLOAT_DTYPE)
    text_model.eval()

    print("Exporting fp16 language modules...")
    #export_llm(model.model.language_model)
    export_allchunk(
        text_model,
        model,
        inputs,
        DEVICE
    )

    print("\n✅ All export done!")

if __name__ == "__main__":
    main()
