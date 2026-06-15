import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration,AutoProcessor
from PIL import Image
import numpy as np
import functools
from typing import Any, Callable

from export_config import ExportProfile, get_export_profile

class Config:
        pass

PROFILE = get_export_profile()
EXPORT_DIR = PROFILE.export_dir
DEFAULT_MODEL_PATH = "./Qwen3-VL-2B-Instruct"
MAX_SEQ_LEN = PROFILE.max_seq_len
FLOAT_DTYPE = torch.float16
INT_DTYPE = torch.int32


def apply_export_profile(profile: ExportProfile) -> None:
    global PROFILE, EXPORT_DIR, MAX_SEQ_LEN
    PROFILE = profile
    EXPORT_DIR = profile.export_dir
    MAX_SEQ_LEN = profile.max_seq_len

QWEN3_TEXT_CONFIG = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "pad_token_id": 151643,
    "dtype": "float16",
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 6144,
    "max_position_embeddings": 262144,
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
        "mrope_interleaved": True,
        "mrope_section": [24, 20, 20],
    },
    "rope_theta": 5000000,
    "tie_word_embeddings": True,
    "vocab_size": 151936,
}

def build_real_inputs(processor, image_path, device, image_size: int | None = None):
    size = image_size if image_size is not None else PROFILE.image_size
    image = Image.open(image_path).convert("RGB").resize((size, size))

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What's the main object in this picture?"}
        ],
    }]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        max_length=MAX_SEQ_LEN,
        padding="max_length",
    ).to(device)

    inputs["input_ids"] = inputs["input_ids"].to(INT_DTYPE)
    inputs["attention_mask"] = inputs["attention_mask"].to(INT_DTYPE)
    if "image_grid_thw" in inputs:
        inputs["image_grid_thw"] = inputs["image_grid_thw"].to(INT_DTYPE)

    return inputs

# def create_causal_mask(attention_mask, dtype=None):
#     attention_mask = attention_mask.to(torch.float32)
#     device = attention_mask.device
#     B, L = attention_mask.shape

#     # causal（bool）
#     causal = torch.arange(L, device=device)
#     causal = causal[None, :] <= causal[:, None]   # [L, L]

#     # key padding
#     key_pad = attention_mask[:, None, None, :] == 1

#     # 合并
#     full_mask = causal[None, None, :, :] & key_pad

#     # 转 float
#     min_val = -1e9
#     full_mask = torch.where(
#         full_mask,
#         torch.zeros((), dtype=dtype, device=device),
#         torch.full((), min_val, dtype=dtype, device=device)
#     )

#     return full_mask

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
        self.act_fn = SiLUActivation()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

def compute_default_rope_parameters(
        device = None,
        seq_len = None,
    ):
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
        base = 5000000
        dim = 128

        attention_factor = 1.0

        # inv_freq must be computed in fp32: base=5e6 overflows in fp16
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        return inv_freq.to(FLOAT_DTYPE), attention_factor

class Qwen3VLTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, device=None):
        super().__init__()
        self.rope_type = "default"
        rope_init_fn: Callable = compute_default_rope_parameters
        inv_freq, self.attention_scaling = rope_init_fn(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

        self.mrope_section = [24, 20, 20]


    # def apply_interleaved_mrope(self, freqs, mrope_section):
    #     """Apply interleaved MRoPE to 3D rotary embeddings.
    #     Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
    #     interleaved [THTHWHTHW...TT], preserving frequency continuity.
    #     args:
    #         x: (3, bs, seq_len, head_dim // 2)
    #         mrope_section: (3,)
    #     returns:
    #         x_t: (bs, seq_len, head_dim // 2)
    #     """
    #     freqs_t = freqs[0]  # just overwrite the first dimension T
    #     for dim, offset in enumerate((1, 2), start=1):  # H, W
    #         length = mrope_section[dim] * 3
    #         idx = slice(offset, length, 3)
    #         freqs_t[..., idx] = freqs[dim, ..., idx]
    #     return freqs_t

    def apply_interleaved_mrope(self, freqs):
        t = freqs[0]
        h = freqs[1]
        w = freqs[2]  # [B,L,D]

        out = torch.empty_like(t)

        d = 60

        out[..., 0:d:3] = t[..., 0:d:3]
        out[..., 1:d:3] = h[..., 1:d:3]
        out[..., 2:d:3] = w[..., 2:d:3]

        out[..., d:] = t[..., d:]

        return out


    # def forward(self, x, position_ids):
    #     # In contrast to other models, Qwen3VL has different position ids for the grids
    #     # So we expand the inv_freq to shape (3, ...)
    #     inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
    #     position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

    #     device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    #     #with torch.autocast(device_type=device_type, enabled=False):  # Force float32
    #         freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3) #[3,1,512,64]
    #         freqs = self.apply_interleaved_mrope(freqs, self.mrope_section) #[1,512,64]
    #         emb = torch.cat((freqs, freqs), dim=-1) #[1,512,128]
    #         cos = emb.cos() * self.attention_scaling
    #         sin = emb.sin() * self.attention_scaling

    #     return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


    def forward(self, x, position_ids):
        # position_ids: [3, B, L]

        inv_freq = self.inv_freq.to(x.dtype)  # [D/2]

        # ===== Step1：计算 freqs=====
        #freqs = position_ids[:, :, :, None] * inv_freq[None, None, None, :]
        inv_freq_expanded = self.inv_freq[None, None, :, None].to(x.dtype)  # [1,1,D/2,1]
        position_ids_expanded = position_ids[:, :, None, :].to(x.dtype)   # [3,B,1,L]

        freqs = torch.matmul(inv_freq_expanded, position_ids_expanded)    # [3,B,D/2,L]
        freqs = freqs.transpose(2, 3)                                     # [3,B,L,D/2]
        # → [3, B, L, D/2]

        # ===== Step2：MRoPE 融合=====
        freqs = self.apply_interleaved_mrope(freqs)
        # → [B, L, D/2]

        # ===== Step3：cos / sin=====
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = torch.cos(emb) * self.attention_scaling
        sin = torch.sin(emb) * self.attention_scaling

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
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # 纯 fp16 attention（与 gemma4 一致；OM 只能 fp16）
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask.to(dtype=query.dtype)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
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
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        # Qwen3 官方只有 q_norm/k_norm，无 v_norm（gemma4 有 v_norm 但权重结构不同，不能加）
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
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

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            sliding_window=None,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

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


class Qwen3VLTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding()
        self.gradient_checkpointing = False
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def load_from_pretrained(self, model):
        missing, unexpected = self.load_state_dict(
            model.state_dict(),
            strict=False
        )
        print("✅ TextModel 权重加载完成")
        print("missing keys:", missing)
        print("unexpected keys:", unexpected)
    
    def get_rope_index(
        self,
        input_ids = None,
        image_grid_thw = None,
        attention_mask = None,
    ):
        """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

        # Since we use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
        spatial_merge_size = 2
        image_token_id = 151655
        video_token_id = 151656
        vision_start_token_id = 151652
        mrope_position_deltas = []
        total_input_ids = input_ids[0]
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=INT_DTYPE,
            device=input_ids.device,
        )
        attention_mask = attention_mask[0].to(total_input_ids.device)
        #input_ids = input_ids[attention_mask[0] == 1]
        input_ids = input_ids[0]
        vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
        vision_tokens = input_ids[vision_start_indices + 1]
        #input_tokens = input_ids.tolist()
        llm_pos_ids_list: list = []
        st = 0
        ed_image = PROFILE.image_prefix_len
        ed = ed_image
        llm_grid_t = 1
        llm_grid_h = PROFILE.merged_grid
        llm_grid_w = PROFILE.merged_grid
        text_len = ed - st

        st_idx = 0
        llm_pos_ids_list.append(
            torch.arange(text_len, dtype=INT_DTYPE, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx
        )

        # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
        t_index = torch.arange(llm_grid_t, dtype=INT_DTYPE, device=input_ids.device).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
        h_index = torch.arange(llm_grid_h, dtype=INT_DTYPE, device=input_ids.device).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
        w_index = torch.arange(llm_grid_w, dtype=INT_DTYPE, device=input_ids.device).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
        st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
        text_len = MAX_SEQ_LEN - st
        llm_pos_ids_list.append(
            torch.arange(text_len, dtype=INT_DTYPE, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx
        )

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)

        position_ids = llm_positions.unsqueeze(1).to(INT_DTYPE)
        # position_ids[:, 0, :] = torch.where(
        #     mask == 1,
        #     llm_positions,
        #     position_ids[:, 0, :]
        # )
        #mask = attention_mask.expand(3, -1)
        #position_ids[..., 0, attention_mask[0] == 1] = llm_positions.to(position_ids.device)
        # mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        # mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        #return position_ids, mrope_position_deltas
        return position_ids

    # def _deepstack_process(
    #     self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    # ):
    #     visual_pos_masks = visual_pos_masks.to(hidden_states.device)
    #     visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
    #     hidden_states = hidden_states.clone()
    #     local_this = hidden_states[visual_pos_masks, :] + visual_embeds
    #     hidden_states[visual_pos_masks, :] = local_this
    #     return hidden_states

    def _deepstack_process(self, hidden_states, visual_embeds):
        # hidden_states: [B, L, C]
        # visual_embeds: [196, C]

        B, L, C = hidden_states.shape

        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)

        # 扩展 batch
        visual_embeds = visual_embeds.unsqueeze(0).expand(B, -1, -1)  # [B,196,C]

        # 固定
        hidden_states[:, PROFILE.image_token_start:PROFILE.image_token_end, :] += visual_embeds

        return hidden_states

    def forward(
        self,
        input_ids: torch.LongTensor | None = None, #[1,256]
        attention_mask: torch.Tensor | None = None, #[1,256]
        position_ids: torch.LongTensor | None = None,#[3,1,256]
        past_key_values = None,
        inputs_embeds: torch.FloatTensor | None = None, #[1,256,2048]
        use_cache: bool | None = None,
        # args for deepstack
        #visual_pos_masks: torch.Tensor | None = None, #[1,256]
        ds_0 = None,
        ds_1 = None,
        ds_2 = None, # 3*[1,256,2048]
        **kwargs,
    ):
        r"""
        visual_pos_masks (`torch.Tensor` of shape `(batch_size, seqlen)`, *optional*):
            The mask of the visual positions.
        deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):
            The deepstack visual embeddings. The shape is (num_layers, visual_seqlen, embed_dim).
            The feature is extracted from the different visual encoder layers, and fed to the decoder
            hidden states. It's from the paper DeepStack(https://arxiv.org/abs/2406.04334).
        """
        # if (input_ids is None) ^ (inputs_embeds is not None):
        #     raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # # # torch.jit.trace() doesn't support cache objects in the output
        # # if use_cache and past_key_values is None and not torch.jit.is_tracing():
        # #     past_key_values = DynamicCache(config=self.config)

        # if inputs_embeds is None:
        #     inputs_embeds = self.embed_tokens(input_ids)

        # the hard coded `4` is for text, temporal, height and width.
        # if position_ids is None:
        #     past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        #     position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        #     position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        # elif position_ids.ndim == 2:
        #     position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        # if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        #     text_position_ids = position_ids[0]
        #     position_ids = position_ids[1:]
        # else:
        #     text_position_ids = None
        deepstack_visual_embeds = [ds_0, ds_1, ds_2]
        text_position_ids = position_ids[0]
        # attention_mask = create_causal_mask(
        #     config=self.config,
        #     inputs_embeds=inputs_embeds,
        #     attention_mask=attention_mask,
        #     past_key_values=past_key_values,
        #     position_ids=text_position_ids,
        # )

        attention_mask = create_causal_mask(attention_mask,inputs_embeds.dtype)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids) #(2,)

        # decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs

            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    #visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)

        return hidden_states


def build_qwen3_config() -> Config:
    config = Config()
    for key, value in QWEN3_TEXT_CONFIG.items():
        setattr(config, key, value)
    return config


def load_hf_qwen3_vl(model_path: str | None = None):
    path = model_path or DEFAULT_MODEL_PATH
    return Qwen3VLForConditionalGeneration.from_pretrained(
        path,
        torch_dtype=FLOAT_DTYPE,
        attn_implementation="eager",
        device_map=None,
    ).eval()


def load_text_model(
    *,
    model_path: str | None = None,
    device: str = "cpu",
    language_model=None,
) -> Qwen3VLTextModel:
    """Build Qwen3VLTextModel and load weights from an HF language_model submodule."""
    if language_model is None:
        language_model = load_hf_qwen3_vl(model_path).model.language_model

    text_model = Qwen3VLTextModel(build_qwen3_config())
    text_model.load_from_pretrained(language_model)
    return text_model.to(device=device, dtype=FLOAT_DTYPE).eval()


def export_llm(model, path=None):
    if path is None:
        path = os.path.join(EXPORT_DIR, "llm.onnx")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    module = load_text_model(language_model=model, device="cpu")

    B, L, C = 1, MAX_SEQ_LEN, 2048

    # ===== dummy inputs =====
    inputs_embeds = torch.randn(B, L, C, dtype=FLOAT_DTYPE)
    attention_mask = torch.ones(B, L, dtype=INT_DTYPE)

    position_ids = torch.arange(L, dtype=INT_DTYPE).unsqueeze(0).unsqueeze(0)
    position_ids = position_ids.repeat(3, B, 1)  # [3, B, L]

    ds_0 = torch.randn(196, C, dtype=FLOAT_DTYPE)
    ds_1 = torch.randn(196, C, dtype=FLOAT_DTYPE)
    ds_2 = torch.randn(196, C, dtype=FLOAT_DTYPE)

    print("\n模型输入：")
    print("inputs_embeds:", inputs_embeds.shape)
    print("attention_mask:", attention_mask.shape)
    print("position_ids:", position_ids.shape)


    torch.onnx.export(
        module,
        (None,attention_mask,position_ids,None,inputs_embeds,None,ds_0,ds_1,ds_2),
        path,
        input_names=[
            "attention_mask",
            "position_ids",
            "inputs_embeds",
            #"visual_mask",
            "ds_0",
            "ds_1",
            "ds_2"
        ],
        output_names=["last_hidden_state"],
        opset_version=11,
    )

    print(f"✅ ONNX 导出完成: {path}")


class LLMPreBlockWrapper(nn.Module):
    def __init__(self, model, profile: ExportProfile):
        super().__init__()
        self.profile = profile
        self.embed = model.embed_tokens
        self.rotary_emb = model.rotary_emb

    def forward(self, input_ids, image_embeds, attention_mask, position_ids):
        # embedding
        inputs_embeds = self.embed(input_ids)

        # 拼接
        prefix = inputs_embeds[:, :self.profile.image_token_start, :]
        suffix = inputs_embeds[:, self.profile.image_token_end:, :]

        image_embeds = image_embeds.unsqueeze(0)

        inputs_embeds = torch.cat([
            prefix,
            image_embeds,
            suffix
        ], dim=1)

        # position_ids 外部输入，避免 get_rope_index / rotary 被 OM 常量折叠
        position_ids = position_ids.to(
            device=inputs_embeds.device, dtype=INT_DTYPE
        )
        attention_mask = create_causal_mask(attention_mask, inputs_embeds.dtype)
        cos, sin = self.rotary_emb(inputs_embeds, position_ids)

        return inputs_embeds, attention_mask, cos, sin


def compute_static_position_ids(input_ids, attention_mask, device="cpu"):
    """固定 image_size + MAX_SEQ_LEN 布局下的 mrope position_ids [3,1,L]。"""
    return Qwen3VLTextModel.get_rope_index(
        None,
        input_ids=input_ids.to(device=device, dtype=INT_DTYPE),
        image_grid_thw=None,
        attention_mask=attention_mask.to(device=device, dtype=INT_DTYPE),
    )


def export_preblock(model, inputs, path, device, profile: ExportProfile):

    wrapper = LLMPreBlockWrapper(model, profile).to(device=device, dtype=FLOAT_DTYPE).eval()

    input_ids = inputs["input_ids"].to(INT_DTYPE).to(device)
    attention_mask = inputs["attention_mask"].to(INT_DTYPE).to(device)

    image_embeds = torch.randn(
        profile.num_image_tokens, 2048, dtype=FLOAT_DTYPE, device=device
    )
    position_ids = compute_static_position_ids(input_ids, attention_mask, device)

    torch.onnx.export(
        wrapper,
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
            "sin",
        ],
        opset_version=11,
    )

    print(f"exported fp16 ONNX -> {path}")
    print(f"  position_ids shape: {tuple(position_ids.shape)} (static dump for OM)")


class LLMHeadWrapper(nn.Module):
    """单 token lm_head，静态 [1,1,2048] -> [1,1,vocab]，decode 时切 last hidden[:, cur_len-1]。"""

    def __init__(self, lm_head):
        super().__init__()
        self.lm_head = lm_head

    def forward(self, hidden_states):
        return self.lm_head(hidden_states)


def export_lm_head(lm_head, path, device, hidden_size=2048):
    wrapper = LLMHeadWrapper(lm_head).to(device=device, dtype=FLOAT_DTYPE).eval()
    dummy = torch.randn(1, 1, hidden_size, dtype=FLOAT_DTYPE, device=device)
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
    def __init__(
        self,
        layers,
        start_idx,
        deepstack_map,
        profile: ExportProfile,
        norm=None,
        lm_head=None,
    ):
        super().__init__()
        self.layers = layers
        self.start_idx = start_idx
        self.deepstack_map = deepstack_map
        self.profile = profile
        self.norm = norm
        self.lm_head = lm_head

    def forward(
        self,
        hidden_states,
        attention_mask,
        cos,
        sin,
        ds_0,
        ds_1,
        ds_2,
    ):
        ds_embeds = [ds_0, ds_1, ds_2]

        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=(cos,sin),
            )

            # deepstack
            layer_idx = self.start_idx + i
            if layer_idx in self.deepstack_map:
                ds_id = self.deepstack_map[layer_idx]
                visual = ds_embeds[ds_id]

                B, _, C = hidden_states.shape
                visual = visual.unsqueeze(0).expand(B, -1, -1)

                hidden_states[
                    :, self.profile.image_token_start:self.profile.image_token_end, :
                ] += visual

        # 只在最后 block 执行
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)   # [B, L, V]
            return hidden_states, logits

        return hidden_states

def export_block(
    layers,
    start_idx,
    path,
    inputs,
    device,
    profile: ExportProfile,
    norm=None,
    lm_head=None,
):

    deepstack_map = {5:0,11:1,17:2}
    model = LLMBlockWrapper(
        layers, start_idx, deepstack_map, profile, norm, lm_head
    ).to(device=device, dtype=FLOAT_DTYPE).eval()

    input_ids = inputs["input_ids"]
    attention_mask_raw = inputs["attention_mask"]

    B = input_ids.shape[0]
    L = MAX_SEQ_LEN
    C = 2048

    hidden_states = torch.randn(B, L, C, dtype=FLOAT_DTYPE, device=device)

    attention_mask = create_causal_mask(
        attention_mask_raw,
        hidden_states.dtype
    )

    cos = torch.randn(B, L, 128, dtype=FLOAT_DTYPE, device=device)
    sin = torch.randn(B, L, 128, dtype=FLOAT_DTYPE, device=device)

    ds_0 = torch.randn(profile.num_image_tokens, C, dtype=FLOAT_DTYPE, device=device)
    ds_1 = torch.randn(profile.num_image_tokens, C, dtype=FLOAT_DTYPE, device=device)
    ds_2 = torch.randn(profile.num_image_tokens, C, dtype=FLOAT_DTYPE, device=device)

    output_names = (
        ["hidden_states_out", "logits"]
        if lm_head is not None else ["hidden_states_out"]
    )

    torch.onnx.export(
        model,
        (hidden_states, attention_mask, cos, sin, ds_0, ds_1, ds_2),
        path,
        input_names=[
            "hidden_states",
            "attention_mask",
            "cos",
            "sin",
            "ds_0",
            "ds_1",
            "ds_2",
        ],
        output_names=output_names,
        opset_version=11,
    )

    print(f"exported fp16 ONNX -> {path}")


def export_allchunk(module, model, inputs, device, profile: ExportProfile):
    os.makedirs(profile.export_dir, exist_ok=True)

    export_preblock(
        module,
        inputs,
        os.path.join(profile.export_dir, "llm_preblock.onnx"),
        device,
        profile,
    )

    export_block(
        module.layers[:10], 0,
        os.path.join(profile.export_dir, "llm_block1.onnx"),
        inputs, device, profile,
    )

    export_block(
        module.layers[10:20], 10,
        os.path.join(profile.export_dir, "llm_block2.onnx"),
        inputs, device, profile,
    )

    export_block(
        module.layers[20:], 20,
        os.path.join(profile.export_dir, "llm_block3.onnx"),
        inputs, device, profile,
        norm=module.norm,
        lm_head=None,
    )

    export_lm_head(
        model.lm_head,
        os.path.join(profile.export_dir, "lm_head.onnx"),
        device,
        hidden_size=2048,
    )

def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-VL LLM ONNX chunks")
    parser.add_argument(
        "--profile",
        choices=("256_256", "448_512"),
        default=os.environ.get("QWEN3_EXPORT_PROFILE", "256_256"),
        help="export layout profile (default: QWEN3_EXPORT_PROFILE or 256_256)",
    )
    parser.add_argument(
        "--image",
        default="/e-vepfs-01/perception/wuhui/InternVL3_5-1B/InternVL3_5-1B-HF/examples/image1.jpg",
        help="calibration image for preblock export",
    )
    args = parser.parse_args()
    profile = get_export_profile(args.profile)
    apply_export_profile(profile)

    print(f"profile: {profile.name}")
    print(f"image_size: {profile.image_size}")
    print(f"max_seq_len: {profile.max_seq_len}")
    print(f"num_image_tokens: {profile.num_image_tokens}")
    print(f"export_dir: {profile.export_dir}")
    print("Loading model ...")

    DEVICE = "cpu"
    processor = AutoProcessor.from_pretrained(DEFAULT_MODEL_PATH)
    model = load_hf_qwen3_vl(DEFAULT_MODEL_PATH)
    inputs = build_real_inputs(processor, args.image, DEVICE, profile.image_size)
    text_model = load_text_model(language_model=model.model.language_model, device=DEVICE)

    print("Exporting fp16 language modules...")
    export_allchunk(text_model, model, inputs, DEVICE, profile)

    print("\n✅ All export done!")

if __name__ == "__main__":
    main()
