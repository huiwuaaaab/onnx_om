import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration,AutoProcessor
from PIL import Image
import numpy as np
import functools
import inspect

from export_config import ExportProfile, get_export_profile

MODEL_PATH = "./Qwen3-VL-2B-Instruct"
PROFILE = get_export_profile()
EXPORT_DIR = PROFILE.export_dir
DEVICE = "cpu"
OPSET = 11
FLOAT_DTYPE = torch.float16
INT_DTYPE = torch.int32

class Config:
        pass

class Qwen3VLVisionLayerNorm(nn.Module):
    """纯 fp16 LayerNorm，导出 ONNX/OM 全程 fp16（AOE CastRemove 后与 NPU 实际路径一致）。"""

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape, dtype=FLOAT_DTYPE))
        self.bias = nn.Parameter(torch.zeros(normalized_shape, dtype=FLOAT_DTYPE))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 纯 fp16 LayerNorm；x/amax 后再减均值、平方，避免 fp16 上溢（OM 无 fp32 Cast）
        x = hidden_states.to(FLOAT_DTYPE)
        eps = torch.full((), self.eps, dtype=x.dtype, device=x.device)
        # amax + eps 代替 clamp(min=eps)，避免 ONNX Clip/ClipByValue（Ascend OM 不支持）
        amax = x.abs().amax(dim=-1, keepdim=True) + eps
        xs = x / amax
        mean = xs.mean(dim=-1, keepdim=True)
        xm = xs - mean
        var = xm.pow(2).mean(dim=-1, keepdim=True) + eps
        x = xm * torch.pow(var, -0.5)
        x = x * self.weight + self.bias
        return x

class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)
    
    def load_from_pretrained(self, pretrained_block):
        self.proj.weight.data.copy_(pretrained_block.proj.weight.data)
        self.proj.bias.data.copy_(pretrained_block.proj.bias.data)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        ) #[784,3,2,16,16]
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states

def export_patch_embed(model, path=None):
    if path is None:
        path = os.path.join(EXPORT_DIR, "patch_embed.onnx")
    config = Config()
    config.hidden_size = 1024
    config.in_channels = 3
    config.patch_size = 16
    config.temporal_patch_size = 2

    module = Qwen3VLVisionPatchEmbed(config)
    module.load_from_pretrained(model.patch_embed)
    module.eval()

    dummy = torch.randn(1, 3, 448, 448, dtype=FLOAT_DTYPE)

    torch.onnx.export(
        module,
        (dummy,),
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
    )

# class Qwen3VLVisionRotaryEmbedding(nn.Module):
#     inv_freq: torch.Tensor  # fix linting for `register_buffer`

#     def __init__(self, dim: int, theta: float = 10000.0) -> None:
#         super().__init__()
#         self.dim = dim
#         self.theta = theta
#         inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
#         self.register_buffer("inv_freq", inv_freq, persistent=False)

#     def forward(self, seqlen: int) -> torch.Tensor:
#         seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
#         #freqs = torch.outer(seq, self.inv_freq)
#         freqs = seq[:, None] * self.inv_freq[None, :]
#         return freqs

class Qwen3VLVisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

        index = torch.arange(0, dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (index / dim))
        self.register_buffer("inv_freq", inv_freq.to(FLOAT_DTYPE), persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.ones((seqlen, 1), dtype=FLOAT_DTYPE, device=self.inv_freq.device).cumsum(dim=0)
        seq = seq - 1.0
        seq = seq.squeeze(-1)

        freqs = seq[:, None] * self.inv_freq[None, :]
        return freqs

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


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)

class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, config, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = 64
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        #self.scaling = self.head_dim**-0.5
        self.register_buffer(
                        "scaling",
                        torch.tensor(0.125, dtype=FLOAT_DTYPE),  # 1 / sqrt(64)
                        persistent=False
                    )
        self.config = config
        self.is_causal = False
    
    def load_from_pretrained(self, pretrained_attn):
        self.qkv.weight.data.copy_(pretrained_attn.qkv.weight.data)
        self.qkv.bias.data.copy_(pretrained_attn.qkv.bias.data)
        self.proj.weight.data.copy_(pretrained_attn.proj.weight.data)
        self.proj.bias.data.copy_(pretrained_attn.proj.bias.data)


    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     cos: torch.Tensor,
    #     sin: torch.Tensor,
    #     **kwargs,
    # ) -> torch.Tensor:
    #     seq_length = hidden_states.shape[0]
    #     query_states, key_states, value_states = (
    #         self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    #     )
    #     query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

    #     query_states = query_states.transpose(0, 1).unsqueeze(0)
    #     key_states = key_states.transpose(0, 1).unsqueeze(0)
    #     value_states = value_states.transpose(0, 1).unsqueeze(0)

    #     attn_scores = torch.matmul(
    #         query_states, key_states.transpose(-2, -1)
    #     ) * self.scaling  # (1, heads, seq, seq)

    #     attn_probs = torch.softmax(attn_scores, dim=-1)

    #     attn_output = torch.matmul(attn_probs, value_states)  # (1, heads, seq, dim)
    #     attn_output = attn_output.squeeze(0).transpose(0, 1)
    #     attn_output = attn_output.reshape(seq_length, -1).contiguous()
    #     attn_output = self.proj(attn_output)
    #     return attn_output
    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:

        S = self.seq_len
        # =========================
        # QKV（严格官方路径）
        # =========================
        qkv = self.qkv(hidden_states)

        q, k, v = (
            qkv.reshape(S, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )  # (S, H, D)

        # =========================
        # RoPE
        # =========================
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # =========================
        # layout 变换
        # =========================
        q = q.transpose(0, 1).unsqueeze(0)  # (1, H, S, D)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        scale = self.scaling.to(dtype=q.dtype, device=q.device)
        attn_scores = attn_scores * scale  # 对齐PyTorch

        max_val = attn_scores.max(dim=-1, keepdim=True)[0]
        attn_scores = attn_scores - max_val

        attn_probs = torch.softmax(attn_scores, dim=-1)

        # =========================
        # 输出
        # =========================
        attn_output = torch.matmul(attn_probs, v)

        attn_output = attn_output.squeeze(0).transpose(0, 1)  # (S, H, D)
        attn_output = attn_output.reshape(S, -1).contiguous()

        out = self.proj(attn_output)

        return out


class GELUTanh(nn.Module):
    """
    A fast C implementation of the tanh approximation of the GeLU activation function. See
    https://huggingface.co/papers/1606.08415.

    This implementation is equivalent to NewGELU and FastGELU but much faster. However, it is not an exact numerical
    match due to rounding errors.
    """

    def __init__(self, use_gelu_tanh_python: bool = False):
        super().__init__()
        if use_gelu_tanh_python:
            self.act = self._gelu_tanh_python
        else:
            self.act = functools.partial(nn.functional.gelu, approximate="tanh")

    def _gelu_tanh_python(self, input):
        return input * 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

    def forward(self, input):
        return self.act(input)

class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = GELUTanh()
    
    def load_from_pretrained(self, pretrained_mlp):
        self.linear_fc1.weight.data.copy_(pretrained_mlp.linear_fc1.weight.data)
        self.linear_fc1.bias.data.copy_(pretrained_mlp.linear_fc1.bias.data)
        self.linear_fc2.weight.data.copy_(pretrained_mlp.linear_fc2.weight.data)
        self.linear_fc2.bias.data.copy_(pretrained_mlp.linear_fc2.bias.data)

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3VLVisionBlock(nn.Module):
    def __init__(self, config, seq_len: int, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = Qwen3VLVisionLayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Qwen3VLVisionLayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config=config, seq_len=seq_len)
        self.mlp = Qwen3VLVisionMLP(config=config)
    
    def load_from_pretrained(self, pretrained_block):
        self.norm1.load_state_dict(pretrained_block.norm1.state_dict())
        self.norm2.load_state_dict(pretrained_block.norm2.state_dict())
        self.attn.load_from_pretrained(pretrained_block.attn)
        self.mlp.load_from_pretrained(pretrained_block.mlp)


    def forward(
        self,
        hidden_states: torch.Tensor,
        cos,
        sin,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        rotary_pos_emb (`torch.Tensor`, *optional*):
            Precomputed rotary positional embeddings applied to the vision attention query/key states.
        """
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cos = cos,
            sin = sin,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

def export_block(model, path=None):
    if path is None:
        path = os.path.join(EXPORT_DIR, "block.onnx")
    config = Config()
    config.hidden_size = 1024
    config.intermediate_size = 4096
    config.num_heads = 16
    config.hidden_act = "gelu_pytorch_tanh"

    module = Qwen3VLVisionBlock(config)
    module.load_from_pretrained(model.blocks[0])
    module.eval()

    seq = 392
    head_dim = 64
    hidden_size = 1024

    dummy_hidden = torch.randn(seq, hidden_size, dtype=FLOAT_DTYPE)
    dummy_cos = torch.randn(seq, head_dim, dtype=FLOAT_DTYPE)
    dummy_sin = torch.randn(seq, head_dim, dtype=FLOAT_DTYPE)

    torch.onnx.export(
        module,
        (dummy_hidden, dummy_cos, dummy_sin),
        path,
        input_names=["hidden", "cos", "sin"],
        output_names=["output"],
        opset_version=11,
    )

class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, config, use_postshuffle_norm=False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = Qwen3VLVisionLayerNorm(
            self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6
        )
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def load_from_pretrained(self, pretrained_merger):
        self.norm.load_state_dict(pretrained_merger.norm.state_dict())
        self.linear_fc1.weight.data.copy_(pretrained_merger.linear_fc1.weight.data)
        self.linear_fc1.bias.data.copy_(pretrained_merger.linear_fc1.bias.data)
        self.linear_fc2.weight.data.copy_(pretrained_merger.linear_fc2.weight.data)
        self.linear_fc2.bias.data.copy_(pretrained_merger.linear_fc2.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x

def export_merger(model, path=None):
    if path is None:
        path = os.path.join(EXPORT_DIR, "merger.onnx")
    config = Config()
    config.hidden_size = 1024
    config.spatial_merge_size = 2
    config.out_hidden_size = 2048

    module = Qwen3VLVisionPatchMerger(config)
    module.load_from_pretrained(model.merger)

    module.eval()

    dummy = torch.randn(392, 1024, dtype=FLOAT_DTYPE)

    torch.onnx.export(
        module,
        (dummy,),
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
    )


class Qwen3VLVisionModel(nn.Module):
    input_modalities = ("image", "video")
    _no_split_modules = ["Qwen3VLVisionBlock"]
    _can_record_outputs = {
        "hidden_states": Qwen3VLVisionBlock,
        "attentions": Qwen3VLVisionAttention,
    }

    def __init__(self, config, profile: ExportProfile, *inputs, **kwargs) -> None:
        super().__init__()
        self.profile = profile
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3VLVisionPatchEmbed(
            config=config,
        )

        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([
            Qwen3VLVisionBlock(config, profile.num_vision_patches)
            for _ in range(config.depth)
        ])
        self.merger = Qwen3VLVisionPatchMerger(
            config=config,
            use_postshuffle_norm=False,
        )

        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(
                    config=config,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(config.deepstack_visual_indexes))
            ]
        )

        self.gradient_checkpointing = False

        t = 1
        h = profile.patch_grid
        w = profile.patch_grid
        self.max_hw = profile.patch_grid

        m = self.spatial_merge_size
        merged_h = h // m
        merged_w = w // m

        # ===== coords=====
        row = torch.arange(merged_h, dtype=INT_DTYPE)
        col = torch.arange(merged_w, dtype=INT_DTYPE)
        intra_row = torch.arange(m, dtype=INT_DTYPE)
        intra_col = torch.arange(m, dtype=INT_DTYPE)

        row_idx = row[:, None, None, None] * m + intra_row[None, None, :, None]
        col_idx = col[None, :, None, None] * m + intra_col[None, None, None, :]

        coords = torch.stack((
            row_idx + 0 * col_idx,
            col_idx + 0 * row_idx
        ), dim=-1)

        coords = coords.reshape(1, -1, 2)
        coords = coords.repeat(t, 1, 1)
        coords = coords.reshape(-1, 2)   # (N, 2)

        # ===== RoPE freq_table =====
        head_dim = config.hidden_size // config.num_heads
        half_dim = head_dim // 2

        index = torch.arange(0, half_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (10000 ** (index / half_dim))

        seq = torch.arange(self.max_hw, dtype=FLOAT_DTYPE)
        freq_table = (seq[:, None].float() * inv_freq[None, :]).to(FLOAT_DTYPE)

        # ===== gather =====
        emb = freq_table[coords]   # (N, 2, dim/2)

        # ===== flatten（xy拼接）=====
        emb = emb.reshape(emb.shape[0], -1)             # (N, dim)
        self.register_buffer("rope_emb", emb, persistent=False)

        h_idxs = torch.linspace(0, self.num_grid_per_side - 1, steps=h, dtype=FLOAT_DTYPE)
        w_idxs = torch.linspace(0, self.num_grid_per_side - 1, steps=w, dtype=FLOAT_DTYPE)

        N = self.num_grid_per_side  # 48

        # ===== floor / ceil =====
        h_floor = torch.floor(h_idxs).to(INT_DTYPE)
        w_floor = torch.floor(w_idxs).to(INT_DTYPE)

        max_val = float(N - 1)

        h_ceil = torch.minimum(
            (h_floor + 1).to(FLOAT_DTYPE),
            torch.full((), max_val, dtype=FLOAT_DTYPE)
        ).to(INT_DTYPE)

        w_ceil = torch.minimum(
            (w_floor + 1).to(FLOAT_DTYPE),
            torch.full((), max_val, dtype=FLOAT_DTYPE)
        ).to(INT_DTYPE)

        # ===== 权重 =====
        dh = h_idxs - h_floor.to(FLOAT_DTYPE)
        dw = w_idxs - w_floor.to(FLOAT_DTYPE)

        dh_exp = dh[:, None]
        dw_exp = dw[None, :]

        w00 = ((1 - dh_exp) * (1 - dw_exp)).reshape(-1)
        w01 = ((1 - dh_exp) * dw_exp).reshape(-1)
        w10 = (dh_exp * (1 - dw_exp)).reshape(-1)
        w11 = (dh_exp * dw_exp).reshape(-1)

        weight = torch.stack([w00, w01, w10, w11], dim=0)  # (4, HW)

        # ===== index =====
        base_h = h_floor * N
        base_h_ceil = h_ceil * N

        idx00 = (base_h[:, None] + w_floor[None]).reshape(-1)
        idx01 = (base_h[:, None] + w_ceil[None]).reshape(-1)
        idx10 = (base_h_ceil[:, None] + w_floor[None]).reshape(-1)
        idx11 = (base_h_ceil[:, None] + w_ceil[None]).reshape(-1)

        idx = torch.stack([idx00, idx01, idx10, idx11], dim=0).to(INT_DTYPE)  # (4, HW)

        # # ===== register =====
        self.register_buffer("interp_idx", idx, persistent=False)
        self.register_buffer("interp_weight", weight.to(FLOAT_DTYPE), persistent=False)


    def load_from_pretrained(self, pretrained_vis_model):
        # 加载所有子模块权重
        self.patch_embed.load_from_pretrained(pretrained_vis_model.patch_embed)
        self.pos_embed.load_state_dict(pretrained_vis_model.pos_embed.state_dict())
        self.merger.load_from_pretrained(pretrained_vis_model.merger)
        
        # 加载所有 block
        for i, blk in enumerate(self.blocks):
            blk.load_from_pretrained(pretrained_vis_model.blocks[i])
        
        # 加载 deepstack mergers
        for i, m in enumerate(self.deepstack_merger_list):
            m.load_from_pretrained(pretrained_vis_model.deepstack_merger_list[i])


    # def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
    #     merge_size = self.spatial_merge_size
    #     grid_thw_list = grid_thw.tolist()

    #     max_hw = max(max(h, w) for _, h, w in grid_thw_list)
    #     max_hw = torch.max(h, w)
    #     freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
    #     device = freq_table.device

    #     total_tokens = sum(t * h * w for t, h, w in grid_thw_list)
    #     pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

    #     offset = 0
    #     for num_frames, height, width in grid_thw_list:
    #         merged_h, merged_w = height // merge_size, width // merge_size

    #         block_rows = torch.arange(merged_h, device=device)  # block row indices
    #         block_cols = torch.arange(merged_w, device=device)  # block col indices
    #         intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
    #         intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

    #         # Compute full-resolution positions
    #         row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
    #         col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

    #         row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
    #         col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

    #         coords = torch.stack((row_idx, col_idx), dim=-1)

    #         if num_frames > 1:
    #             coords = coords.repeat(num_frames, 1)

    #         num_tokens = coords.shape[0]
    #         pos_ids[offset : offset + num_tokens] = coords
    #         offset += num_tokens

    #     embeddings = freq_table[pos_ids]  # lookup rotary embeddings
    #     embeddings = embeddings.flatten(1)
    #     return embeddings

    # def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
    #     device = self.pos_embed.weight.device
    #     merge_size = self.spatial_merge_size

    #     # === 单图 ===
    #     t = grid_thw[0, 0]
    #     h = grid_thw[0, 1]
    #     w = grid_thw[0, 2]

    #     h = h.to(torch.int64)
    #     w = w.to(torch.int64)

    #     # === 最大尺寸 ===
    #     #max_hw = torch.max(h, w)
    #     max_hw = torch.max(h.float(), w.float())

    #     freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim//2)

    #     # === 构建坐标===
    #     merged_h = h // merge_size
    #     merged_w = w // merge_size

    #     row = torch.arange(merged_h, device=device)
    #     col = torch.arange(merged_w, device=device)

    #     intra_row = torch.arange(merge_size, device=device)
    #     intra_col = torch.arange(merge_size, device=device)


    #     row_idx = row[:, None, None, None] * merge_size + intra_row[None, None, :, None]
    #     col_idx = col[None, :, None, None] * merge_size + intra_col[None, None, None, :]


    #     # row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
    #     # col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
    #     # row_idx = row_idx.repeat(1, merged_w, 1, merge_size).reshape(-1)
    #     # col_idx = col_idx.repeat(merged_h, 1, merge_size, 1).reshape(-1)

    #     # broadcast
    #     coords = torch.stack((
    #         row_idx + 0 * col_idx,
    #         col_idx + 0 * row_idx
    #     ), dim=-1)   # (H, W, m, m, 2)

    #     # flatten
    #     coords = coords.reshape(1, -1, 2)   # (1, N, 2)

    #     # ✅ 用同 dtype 的 0 做 broadcast（更安全）
    #     coords = coords + coords.new_zeros(t, 1, 1)

    #     coords = coords.reshape(-1, 2)

    #     # === t 维展开 ===
    #     #coords = coords.repeat(t, 1)

    #     # === lookup ===
    #     embeddings = freq_table[coords]  # (N, 2, dim//2)
    #     embeddings = embeddings.reshape(embeddings.shape[0], -1)

    #     return embeddings

    # def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
    #     device = self.pos_embed.weight.device

    #     freq_table = self.rotary_pos_emb(self.max_hw)  # float

    #     # === 直接用预计算 coords（int64）===
    #     coords = self.coords_static.to(device)

    #     # === gather ===
    #     embeddings = freq_table[coords]

    #     # 🔥 防止后面 concat 冲突（保险）
    #     embeddings = embeddings.to(freq_table.dtype)

    #     embeddings = embeddings.reshape(embeddings.shape[0], -1)
    #     return embeddings

    # def fast_pos_embed_interpolate(self, grid_thw):
    #     grid_thw_list = grid_thw.tolist()
    #     grid_ts = [row[0] for row in grid_thw_list]
    #     grid_hs = [row[1] for row in grid_thw_list]
    #     grid_ws = [row[2] for row in grid_thw_list]
    #     device = self.pos_embed.weight.device

    #     idx_list = [[] for _ in range(4)]
    #     weight_list = [[] for _ in range(4)]

    #     for t, h, w in grid_thw_list:
    #         h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
    #         w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

    #         h_idxs_floor = h_idxs.int()
    #         w_idxs_floor = w_idxs.int()
    #         h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
    #         w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

    #         dh = h_idxs - h_idxs_floor
    #         dw = w_idxs - w_idxs_floor

    #         base_h = h_idxs_floor * self.num_grid_per_side
    #         base_h_ceil = h_idxs_ceil * self.num_grid_per_side

    #         indices = [
    #             (base_h[None].T + w_idxs_floor[None]).reshape(-1),
    #             (base_h[None].T + w_idxs_ceil[None]).reshape(-1),
    #             (base_h_ceil[None].T + w_idxs_floor[None]).reshape(-1),
    #             (base_h_ceil[None].T + w_idxs_ceil[None]).reshape(-1),
    #         ]

    #         weights = [
    #             ((1 - dh)[None].T * (1 - dw)[None]).reshape(-1),
    #             ((1 - dh)[None].T * dw[None]).reshape(-1),
    #             (dh[None].T * (1 - dw)[None]).reshape(-1),
    #             (dh[None].T * dw[None]).reshape(-1),
    #         ]

    #         for i in range(4):
    #             idx_list[i].extend(indices[i].tolist())
    #             weight_list[i].extend(weights[i].tolist())

    #     idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
    #     weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
    #     pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
    #     patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

    #     patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

    #     patch_pos_embeds_permute = []
    #     merge_size = self.config.spatial_merge_size
    #     for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
    #         pos_embed = pos_embed.repeat(t, 1)
    #         pos_embed = (
    #             pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
    #             .permute(0, 1, 3, 2, 4, 5)
    #             #.flatten(0, 4)
    #             .reshape(t * (h // merge_size) * (w // merge_size) * merge_size * merge_size, -1)
    #         )
    #         patch_pos_embeds_permute.append(pos_embed)
    #     patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
    #     return patch_pos_embeds

    def fast_pos_embed_interpolate(self, grid_thw):

        idx = self.interp_idx          # (4, HW)
        weight = self.interp_weight    # (4, HW)

        pos = self.pos_embed(idx) * weight[:, :, None]
        pos = pos.sum(dim=0)

        # t = 1
        # h_m = 14
        # w_m = 14
        # C = 1024

        m = self.spatial_merge_size
        mh = self.profile.merged_grid
        mw = self.profile.merged_grid
        pos = pos.view(1, mh, m, mw, m, 1024).permute(0, 1, 3, 2, 4, 5)

        pos = pos.reshape(-1, pos.shape[-1])

        return pos

        # device = self.pos_embed.weight.device

        # # === 构建插值坐标 ===
        # # h_idxs = torch.linspace(
        # #     0, self.num_grid_per_side - 1,
        # #     steps=h,
        # #     device=device,
        # #     dtype=torch.float32
        # # )

        # # w_idxs = torch.linspace(
        # #     0, self.num_grid_per_side - 1,
        # #     steps=w,
        # #     device=device,
        # #     dtype=torch.float32
        # # )

        # h_idxs = self.h_idxs
        # w_idxs = self.w_idxs

        # h_floor = torch.floor(h_idxs).to(torch.int64)
        # w_floor = torch.floor(w_idxs).to(torch.int64)

        # # h_ceil = torch.clamp(h_floor + 1, max=self.num_grid_per_side - 1)
        # # w_ceil = torch.clamp(w_floor + 1, max=self.num_grid_per_side - 1)
        # max_val = float(self.num_grid_per_side - 1)

        # h_ceil = torch.minimum(
        #     (h_floor + 1).float(),
        #     torch.full_like(h_floor, max_val, dtype=torch.float32)
        # ).to(torch.int64)

        # w_ceil = torch.minimum(
        #     (w_floor + 1).float(),
        #     torch.full_like(w_floor, max_val, dtype=torch.float32)
        # ).to(torch.int64)


        # dh = h_idxs - h_floor.float()
        # dw = w_idxs - w_floor.float()

        # base_h = h_floor * self.num_grid_per_side
        # base_h_ceil = h_ceil * self.num_grid_per_side

        # # === 4点插值 index ===
        # idx00 = (base_h[:, None] + w_floor[None]).reshape(-1)
        # idx01 = (base_h[:, None] + w_ceil[None]).reshape(-1)
        # idx10 = (base_h_ceil[:, None] + w_floor[None]).reshape(-1)
        # idx11 = (base_h_ceil[:, None] + w_ceil[None]).reshape(-1)

        # idx = torch.stack([idx00, idx01, idx10, idx11], dim=0)

        # # === 权重 ===
        # # w00 = ((1 - dh)[:, None] * (1 - dw)[None]).reshape(-1)
        # # w01 = ((1 - dh)[:, None] * dw[None]).reshape(-1)
        # # w10 = (dh[:, None] * (1 - dw)[None]).reshape(-1)
        # # w11 = (dh[:, None] * dw[None]).reshape(-1)

        # H = dh.shape[0]
        # W = dw.shape[0]

        # dh_exp = dh.unsqueeze(1).expand(H, W)
        # dw_exp = dw.unsqueeze(0).expand(H, W)

        # w00 = ((1 - dh_exp) * (1 - dw_exp)).reshape(-1)
        # w01 = ((1 - dh_exp) * dw_exp).reshape(-1)
        # w10 = (dh_exp * (1 - dw_exp)).reshape(-1)
        # w11 = (dh_exp * dw_exp).reshape(-1)

        # weight = torch.stack([w00, w01, w10, w11], dim=0)

        # # === gather embedding ===
        # pos = self.pos_embed(idx) * weight[:, :, None]
        # pos = pos.sum(dim=0)   # (HW, dim)

        # # === merge reshape ===
        # merge_size = self.config.spatial_merge_size

        # pos = pos.repeat(1, 1)

        # # pos = pos.view(
        # #     t,
        # #     h // merge_size,
        # #     merge_size,
        # #     w // merge_size,
        # #     merge_size,
        # #     -1
        # # ).permute(0,1,3,2,4,5)

        # # 假设你固定 28×28
        # t = 1
        # h_m = 28 // merge_size
        # w_m = 28 // merge_size
        # C = 1024

        # pos = pos.view(
        #     t,
        #     h_m,
        #     merge_size,
        #     w_m,
        #     merge_size,
        #     C
        # ).permute(0,1,3,2,4,5)

        # pos = pos.reshape(-1, pos.shape[-1])

        # return pos

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs
    ):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        deepstack_feat_5 = deepstack_feat_11 = deepstack_feat_17 = None
        hidden_states = self.patch_embed(hidden_states) #[seq_len, hidden_size]

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rope_emb

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        deepstack_feature_lists = []
        cos, sin = position_embeddings[0],position_embeddings[1]
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cos,
                sin,
                **kwargs,
            )
        #     if layer_num in self.deepstack_visual_indexes:
        #         deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
        #             hidden_states
        #         )
        #         deepstack_feature_lists.append(deepstack_feature)

        # merged_hidden_states = self.merger(hidden_states)

        # return hidden_states, merged_hidden_states, deepstack_feature_lists
            if layer_num == 5:
                deepstack_feat_5 = self.deepstack_merger_list[0](hidden_states)
            if layer_num == 11:
                deepstack_feat_11 = self.deepstack_merger_list[1](hidden_states)
            if layer_num == 17:
                deepstack_feat_17 = self.deepstack_merger_list[2](hidden_states)

        merged_hidden_states = self.merger(hidden_states)

        return merged_hidden_states, deepstack_feat_5, deepstack_feat_11, deepstack_feat_17

def export_vision_encoder(model, profile: ExportProfile, path=None):
    if path is None:
        path = os.path.join(profile.export_dir, profile.vision_onnx_name)
    os.makedirs(profile.export_dir, exist_ok=True)
    config = Config()
    config.deepstack_visual_indexes = [5,11,17]
    config.depth = 24
    config.hidden_size = 1024
    config.in_channels = 3
    config.initializer_range = 0.02
    config.intermediate_size = 4096
    config.model_type = "qwen3_vl"
    config.num_heads = 16
    config.num_position_embeddings = 2304
    config.out_hidden_size = 2048
    config.patch_size = 16
    config.spatial_merge_size = 2
    config.temporal_patch_size = 2

    module = Qwen3VLVisionModel(config, profile).to("cpu", FLOAT_DTYPE)
    module.load_from_pretrained(model)

    module.eval()
    size = profile.image_size
    np_img = np.ones((size, size, 3), dtype=np.uint8) * 255
    img = Image.fromarray(np_img).resize((size, size))
    processor = AutoProcessor.from_pretrained("./Qwen3-VL-2B-Instruct")
    inputs = processor(text=[""], images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(FLOAT_DTYPE)       # 视觉模型输入 1
    image_grid_thw = inputs["image_grid_thw"].to(INT_DTYPE)   # 视觉模型输入 2
    print("\n模型输入：")
    print(f"pixel_values shape: {pixel_values.shape}")
    print(f"image_grid_thw:      {image_grid_thw}")

    torch.onnx.export(
        module,
        (pixel_values,image_grid_thw,),
        path,
        input_names=["hidden_states","grid_thw"],
        output_names=["merged_hidden_states",
                    "deepstack_feat_5",
                    "deepstack_feat_11",
                    " deepstack_feat_17"],
        opset_version=11,
    )

    print(f"exported fp16 ONNX -> {path}")

def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-VL vision ONNX")
    parser.add_argument(
        "--profile",
        choices=("256_256", "448_512"),
        default=os.environ.get("QWEN3_EXPORT_PROFILE", "256_256"),
        help="export layout profile (default: QWEN3_EXPORT_PROFILE or 256_256)",
    )
    args = parser.parse_args()
    profile = get_export_profile(args.profile)

    print(f"profile: {profile.name}")
    print(f"image_size: {profile.image_size}")
    print(f"export_dir: {profile.export_dir}")
    print("Loading model ...")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=FLOAT_DTYPE,
        attn_implementation="eager",
        device_map=None
    ).eval()

    print("Exporting fp16 vision modules...")
    export_vision_encoder(model.model.visual, profile)

    print("\n✅ All export done!")


if __name__ == "__main__":
    main()