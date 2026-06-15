from transformers import InternVLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration,AutoProcessor
from PIL import Image
import numpy as np
import functools
import inspect

FLOAT_DTYPE = torch.float16

class Config:
    pass


class InternVLVisionLayerNorm(nn.Module):
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
        amax = x.abs().amax(dim=-1, keepdim=True) + eps
        xs = x / amax
        mean = xs.mean(dim=-1, keepdim=True)
        xm = xs - mean
        var = xm.pow(2).mean(dim=-1, keepdim=True) + eps
        x = xm * torch.pow(var, -0.5)
        x = x * self.weight + self.bias
        return x


class InternVLVisionPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """
    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_shape = patch_shape

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        embeddings = self.projection(pixel_values.to(self.projection.weight.dtype))
        embeddings = embeddings.flatten(2).transpose(1, 2)

        return embeddings


class InternVLVisionEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.

    """

    def __init__(self, config) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = InternVLVisionPatchEmbeddings(config)
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size[0]
        new_width = width // self.patch_size[1]

        sqrt_num_positions = (num_positions**0.5).to(torch.int64)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor
    ) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)
        batch_size, seq_len, _ = embeddings.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  #[B,1,hidden_size]
        embeddings = torch.cat((cls_tokens, embeddings), dim=1) #[B,seq_len+1,hidden_size]

        #embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)  
        embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float | int = 0.0,
    **kwargs,
):
    key_states = key
    value_states = value

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # No upcasting of the attention weights to float32 in this implementation
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

class InternVLVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = nn.functional.gelu
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class InternVLVisionRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=FLOAT_DTYPE))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        eps = torch.full((), self.variance_epsilon, dtype=x.dtype, device=x.device)
        amax = x.abs().amax(dim=-1, keepdim=True) + eps
        xs = x / amax
        mean_squared = xs.pow(2).mean(-1, keepdim=True) + eps
        x = xs * torch.pow(mean_squared, -0.5)
        x = x * self.weight
        return x


class InternVLVisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.projection_layer = nn.Linear(self.embed_dim, self.embed_dim)
        # Needed for flash attention
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        batch_size, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scale,
            is_causal=False,
            **kwargs,
        )
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        output = self.projection_layer(attn_output)

        return output, attn_weights


class InternVLVisionLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config) -> None:
        super().__init__()
        self.seq_len_dim = 1
        self.attention = InternVLVisionAttention(config)
        self.mlp = InternVLVisionMLP(config)
        self.layernorm_before = InternVLVisionLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = InternVLVisionLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        init_values = config.layer_scale_init_value
        self.lambda_1 = nn.Parameter(init_values * torch.ones(config.hidden_size), requires_grad=True)
        self.lambda_2 = nn.Parameter(init_values * torch.ones(config.hidden_size), requires_grad=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self,hidden_states: torch.Tensor):
        attention_output, _ = self.attention(
            self.layernorm_before(hidden_states),  # in InternVLVision, layernorm is applied before self-attention
        )

        attention_output = self.lambda_1 * attention_output

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in InternVLVision, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)

        layer_output = self.mlp(layer_output)
        layer_output = self.dropout(layer_output)

        if self.lambda_2 is not None:
            layer_output = self.lambda_2 * layer_output

        # second residual connection
        layer_output = layer_output + hidden_states

        return layer_output


class InternVLVisionEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([InternVLVisionLayer(config) for i in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)

        return hidden_states


class InternVLVisionModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.embeddings = InternVLVisionEmbeddings(config)
        self.encoder = InternVLVisionEncoder(config)

        self.layernorm = (
            nn.Identity() if config.use_mean_pooling
            else InternVLVisionLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
    
    def load_from_pretrained(self, pretrained_vis_model):
        vision_state_dict = pretrained_vis_model.state_dict()
        missing, unexpected = self.load_state_dict(vision_state_dict, strict=False)

        print("missing:", missing)
        print("unexpected:", unexpected)

    def forward(
        self,
        pixel_values: torch.Tensor
    ):
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(embedding_output)
        sequence_output = self.layernorm(encoder_outputs)

        return sequence_output

def export_vision_encoder(model,path="/e-vepfs-01/ppdc/guanxj/ENetQuery/work_dirs/InternVL3_5/onnx_export/vision_448_notchunk.onnx"):
    config = Config()
    config.attention_bias = True
    config.hidden_size = 1024
    config.hidden_dropout_prob = 0.0
    config.image_size = [
      448,
      448
    ]
    config.initializer_factor = 0.1
    config.initializer_range = 1e-10
    config.intermediate_size = 4096
    config.layer_norm_eps = 1e-06
    config.layer_scale_init_value = 0.1
    config.num_attention_heads = 16
    config.num_channels = 3
    config.num_hidden_layers = 24
    config.patch_size = [
      14,
      14
    ]
    config.use_absolute_position_embeddings = True
    config.use_mean_pooling = True
    config.vision_feature_layer = -1
    config.vision_feature_select_strategy = "default"

    module = InternVLVisionModel(config).to("cuda", FLOAT_DTYPE)
    module.load_from_pretrained(model)

    module.eval()
    processor = AutoProcessor.from_pretrained("./InternVL3_5-1B-HF")
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

    # inputs = processor(
    #     text=[prompt],
    #     images=[image],
    #     return_tensors="pt"
    # ).to("cuda", torch.float32)
    # pixel_values = inputs["pixel_values"]
    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors=None,
        images_kwargs={"min_patches": 1,
                    "max_patches": 1,},
    ).to("cuda", FLOAT_DTYPE)

    pixel_values = inputs["pixel_values"]

    if isinstance(pixel_values, list):
        pixel_values = torch.cat(pixel_values, dim=0)

    pixel_values = pixel_values.unsqueeze(0).to("cuda", FLOAT_DTYPE)

    print("\n模型输入：")
    print(f"pixel_values shape: {pixel_values.shape}")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.onnx.export(
        module,
        (pixel_values,),
        path,
        input_names=["pixel_values"],
        output_names=["hidden_states"],
        opset_version=11,
    )
    print(f"exported fp16 ONNX -> {path}")

def main():
    print("Loading model ...")
    MODEL_PATH = "./InternVL3_5-1B-HF"
    model = InternVLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=FLOAT_DTYPE,
        attn_implementation="eager",
        device_map=None
    ).eval()

    print("Exporting fp16 vision modules...")
    # export_patch_embed(model.model.visual)
    # export_block(model.model.visual)
    # export_merger(model.model.visual)
    export_vision_encoder(model.model.vision_tower)

    print("\n✅ All export done!")


if __name__ == "__main__":
    main()