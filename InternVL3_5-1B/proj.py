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


class InternVLLayerNorm(nn.Module):
    """纯 fp16 LayerNorm，导出 ONNX/OM 全程 fp16。"""

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape, dtype=FLOAT_DTYPE))
        self.bias = nn.Parameter(torch.zeros(normalized_shape, dtype=FLOAT_DTYPE))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
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


class InternVLMultiModalProjector(nn.Module):
    def __init__(self, norm_size: int = 4096, hidden_size: int = 1024) -> None:
        super().__init__()
        self.layer_norm = InternVLLayerNorm(norm_size)
        self.linear_1 = nn.Linear(norm_size, hidden_size, bias=True)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def load_from_pretrained(self, pretrained_mm_proj):
        missing, unexpected = self.load_state_dict(pretrained_mm_proj.state_dict(), strict=True)
        print("mm_proj missing:", missing)
        print("mm_proj unexpected:", unexpected)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.layer_norm(image_features)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


def pixel_shuffle(vision_features: torch.Tensor, scale_factor: float = 0.5):
    """Perform pixel shuffle downsampling on vision features.

    Args:
        vision_features (`torch.Tensor`):
            Input tensor of shape (batch_size, width, height, channels).
        scale_factor (`float`, *optional*, defaults to `0.5`):
            Factor by which to downsample. Default is 0.5, which halves the dimensions.

    Returns:
        vision_features (`torch.Tensor`):
            Downsampled tensor of shape (batch_size, height*scale_factor, width*scale_factor, channels/(scale_factor^2)).
    """
    batch_size, width, height, channels = vision_features.size()
    height_out = 16
    width_out = 16
    channels_tmp = 2048
    channels_out = 4096
    # Reshape to allow downsampling
    vision_features = vision_features.view(
        batch_size, width, height_out,channels_tmp
    )
    # Permute dimensions to align downsampled axis correctly
    vision_features = vision_features.permute(0, 2, 1, 3).contiguous()
    # Reshape to achieve final downsampled dimensions
    vision_features = vision_features.view(
        batch_size, height_out, width_out, channels_out
    )

    # Swap height and width back for proper orientation
    vision_features = vision_features.permute(0, 2, 1, 3).contiguous()

    return vision_features

class multi_modal_projector(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.mm_proj = InternVLMultiModalProjector()
        self.mm_proj.load_from_pretrained(model.multi_modal_projector)

    def forward(self, vision_output):
        vision_features = vision_output[:,1:,:]
        channels = vision_features.shape[1]
        feature_size = 32   #int(channels**0.5)
        batch_size = vision_features.shape[0]

        # Reshape tensor to spatial dimensions
        vision_features = vision_features.reshape(batch_size, feature_size, feature_size, -1)   #[1,32,32,1024]

        # Apply downsampling using pixel shuffle
        vision_features = pixel_shuffle(vision_features, scale_factor=0.5)  #[1,16,16,4096]

        # Reshape tensor to prepare for projection
        vision_features = vision_features.reshape(batch_size, -1, vision_features.shape[-1])
        
        vision_features = self.mm_proj(vision_features)
        return vision_features

def export_projection(model,path="./onnx_export/mm_proj.onnx"):
    model = model.to("cuda", FLOAT_DTYPE)
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

    inputs["pixel_values"] = pixel_values.unsqueeze(0).to("cuda", FLOAT_DTYPE)

    output = model.model.vision_tower(pixel_values=inputs["pixel_values"])
    vision_features = output.last_hidden_state

    mm_model = multi_modal_projector(model.model).to("cuda", FLOAT_DTYPE).eval()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.onnx.export(
        mm_model,
        (vision_features,),
        path,
        input_names=["vision_features"],
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

    print("Exporting fp16 projection modules...")
    export_projection(model)

    print("\n✅ All export done!")


if __name__ == "__main__":
    main()