"""Shared ONNX export layout profiles for Qwen3-VL."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ExportProfile:
    name: str
    image_size: int
    max_seq_len: int
    export_dir: str
    vision_onnx_name: str

    @property
    def patch_grid(self) -> int:
        return self.image_size // 16

    @property
    def merged_grid(self) -> int:
        return self.patch_grid // 2

    @property
    def num_vision_patches(self) -> int:
        return self.patch_grid * self.patch_grid

    @property
    def num_image_tokens(self) -> int:
        return self.merged_grid * self.merged_grid

    @property
    def image_prefix_len(self) -> int:
        return 4

    @property
    def image_token_start(self) -> int:
        return self.image_prefix_len

    @property
    def image_token_end(self) -> int:
        return self.image_prefix_len + self.num_image_tokens


_PROFILES = {
    "256_256": ExportProfile(
        name="256_256",
        image_size=256,
        max_seq_len=256,
        export_dir=(
            "/e-vepfs-01/ppdc/guanxj/ENetQuery/work_dirs/Qwen3-vlTest/"
            "qwen_256_256/onnx_export_256_256"
        ),
        vision_onnx_name="vision_256.onnx",
    ),
    "448_512": ExportProfile(
        name="448_512",
        image_size=448,
        max_seq_len=512,
        export_dir=(
            "/e-vepfs-01/ppdc/guanxj/ENetQuery/work_dirs/Qwen3-vlTest/"
            "qwen_448_512/onnx_export_448_512"
        ),
        vision_onnx_name="vision_448.onnx",
    ),
}


def get_export_profile(name: str | None = None) -> ExportProfile:
    key = name or os.environ.get("QWEN3_EXPORT_PROFILE", "256_256")
    if key not in _PROFILES:
        raise ValueError(f"unknown export profile {key!r}, choose from {list(_PROFILES)}")
    profile = _PROFILES[key]
    override = os.environ.get("QWEN3_ONNX_EXPORT_DIR")
    if override:
        return ExportProfile(
            name=profile.name,
            image_size=profile.image_size,
            max_seq_len=profile.max_seq_len,
            export_dir=override,
            vision_onnx_name=profile.vision_onnx_name,
        )
    return profile
