# VLM GPU 全量推理（ThorU / ORT CUDA）

三个多模态模型的 **ONNX Runtime CUDA 全 GPU** 推理：`vision` + `LLM` 均在 GPU 上运行（Thor stream 模式：preblock 常驻，block/head 逐步 load/run）。

| 模型 | 入口 | ONNX 环境变量 |
|------|------|---------------|
| InternVL3_5-1B | `InternVL3_5-1B/test/test.py` | `INTERNVL_ONNX_EXPORT` |
| Qwen3-VL-2B | `Qwen3-VL/test/test.py` | `QWEN3_ONNX_EXPORT` |
| Gemma-4 E2B-it | `gemma-4/test/test.py` | `GEMMA4_ONNX_EXPORT` |

## 目录结构

```
InternVL3_5-1B/  Qwen3-VL/  gemma-4/
  test/test.py          # CUDA 推理入口
  om/
    dump_vision_om_inputs.py
    dump_llm_preblock_inputs.py
    parse_state.py      # decode
    vision_bin/         # 图像输入 bin
    prompt_bin/         # 文本 prefill bin
  *-HF/ 或 *-Instruct/  # tokenizer + config（无 safetensors）

scripts/
  thoru_ssh.sh          # SSH 到 ThorU
  thoru_rsync.sh        # 同步到板端
  thoru_install_verify_ort_gpu.sh
  thoru_build_ort_gpu_docker.sh
  thoru_vlm_forward_only_timing.py
  thoru_gemma_knorm_verify.py

imgs/                   # 测试图片（dump 用）
```

ONNX 权重不在本仓库，默认指向外部 `onnx_export/` 目录；ThorU 部署路径见下。

**导出脚本**（本仓库）：各模型根目录下 `vision.py` / `llm.py` / `proj.py` 等；Gemma 另有 `export_llm_onnx_all.py`。

> 整理 GPU 推理环境时，仅清理 **ThorU 板端**（`/cus_app_data/guanxj/`），**不要删本仓库**的导出与开发脚本。

## 运行（ThorU）

```bash
export ORT_USE_GPU=1
export PYTHONPATH=/cus_app_data/guanxj/py312-site-packages
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/thor/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu

# InternVL 示例
export INTERNVL_ONNX_EXPORT=/cus_app_data/guanxj/internvl3_5/onnx_export
cd InternVL3_5-1B/test && python3 test.py
```

Gemma 额外需要 `om/ple_table/embed_tokens_per_layer.bin`（`dump_llm_preblock_inputs.py --ple-only` 生成）。

## 输入准备

```bash
cd <model>/om
python dump_vision_om_inputs.py --image ../../imgs/your.png
python dump_llm_preblock_inputs.py --prompt "描述这张图片"
```

## GPU 实现要点

- **Vision**：InternVL/Gemma 的 vision 与 mm_proj **顺序加载**（不可同时 preload 两个 CUDA session，否则 Thor 上 Gather/CUDNN 报错）
- **LLM**：preblock session 常驻 GPU，各 block + lm_head 每步 load → run → unload
- 已移除 CPU/hybrid/MDC OM/assistant 推测解码/benchmark 遗留

## 同步到 ThorU

```bash
./scripts/thoru_rsync.sh InternVL3_5-1B /cus_app_data/guanxj/internvl3_5
./scripts/thoru_rsync.sh Qwen3-VL /cus_app_data/guanxj/qwen3-vl
./scripts/thoru_rsync.sh gemma-4 /cus_app_data/guanxj/gemma4
```
