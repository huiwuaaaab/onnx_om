# onnx_om — 多模态 VLM ONNX / OM 推理工程

本仓库包含三个图文理解模型的 **ONNX 导出、数值对齐、板端推理** 全流程代码：

| 模型 | 参数量 | 目录 | 详细文档 |
|------|--------|------|----------|
| InternVL3.5-1B | ~1B | [`InternVL3_5-1B/`](InternVL3_5-1B/) | [README](InternVL3_5-1B/README.md) · [om/](InternVL3_5-1B/om/README.md) |
| Qwen3-VL-2B-Instruct | ~2B | [`Qwen3-VL/`](Qwen3-VL/) | [om/](Qwen3-VL/om/README.md) |
| Gemma-4 E2B-it | ~2B | [`gemma-4/`](gemma-4/) | [README](gemma-4/README.md) · [om/](gemma-4/om/README.md) |

支持两条部署路径：

1. **ThorU / ORT CUDA**（当前主推）：`vision` + `LLM` 全 GPU 推理，`test/test.py` 为端到端入口
2. **MDC / Ascend OM**（历史链路）：`om/` 下 dump 静态 bin → 板端 `run_om_pipeline.sh` → `parse_state.py` 解析文本

```
开发机                          板端
──────                          ────
vision.py / llm.py  ──导出──►  onnx_export/*.onnx
dump_*_inputs.py    ──bin──►   vision_bin/ + prompt_bin/
test/test.py        ◄─对齐─    ORT CUDA 或 OM pipeline
```

---

## 目录结构

```
onnx_om/
├── InternVL3_5-1B/          # InternVL3.5-1B
├── Qwen3-VL/                  # Qwen3-VL-2B-Instruct
├── gemma-4/                   # Gemma-4 E2B-it
│   ├── vision.py / llm.py / proj.py   # ONNX 导出脚本
│   ├── *-HF/ 或 *-Instruct/           # tokenizer + config（权重不入库）
│   ├── test/
│   │   ├── test.py            # ThorU ORT CUDA 端到端推理 ★
│   │   ├── onnx_torch_test.py # ONNX vs PyTorch 数值对齐
│   │   └── vision_test.py / llm_test.py
│   └── om/
│       ├── dump_vision_om_inputs.py
│       ├── dump_llm_preblock_inputs.py
│       ├── parse_state.py     # decode 状态解析
│       ├── vision_bin/          # 图像输入 bin
│       ├── prompt_bin/          # 文本 prefill bin
│       ├── pipeline/            # OM 并行 / resident worker
│       └── run_om_pipeline.sh
├── scripts/                   # ThorU 部署与验证工具
├── imgs/                      # 测试图片（dump 用）
└── README.md
```

各模型目录结构一致，差异见下表。

---

## 模型差异速查

| 项目 | InternVL3.5-1B | Qwen3-VL-2B | Gemma-4 E2B-it |
|------|----------------|-------------|----------------|
| 推理入口 | `InternVL3_5-1B/test/test.py` | `Qwen3-VL/test/test.py` | `gemma-4/test/test.py` |
| ONNX 路径变量 | `INTERNVL_ONNX_EXPORT` | `QWEN3_ONNX_EXPORT` | `GEMMA4_ONNX_EXPORT` |
| 输入方式 | `om/` 静态 bin | `transformers` processor 在线预处理 | `om/` 静态 bin |
| LLM 分块 | 3 block + lm_head | 3 block + lm_head | 7 block + lm_head |
| 特殊依赖 | — | `QWEN3_EXPORT_PROFILE`（默认 `448_512`） | `om/ple_table/embed_tokens_per_layer.bin` |
| ThorU 部署目录 | `/cus_app_data/guanxj/internvl3_5` | `/cus_app_data/guanxj/qwen3-vl` | `/cus_app_data/guanxj/gemma4` |

---

## 仓库包含 / 不包含

**包含**（约 25MB）：

- ONNX 导出与对齐脚本
- ORT CUDA 推理入口
- MDC OM 流水线（dump / pipeline / parse）
- tokenizer / config 占位目录（`.gitkeep`）
- ThorU 部署脚本与测试图

**不包含**（见 [`.gitignore`](.gitignore)）：

| 类型 | 说明 |
|------|------|
| `*.safetensors` / `*.bin` | 模型权重与运行时 dump 数据 |
| `transformers/` | 本地 transformers 源码副本 |
| `onnx_export/` | ONNX 权重目录，需单独准备或导出 |

克隆后需自行下载 HF 权重到各模型的 `*-HF/` / `*-Instruct/` 目录，或将 ONNX 放到外部 `onnx_export/` 路径。

---

## 环境要求

### 开发机（导出 / 对齐）

- Python 3.10+
- PyTorch、transformers、onnx、onnxruntime
- 各模型 HF 权重（用于 `demo.py` 与 `onnx_torch_test.py`）

### ThorU 板端（ORT CUDA 推理）

- Python 3.12
- **onnxruntime-gpu**（aarch64，需在板端源码编译或离线安装 wheel）
- CUDA 12.8 runtime + cuDNN
- 环境变量示例：

```bash
export ORT_USE_GPU=1
export PYTHONPATH=/cus_app_data/guanxj/py312-site-packages
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/thor/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu
```

### MDC 板端（OM 推理）

- CANN / ACL 环境
- `msame` 或 resident worker 二进制
- 详见各模型 `om/README.md`

---

## 快速开始：ThorU GPU 推理

以 InternVL 为例，Qwen3-VL / Gemma-4 仅替换路径与环境变量。

```bash
# 1. 设置 ONNX 权重路径
export INTERNVL_ONNX_EXPORT=/cus_app_data/guanxj/internvl3_5/onnx_export
export ORT_USE_GPU=1

# 2. 准备输入 bin（开发机上执行，再 rsync 到板端）
cd InternVL3_5-1B/om
python dump_vision_om_inputs.py --image ../../imgs/20260616-111801.jpg
python dump_llm_preblock_inputs.py --prompt "描述这张图片"

# 3. 板端运行
cd InternVL3_5-1B/test
python3 test.py
```

Gemma-4 额外步骤：

```bash
cd gemma-4/om
python dump_llm_preblock_inputs.py --ple-only   # 生成 ple_table/
```

Qwen3-VL 使用 `transformers` 在线预处理，不依赖 `om/` bin（但仍保留 OM 流水线）。

---

## 快速开始：ONNX 导出与对齐

```bash
cd <model>/

# 导出（各模型脚本名略有不同）
python vision.py
python llm.py
# Gemma 另有：python export_llm_onnx_all.py

# 数值对齐
cd test/
python onnx_torch_test.py    # ONNX vs PyTorch
python vision_test.py        # 分模块单测
python llm_test.py
```

导出产物默认写到外部 `work_dirs/<model>/onnx_export/`，通过环境变量覆盖。

---

## 同步到 ThorU

```bash
# 代码同步（不含 ONNX 权重）
./scripts/thoru_rsync.sh InternVL3_5-1B /cus_app_data/guanxj/internvl3_5
./scripts/thoru_rsync.sh Qwen3-VL       /cus_app_data/guanxj/qwen3-vl
./scripts/thoru_rsync.sh gemma-4        /cus_app_data/guanxj/gemma4

# SSH 登录板端
./scripts/thoru_ssh.sh
./scripts/thoru_ssh.sh 'ls /cus_app_data/guanxj/'
```

ONNX 权重大（数 GB），需单独 `rsync` 到板端 `onnx_export/` 目录。

---

## GPU 实现要点（ThorU）

推理采用 **stream 模式**，在显存受限的 Thor 板上逐段加载 ONNX session：

```
prefill:  vision → mm_proj → llm_preblock（preblock 常驻 GPU）
decode:   llm_block_i（load → run → unload）→ lm_head → argmax → 下一步
```

关键约束：

- **Vision 顺序加载**：InternVL / Gemma 的 `vision` 与 `mm_proj` 不可同时 preload 两个 CUDA session，否则 Thor 上 Gather/CUDNN 报错
- **LLM stream**：`llm_preblock` session 常驻 GPU；各 `llm_block` + `lm_head` 每步 load → run → unload
- **Gemma4 RMSNorm**：`llm.py` 使用 FP16-safe amax 归一化，避免 Thor GPU 上 `k_norm` 溢出；re-export 前勿改回旧版
- 已移除 CPU/hybrid 模式、assistant 投机解码、benchmark 遗留代码

验证工具：

```bash
python scripts/thoru_gemma_knorm_verify.py       # Gemma k_norm 无 Inf 检查
python scripts/thoru_vlm_forward_only_timing.py  # 分模型 forward 计时
```

ORT GPU 编译与安装见 `scripts/thoru_build_ort_gpu_docker.sh`、`scripts/thoru_install_verify_ort_gpu.sh`。

---

## MDC OM 流水线

`om/` 目录提供完整的 Ascend OM 推理链路，三个模型流程一致：

```
dump_vision_om_inputs.py  →  vision_bin/
dump_llm_preblock_inputs.py → prompt_bin/
         ↓ scp
run_om_pipeline.sh（或 run_om_pipeline_pipe.sh 并行版）
         ↓
om_output/state/ + final_*
         ↓ scp
parse_state.py  →  生成文本
```

并行 / resident 模式见各模型 `om/pipeline/README.md`。

---

## 环境变量参考

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ORT_USE_GPU` | `0` | `1` 启用 CUDA EP |
| `INTERNVL_ONNX_EXPORT` | 开发机 work_dirs 路径 | InternVL ONNX 目录 |
| `QWEN3_ONNX_EXPORT` | 开发机 work_dirs 路径 | Qwen3-VL ONNX 目录 |
| `QWEN3_EXPORT_PROFILE` | `448_512` | Qwen3 导出 profile |
| `GEMMA4_ONNX_EXPORT` | 开发机 work_dirs 路径 | Gemma-4 ONNX 目录 |
| `MAX_NEW_TOKENS` / `QWEN3_MAX_NEW_TOKENS` | 各 test.py 内默认 | 生成长度 |

---

## 注意事项

- 整理 ThorU 板端环境时，仅清理 `/cus_app_data/guanxj/`，**不要删除本仓库**的导出与开发脚本
- `rand/`、`cuda-samples-12.8/` 等实验目录不在本仓库核心范围内
- 离线传输：开发机无法直连 GitHub 时，可用 `git bundle` 打包后在本机 push

---

## License

各模型权重遵循其上游 HuggingFace 仓库的 License（InternVL、Qwen、Gemma 等）。本仓库代码仅包含导出与推理工程脚本。
