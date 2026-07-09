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
├── Qwen3-VL/                # Qwen3-VL-2B-Instruct
├── gemma-4/                 # Gemma-4 E2B-it
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
│       ├── vision_bin/        # 图像输入 bin
│       ├── prompt_bin/        # 文本 prefill bin
│       ├── pipeline/          # OM 并行 / resident worker
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
| ThorU 部署目录 | `/opt/vlm/internvl3_5` | `/opt/vlm/qwen3-vl` | `/opt/vlm/gemma4` |

---

## 仓库包含 / 不包含

**包含**（约 25MB）：

- ONNX 导出与对齐脚本
- ORT CUDA 推理入口
- MDC OM 流水线（dump / pipeline / parse）
- tokenizer / config 占位目录（`.gitkeep`）
- ThorU 部署脚本

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

### MDC 板端（OM 推理）

- CANN / ACL 环境
- `msame` 或 resident worker 二进制
- 可选：pyACL（Python）或 C++ OM 常驻 daemon（`om/pipeline/om_resident_cpp/`）
- 详见各模型 `om/README.md`

---

## 快速开始：ThorU GPU 推理

以 InternVL 为例，Qwen3-VL / Gemma-4 仅替换路径与环境变量。

```bash
export ORT_USE_GPU=1
export PYTHONPATH=/opt/vlm/py312-site-packages
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/thor/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu

# 1. 设置 ONNX 权重路径
export INTERNVL_ONNX_EXPORT=/opt/vlm/internvl3_5/onnx_export

# 2. 准备输入 bin（开发机上执行，再 rsync 到板端）
cd InternVL3_5-1B/om
python dump_vision_om_inputs.py --image ../../imgs/example.jpg
python dump_llm_preblock_inputs.py --prompt "描述这张图片"

# 3. 板端运行
cd ../test && python3 test.py
```

Qwen3-VL 示例：

```bash
export QWEN3_ONNX_EXPORT=/opt/vlm/qwen3-vl/onnx_export
export QWEN3_EXPORT_PROFILE=448_512   # 或 256_256
cd Qwen3-VL/test && python3 test.py
```

Gemma-4 额外需要 PLE 查表：

```bash
cd gemma-4/om
python dump_llm_preblock_inputs.py --ple-only   # 生成 ple_table/
export GEMMA4_ONNX_EXPORT=/opt/vlm/gemma4/onnx_export
cd ../test && python3 test.py
```

Qwen3-VL 使用 `transformers` 在线预处理，不依赖 `om/` bin（但仍保留 OM 流水线）。

---

## 快速开始：ONNX 导出与对齐

```bash
cd <model>/

# 导出（各模型脚本名略有不同）
python vision.py
python llm.py
# Gemma 另有：python assist_model.py

# 数值对齐
cd test/
python onnx_torch_test.py    # ONNX vs PyTorch
python vision_test.py        # 分模块单测
python llm_test.py
```

导出产物默认写到各模型下的 `./onnx_export/`，通过环境变量覆盖。

---

## 快速开始：MDC OM 推理

以 Qwen3-VL 为例（完整命令见 [Qwen3-VL/om/README.md](Qwen3-VL/om/README.md)）：

```bash
cd Qwen3-VL/om

# 本地：生成静态 bin
python dump_vision_om_inputs.py --image ../../imgs/example.jpg
python dump_llm_preblock_inputs.py --prompt "What is shown in this image?"

# 拷到板端（替换为实际设备地址）
scp -r vision_bin prompt_bin user@<device-ip>:/opt/vlm/qwen3-vl/

# 板端：推理
RUN_MSAME=1 bash run_om_pipeline.sh

# 拷回并解析
scp -r user@<device-ip>:/opt/vlm/qwen3-vl/om_output .
python parse_state.py --output-dir om_output --dump-dir prompt_bin
```

批量多图流水线：

```bash
RUN_MSAME=1 bash run_om_pipeline_pipe.sh ./batch
```

---

## 同步到 ThorU

SSH 连接信息通过环境变量配置（见 `scripts/thoru_ssh.sh` / `scripts/thoru_rsync.sh`）：

```bash
export THORU_JUMP1=user@jump1.example.com
export THORU_JUMP1_PASS=...
export THORU_JUMP2=user@jump2.example.com
export THORU_JUMP2_PASS=...
export THORU_TARGET=user@device.example.com
export THORU_TARGET_PASS=...

# 代码同步（不含 ONNX 权重）
./scripts/thoru_rsync.sh InternVL3_5-1B /opt/vlm/internvl3_5
./scripts/thoru_rsync.sh Qwen3-VL       /opt/vlm/qwen3-vl
./scripts/thoru_rsync.sh gemma-4        /opt/vlm/gemma4

# SSH 登录板端
./scripts/thoru_ssh.sh
./scripts/thoru_ssh.sh 'ls /opt/vlm/'
```

ONNX 权重大（数 GB），需单独 `rsync` 到板端 `onnx_export/` 目录。

---

## ThorU 工具脚本

| 脚本 | 用途 |
|------|------|
| `thoru_ssh.sh` | 经双跳 SSH 登录 ThorU 或执行远程命令 |
| `thoru_rsync.sh` | 将本地目录 tar 流式同步到 ThorU |
| `thoru_build_ort_gpu_docker.sh` | 在 Docker 内交叉编译 ORT GPU wheel |
| `thoru_install_verify_ort_gpu.sh` | 安装并验证 ORT GPU |
| `thoru_vlm_forward_only_timing.py` | 三模型 pure forward 耗时 benchmark |
| `thoru_gemma_knorm_verify.py` | Gemma k-norm 数值验证 |

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

诊断脚本：

```bash
bash om/pipeline/diagnose_acl.sh          # pyACL 环境
bash om/pipeline/om_resident_cpp/diagnose_cann_build.sh  # C++ 编译环境
```

---

## 模型 ONNX 切分概览

| 模型 | Vision | LLM 切分 | 特殊 |
|------|--------|----------|------|
| InternVL3.5-1B | `vision_448` | preblock + block1~3 + lm_head | 448px, seq=512 |
| Qwen3-VL-2B | `vision_256` / `vision_448` | preblock + block1~3 + lm_head | 两种 profile |
| Gemma-4 E2B-it | `vision` + `mm_proj` | preblock + block×7 + lm_head | PLE 查表；可选 assistant |

各段 I/O shape、dtype 与链式拼接规则详见各模型 README 中的 ONNX I/O 表格。

---

## 环境变量参考

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ORT_USE_GPU` | `0` | `1` 启用 CUDA EP |
| `INTERNVL_ONNX_EXPORT` | `./onnx_export` | InternVL ONNX 目录 |
| `QWEN3_ONNX_EXPORT` | `./onnx_export` | Qwen3-VL ONNX 目录 |
| `QWEN3_EXPORT_PROFILE` | `448_512` | Qwen3 导出 profile |
| `GEMMA4_ONNX_EXPORT` | `./onnx_export` | Gemma-4 ONNX 目录 |
| `GEMMA4_OM` | — | Gemma-4 OM 工作目录 |
| `THORU_DATA_DIR` | `/opt/vlm` | ThorU 板端数据根目录 |
| `THORU_JUMP1` / `THORU_JUMP1_PASS` | — | 第一跳 SSH |
| `THORU_JUMP2` / `THORU_JUMP2_PASS` | — | 第二跳 SSH |
| `THORU_TARGET` / `THORU_TARGET_PASS` | — | ThorU 目标设备 |
| `SUDO_PASS` | — | `thoru_build_ort_gpu_docker.sh` 中 sudo 密码 |

---

## 注意事项

- 整理 ThorU 板端环境时，仅清理 `/opt/vlm/`（可通过 `THORU_DATA_DIR` 覆盖），**不要删除本仓库**的导出与开发脚本
- 离线传输：开发机无法直连 GitHub 时，可用 `git bundle` 打包后在本机 push

---

## 子文档索引

| 文档 | 内容 |
|------|------|
| [InternVL3_5-1B/README.md](InternVL3_5-1B/README.md) | InternVL ONNX I/O、test、工作流 |
| [InternVL3_5-1B/om/README.md](InternVL3_5-1B/om/README.md) | InternVL MDC OM 流水线 |
| [gemma-4/README.md](gemma-4/README.md) | Gemma-4 ONNX I/O、PLE、assistant |
| [gemma-4/om/README.md](gemma-4/om/README.md) | Gemma-4 MDC OM 流水线 |
| [Qwen3-VL/om/README.md](Qwen3-VL/om/README.md) | Qwen3-VL MDC OM 流水线 |
| [gemma-4/om/pipeline/om_resident_cpp/README.md](gemma-4/om/pipeline/om_resident_cpp/README.md) | C++ OM 常驻 daemon |

---

## License

各模型权重遵循其上游 HuggingFace 仓库的 License（InternVL、Qwen、Gemma 等）。Qwen3-VL 子目录含 [Qwen3-VL/LICENSE](Qwen3-VL/LICENSE)（Apache 2.0）。本仓库代码仅包含导出与推理工程脚本。
