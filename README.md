# ONNX → OM 多模态推理工程

本仓库收录 **Gemma-4**、**Qwen3-VL**、**InternVL3.5-1B** 三个视觉语言（VL）模型的 ONNX 导出、数值对齐与华为 MDC 板端 OM 推理流水线。

```
本地机：导出 ONNX / dump 静态 bin → MDC：OM 链式推理 → 本地机：parse 文本
```

---

## 目录结构

| 目录 | 模型 | 说明 |
|------|------|------|
| [`gemma-4/`](gemma-4/) | Gemma-4 E2B-it | 含 assistant 投机解码；vision + mm_proj + 7 段 LLM block |
| [`Qwen3-VL/`](Qwen3-VL/) | Qwen3-VL-2B-Instruct | vision_256 + 3 段 LLM block |
| [`InternVL3_5-1B/`](InternVL3_5-1B/) | InternVL3.5-1B | vision_448 + mm_proj + 3 段 LLM block |

每个模型目录结构一致：

| 子目录 / 文件 | 用途 |
|---------------|------|
| 根目录 `*.py` | PyTorch 模型加载、ONNX 导出、本地 demo |
| `test/` | ONNX 与 PyTorch 数值对齐、分模块单测 |
| `om/` | MDC 板端 OM 推理：`dump_om_inputs.py` → `run_om_pipeline.sh` → `parse_state.py` |
| `<model>-HF/` 或类似 | HuggingFace 格式权重目录（需自行下载，见 `.gitignore`） |

各模型详细文档：

- [gemma-4/README.md](gemma-4/README.md) — ONNX I/O 规格、投机解码
- [gemma-4/om/README.md](gemma-4/om/README.md) — Gemma-4 OM 流水线
- [Qwen3-VL/om/README.md](Qwen3-VL/om/README.md) — Qwen3-VL OM 流水线
- [InternVL3_5-1B/README.md](InternVL3_5-1B/README.md) — InternVL ONNX I/O
- [InternVL3_5-1B/om/README.md](InternVL3_5-1B/om/README.md) — InternVL OM 流水线

---

## 环境依赖

- Python 3.8+
- PyTorch、ONNX Runtime（本地开发与对齐）
- `tokenizers`（`parse_state.py` 解码，不依赖 transformers）
- 华为 MDC 环境 + `msame`（板端 OM 推理）

模型权重不在仓库内，请将 HF 权重放入各模型目录下的占位文件夹（如 `gemma-4/gemma-4-E2B-it/`），再运行导出与 dump 脚本。

---

## 典型工作流

### 1. 本地：ONNX 导出与对齐

```bash
cd gemma-4   # 或 Qwen3-VL / InternVL3_5-1B

# 导出 ONNX（按需）
python vision.py
python proj.py      # gemma-4 / InternVL 需要
python llm.py
python assist_model.py   # 仅 gemma-4

# 数值对齐
python test/onnx_torch_test_it.py
```

ONNX 默认输出到各模型目录下的 `./onnx_export/`。测试脚本中的样例图像路径为 `path/to/image.jpg`，请替换为本地实际路径。

### 2. 板端：OM 推理

以 Gemma-4 为例（其余模型见对应 `om/README.md`）：

```bash
cd gemma-4/om

# 本地：生成静态 bin
python dump_om_inputs.py --image path/to/image.jpg

# 拷到 MDC（替换 <mdc-host> 与板端路径）
scp -r dump ple_table user@<mdc-host>:/path/to/mdc/gemma4

# MDC 上推理
RUN_MSAME=1 bash run_om_pipeline.sh --dump-dir dump

# 拷回输出并解析
scp -r user@<mdc-host>:/path/to/mdc/gemma4/om_output .
python parse_state.py --output-dir om_output
```

批量推理：使用 `--image-dir path/to/images` dump，再通过 `--batch-root batch` 跑 pipeline 与 parse。

---

## 路径说明

代码与文档中的路径均为**占位符**，使用前请按实际环境替换：

| 占位符 | 含义 |
|--------|------|
| `path/to/image.jpg` | 本地测试图像 |
| `path/to/images/` | 批量图像目录 |
| `./onnx_export/` | ONNX 模型输出目录 |
| `user@<mdc-host>` | MDC 板端 SSH 地址 |
| `/path/to/mdc/<model>/` | MDC 上工程根目录 |

---

## 不上传的内容

见 [`.gitignore`](.gitignore)：模型权重（`*.safetensors` 等）、运行时 bin（`*.bin`）、HuggingFace 缓存、`transformers/` 源码等。
