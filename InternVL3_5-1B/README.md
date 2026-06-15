# InternVL3_5-1B

InternVL3.5-1B 图文理解模型的本地开发与 MDC 板端 OM 推理工程。整体分两条线：

1. **PyTorch / ONNX 开发与对齐**（根目录脚本 + `test/`）
2. **MDC OM 推理流水线**（`om/`，详见 [om/README.md](om/README.md)）

```
本地机：dump 静态 bin → MDC：OM 链式推理 → 本地机：parse 文本
```

---

## 目录概览

| 目录 | 用途 |
|------|------|
| `InternVL3_5-1B-HF/` | 模型权重（HF 格式）：vision + LLM + projector，`model.safetensors`、tokenizer、chat template |
| `om/` | MDC 板端 OM 推理全流程：dump 输入、跑 pipeline、解析输出 |
| `test/` | ONNX 与 PyTorch 数值对齐、分模块单测、端到端参考实现 |

### `om/` 子目录

| 目录 | 谁写 | 谁读 | 说明 |
|------|------|------|------|
| `dump/` | 本地 | MDC | 单张静态输入：`vision/` + `llm_preblock/`（preblock 从 `prompt_bin/` 复制） |
| `prompt_bin/` | 本地 | — | 默认 prompt bin（`What is shown in this image?`） |
| `batch/` | 本地 | MDC | 批量：`batch/<stem>/dump/` + `batch/<stem>/om_output/` |
| `om_output/` | MDC | 本地 | 单张统一输出：`state/`、`final_*`（`work/` 默认跑完删） |
| `om_export/` | — | MDC | `*.om` 模型目录（部署时放在 MDC，本地可无） |

---

## 根目录文件

| 文件 | 用途 |
|------|------|
| `demo.py` | HuggingFace 端到端 demo，验证图文对话 |
| `vision.py` | 导出 `vision_448` ONNX |
| `proj.py` | （多模态投影层）ONNX 导出 |
| `llm.py` | 导出llm ONNX |

导出 ONNX 后，用 `test/` 下的脚本与原始 PyTorch 输出做数值对比。

---

## ONNX 模型 I/O（fp16）

主链按顺序串联，**粗体**为各段主输出；带 * 的输入由上游 OM 输出拼装，不由 `dump/` 静态写出。

```
pixel_values [1,3,448,448]
  → vision_448 → [1,1025,1024]
  → mm_proj → [1,256,1024] (image_embeds)
  → llm_preblock → embeds + mask + cos/sin
  → llm_block1..3
  → lm_head → [1,1,151936] logits
```

### 公共常量

| 符号 | 值 | 说明 |
|------|-----|------|
| `L` | 512 | LLM 固定序列长（prompt pad 到 512） |
| `H` | 1024 | LLM hidden dim |
| `V` | 151936 | 词表大小 |
| `T_img` | 256 | 图像 soft token 数（`image_seq_length`） |
| `T_vis` | 1025 | vision encoder 输出 token 数 |

图像 token 在 `input_ids` 中占 `[4:260)`（256 个 image placeholder）。

### `vision_448.onnx`

| | 名称 | shape | dtype | 来源 |
|--|------|-------|-------|------|
| in | `pixel_values` | `[1, 3, 448, 448]` | fp16 | `dump/vision/` |
| out | **`last_hidden_state`** | `[1, 1025, 1024]` | fp16 | → mm_proj |

### `mm_proj.onnx`

| | 名称 | shape | dtype | 来源 |
|--|------|-------|-------|------|
| in | `vision_features` | `[1, 1025, 1024]` | fp16 | * vision 输出 |
| out | **`hidden_states`** | `[1, 256, 1024]` | fp16 | → llm_preblock `image_embeds` |

### `llm_preblock.onnx`

| | 名称 | shape | dtype | 来源 |
|--|------|-------|-------|------|
| in | `input_ids` | `[1, 512]` | int32 | `dump/llm_preblock/` |
| in | `image_embeds` | `[1, 256, 1024]` | fp16 | * mm_proj 输出 |
| in | `attention_mask` | `[1, 512]` | int32 | `dump/llm_preblock/` |
| in | `position_ids` | `[1, 512]` | int32 | `dump/llm_preblock/`（固定 arange） |
| out | `inputs_embeds_out` | `[1, 512, 1024]` | fp16 | → block_1 |
| out | `attention_mask_out` | `[1, 1, 512, 512]` | fp16 | 因果 mask |
| out | `cos` / `sin` | `[1, 512, 128]` | fp16 | RoPE |

### `llm_block*.onnx`（3 段）

| 模型文件 | 层范围 | 说明 |
|----------|--------|------|
| `llm_block1` | 0–9 | 10 层 |
| `llm_block2` | 10–19 | 10 层 |
| `llm_block3` | 20–27 + norm | 8 层 + RMSNorm |

每段输入（4 入 1 出）：

| | 名称 | shape | 来源 |
|--|------|-------|------|
| in | `hidden_states` | `[1, 512, 1024]` | preblock 或上游 block |
| in | `attention_mask` | `[1, 1, 512, 512]` | preblock |
| in | `cos` / `sin` | `[1, 512, 128]` | preblock |
| out | **`hidden_states_out`** | `[1, 512, 1024]` | → 下一 block |

### `lm_head.onnx`

| | 名称 | shape | dtype | 说明 |
|--|------|-------|-------|------|
| in | `hidden_states` | `[1, 1, 1024]` | fp16 | 从 b3 `hidden[:, cur_len-1]` 切片 |
| out | **`logits`** | `[1, 1, 151936]` | fp16 | decode 每步 argmax |

---

## `om/` 文件

| 文件 | 运行位置 | 用途 |
|------|----------|------|
| `dump_om_inputs.py` | 本地 | 图像/prompt 预处理 → 静态 `.bin` |
| `run_om_pipeline.sh` | MDC | OM 链式推理入口 |
| `om_bin_utils.py` | MDC | bin 拼装：prepare-*-input、sync-state、update-decode-state |
| `parse_state.py` | 本地 | 读取 MDC 拷回的 `om_output/`，解码生成文本 |
| `compare_om_onnx.py` | 本地 | 开发：OM vs ONNX 数值对比 |
| `run.sh` | MDC | msame 包装脚本，由 pipeline 动态生成 |
| `README.md` | — | OM 流水线详细说明、快速开始、scp 命令 |

---

## `test/` 文件

| 文件 | 用途 |
|------|------|
| `test.py` | ONNX 端到端 generate 参考 |
| `onnx_torch_test.py` | ONNX vs PyTorch 推理与误差对比 |
| `onnx_common.py` | 共享常量、路径、pad/position_ids 等工具 |
| `vision_test.py` | Vision ONNX vs `vision.py` PyTorch 对齐 |
| `mm_proj_test.py` | mm_proj ONNX 对齐 |
| `llm_test.py` | LLM preblock / block ONNX 对齐 |

---

## 模型权重目录（`InternVL3_5-1B-HF/`）

| 文件 | 说明 |
|------|------|
| `model.safetensors` | 模型权重 |
| `config.json` | 模型结构配置（vision + text） |
| `tokenizer.json` / `tokenizer_config.json` / `vocab.json` | 分词器 |
| `generation_config.json` | 生成参数 |
| `preprocessor_config.json` / `processor_config.json` | 图像/多模态处理器配置 |
| `chat_template.jinja` | 对话模板 |
| `examples/` | 示例图片/视频 |

---

## 典型工作流

### 开发 / 对齐（本地 GPU/CPU）

```bash
cd InternVL3_5-1B

# 导出 ONNX（按需）
python vision.py
python proj.py
python llm.py

# 数值对齐
python test/onnx_torch_test.py
python test/test.py
```

### 板端推理（本地 + MDC）

```bash
cd InternVL3_5-1B/om
# 详见 om/README.md
python dump_om_inputs.py --image path/image.jpg
# scp dump/ + 脚本 + om_export/ → MDC（首次）
# MDC: RUN_MSAME=1 bash run_om_pipeline.sh --dump-dir dump
# scp om_output/ ← MDC
python parse_state.py --output-dir om_output --dump-dir dump
```
