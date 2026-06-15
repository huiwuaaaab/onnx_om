# Gemma-4 E2B-it-assistant

Gemma-4 图文理解模型的本地开发与 MDC 板端 OM 推理工程。整体分两条线：

1. **PyTorch / ONNX 开发与对齐**（根目录脚本 + `test/`）
2. **MDC OM 推理流水线**（`om/`，详见 [om/README.md](om/README.md)）

```
本地机：dump 静态 bin → MDC：OM 链式推理 → 本地机：parse 文本
```

---

## 目录概览

| 目录 | 用途 |
|------|------|
| `gemma-4-E2B-it/` | 主模型权重（HF 格式）：vision + LLM，`model.safetensors`、tokenizer、chat template |
| `gemma-4-E2B-it-assistant/` | 投机解码（speculative decoding）用的 assistant 模型权重 |
| `om/` | MDC 板端 OM 推理全流程：dump 输入、跑 pipeline、解析输出 |
| `test/` | ONNX 与 PyTorch 数值对齐、分模块单测、端到端参考实现 |

### `om/` 子目录

| 目录 | 谁写 | 谁读 | 说明 |
|------|------|------|------|
| `dump/` | 本地 | MDC | 单张静态输入：`vision/` + `llm_preblock/`（preblock 从 `prompt_bin/` 复制） |
| `prompt_bin/` | 本地 | — | 默认 prompt bin（`What is shown in this image?`） |
| `ple_table/` | 本地 | MDC | 全工程共享 PLE 查表，与图像无关，scp 一次即可 |
| `batch/` | 本地 | MDC | 批量：`batch/<stem>/dump/` + `batch/<stem>/om_output/` |
| `om_output/` | MDC | 本地 | 单张统一输出：`state/`、`final_*`（`work/` 默认跑完删） |
| `om_export/` | — | MDC | `*.om` 模型目录（部署时放在 MDC，本地可无） |

---

## 根目录文件

| 文件 | 用途 |
|------|------|
| `demo.py` | HuggingFace 端到端 demo：主模型 + assistant 投机解码，验证图文对话 |
| `vision.py` | 导出 vision ONNX |
| `proj.py` | （多模态投影层）ONNX 导出 |
| `llm.py` | 导出 llm ONNX |
| `assist_model.py` | 导出 assistant ONNX |

导出 ONNX 后，用 `test/` 下的脚本与原始 PyTorch 输出做数值对比。

---

## ONNX 模型 I/O（fp16）

主链按顺序串联，**粗体**为各段主输出；带 * 的输入由上游 OM 输出拼装，不由 `dump/` 静态写出。

```
pixel_values + image_position_ids
  → vision → [1,256,768]
  → mm_proj → [1,256,1536] (image_embeds)
  → llm_preblock → masks/rope/embeds
  → llm_block_1..3 (无 KV) → llm_block_4..7 (共享 KV)
  → lm_head → [1,1,262144] logits
  → [可选] assistant → projected_state + logits
```

### 公共常量

| 符号 | 值 | 说明 |
|------|-----|------|
| `L` | 512 | LLM 固定序列长（prompt pad 到 512） |
| `H` | 1536 | 主模型 hidden dim |
| `PLE` | 35 × 256 | 35 层，每层 per-layer embedding 256 维 |
| `V` | 262144 | 词表大小 |
| `T_img` | 256 | vision pool 后 soft token 数（768×768 图典型值） |
| `P_max` | 2520 | patch 上限 = `max_soft_tokens(280) × pooling²(9)` |

图像 token 在 `input_ids` 中占 `[5:261)`（256 个 image placeholder）。

### `vision.onnx`

| | 名称 | shape | dtype | 来源 |
|--|------|-------|-------|------|
| in | `pixel_values` | `[1, 2520, 768]` | fp16 | `dump/vision/` |
| in | `image_position_ids` | `[1, 2520, 2]` | int32 | `dump/vision/` |
| out | **`hidden_states`** | `[1, 256, 768]` | fp16 | → mm_proj |

`pixel_values` 每行 768 = `patch_size² × 3`（16×16×3）；不足 `P_max` 的 patch 在 dim=1 pad。

### `mm_proj.onnx`

| | 名称 | shape | dtype | 来源 |
|--|------|-------|-------|------|
| in | `vision_features` | `[1, 256, 768]` | fp16 | * vision 输出 |
| out | **`hidden_states`** | `[1, 256, 1536]` | fp16 | → llm_preblock `image_embeds` |

### `llm_preblock.onnx`

| | 名称 | shape | dtype | 来源 |
|--|------|-------|-------|------|
| in | `input_ids` | `[1, 512]` | int32 | `dump/llm_preblock/` |
| in | `image_embeds` | `[1, 256, 1536]` | fp16 | * mm_proj 输出 |
| in | `attention_mask` | `[1, 512]` | int32 | `dump/llm_preblock/` |
| in | `per_layer_inputs` | `[1, 512, 35, 256]` | fp16 | `dump/llm_preblock/` |
| in | `position_ids` | `[1, 512]` | int32 | `dump/llm_preblock/`（固定） |
| out | `inputs_embeds_out` | `[1, 512, 1536]` | fp16 | → block_1 |
| out | `per_layer_inputs_out` | `[1, 512, 35, 256]` | fp16 | 供各 block 切层 |
| out | `full_mask` / `sliding_mask` | `[1, 1, 512, 512]` | fp16 | 注意力 mask |
| out | `cos_full` / `sin_full` | `[1, 512, 512]` | fp16 | RoPE |
| out | `cos_sliding` / `sin_sliding` | `[1, 512, 256]` | fp16 | sliding RoPE |

### `llm_block_*.onnx`（7 段，每段 5 层）

| 模型文件 | 层范围 | 共享 KV |
|----------|--------|---------|
| `llm_block_0_5` | 0–4 | 否 |
| `llm_block_5_10` | 5–9 | 否 |
| `llm_block_10_15` | 10–14 | 否 |
| `llm_block_15_20` | 15–19 | 是 |
| `llm_block_20_25` | 20–24 | 是 |
| `llm_block_25_30` | 25–29 | 是 |
| `llm_block_30_35` | 30–34 + norm | 是 |

**block 1–3**（8 入 1 出）：

| | 名称 | shape | 来源 |
|--|------|-------|------|
| in | `inputs_embeds` | `[1, 512, 1536]` | preblock 或上游 block |
| in | `full_mask` / `sliding_mask` | `[1, 1, 512, 512]` | preblock |
| in | `cos_full` / `sin_full` | `[1, 512, 512]` | preblock |
| in | `cos_slide` / `sin_slide` | `[1, 512, 256]` | preblock |
| in | `per_layer_input` | `[1, 512, 5, 256]` | preblock PLE 切 5 层 |
| out | **`hidden_states_out`** | `[1, 512, 1536]` | → 下一 block |

**block 4–7** 额外输入/输出 KV cache：

| | 名称 | shape |
|--|------|-------|
| in | `full_k` / `full_v` | `[1, 1, 512, 512]` |
| in | `slide_k` / `slide_v` | `[1, 1, 512, 256]` |
| out | `out_full_k` / `out_full_v` | `[1, 1, 512, 512]` |
| out | `out_slide_k` / `out_slide_v` | `[1, 1, 512, 256]` |

block 4+ 的 KV 初值来自 block 3 的 `out_*`。

### `lm_head.onnx`

| | 名称 | shape | dtype | 说明 |
|--|------|-------|-------|------|
| in | `hidden_states` | `[1, 1, 1536]` | fp16 | 从 b7 `hidden[:, cur_len-1]` 切片 |
| out | **`logits`** | `[1, 1, 262144]` | fp16 | decode 每步 argmax |

### `assistant.onnx`（投机解码，可选）

| | 名称 | shape | dtype | 来源 |
|--|------|-------|-------|------|
| in | `last_token_id` | `[1, 1]` | int32 | 上一步 token |
| in | `last_hidden` | `[1, 1, 1536]` | fp16 | 主链 b7 最后 hidden |
| in | `attention_mask` | `[1, 512]` | int32 | state |
| in | `position_ids` | `[1, 1]` | int32 | 当前位置 |
| in | `full_k` / `full_v` | `[1, 1, 512, 512]` | fp16 | 主链 KV |
| in | `slide_k` / `slide_v` | `[1, 1, 512, 256]` | fp16 | 主链 KV |
| out | `projected_state` | `[1, 1, 1536]` | fp16 | 回注主链 |
| out | `logits` | `[1, 1, 262144]` | fp16 | draft token |
| out | `hidden_states_out` | `[1, 1, 256]` | fp16 | assistant hidden |

---

## `om/` 文件

| 文件 | 运行位置 | 用途 |
|------|----------|------|
| `dump_om_inputs.py` | 本地 | 图像/prompt 预处理 → 静态 `.bin`（不依赖 transformers） |
| `run_om_pipeline.sh` | MDC | OM 链式推理入口：vision → mm_proj → LLM blocks → lm_head [→ assistant] |
| `om_bin_utils.py` | MDC | 主链 bin 拼装/解析：prepare-*-input、sync state、argmax 等 |
| `om_bin_utils_it_assistant.py` | MDC | Assistant 投机解码相关 bin 拼装与解析 |
| `parse_state.py` | 本地 | 读取 MDC 拷回的 `om_output/`，解码生成文本 |
| `run.sh` | MDC | msame 包装脚本，由 pipeline 动态生成 |
| `README.md` | — | OM 流水线详细说明、快速开始、scp 命令 |

---

## `test/` 文件

| 文件 | 用途 |
|------|------|
| `onnx_torch_test_it.py` | 主链 ONNX 端到端参考, 同时包含onnx和torch推理与误差计算 |
| `onnx_torch_test.py` | 主链 + assistant 投机解码 ONNX 端到端参考, 同时包含onnx和torch推理与误差计算 |
| `test.py` | 仅onnx推理 |
| `vision_test.py` | Vision ONNX vs `vision.py` PyTorch 对齐 |
| `mm_proj_test.py` | mm_proj ONNX 对齐 |
| `llm_test.py` | LLM preblock / block ONNX 对齐 |
| `assistant_test.py` | Assistant 模型 ONNX 对齐 |

---

## 模型权重目录内容

`gemma-4-E2B-it/` 与 `gemma-4-E2B-it-assistant/` 均为 HuggingFace 格式：

| 文件 | 说明 |
|------|------|
| `model.safetensors` | 模型权重 |
| `config.json` | 模型结构配置 |
| `tokenizer.json` / `tokenizer_config.json` | 分词器 |
| `generation_config.json` | 生成参数 |
| `processor_config.json` | 仅主模型：图像处理器配置 |
| `chat_template.jinja` | 仅主模型：对话模板 |
| `.cache/` | HuggingFace 下载缓存，可忽略 |

---

## 典型工作流

### 开发 / 对齐（本地 GPU/CPU）

```bash
cd gemma-4

# 导出 ONNX（按需）
python vision.py
python proj.py
python llm.py
python assist_model.py

# 数值对齐
python test/onnx_torch_test_it.py      # 主链
python test/onnx_torch_test.py         # 主链 + assistant
```

### 板端推理（本地 + MDC）

```bash
cd gemma-4/om
# 详见 om/README.md
python dump_om_inputs.py --image path/image.jpg
# scp dump/ ple_table/ → MDC
# MDC: RUN_MSAME=1 bash run_om_pipeline.sh --dump-dir dump
# scp om_output/ ← MDC
python parse_state.py --output-dir om_output
```
