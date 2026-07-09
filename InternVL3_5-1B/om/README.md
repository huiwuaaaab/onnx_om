# InternVL3_5-1B OM 推理

**本地机**：`dump_vision_om_inputs.py` + `dump_llm_preblock_inputs.py` 生成静态 bin → **MDC**：`run_om_pipeline.sh` 跑 OM → **本地机**：`parse_state.py` 解析文本。

```
vision_448 → mm_proj → llm_preblock → llm_block1..3 → lm_head
```

## 快速开始

### 单张

```bash
cd InternVL3_5-1B/om

# 本地：vision + prompt 分开 dump
python dump_vision_om_inputs.py --image path/image.jpg
python dump_llm_preblock_inputs.py --prompt "What is shown in this image?"

# 拷到 MDC（vision_bin + prompt_bin + 脚本 + om_export/）
scp -r vision_bin prompt_bin user@<device-ip>:/opt/vlm/internvl

# MDC
RUN_MSAME=1 bash run_om_pipeline.sh

# 输出拷到本地
scp -r user@<device-ip>:/opt/vlm/internvl/om_output .

# 本地 parse
python parse_state.py --output-dir om_output --dump-dir prompt_bin
```

### 批量

```bash
cd InternVL3_5-1B/om

# 本地 dump
python dump_vision_om_inputs.py --image-dir path/images   # → batch/<stem>/vision_bin/
python dump_llm_preblock_inputs.py --prompt "What is shown in this image?"  # → prompt_bin/

# 拷到 MDC
scp -r batch prompt_bin user@<device-ip>:/opt/vlm/internvl

# MDC — 串行 batch
RUN_MSAME=1 bash run_om_pipeline.sh --batch-root batch

# MDC — 流水线（7 OM 常驻）
RUN_MSAME=1 bash run_om_pipeline_pipe.sh ./batch

# 输出拷回 → om/batch/
scp -r user@<device-ip>:/opt/vlm/internvl/batch .
python parse_state.py --batch-root batch --write-response
```

---

## 数据目录

```
om/
├── vision_bin/              [本地] 单张 vision：pixel_values.bin
├── prompt_bin/              [本地] 共享 prompt：input_ids.bin 等
├── batch/                   [本地→MDC] 批量：<stem>/vision_bin/ + <stem>/om_output/
├── pipeline/                [MDC] serial.sh / pipe.sh / worker.sh / paths.sh
├── om_export/               [MDC] *.om（vision_448, mm_proj, llm_*）
└── om_output/               [MDC→本地] final_* + state/
```

**新布局 vs 旧 `dump/`**

| 用途 | 旧 | 新 |
|------|-----|-----|
| Vision | `dump/vision/pixel_values.bin` | `vision_bin/pixel_values.bin` |
| Preblock | `dump/llm_preblock/*.bin` | `prompt_bin/*.bin`（批量共用 `om/prompt_bin/`） |

legacy `dump/` 路径仍被 `pipeline/paths.sh` 兼容。

---

## 脚本说明

| 文件 | 位置 | 用途 |
|------|------|------|
| `dump_vision_om_inputs.py` | 本地 | 图像 → `vision_bin/pixel_values.bin` |
| `dump_llm_preblock_inputs.py` | 本地 | prompt → `prompt_bin/*.bin` |
| `run_om_pipeline.sh` | MDC | **串行**入口 |
| `run_om_pipeline_pipe.sh` | MDC | **流水线**入口（7 worker，OM 常驻 ctypes） |
| `pipeline/` | MDC | `serial.sh` / `pipe.sh` / `worker*.sh` / `acl_ctypes.py` |
| `om_bin_utils.py` | MDC | bin 拼装、decode 状态更新（含 EOS） |
| `parse_state.py` | 本地 | JSON vocab 解码（无 tokenizers） |
| `msame` | MDC | 推理二进制 |

---

## `run_om_pipeline.sh` / `run_om_pipeline_pipe.sh`

```bash
# 单张（默认 om/vision_bin + om/prompt_bin）
RUN_MSAME=1 bash run_om_pipeline.sh

# 批量串行
RUN_MSAME=1 bash run_om_pipeline.sh --batch-root batch

# 批量流水线（7 worker + OM 常驻 ctypes）
RUN_MSAME=1 bash run_om_pipeline_pipe.sh ./batch
RUN_MSAME=1 bash run_om_pipeline_pipe.sh ./batch 50

# decode 步数
GEN_STEPS=100 RUN_MSAME=1 bash run_om_pipeline.sh batch 100

# 遇 EOS 提前停止（默认 STOP_ON_EOS=1）
STOP_ON_EOS=0 RUN_MSAME=1 bash run_om_pipeline.sh batch
```

| 参数 / 环境变量 | 默认 | 说明 |
|----------------|------|------|
| `--vision-bin` / `--prompt-bin` | `vision_bin/` / `prompt_bin/` | 单张静态 input |
| `--batch-root` | — | 批量（每 item 需 `vision_bin/`） |
| `MODE` | `full` | `prefill_only` / `full` / `decode` |
| `GEN_STEPS` | `50` | decode 步数 |
| `STOP_ON_EOS` | `1` | decode 遇 token 151645 停止 |
| `RUN_MSAME` | `0` | MDC 上设为 `1` |
| `OM_RESIDENT` | `1`（pipe 入口） | `0` 改用 per-job msame |
| `SKIP_EXIST` | `0` | 批量跳过已有 output |

Pipe workers：`batch/.om_pipe/<vision|mm_proj|preblock|block*>/worker.log`  
确认常驻：`grep resident= batch/.om_pipe/*/worker.log` → `resident=ctypes`

### `msame`

默认 `MSAME_BIN=./msame`，或 `export MSAME_BIN=/path/to/msame_elf`

---

## `parse_state.py`

```bash
python parse_state.py --output-dir om_output --dump-dir prompt_bin
python parse_state.py --batch-root batch --write-response
```

`--dump-dir` 指向 `prompt_bin/`（读 `meta.json` 中的 `seq_len` 确定 prefill 长度）。

---

## ONNX I/O 概要（fp16）

| 模型 | 主要输入 | 主要输出 |
|------|----------|----------|
| `vision_448` | `pixel_values` [1,3,448,448] | `last_hidden_state` [1,1025,1024] |
| `mm_proj` | `vision_features` [1,1025,1024] | `hidden_states` [1,256,1024] |
| `llm_preblock` | `input_ids` [1,512], `image_embeds`*, mask, position_ids | `inputs_embeds_out`, `attention_mask_out`, cos/sin |
| `llm_block1..3` | hidden [1,512,1024], mask, cos, sin | `hidden_states_out` |
| `lm_head` | `hidden_states` [1,1,1024] | `logits` [1,1,151936] |

\* `image_embeds` 由 mm_proj OM 输出在板端拼装。
