# Qwen3-VL OM 推理

**本地机**：`dump_om_inputs.py` 生成静态 bin → **MDC**：`run_om_pipeline.sh` 跑 OM → **本地机**：`parse_state.py` 解析文本。

```
vision_256 → llm_preblock → llm_block1..3 → lm_head
```

## 快速开始

### 单张

```bash
cd Qwen3-VL/om

# 本地：生成静态 bin（默认 image-only：只写 vision/，prompt 复用 prompt_bin/llm_preblock/）
python dump_om_inputs.py --image path/image.jpg
# 换 prompt：--mode full --prompt "..."
# python dump_om_inputs.py --mode full --prompt "What is shown in this image?" --image path/image.jpg

# 拷到 MDC（路径按实际板端目录调整）
scp -r dump user@<mdc-host>:/path/to/mdc/qwen3-vl

# MDC：在 om 目录下跑推理
RUN_MSAME=1 bash run_om_pipeline.sh --dump-dir dump

# 输出拷到本地
scp -r user@<mdc-host>:/path/to/mdc/qwen3-vl/om_output .

# 本地：解析
python parse_state.py --output-dir om_output --dump-dir dump
```

### 批量

```bash
cd Qwen3-VL/om

# 本地 dump → 默认 batch/<stem>/dump/
python dump_om_inputs.py --image-dir path/images

# 拷到 MDC
scp -r batch user@<mdc-host>:/path/to/mdc/qwen3-vl

# MDC
RUN_MSAME=1 bash run_om_pipeline.sh --batch-root batch

# 输出拷回本地 → om/batch/
scp -r user@<mdc-host>:/path/to/mdc/qwen3-vl/batch .

# 本地 parse（在 om/ 下）
python parse_state.py --batch-root batch
python parse_state.py --batch-root batch --stem images2
python parse_state.py --batch-root batch --write-response
```

单张等价：

```bash
python parse_state.py --output-dir batch/images2/om_output --dump-dir batch/images2/dump
```

---

## 数据目录

```
om/
├── dump/                    [本地写 → MDC 读] 单张静态 input（vision/）
│   └── vision/
├── prompt_bin/              [本地] 默认 prompt（What is shown in this image?）
│   └── llm_preblock/
├── batch/                   [本地写 → MDC 读] 批量：<stem>/dump/ + <stem>/om_output/
├── om_export/               [MDC] *.om 模型（vision_256, llm_*）
└── om_output/               [MDC → 本地 parse] 单张输出
    ├── work/
    ├── state/
    └── final_*
```

**链式输入**（不由 dump 写出）：`llm_preblock.image_embeds` ← vision OM `merged_hidden_states`；block1/2 还需 vision deepstack 输出。

**默认 prompt**：`prompt_bin/llm_preblock/` 已预生成（`What is shown in this image?`）。默认 `image-only` 只写 `vision/`，并将 `prompt_bin/llm_preblock/` **复制**到输出目录；换 prompt 时用 `--mode full` 更新 `prompt_bin/`。

---

## 脚本说明

| 文件 | 位置 | 用途 |
|------|------|------|
| `dump_om_inputs.py` | 本地 | 图像/prompt → `dump/` 或 `batch/` 静态 bin |
| `run_om_pipeline.sh` | MDC | OM 链式推理入口（含 batch） |
| `om_bin_utils.py` | MDC | bin 拼装：prepare-*-input、update-decode-state |
| `parse_state.py` | 本地 | 读 `om_output/final_*` 解码文本（tokenizers-only） |

---

## 启动命令速查

以下命令均在各自目录下执行：`cd .../Qwen3-VL/om`（本地）或 MDC 上的 `om/`。

### `dump_om_inputs.py`（本地）

```bash
# 单张 full（图 + prompt → dump/）
python dump_om_inputs.py --mode full --image path/image.jpg

# 单张（默认 image-only，从 prompt_bin/ 复制 preblock）
python dump_om_inputs.py --image path/new.jpg

# 批量（→ batch/<stem>/dump/，默认只写 vision/）
python dump_om_inputs.py --image-dir path/images

# 批量换图
python dump_om_inputs.py --image-dir path/images

# 批量跳过已存在项
python dump_om_inputs.py --image-dir path/images --skip-exist

# 自定义输出路径
python dump_om_inputs.py --image path/image.jpg --out-dir /tmp/my_dump
python dump_om_inputs.py --image-dir path/images --batch-root /tmp/my_batch
```

### `run_om_pipeline.sh`（MDC）

```bash
# 单张（默认 dump/ → om_output/，50 decode 步）
RUN_MSAME=1 bash run_om_pipeline.sh

# 单张，显式指定 input
RUN_MSAME=1 bash run_om_pipeline.sh --dump-dir dump

# 单张，指定 decode 步数（须用环境变量；不支持 --GEN_STEPS）
GEN_STEPS=20 RUN_MSAME=1 bash run_om_pipeline.sh --dump-dir dump

# 单张，positional 写法（path 为 dump 根目录）
RUN_MSAME=1 bash run_om_pipeline.sh dump 20

# 批量
RUN_MSAME=1 bash run_om_pipeline.sh --batch-root batch
RUN_MSAME=1 bash run_om_pipeline.sh batch          # 同上，自动识别 batch 根

# 批量 + decode 步数（第 2 个 positional）
RUN_MSAME=1 bash run_om_pipeline.sh batch 100
GEN_STEPS=100 RUN_MSAME=1 bash run_om_pipeline.sh --batch-root batch

# 只跑 prefill，不 decode
MODE=prefill_only RUN_MSAME=1 bash run_om_pipeline.sh --dump-dir dump

# 保留中间 work/；批量跳过已有结果
KEEP_INTERMEDIATE=1 RUN_MSAME=1 bash run_om_pipeline.sh --dump-dir dump
SKIP_EXIST=1 RUN_MSAME=1 bash run_om_pipeline.sh --batch-root batch

# 自定义输出目录
RUN_MSAME=1 bash run_om_pipeline.sh --dump-dir dump --output-dir om_output
```

`GEN_STEPS` 仅支持**环境变量**或 **positional**（`bash run_om_pipeline.sh <path> <步数>`），`--dump-dir` 与步数同用时请写 `GEN_STEPS=N`。

### `parse_state.py`（本地）

```bash
# 单张（默认 om_output/ + dump/）
python parse_state.py

# 单张，显式路径
python parse_state.py --output-dir om_output --dump-dir dump

# 批量（默认 batch/）
python parse_state.py --batch-root
python parse_state.py --batch-root batch

# 批量，只解析一张
python parse_state.py --batch-root --stem images2

# 批量，并写各 item/response.txt
python parse_state.py --batch-root --write-response

# 写出到指定文件
python parse_state.py --output-dir om_output --response-out out.txt

# 批量中单张等价写法
python parse_state.py \
  --output-dir batch/images2/om_output \
  --dump-dir batch/images2/dump
```

### 端到端复制块

```bash
# ── 本地 dump ──
python dump_om_inputs.py --image path/image.jpg
# scp -r dump root@MDC:.../qwen_256_256

# ── MDC 推理 ──
GEN_STEPS=20 RUN_MSAME=1 bash run_om_pipeline.sh --dump-dir dump
# scp -r root@MDC:.../om_output .

# ── 本地 parse ──
python parse_state.py --output-dir om_output --dump-dir dump
```

---

## `run_om_pipeline.sh` 参数

| 参数 / 环境变量 | 默认 | 说明 |
|----------------|------|------|
| `--dump-dir` / `DUMP_ROOT` | `dump/` | 静态 input 根目录 |
| `--output-dir` / `OUTPUT_ROOT` | `om_output/` | 统一输出根 |
| `--batch-root` | — | 批量模式（`<stem>/dump/` + `<stem>/om_output/`） |
| `MODE` | `full` | `prefill_only` / `full` / `decode` |
| `GEN_STEPS` | `50` | decode 步数；`GEN_STEPS=N` 环境变量，或 positional：`bash run_om_pipeline.sh batch 100`（不支持 `--GEN_STEPS`） |
| `RUN_MSAME` | `0` | MDC 上设为 `1` |
| `KEEP_INTERMEDIATE` | `0` | `1` 保留 `work/` |
| `SKIP_EXIST` | `0` | 批量：已有 `om_output/work/step_0000` 则跳过 |

---

## ONNX I/O 概要（fp16）

| 模型 | 主要输入 | 主要输出 |
|------|----------|----------|
| `vision_256` | `hidden_states` [256,1536] | `merged_hidden_states` [64,2048], deepstack ×3 |
| `llm_preblock` | `input_ids` [1,256], `image_embeds`*, `attention_mask`, `position_ids` [3,1,256] | `inputs_embeds_out`, `attention_mask_out`, `cos`/`sin` |
| `llm_block1` | hidden, mask, cos, sin, `ds_0` | `hidden_states_out` |
| `llm_block2` | hidden, mask, cos, sin, `ds_0`, `ds_1` | `hidden_states_out` |
| `llm_block3` | hidden, mask, cos, sin | `hidden_states_out` |
| `lm_head` | `hidden_states` [1,1,2048] | `logits` [1,1,151936] |

\* `image_embeds` 由 vision OM 输出在板端拼装，不在 `dump/` 中。
