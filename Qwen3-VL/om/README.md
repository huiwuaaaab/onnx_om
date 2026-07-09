# Qwen3-VL OM 推理

**本地机**：`dump_vision_om_inputs.py` + `dump_llm_preblock_inputs.py` 生成静态 bin → **MDC**：`run_om_pipeline.sh` 跑 OM → **本地机**：`parse_state.py` 解析文本。

```
vision_448 → llm_preblock → llm_block1..3 → lm_head
```

## 快速开始

### 单张

```bash
cd Qwen3-VL/om

# 本地：生成静态 bin
python dump_vision_om_inputs.py --image path/image.jpg
python dump_llm_preblock_inputs.py --prompt "What is shown in this image?"

# 拷到 MDC（路径按实际板端目录调整）
scp -r vision_bin prompt_bin user@<device-ip>:/opt/vlm/qwen3-vl/

# MDC：在 om 目录下跑推理（默认读 om/vision_bin + om/prompt_bin）
RUN_MSAME=1 bash run_om_pipeline.sh

# 输出拷到本地
scp -r user@<device-ip>:/opt/vlm/qwen3-vl/om_output .

# 本地：解析
python parse_state.py --output-dir om_output --dump-dir prompt_bin
```

### 批量

```bash
cd Qwen3-VL/om

# 本地：每张图写 batch/<stem>/vision_bin/；prompt 共用 om/prompt_bin/（只需 dump 一次）
python dump_vision_om_inputs.py --image-dir path/images
python dump_llm_preblock_inputs.py --prompt "What is shown in this image?"

# 拷到 MDC
scp -r batch prompt_bin user@<device-ip>:/opt/vlm/qwen3-vl/

# MDC：串行 batch
RUN_MSAME=1 bash run_om_pipeline.sh --batch-root batch

# MDC：多图流水线
RUN_MSAME=1 bash run_om_pipeline_pipe.sh ./batch

# 输出拷回本地 → om/batch/
scp -r user@<device-ip>:/opt/vlm/qwen3-vl/batch .

# 本地 parse（在 om/ 下）
python parse_state.py --batch-root batch
python parse_state.py --batch-root batch --stem images2
python parse_state.py --batch-root batch --write-response
```

单张等价：

```bash
python parse_state.py \
  --output-dir batch/images2/om_output \
  --dump-dir prompt_bin
```

---

## 数据目录

```
om/
├── vision_bin/              [本地写 → MDC 读] 单张图像 input
│   └── pixel_values.bin
├── prompt_bin/              [本地写 → MDC 读] 文本 preblock 静态 bin（扁平目录）
│   ├── input_ids.bin
│   ├── attention_mask.bin
│   └── position_ids.bin
├── batch/                   [本地写 → MDC 读] 批量：<stem>/vision_bin/ + <stem>/om_output/
│   └── <stem>/
│       ├── vision_bin/
│       └── om_output/
├── om_export/               [MDC] *.om 模型（vision_448, llm_*）
└── om_output/               [MDC → 本地 parse] 单张输出
    ├── work/
    ├── state/
    └── final_*
```

**链式输入**（不由 dump 写出）：`llm_preblock.image_embeds` ← vision OM `merged_hidden_states`；block1/2 还需 vision deepstack 输出。

**批量 prompt**：`dump_llm_preblock_inputs.py` 默认写到 `prompt_bin/`，所有 batch item 共用；换 prompt 时重新 dump 一次即可。

**旧布局兼容**：`dump/vision/` + `dump/llm_preblock/` 仍可通过 `--dump-dir dump` 使用。

---

## 脚本说明

| 文件 | 位置 | 用途 |
|------|------|------|
| `dump_vision_om_inputs.py` | 本地 | 图像 → `vision_bin/` 或 `batch/<stem>/vision_bin/` |
| `dump_llm_preblock_inputs.py` | 本地 | prompt → `prompt_bin/` 静态 preblock bin |
| `run_om_pipeline.sh` | MDC | **串行**入口（单张 / batch 逐张） |
| `run_om_pipeline_pipe.sh` | MDC | **流水线**入口（6 worker，OM 常驻） |
| `pipeline/` | MDC | 实现：`serial.sh` / `pipe.sh` / `worker.sh` / `paths.sh` |
| `msame` | MDC | 推理二进制；默认 `./msame`，可用 `MSAME_BIN` 指定 |
| `om_bin_utils.py` | MDC | bin 拼装：prepare-*-input、update-decode-state |
| `parse_state.py` | 本地 | 读 `om_output/final_*` 解码文本（tokenizer.json） |

---

## 启动命令速查

以下命令均在各自目录下执行：`cd .../Qwen3-VL/om`（本地）或 MDC 上的 `om/`。

### `dump_vision_om_inputs.py`（本地）

```bash
# 单张 → vision_bin/
python dump_vision_om_inputs.py --image path/image.jpg

# 批量 → batch/<stem>/vision_bin/
python dump_vision_om_inputs.py --image-dir path/images

# 跳过已存在项
python dump_vision_om_inputs.py --image-dir path/images --skip-exist

# 自定义输出
python dump_vision_om_inputs.py --image path/image.jpg --out-dir /tmp/vision_bin
python dump_vision_om_inputs.py --image-dir path/images --batch-root /tmp/my_batch

# profile：256_256 | 448_512
python dump_vision_om_inputs.py --profile 448_512 --image path/image.jpg
```

### `dump_llm_preblock_inputs.py`（本地）

```bash
# 默认 prompt → prompt_bin/
python dump_llm_preblock_inputs.py --prompt "What is shown in this image?"

# 从文件读 prompt
python dump_llm_preblock_inputs.py --prompt-file prompt.txt

# 自定义输出目录
python dump_llm_preblock_inputs.py --prompt "..." --out-dir /tmp/prompt_bin

# profile：256_256 | 448_512
python dump_llm_preblock_inputs.py --profile 448_512 --prompt "..."
```

### `run_om_pipeline.sh` / `run_om_pipeline_pipe.sh`（MDC）

| 入口 | 行为 |
|------|------|
| `run_om_pipeline.sh` | 单张或 batch **串行**，每步 inline 调 msame |
| `run_om_pipeline_pipe.sh` | batch **多图 OM 流水线**，6 worker + **OM 常驻**（默认） |

```bash
# 单张（默认 vision_bin/ + prompt_bin/ → om_output/）
RUN_MSAME=1 bash run_om_pipeline.sh

# 单张，显式指定 input
RUN_MSAME=1 bash run_om_pipeline.sh --vision-bin vision_bin --prompt-bin prompt_bin

# 单张，decode 步数
GEN_STEPS=20 RUN_MSAME=1 bash run_om_pipeline.sh
RUN_MSAME=1 bash run_om_pipeline.sh . 20

# 批量（串行）
RUN_MSAME=1 bash run_om_pipeline.sh --batch-root batch
RUN_MSAME=1 bash run_om_pipeline.sh batch 100

# 批量（多图流水线）
RUN_MSAME=1 bash run_om_pipeline_pipe.sh ./batch
RUN_MSAME=1 bash run_om_pipeline_pipe.sh ./batch 50

# 流水线但不常驻 OM（调试用）
OM_RESIDENT=0 RUN_MSAME=1 bash run_om_pipeline_pipe.sh ./batch

# 只跑 prefill
MODE=prefill_only RUN_MSAME=1 bash run_om_pipeline.sh

# 保留中间 work/；批量跳过已有结果
KEEP_INTERMEDIATE=1 RUN_MSAME=1 bash run_om_pipeline.sh
SKIP_EXIST=1 RUN_MSAME=1 bash run_om_pipeline.sh --batch-root batch

# 旧布局
RUN_MSAME=1 bash run_om_pipeline.sh --dump-dir dump
```

`GEN_STEPS` 支持**环境变量**或 **positional**（`bash run_om_pipeline.sh batch 100`）。

Pipe 模式 worker 日志：`<batch>/.om_pipe/<vision|preblock|block*>/worker.log`

**OM 常驻**（`run_om_pipeline_pipe.sh` 默认开启）：每个 worker 用 pyACL 预加载对应 `.om`，6 个 OM 同时驻留。

Worker 启动时会 **source 与 msame 相同的 CANN 环境**（`pipeline/acl_env.sh` → `/var/set_env.sh` 等），否则 `import acl` 失败 → **msame-fallback**（每个 job 重新 load OM，vision 极慢）。

```bash
# MDC 上诊断 pyACL（必跑）
bash pipeline/diagnose_acl.sh

# 若诊断通过，按输出设置 ACL_PYTHON 后跑 pipe
export ACL_PYTHON=/path/from/diagnose
RUN_MSAME=1 bash run_om_pipeline_pipe.sh ./batch

# worker.log 应出现 resident=acl，不是 msame-fallback
grep resident= batch/.om_pipe/vision/worker.log
```

**若 diagnose 报 pyACL NOT available**：用 **C++ 常驻 daemon**（推荐，无需 Python acl）：

```bash
cd pipeline/om_resident_cpp && bash build.sh
cd ../..
RUN_MSAME=1 bash run_om_pipeline_pipe.sh ./batch
grep resident= batch/.om_pipe/vision/worker.log   # 应为 resident=cpp
```

或改用串行（无加速，但简单）：

```bash
RUN_MSAME=1 bash run_om_pipeline.sh --batch-root batch
```

| 环境变量 | 默认 | 说明 |
|---------|------|------|
| `OM_RESIDENT` | `1`（pipe 入口） | `0` 改用 per-job msame |
| `OM_RESIDENT_ACL` | `1` | `0` 强制 msame fallback |
| `ASCEND_ENV_SH` | 自动搜索 | CANN `set_env.sh` 路径 |
| `ACL_PYTHON` | 自动检测 | 能 `import acl` 的 python |

### `parse_state.py`（本地）

```bash
# 单张（默认 om_output/ + prompt_bin/）
python parse_state.py

# 单张，显式路径
python parse_state.py --output-dir om_output --dump-dir prompt_bin

# 批量
python parse_state.py --batch-root batch
python parse_state.py --batch-root batch --stem images2
python parse_state.py --batch-root batch --write-response

# 写出到指定文件
python parse_state.py --output-dir om_output --response-out out.txt
```

### 端到端复制块

```bash
# ── 本地 dump ──
python dump_vision_om_inputs.py --image path/image.jpg
python dump_llm_preblock_inputs.py --prompt "What is shown in this image?"
# scp -r vision_bin prompt_bin root@MDC:.../qwen_448_512

# ── MDC 推理 ──
GEN_STEPS=20 RUN_MSAME=1 bash run_om_pipeline.sh
# scp -r root@MDC:.../om_output .

# ── 本地 parse ──
python parse_state.py --output-dir om_output --dump-dir prompt_bin
```

---

## `run_om_pipeline.sh` 参数

| 参数 / 环境变量 | 默认 | 说明 |
|----------------|------|------|
| `--vision-bin` | `vision_bin/` | 图像 input 目录 |
| `--prompt-bin` | `prompt_bin/` | preblock 静态 bin 目录 |
| `--dump-dir` | — | 旧布局兼容（自动识别 `dump/` 或 `vision_bin`+`prompt_bin`） |
| `--output-dir` / `OUTPUT_ROOT` | `om_output/` | 统一输出根 |
| `--batch-root` | — | 批量模式（`<stem>/vision_bin/` + `<stem>/om_output/`） |
| `SHARED_PROMPT_BIN` | `prompt_bin/` | batch 共用 prompt 目录 |
| `MODE` | `full` | `prefill_only` / `full` / `decode` |
| `GEN_STEPS` | `50` | decode 最大步数（遇 EOS 会提前停） |
| `STOP_ON_EOS` | `1` | `0` 时不因 EOS 提前停止 |
| `RUN_MSAME` | `0` | MDC 上设为 `1` |
| `KEEP_INTERMEDIATE` | `0` | `1` 保留 `work/` |
| `SKIP_EXIST` | `0` | 批量：已有 `om_output/work/step_0000` 则跳过 |

---

## ONNX I/O 概要（fp16，profile 448_512）

| 模型 | 主要输入 | 主要输出 |
|------|----------|----------|
| `vision_448` | `pixel_values` [784,1536] | `merged_hidden_states` [196,2048], deepstack ×3 |
| `llm_preblock` | `input_ids` [1,512], `image_embeds`*, `attention_mask`, `position_ids` [3,1,512] | `inputs_embeds_out`, `attention_mask_out`, `cos`/`sin` |
| `llm_block1` | hidden, mask, cos, sin, `ds_0` | `hidden_states_out` |
| `llm_block2` | hidden, mask, cos, sin, `ds_0`, `ds_1` | `hidden_states_out` |
| `llm_block3` | hidden, mask, cos, sin | `hidden_states_out` |
| `lm_head` | `hidden_states` [1,1,2048] | `logits` [1,1,151936] |

\* `image_embeds` 由 vision OM 输出在板端拼装，不在 `prompt_bin/` 中。

### pipe 报 `worker vision failed`

1. 看 worker 日志：`batch/.om_pipe/vision/worker.log`（新版失败时 orchestrator 会 tail 最后 20 行）。
2. 常见原因：`msprof`/`msame` 需在 `om/` 目录下执行。请同步最新 `pipeline/worker.sh`。
3. 单张先验证：`RUN_MSAME=1 bash run_om_pipeline.sh --vision-bin batch/<stem>/vision_bin --prompt-bin prompt_bin --output-dir batch/<stem>/om_output`

---

## 常见问题

### batch 全部 `SKIP ... missing dump`

1. **脚本未更新**：请同步 `run_om_pipeline.sh` 和 `pipeline/` 目录到 MDC。
2. **只有 meta、没有 bin**：`dump/` 或 `prompt_bin/` 里若只有 `meta.json` 没有 `.bin`，需要先在本机跑 dump：
   ```bash
   python dump_vision_om_inputs.py --image-dir path/images   # → batch/<stem>/vision_bin/
   python dump_llm_preblock_inputs.py --prompt "What is shown in this image?"  # → prompt_bin/*.bin
   ```
3. **拷到 MDC 时漏了 `prompt_bin`**：batch 各 item 只有 `vision_bin/`，prompt 共用 `om/prompt_bin/input_ids.bin` 等，需与 `batch/` 一起 scp。

同步后重跑应看到具体原因，例如 `SKIP images1: missing vision: ...` 或 `missing prompt: ...`。

### `msame` 部署

pipeline 通过 `pmupload ${MSAME_BIN} --model ...` 调用 msame。

- **`msame`**：bash 启动器，自动 source CANN 环境并转发到 ELF 二进制
- 若你拷贝的是 **ELF 二进制**，可直接：`export MSAME_BIN=/path/to/msame_elf`
- 若用启动器，ELF 放同目录 **`msame.bin`**，或 `export MSAME_REAL=/path/to/msame_elf`

```bash
chmod +x ./msame
./msame --help

# 或直接指定 ELF
MSAME_BIN=/path/to/msame RUN_MSAME=1 bash run_om_pipeline_pipe.sh ./batch
```
