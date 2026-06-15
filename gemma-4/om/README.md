# Gemma-4 OM 推理

**本地机**：`dump_om_inputs.py` 生成静态 bin → **MDC**：`run_om_pipeline.sh` 跑 OM → **本地机**：`parse_state.py` 解析文本。


## 快速开始

### 单张

```bash
cd gemma-4/om

# 本地：生成静态 bin（默认 image-only：只写 vision/，prompt 复用 prompt_bin/llm_preblock/）
python dump_om_inputs.py --image path/image.jpg
# 换 prompt 或 num_soft_tokens 变化时：--mode full --prompt-text "..."
# python dump_om_inputs.py --mode full --prompt-text "What is shown in this image?" --image path/image.jpg

# 拷到 MDC：dump/ + ple_table/（PLE 全工程一份，只需 scp 一次）
# 首次运行
scp -r dump ple_table user@<mdc-host>:/path/to/mdc/gemma4
# 非首次
scp -r dump user@<mdc-host>:/path/to/mdc/gemma4

# MDC：OM 推理
RUN_MSAME=1 bash run_om_pipeline.sh --dump-dir dump

# 输出拷到本地（含 state/、final_*；work/ 默认跑完已删）
scp -r user@<mdc-host>:/path/to/mdc/gemma4/om_output .

# 本地：解析
python parse_state.py --output-dir om_output
```

### 批量

```bash
cd gemma-4/om

# 本地 dump → 默认 batch/<stem>/dump/（只写 vision/，prompt 从 prompt_bin/ 复制）
python dump_om_inputs.py --image-dir path/images
# 换 prompt：--mode full 更新 prompt_bin/llm_preblock/，或指定 --prompt-dir

# 拷到 MDC：batch/ + ple_table/
# 首次运行
scp -r batch ple_table user@<mdc-host>:/path/to/mdc/gemma4
# 非首次
scp -r batch user@<mdc-host>:/path/to/mdc/gemma4

# MDC
RUN_MSAME=1 bash run_om_pipeline.sh --batch-root batch

# 输出拷到本地 → om/batch/
scp -r user@<mdc-host>:/path/to/mdc/gemma4/batch .
python parse_state.py --batch-root batch
python parse_state.py --batch-root batch --stem images2
python parse_state.py --batch-root batch --write-response
```

单张等价：

```bash
python parse_state.py --output-dir batch/<stem>/om_output --dump-dir batch/<stem>/dump
```

---

## 数据目录

```
om/
├── dump/                          [本地写 → MDC 读] 单张静态 input（vision/）
│   └── vision/
│
├── prompt_bin/                    [本地] 默认 prompt（What is shown in this image?）
│   └── llm_preblock/
│
├── batch/                         [本地写 → MDC 读] 批量：<stem>/dump/ + <stem>/om_output/
│
├── ple_table/                     [本地写 → MDC 读] 全工程共享一份 PLE
│   └── embed_tokens_per_layer.bin
│
├── om_export/                     [MDC] *.om 模型
└── om_output/                     [MDC → 本地 parse] 单张统一输出根
    ├── work/                      中间 OM scratch，默认跑完删
    ├── state/                     decode 可变状态
    ├── final_logits.bin
    ├── final_input_ids.bin
    ├── final_attention_mask.bin
    ├── final_cur_len.txt
    └── final.meta.json
```

| 目录 | 机器 | 何时产生 | 说明 |
|------|------|----------|------|
| `dump/` | 本地 | `dump_om_inputs.py` | 静态 frontend input；MDC 只读 |
| `ple_table/` | 本地 | dump / `--ple-only` | 与 `dump/` 同级，全工程一份 |
| `batch/<stem>/dump/` | 本地 | 批量 dump | 每图 vision + llm_preblock |
| `om_output/` | MDC | pipeline | **统一输出根**：`work/` + `state/` + `final_*` |
| `om_output/work/` | MDC | pipeline | 中间 OM scratch，`KEEP_INTERMEDIATE=0` 跑完删 |
| `om_output/state/` | MDC | pipeline | decode 可变状态 |
| `om_output/final_*` | MDC | pipeline | 最终 bin，拷回本地 `parse_state.py` |

**PLE 说明**：PLE 是模型级权重表，与图像无关。放在 `om/ple_table/`（不在 `dump/` 内）。本地 `dump_om_inputs.py` full 模式或 `--ple-only` 会自动 ensure；**scp 到 MDC 时与 `dump/`/`batch/` 一并拷贝，只需一份**。

**scp 要点**：本地 → MDC 拷 `dump/`/`batch/` + `ple_table/` + 脚本；MDC → 本地拷 `om_output/`（已含 work/、state/、final_*）。

---

## 启动命令详解

### `dump_om_inputs.py`（本地）

```bash
python dump_om_inputs.py [选项]
```

| 场景 | 命令 | 输出路径 |
|------|------|----------|
| 单张（默认） | `--image img.jpg` | `dump/vision/` + 从 `prompt_bin/` 复制 preblock |
| 单张 full | `--mode full --image img.jpg` | `dump/` + 更新 `prompt_bin/` |
| 批量 | `--image-dir path/images` | `batch/<stem>/dump/` |
| 仅 PLE | `--ple-only` | `ple_table/` |

**默认 prompt**：`prompt_bin/llm_preblock/` 已预生成（`What is shown in this image?`）。默认 `image-only` 只写 `vision/`，并将 `prompt_bin/llm_preblock/` **复制**到输出目录（`dump/` 或 `batch/<stem>/dump/`）；换 prompt 时用 `--mode full` 更新 `prompt_bin/`。

**关键选项**

| 选项 | 默认 | 含义 |
|------|------|------|
| `--mode full\|image-only` | `image-only` | 默认只 dump 图；full 重新生成 prompt/preblock |
| `--prompt-dir` | `prompt_bin` | image-only：读 `<dir>/llm_preblock/` |
| `--out-dir` | `dump` | 单张输出根目录 |
| `--batch-root` | `batch` | 批量输出根目录 |
| `--ple-only` | — | 只 ensure PLE bin |

PLE：`ple_table/embed_tokens_per_layer.bin` 与 `dump/` 同级；不存在时自动生成一次，已存在则跳过。

---

### `run_om_pipeline.sh`（MDC）

```bash
bash run_om_pipeline.sh [options] [路径] [GEN_STEPS] [NUM_ASSISTANT_TOKENS]
```

**指定路径**

| 选项 / 环境变量 | 说明 |
|----------------|------|
| `--dump-dir PATH` / `DUMP_ROOT` | 静态 input bin（`vision/`、`llm_preblock/`） |
| `--output-dir PATH` / `OUTPUT_ROOT` | **统一输出根**（默认 `om_output/`） |
| `--work-dir PATH` | 覆盖 scratch（默认 `<output-dir>/work`） |
| `--state-dir PATH` | 覆盖 state（默认 `<output-dir>/state`） |
| `--ple-table-dir PATH` | PLE 表（默认 `ple_table/`） |
| `--batch-root PATH` | 批量根目录 |

```bash
# 指定 input + 统一输出根（work/ state/ final_* 都在 om_output/ 下）
RUN_MSAME=1 bash run_om_pipeline.sh --dump-dir dump --output-dir om_output
```

**路径 positional 自动识别**（未用 `--dump-dir` 时）

| 传入路径 | 模式 | 路径映射 |
|----------|------|----------|
| 无参数 | 单张 | 默认 `DUMP_ROOT=dump` |
| `./batch/` | 批量 | 遍历 `batch/<stem>/` |
| `./batch/foo/` | 单 item | `dump/` + `om_output/`（含 work/ state/） |
| `./dump/` | 单张 | `DUMP_ROOT=./dump` |

批量判定：目录下存在任意 `<子目录>/dump/vision/pixel_values.bin`。

**必设环境变量（MDC）**

```bash
RUN_MSAME=1 bash run_om_pipeline.sh          # 不设则只打印 msame 命令（dry-run）
```

**常用环境变量**

| 变量 | 默认 | 作用 |
|------|------|------|
| `MODE` | `full` | `full` 投机解码 / `main_decode` 纯主链 / `prefill_only` 只 prefill |
| `GEN_STEPS` | `50` | decode 循环次数；也可第 2 个 positional 参数 |
| `NUM_ASSISTANT_TOKENS` | `6` | 每步 assistant draft 数；也可第 3 个 positional |
| `KEEP_INTERMEDIATE` | `0` | `1` 保留 `om_work/` |
| `SKIP_EXIST` | `0` | 批量：已有 `om_work/step_0000` 则跳过该项 |
| `PLE_TABLE_DIR` | `ple_table` | decode 查表（与 dump 同级，全工程一份） |

**示例**

```bash
# 单张 full decode（默认 50 step + assistant）
RUN_MSAME=1 bash run_om_pipeline.sh

# 批量，100 step，8 draft tokens
RUN_MSAME=1 bash run_om_pipeline.sh ./batch 100 8

# 主链 decode，无 assistant
MODE=main_decode RUN_MSAME=1 bash run_om_pipeline.sh

# 等价写法
GEN_STEPS=100 NUM_ASSISTANT_TOKENS=8 RUN_MSAME=1 bash run_om_pipeline.sh ./batch
```

**MDC 内部一步 decode（MODE=full）数据变化**

```
1. om_bin_utils 从 dump/ + state/ 拼 vision / preblock / block 输入
2. msame 跑 vision → mm_proj → preblock → b1..b7 → lm_head
3. assistant 从 b7/b3 + state 拼 draft 输入，跑 assistant OM → 候选 token
4. verify 链对候选逐位跑 lm_head → main_preds
5. process-speculative-accept 更新 state/，cur_len += accept_count
6. 写 om_output/final_*；删 om_work/（KEEP_INTERMEDIATE=0）
```

---

### `parse_state.py`（本地）

```bash
python parse_state.py [--output-dir om_output] [--dump-dir dump] [--response-out out.txt]
```

| 选项 | 默认 | 用途 |
|------|------|------|
| `--output-dir` | `om_output` | MDC 拷回的 final bin |
| `--dump-dir` | `dump` | 读 `llm_preblock/attention_mask.bin` 算 prefill 长度 |
| `--response-out` | — | 写生成段文本到文件 |

读 `final_input_ids.bin` + prefill_len → 解码 **生成段**（去掉 prompt/image token 部分）。

---

## Pipeline 命令速查

以下命令均在 `cd gemma-4/om` 下执行（本地或 MDC 对应小节）。

### `dump_om_inputs.py`（本地）

```bash
# 单张 full
python dump_om_inputs.py --mode full --image path/image.jpg

# 单张（默认 image-only）
python dump_om_inputs.py --image path/new.jpg

# 批量
python dump_om_inputs.py --image-dir path/images

# 仅生成 PLE（全工程一份，首次 scp 到 MDC）
python dump_om_inputs.py --ple-only
```

### `run_om_pipeline.sh`（MDC）

```bash
# 单张 full decode（默认 50 step + 6 assistant draft）
RUN_MSAME=1 bash run_om_pipeline.sh
RUN_MSAME=1 bash run_om_pipeline.sh --dump-dir dump

# 指定 decode 步数 / draft 数（环境变量；不支持 --GEN_STEPS）
GEN_STEPS=20 RUN_MSAME=1 bash run_om_pipeline.sh --dump-dir dump
GEN_STEPS=100 NUM_ASSISTANT_TOKENS=8 RUN_MSAME=1 bash run_om_pipeline.sh --dump-dir dump

# positional：path [GEN_STEPS] [NUM_ASSISTANT_TOKENS]
RUN_MSAME=1 bash run_om_pipeline.sh dump 20
RUN_MSAME=1 bash run_om_pipeline.sh batch 100 8

# 批量
RUN_MSAME=1 bash run_om_pipeline.sh --batch-root batch
SKIP_EXIST=1 RUN_MSAME=1 bash run_om_pipeline.sh --batch-root batch

# 主链 decode（无 assistant 投机）
MODE=main_decode RUN_MSAME=1 bash run_om_pipeline.sh --dump-dir dump

# 只 prefill
MODE=prefill_only RUN_MSAME=1 bash run_om_pipeline.sh --dump-dir dump

# 保留 work/
KEEP_INTERMEDIATE=1 RUN_MSAME=1 bash run_om_pipeline.sh --dump-dir dump
```

### `parse_state.py`（本地）

```bash
python parse_state.py --output-dir om_output
python parse_state.py --batch-root
python parse_state.py --batch-root --stem images2
python parse_state.py --batch-root --write-response
```

### 端到端复制块

```bash
# ── 本地 dump ──
python dump_om_inputs.py --image path/image.jpg
# 首次：scp -r dump ple_table root@MDC:.../gemma4
# 非首次：scp -r dump root@MDC:.../gemma4

# ── MDC 推理 ──
GEN_STEPS=20 RUN_MSAME=1 bash run_om_pipeline.sh --dump-dir dump
# scp -r root@MDC:.../om_output .

# ── 本地 parse ──
python parse_state.py --output-dir om_output
```

---

## 附录

模型路径：`../gemma-4-E2B-it/`（本地 dump / parse 用）。

**MODE 一览**

| MODE | assistant | decode |
|------|-----------|--------|
| `full` | ✓ 投机 | ✓ |
| `prefill_only` | 可选 | ✗ |
| `main_only` | ✗ | ✗ |
| `main_decode` | ✗ | ✓ 逐步 +1 |

**链式喂数**：仅 `dump/vision` 和 `dump/llm_preblock` 是静态 bin；mm_proj / block / assistant 输入在 MDC 上由 OM 输出 + `om_bin_utils*` 运行时拼装。
