# Gemma-4 OM 推理

**本地机**：`dump_vision_om_inputs.py` + `dump_llm_preblock_inputs.py` → **MDC**：`run_om_pipeline.sh` / `run_om_pipeline_pipe.sh` → **本地机**：`parse_state.py`

```
Main: vision → mm_proj → llm_preblock → b1..b7 → lm_head
Assist: assistant speculative decode (serial 模式, MODE=full)
```

## 快速开始

### 单张

```bash
cd /e-vepfs-01/perception/wuhui/gemma-4/om

python dump_vision_om_inputs.py --image path/image.jpg
python dump_llm_preblock_inputs.py --prompt "What is shown in this image?"
# 首次需 PLE: python dump_llm_preblock_inputs.py --ple-only

# scp: vision_bin + prompt_bin + ple_table/ + om_export/
RUN_MSAME=1 bash run_om_pipeline.sh

python parse_state.py --output-dir om_output --dump-dir prompt_bin
```

### 批量

```bash
python dump_vision_om_inputs.py --image-dir path/images
python dump_llm_preblock_inputs.py --prompt "What is shown in this image?"

# 串行（含 assistant 投机解码，默认 MODE=full）
RUN_MSAME=1 bash run_om_pipeline.sh --batch-root batch

# 流水线 main chain（无 assistant，11 OM 常驻）
MODE=main_decode RUN_MSAME=1 bash run_om_pipeline_pipe.sh ./batch

python parse_state.py --batch-root batch --write-response
```

## 数据目录

```
om/
├── vision_bin/          pixel_values.bin + image_position_ids.bin
├── prompt_bin/          input_ids + attention_mask + per_layer_inputs + position_ids
├── ple_table/           embed_tokens_per_layer.bin（全工程共享）
├── pipeline/            serial.sh / pipe.sh / worker*.sh / acl_ctypes.py
├── om_export/           *.om
└── batch/<stem>/vision_bin/ + om_output/
```

legacy `dump/` 仍被 `pipeline/paths.sh` 兼容。

## run_om_pipeline.sh / run_om_pipeline_pipe.sh

| 脚本 | 说明 |
|------|------|
| `run_om_pipeline.sh` | **串行**：inline msame + assistant 投机解码 |
| `run_om_pipeline_pipe.sh` | **流水线**：11 worker 常驻，每 job ctypes 加载/卸载 OM |

```bash
RUN_MSAME=1 bash run_om_pipeline.sh --batch-root batch
MODE=main_decode RUN_MSAME=1 bash run_om_pipeline_pipe.sh ./batch
STOP_ON_EOS=1 RUN_MSAME=1 bash run_om_pipeline.sh batch   # 遇 <eos> (id=1) 停止
```

| 环境变量 | 默认 | 说明 |
|----------|------|------|
| `MODE` | `full`（serial）/ `main_decode`（pipe 必需） | `prefill_only` / `full` / `main_decode` |
| `GEN_STEPS` | `50` | decode 步数 |
| `NUM_ASSISTANT_TOKENS` | `6` | serial 投机解码 |
| `STOP_ON_EOS` | `1` | 遇 token 1 停止 |
| `RUN_MSAME` | `0` | MDC 上设为 `1` |
| `OM_RESIDENT` | `1`（pipe） | `0` 改用 per-job msame |
| `OM_PER_JOB_LOAD` | `1`（pipe） | `0` worker 内 OM 常驻（11 段易 OOM） |

Pipe workers: `batch/.om_pipe/<stage>/worker.log`  
确认: `grep resident= batch/.om_pipe/*/worker.log` → **`resident=ctypes-perjob`**

## parse_state.py

JSON vocab 解码（无 tokenizers）。`--dump-dir` 指向 `prompt_bin/`（读 `meta.json` 的 `seq_len`）。

## msame

默认 `MSAME_BIN=./msame`，或 `export MSAME_BIN=/path/to/msame_elf`
