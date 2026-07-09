# OM pipeline scripts

| 入口 | 脚本 | 说明 |
|------|------|------|
| 串行 | `../run_om_pipeline.sh` | 单张 / batch 逐张 inline msame |
| 流水线 | `../run_om_pipeline_pipe.sh` | 7 worker + OM 常驻（ctypes，默认） |

```
vision → mm_proj → preblock → block1..3 → lm_head
```

| File | Role |
|------|------|
| `paths.sh` | `vision_bin/` + `prompt_bin/` |
| `serial.sh` | serial backend |
| `pipe.sh` | pipe backend |
| `worker.sh` | per-job msame（`OM_RESIDENT=0`） |
| `worker_resident.sh` | ctypes/pyACL 常驻 worker |
| `om_resident_worker.py` | resident 推理 daemon |
| `acl_env.sh` / `acl_ctypes.py` | CANN 环境 + libascendcl ctypes |

```bash
# 串行
RUN_MSAME=1 bash ../run_om_pipeline.sh --batch-root batch

# 流水线（7 OM 常驻）
RUN_MSAME=1 bash ../run_om_pipeline_pipe.sh ./batch

# 不常驻
OM_RESIDENT=0 RUN_MSAME=1 bash ../run_om_pipeline_pipe.sh ./batch
```

Workers: `batch/.om_pipe/<vision|mm_proj|preblock|block*>/worker.log`  
确认常驻: `grep resident= batch/.om_pipe/*/worker.log` → `resident=ctypes`
