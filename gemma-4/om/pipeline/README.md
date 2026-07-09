# OM pipeline scripts (Gemma-4)

| 入口 | 脚本 | 说明 |
|------|------|------|
| 串行 | `../run_om_pipeline.sh` | main + assistant 投机解码 |
| 流水线 | `../run_om_pipeline_pipe.sh` | 11 worker 常驻，每 job 加载/卸载 OM |

```
vision → mm_proj → preblock → block1..7 → lm_head
```

| File | Role |
|------|------|
| `worker_resident.sh` | 启动 ctypes worker（`OM_PER_JOB_LOAD=1`） |
| `om_resident_worker.py` | worker 进程常驻，OM 按 job load/unload |
| `acl_ctypes.py` | `AclPerJobRunner` |

```bash
# 串行（含 assistant）
RUN_MSAME=1 bash ../run_om_pipeline.sh --batch-root batch

# 流水线 main chain
MODE=main_decode RUN_MSAME=1 bash ../run_om_pipeline_pipe.sh ./batch
```

Pipe 不支持 `MODE=full`（assistant 请用 serial）。

Workers: `batch/.om_pipe/<stage>/worker.log`  
确认: `grep resident= batch/.om_pipe/*/worker.log` → **`resident=ctypes-perjob`**

每 job 日志应有 `load OM for job` → `OM unloaded after job`。
