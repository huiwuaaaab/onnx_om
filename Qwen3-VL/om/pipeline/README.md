# OM pipeline scripts

| 入口 | 脚本 | 说明 |
|------|------|------|
| 串行 | `../run_om_pipeline.sh` | 单张 / batch 逐张 inline msame |
| 流水线 | `../run_om_pipeline_pipe.sh` | 多图 stage 并行，**6 OM 常驻**（默认） |

| File | Role |
|------|------|
| `paths.sh` | Resolve `vision_bin/` + `prompt_bin/` |
| `serial.sh` | serial entry backend |
| `pipe.sh` | pipe entry backend |
| `worker.sh` | per-job msame（`OM_RESIDENT=0` 时） |
| `worker_resident.sh` | pyACL 预加载 OM（pipe 默认，source acl_env.sh） |
| `acl_env.sh` | CANN / pyACL 环境（与 msame 相同 set_env.sh） |
| `worker_cpp_resident.sh` | **C++ AscendCL 常驻**（无 pyACL，AOS 推荐） |
| `om_resident_cpp/` | C++ daemon 源码 + `build.sh` |
| `diagnose_acl.sh` | 检测 pyACL；不可用则编译 C++ daemon |
| `om_resident_worker.py` | resident 推理 daemon |

```bash
# 串行
RUN_MSAME=1 bash ../run_om_pipeline.sh --batch-root batch

# 流水线（6 OM 常驻）
RUN_MSAME=1 bash ../run_om_pipeline_pipe.sh ./batch

# 流水线但不常驻（每 job 起 msame）
OM_RESIDENT=0 RUN_MSAME=1 bash ../run_om_pipeline_pipe.sh ./batch
```

Workers + logs: `batch/.om_pipe/<vision|preblock|block*>/worker.log`
