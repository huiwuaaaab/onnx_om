# C++ OM 常驻 Daemon（无 pyACL）

板端有 **msame / AscendCL（C++）** 但没有 Python `acl` 时，用这个 daemon 替代 `om_resident_worker.py`。

## 原理

```
pipe.sh 调度
  └─ worker_cpp_resident.sh
       └─ om_resident_daemon（C++ 进程）
            ├─ 启动: aclmdlLoadFromFile() 一次
            └─ 循环: 读 FIFO job → aclmdlExecute() → 写 output/*.bin
```

与 `worker.sh` **完全相同的 job 协议**（`.pending` / `.env` / `.done`），pipe 脚本无需改调度逻辑。

## MDC 上编译

```bash
cd /home/mdc/guanxj/qwen3-vl/pipeline/om_resident_cpp

# 先看板子上有没有开发头文件
bash diagnose_cann_build.sh

bash build.sh
```

### AOS 板端只有 runtime（常见）

`msame` 能跑，但 **没有 `acl/acl.h`** → 无法在 AOS 上直接编译。

在 **有 CANN toolkit 的开发机**上编（aarch64，CANN 版本与板端一致），再拷到板子：

```bash
# 开发机
cd pipeline/om_resident_cpp
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 或你的 CANN 路径
bash build.sh

scp out/om_resident_daemon \
  root@AOS:/home/mdc/guanxj/qwen3-vl/pipeline/om_resident_cpp/out/
chmod +x /home/mdc/guanxj/qwen3-vl/pipeline/om_resident_cpp/out/om_resident_daemon
```

也可在与 msame 相同的编译环境编（msame 从哪编，daemon 就在哪编）。

### 手动指定 CANN 路径

```bash
export ASCEND_INCLUDE=/path/to/acllib/include
export ASCEND_LIB=/path/to/acllib/lib64/stub
bash build.sh
```

## 跑 pipe

编译完成后，`run_om_pipeline_pipe.sh` **自动优先**用 C++ daemon：

```bash
cd /home/mdc/guanxj/qwen3-vl
pkill -f om_resident || true
rm -rf batch/.om_pipe

RUN_MSAME=1 bash run_om_pipeline_pipe.sh ./batch
```

worker.log 应出现：

```
[xx:xx:xx][cpp:vision] model resident loaded om=... elapsed=...
[xx:xx:xx][cpp:vision] ready  om=...  resident=cpp  jobs=...
[xx:xx:xx][cpp:vision] acl infer done tag=... elapsed=...
```

**不是** `resident=msame-fallback`。

## 与 msame 源码的关系

本 daemon **不 fork msame 进程**，而是直接调 AscendCL C API（与 msame 内部 `model.cpp` 同类逻辑）：

| msame CLI | om_resident_daemon |
|-----------|-------------------|
| 每次命令 load+unload | 启动 load 一次 |
| 读 CLI 参数 | 读 FIFO job `.env` |
| 写 msame 输出命名 | 写 `{0,1,...}.bin`（`om_bin_utils` 已兼容） |

也可参考官方 msame 源码加深理解：

```bash
git clone https://github.com/Ascend/tools.git
# 看 tools/msame/src/model.cpp
```

## 文件

| 文件 | 说明 |
|------|------|
| `src/main.cpp` | 入口 + job 循环 |
| `src/acl_model.cpp` | load once + execute |
| `src/job_queue.cpp` | FIFO 协议 |
| `build.sh` | MDC 编译 |
| `../worker_cpp_resident.sh` | pipe worker 启动器 |

## 故障排查

```bash
# 手动测 vision worker
source ../acl_env.sh
./out/om_resident_daemon vision \
  /home/mdc/guanxj/qwen3-vl/om_export/vision_448_xxx.om \
  /tmp/test_queue
# 另开终端往 /tmp/test_queue/jobs/ 写 .env + .pending
```

load 失败：检查 `LD_LIBRARY_PATH`、OM 路径、设备是否被占用。
