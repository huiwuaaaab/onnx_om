#!/usr/bin/env python3
"""
Resident OM inference worker for Gemma-4 pipe mode (main chain only).

Loads one .om at startup (pyACL or ctypes libascendcl) and serves FIFO jobs.
Falls back to per-job msame when neither is available.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}][resident:{WORKER_NAME}] {msg}", flush=True)


WORKER_NAME = ""
OM_PATH = ""
QUEUE_DIR = Path()
JOBS_DIR = Path()
MSAME_BIN = os.environ.get("MSAME_BIN", "./msame")
OM_SCRIPT_DIR = Path(os.environ.get("OM_SCRIPT_DIR", "."))

MAX_SEQ_LEN = 512
HIDDEN_DIM = 1536
NUM_LAYERS = 35
PLE_DIM = 256
VISION_OUT_BYTES = 1 * 256 * 768 * 2
MM_PROJ_OUT_BYTES = 1 * 256 * HIDDEN_DIM * 2
BLOCK_HIDDEN_BYTES = MAX_SEQ_LEN * HIDDEN_DIM * 2
PLE_OUT_BYTES = MAX_SEQ_LEN * NUM_LAYERS * PLE_DIM * 2
MASK_BYTES = MAX_SEQ_LEN * MAX_SEQ_LEN * 2
LOGITS_BYTES = 262144 * 2


def msame_input_arg(input_dir: Path, num_inputs: int) -> str:
    if num_inputs <= 1:
        return str(input_dir / "0.bin")
    return ",".join(str(input_dir / f"{i}.bin") for i in range(num_inputs))


def check_output_bins(output_dir: Path) -> bool:
    return any(output_dir.rglob("*.bin"))


def dry_run_outputs(worker: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    def w(name: str, n: int) -> None:
        (output_dir / name).write_bytes(b"\x00" * n)

    if worker == "vision":
        w("0.bin", VISION_OUT_BYTES)
    elif worker == "mm_proj":
        w("0.bin", MM_PROJ_OUT_BYTES)
    elif worker == "preblock":
        w("inputs_embeds_out.bin", BLOCK_HIDDEN_BYTES)
        w("per_layer_inputs_out.bin", PLE_OUT_BYTES)
        w("full_mask.bin", MASK_BYTES)
        w("sliding_mask.bin", MASK_BYTES)
    elif worker in ("block1", "block2", "block3", "block4", "block5", "block6", "block7"):
        w("hidden_states_out.bin", BLOCK_HIDDEN_BYTES)
    elif worker == "lm_head":
        w("logits.bin", LOGITS_BYTES)
    else:
        w("0.bin", 16)


def run_msame_job(
    job_id: str,
    tag: str,
    input_dir: Path,
    output_dir: Path,
    num_inputs: int,
) -> bool:
    t0 = time.time()
    input_arg = msame_input_arg(input_dir, num_inputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_sh = OM_SCRIPT_DIR / f"run_pipe_{WORKER_NAME}.sh"
    line = (
        f"pmupload {MSAME_BIN} --model {OM_PATH} --input \"{input_arg}\" "
        f"--output {output_dir} --outfmt BIN --loop 1"
    )
    run_sh.write_text(line + "\n", encoding="utf-8")
    run_sh.chmod(0o777)
    log(f"msame infer start tag={tag} (each job reloads OM)")
    log(f"  cmd: {line}")

    if os.environ.get("MSPROF_WRAP", "0") == "1":
        msprof = os.environ.get("MSPROF_BIN", "/var/msprof")
        (output_dir / "msprof").mkdir(parents=True, exist_ok=True)
        cmd = [
            msprof,
            f"--application=./run_pipe_{WORKER_NAME}.sh",
            f"--output={output_dir / 'msprof'}",
        ]
    else:
        cmd = ["bash", f"./run_pipe_{WORKER_NAME}.sh"]

    proc = subprocess.run(cmd, cwd=str(OM_SCRIPT_DIR))
    elapsed = time.time() - t0
    if proc.returncode != 0:
        log(f"ERROR: msame failed {job_id} tag={tag} rc={proc.returncode} elapsed={elapsed:.1f}s")
        return False
    if not check_output_bins(output_dir):
        log(f"ERROR: no output bins {job_id} elapsed={elapsed:.1f}s")
        return False
    log(f"msame infer done tag={tag} elapsed={elapsed:.1f}s")
    return True


class AclResidentModel:
    def __init__(self, om_path: str) -> None:
        import acl  # type: ignore

        self.acl = acl
        self.model_id: int | None = None
        self.model_desc = None
        self._output_sizes: list[int] = []
        self.stream = None
        self.context = None

        ret = acl.init()
        if ret != 0:
            raise RuntimeError(f"acl.init failed: {ret}")
        ret = acl.rt.set_device(0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.set_device failed: {ret}")
        self.context, ret = acl.rt.create_context(0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.create_context failed: {ret}")
        self.stream, ret = acl.rt.create_stream()
        if ret != 0:
            raise RuntimeError(f"acl.rt.create_stream failed: {ret}")

        self.model_id, ret = acl.mdl.load_from_file(om_path)
        if ret != 0:
            raise RuntimeError(f"acl.mdl.load_from_file failed: {ret} path={om_path}")

        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        if ret != 0:
            raise RuntimeError(f"acl.mdl.get_desc failed: {ret}")

        n_out = acl.mdl.get_num_outputs(self.model_desc)
        for i in range(n_out):
            size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            self._output_sizes.append(int(size))
            log(f"  output[{i}] size={size}")

        log(f"model resident loaded id={self.model_id} outputs={n_out}")

    def execute_job(self, input_dir: Path, output_dir: Path, num_inputs: int) -> None:
        acl = self.acl
        blobs = []
        for i in range(num_inputs):
            p = input_dir / f"{i}.bin"
            if not p.is_file():
                raise FileNotFoundError(p)
            blobs.append(p.read_bytes())

        in_ds = acl.mdl.create_dataset()
        for blob in blobs:
            dev_ptr, ret = acl.rt.malloc(len(blob), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc input failed: {ret}")
            host_ptr = acl.util.bytes_to_ptr(blob)
            ret = acl.rt.memcpy(dev_ptr, len(blob), host_ptr, len(blob), 1)
            if ret != 0:
                raise RuntimeError(f"acl.rt.memcpy H2D failed: {ret}")
            data_buf = acl.create_data_buffer(dev_ptr, len(blob))
            _, ret = acl.mdl.add_dataset_buffer(in_ds, data_buf)
            if ret != 0:
                raise RuntimeError(f"acl.mdl.add_dataset_buffer input failed: {ret}")

        out_ds = acl.mdl.create_dataset()
        out_ptrs: list[int] = []
        for size in self._output_sizes:
            dev_ptr, ret = acl.rt.malloc(size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc output failed: {ret}")
            out_ptrs.append(dev_ptr)
            data_buf = acl.create_data_buffer(dev_ptr, size)
            _, ret = acl.mdl.add_dataset_buffer(out_ds, data_buf)
            if ret != 0:
                raise RuntimeError(f"acl.mdl.add_dataset_buffer output failed: {ret}")

        try:
            ret = acl.mdl.execute(self.model_id, in_ds, out_ds)
            if ret != 0:
                raise RuntimeError(f"acl.mdl.execute failed: {ret}")
            output_dir.mkdir(parents=True, exist_ok=True)
            for idx, (dev_ptr, size) in enumerate(zip(out_ptrs, self._output_sizes)):
                host_blob = bytearray(size)
                host_ptr = acl.util.bytes_to_ptr(host_blob)
                ret = acl.rt.memcpy(host_ptr, size, dev_ptr, size, 2)
                if ret != 0:
                    raise RuntimeError(f"acl.rt.memcpy D2H failed: {ret}")
                (output_dir / f"{idx}.bin").write_bytes(host_blob)
        finally:
            n = acl.mdl.get_dataset_num_buffers(in_ds)
            for i in range(n):
                buf = acl.mdl.get_dataset_buffer(in_ds, i)
                dev_ptr = acl.get_data_buffer_addr(buf)
                acl.rt.free(dev_ptr)
                acl.destroy_data_buffer(buf)
            acl.mdl.destroy_dataset(in_ds)
            n = acl.mdl.get_dataset_num_buffers(out_ds)
            for i in range(n):
                buf = acl.mdl.get_dataset_buffer(out_ds, i)
                dev_ptr = acl.get_data_buffer_addr(buf)
                acl.rt.free(dev_ptr)
                acl.destroy_data_buffer(buf)
            acl.mdl.destroy_dataset(out_ds)

    def close(self) -> None:
        acl = self.acl
        if self.model_id is not None:
            acl.mdl.unload(self.model_id)
            self.model_id = None
        if self.model_desc is not None:
            acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None
        if self.stream is not None:
            acl.rt.destroy_stream(self.stream)
        if self.context is not None:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(0)
        acl.finalize()
        log("model unloaded, acl finalized")


def parse_job_env(job_env: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in job_env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def resident_backend_name(resident: object | None) -> str:
    if resident is None:
        return "msame-fallback"
    if resident.__class__.__name__ == "AclPerJobRunner":
        return "ctypes-perjob"
    if resident.__class__.__name__ == "AclResidentModelCtypes":
        return "ctypes"
    return "acl"


def create_resident_model(om_path: str) -> object | None:
    if os.environ.get("OM_RESIDENT_ACL", "1") != "1":
        return None
    per_job = os.environ.get("OM_PER_JOB_LOAD", "0") == "1"
    if per_job:
        try:
            from acl_ctypes import try_create_perjob

            log("init acl session (per-job OM load/unload) ...")
            return try_create_perjob(om_path, log)
        except Exception as exc:
            log(f"ERROR: ctypes per-job session failed ({exc})")
            log(f"  python={sys.executable}")
            log(f"  LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}")
            return None
    try:
        log("loading OM via pyACL ...")
        return AclResidentModel(om_path)
    except ImportError as exc:
        log(f"pyACL not available ({exc}), try ctypes libascendcl ...")
    except Exception as exc:
        log(f"pyACL preload failed ({exc}), try ctypes ...")
    try:
        from acl_ctypes import try_create

        log("loading OM via ctypes libascendcl.so ...")
        return try_create(om_path, log)
    except Exception as exc:
        log(f"WARN: ctypes acl failed ({exc}), fallback to per-job msame")
        log(f"  python={sys.executable}")
        log(f"  LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}")
        return None


def job_loop(resident: object | None) -> None:
    backend = resident_backend_name(resident)
    (QUEUE_DIR / "ready").touch()
    log(f"ready  om={OM_PATH}  resident={backend}  jobs={JOBS_DIR}")

    while not (QUEUE_DIR / "exit").is_file():
        pending = sorted(JOBS_DIR.glob("*.pending"))
        if not pending:
            time.sleep(0.005)
            continue

        job_pending = pending[0]
        job_id = job_pending.stem
        job_env = JOBS_DIR / f"{job_id}.env"
        try:
            job_pending.unlink()
        except FileNotFoundError:
            pass

        env = parse_job_env(job_env)
        tag = env.get("TAG", job_id)
        input_dir = Path(env["INPUT_DIR"])
        output_dir = Path(env["OUTPUT_DIR"])
        num_inputs = int(env.get("NUM_INPUTS", "1"))
        run_msame = env.get("RUN_MSAME", "0") == "1"

        log(f"job {job_id}  tag={tag}")
        log(f"  input : {input_dir}")
        log(f"  output: {output_dir}")

        ok = True
        if run_msame:
            try:
                if resident is not None:
                    t0 = time.time()
                    log(f"acl infer start tag={tag}")
                    (output_dir / "msprof").mkdir(parents=True, exist_ok=True)
                    resident.execute_job(input_dir, output_dir, num_inputs)
                    if not check_output_bins(output_dir):
                        raise RuntimeError("no output bins after acl execute")
                    log(f"acl infer done tag={tag} elapsed={time.time() - t0:.1f}s")
                else:
                    ok = run_msame_job(job_id, tag, input_dir, output_dir, num_inputs)
            except Exception as exc:
                log(f"ERROR: {exc}")
                ok = False
        else:
            log("  [dry-run]")
            dry_run_outputs(WORKER_NAME, output_dir)

        if ok:
            (JOBS_DIR / f"{job_id}.done").touch()
        else:
            (JOBS_DIR / f"{job_id}.failed").touch()

    log("exit")


def main() -> None:
    global WORKER_NAME, OM_PATH, QUEUE_DIR, JOBS_DIR

    if len(sys.argv) != 4:
        print("usage: om_resident_worker.py <worker_name> <om_path> <queue_dir>", file=sys.stderr)
        sys.exit(2)

    WORKER_NAME = sys.argv[1]
    OM_PATH = sys.argv[2]
    QUEUE_DIR = Path(sys.argv[3])
    JOBS_DIR = QUEUE_DIR / "jobs"
    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    if not Path(OM_PATH).is_file():
        log(f"ERROR: OM not found: {OM_PATH}")
        sys.exit(1)

    log(f"starting worker om={OM_PATH}")
    resident = create_resident_model(OM_PATH)
    if resident is None and os.environ.get("OM_RESIDENT", "0") == "1":
        log("ERROR: OM resident init failed (would use msame-fallback)")
        sys.exit(1)
    try:
        job_loop(resident)
    finally:
        if resident is not None:
            resident.close()


if __name__ == "__main__":
    main()
