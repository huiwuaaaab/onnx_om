#!/usr/bin/env python3
"""
Single-process resident pool for Gemma pipe mode.

One aclInit; OMs loaded on demand with LRU eviction (OM_RESIDENT_MAX_LOADED).

Usage (invoked by worker_resident_pool.sh):
  python om_resident_pool.py /path/to/resident_pool.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def ts_log(worker: str, msg: str, log_path: Path) -> None:
    line = f"[{time.strftime('%H:%M:%S')}][resident:{worker}] {msg}\n"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line)
    print(line, end="", flush=True)


def parse_job_env(job_env: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in job_env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def check_output_bins(output_dir: Path) -> bool:
    return any(output_dir.rglob("*.bin"))


def dry_run_outputs(worker: str, output_dir: Path) -> None:
    max_seq_len, h = 512, 1536
    num_layers = 35
    ple_dim = 256
    vision_out = 1 * 256 * 768 * 2
    mm_proj = 1 * 256 * h * 2
    block_hidden = max_seq_len * h * 2
    ple_out = max_seq_len * num_layers * ple_dim * 2
    mask = max_seq_len * max_seq_len * 2
    logits = 262144 * 2

    output_dir.mkdir(parents=True, exist_ok=True)

    def w(name: str, n: int) -> None:
        (output_dir / name).write_bytes(b"\x00" * n)

    if worker == "vision":
        w("0.bin", vision_out)
    elif worker == "mm_proj":
        w("0.bin", mm_proj)
    elif worker == "preblock":
        w("inputs_embeds_out.bin", block_hidden)
        w("per_layer_inputs_out.bin", ple_out)
        w("full_mask.bin", mask)
        w("sliding_mask.bin", mask)
    elif worker.startswith("block"):
        w("hidden_states_out.bin", block_hidden)
    elif worker == "lm_head":
        w("logits.bin", logits)
    else:
        w("0.bin", 16)


def process_one_job(
    worker: str,
    queue_dir: Path,
    pool,
    log_path: Path,
) -> bool:
    jobs_dir = queue_dir / "jobs"
    pending = sorted(jobs_dir.glob("*.pending"))
    if not pending:
        return False

    job_pending = pending[0]
    job_id = job_pending.stem
    job_env = jobs_dir / f"{job_id}.env"
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

    ts_log(worker, f"job {job_id}  tag={tag}", log_path)
    ts_log(worker, f"  input : {input_dir}", log_path)
    ts_log(worker, f"  output: {output_dir}", log_path)

    ok = True
    if run_msame:
        try:
            t0 = time.time()
            ts_log(worker, f"acl infer start tag={tag}", log_path)
            (output_dir / "msprof").mkdir(parents=True, exist_ok=True)
            pool.execute_job(worker, input_dir, output_dir, num_inputs)
            if not check_output_bins(output_dir):
                raise RuntimeError("no output bins after acl execute")
            ts_log(worker, f"acl infer done tag={tag} elapsed={time.time() - t0:.1f}s", log_path)
        except Exception as exc:
            ts_log(worker, f"ERROR: {exc}", log_path)
            ok = False
    else:
        ts_log(worker, "  [dry-run]", log_path)
        dry_run_outputs(worker, output_dir)

    if ok:
        (jobs_dir / f"{job_id}.done").touch()
    else:
        (jobs_dir / f"{job_id}.failed").touch()
    return True


def all_exit(queues: list[Path]) -> bool:
    return all((q / "exit").is_file() for q in queues)


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: om_resident_pool.py <manifest.json>", file=sys.stderr)
        sys.exit(2)

    manifest_path = Path(sys.argv[1])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    workers_cfg = manifest.get("workers", [])
    if not workers_cfg:
        print("ERROR: empty workers in manifest", file=sys.stderr)
        sys.exit(1)

    max_loaded = int(
        manifest.get(
            "max_loaded",
            os.environ.get("OM_RESIDENT_MAX_LOADED", "7"),
        )
    )
    preload_all = manifest.get(
        "preload_all",
        os.environ.get("OM_RESIDENT_PRELOAD_ALL", "0") == "1",
    )

    pool_log_path = manifest_path.parent / "pool.log"

    def pool_log_fn(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}][pool] {msg}\n"
        with pool_log_path.open("a", encoding="utf-8") as f:
            f.write(line)
        # stdout is redirected to pool.log by worker_resident_pool.sh — avoid duplicate lines
        if not sys.stdout.isatty():
            return
        print(line, end="", flush=True)

    from acl_ctypes import try_create_lazy_pool, try_create_pool

    worker_specs: list[tuple[str, str, Path, Path]] = []
    load_specs: list[tuple[str, str]] = []
    for item in workers_cfg:
        name = item["name"]
        om_path = item["om"]
        queue_dir = Path(item["queue"])
        log_path = Path(item.get("log", str(queue_dir / "worker.log")))
        queue_dir.mkdir(parents=True, exist_ok=True)
        (queue_dir / "jobs").mkdir(parents=True, exist_ok=True)
        rm = queue_dir / "ready"
        if rm.exists():
            rm.unlink()
        worker_specs.append((name, om_path, queue_dir, log_path))
        load_specs.append((name, om_path))

    pool = None
    session = None
    models = None

    if preload_all:
        pool_log_fn(f"preload all {len(load_specs)} OMs in one acl session ...")
        session, models = try_create_pool(load_specs, pool_log_fn)
        resident_tag = "ctypes-pool"
    else:
        pool_log_fn(
            f"lazy pool: load on demand, max_loaded={max_loaded}, stages={len(load_specs)}"
        )
        pool = try_create_lazy_pool(load_specs, pool_log_fn, max_loaded)
        resident_tag = "ctypes-pool-lazy"

    for name, om_path, queue_dir, log_path in worker_specs:
        ts_log(
            name,
            f"ready  om={om_path}  resident={resident_tag}  jobs={queue_dir / 'jobs'}",
            log_path,
        )
        (queue_dir / "ready").touch()

    pool_log_fn(f"pool ready ({len(load_specs)} stages, tag={resident_tag})")

    try:
        queues = [q for _, _, q, _ in worker_specs]
        while not all_exit(queues):
            did_work = False
            for name, _, queue_dir, log_path in worker_specs:
                if preload_all and models is not None:
                    if process_one_job_legacy(name, queue_dir, models[name], log_path):
                        did_work = True
                elif pool is not None:
                    if process_one_job(name, queue_dir, pool, log_path):
                        did_work = True
            if not did_work:
                time.sleep(0.005)
        pool_log_fn("all queues exited")
    finally:
        if pool is not None:
            pool.close()
        elif session is not None and models is not None:
            for model in models.values():
                model.close()
            session.close()
        pool_log_fn("pool shutdown complete")


def process_one_job_legacy(
    worker: str,
    queue_dir: Path,
    model,
    log_path: Path,
) -> bool:
    jobs_dir = queue_dir / "jobs"
    pending = sorted(jobs_dir.glob("*.pending"))
    if not pending:
        return False

    job_pending = pending[0]
    job_id = job_pending.stem
    job_env = jobs_dir / f"{job_id}.env"
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

    ts_log(worker, f"job {job_id}  tag={tag}", log_path)
    ok = True
    if run_msame:
        try:
            t0 = time.time()
            ts_log(worker, f"acl infer start tag={tag}", log_path)
            (output_dir / "msprof").mkdir(parents=True, exist_ok=True)
            model.execute_job(input_dir, output_dir, num_inputs)
            if not check_output_bins(output_dir):
                raise RuntimeError("no output bins after acl execute")
            ts_log(worker, f"acl infer done tag={tag} elapsed={time.time() - t0:.1f}s", log_path)
        except Exception as exc:
            ts_log(worker, f"ERROR: {exc}", log_path)
            ok = False
    else:
        dry_run_outputs(worker, output_dir)

    if ok:
        (jobs_dir / f"{job_id}.done").touch()
    else:
        (jobs_dir / f"{job_id}.failed").touch()
    return True


if __name__ == "__main__":
    main()
