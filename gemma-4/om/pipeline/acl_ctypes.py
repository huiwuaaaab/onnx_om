"""AscendCL resident OM via ctypes — no pyACL module, no C++ compile.

Uses libascendcl.so already on AOS (/usr/local/Ascend/acllib/lib64/).
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Callable

ACL_SUCCESS = 0
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_H2D = 1
ACL_MEMCPY_D2H = 2


def _load_lib() -> ctypes.CDLL:
    candidates = []
    if os.environ.get("ASCEND_CL_LIB"):
        candidates.append(os.environ["ASCEND_CL_LIB"])
    candidates.extend(
        [
            "/usr/local/Ascend/acllib/lib64/libascendcl.so",
            "libascendcl.so",
        ]
    )
    last_err: Exception | None = None
    for path in candidates:
        try:
            return ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            last_err = exc
    raise RuntimeError(f"libascendcl.so not found: {last_err}")


def _setup_lib(lib: ctypes.CDLL) -> None:
    lib.aclInit.argtypes = [ctypes.c_char_p]
    lib.aclInit.restype = ctypes.c_int
    lib.aclFinalize.restype = ctypes.c_int

    lib.aclrtSetDevice.argtypes = [ctypes.c_int32]
    lib.aclrtSetDevice.restype = ctypes.c_int
    lib.aclrtResetDevice.argtypes = [ctypes.c_int32]
    lib.aclrtResetDevice.restype = ctypes.c_int

    lib.aclrtCreateContext.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int32]
    lib.aclrtCreateContext.restype = ctypes.c_int
    lib.aclrtDestroyContext.argtypes = [ctypes.c_void_p]
    lib.aclrtDestroyContext.restype = ctypes.c_int
    lib.aclrtSetCurrentContext.argtypes = [ctypes.c_void_p]
    lib.aclrtSetCurrentContext.restype = ctypes.c_int
    lib.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.aclrtCreateStream.restype = ctypes.c_int
    lib.aclrtDestroyStream.argtypes = [ctypes.c_void_p]
    lib.aclrtDestroyStream.restype = ctypes.c_int
    lib.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
    lib.aclrtSynchronizeStream.restype = ctypes.c_int

    lib.aclmdlLoadFromFile.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32)]
    lib.aclmdlLoadFromFile.restype = ctypes.c_int
    lib.aclmdlUnload.argtypes = [ctypes.c_uint32]
    lib.aclmdlUnload.restype = ctypes.c_int
    lib.aclmdlCreateDesc.restype = ctypes.c_void_p
    lib.aclmdlDestroyDesc.argtypes = [ctypes.c_void_p]
    lib.aclmdlDestroyDesc.restype = ctypes.c_int
    lib.aclmdlGetDesc.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.aclmdlGetDesc.restype = ctypes.c_int
    lib.aclmdlGetNumOutputs.argtypes = [ctypes.c_void_p]
    lib.aclmdlGetNumOutputs.restype = ctypes.c_size_t
    lib.aclmdlGetOutputSizeByIndex.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.aclmdlGetOutputSizeByIndex.restype = ctypes.c_size_t

    lib.aclmdlCreateDataset.restype = ctypes.c_void_p
    lib.aclmdlDestroyDataset.argtypes = [ctypes.c_void_p]
    lib.aclmdlDestroyDataset.restype = ctypes.c_int
    lib.aclmdlAddDatasetBuffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.aclmdlAddDatasetBuffer.restype = ctypes.c_int
    lib.aclmdlGetDatasetNumBuffers.argtypes = [ctypes.c_void_p]
    lib.aclmdlGetDatasetNumBuffers.restype = ctypes.c_size_t
    lib.aclmdlGetDatasetBuffer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.aclmdlGetDatasetBuffer.restype = ctypes.c_void_p

    lib.aclmdlExecute.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p]
    lib.aclmdlExecute.restype = ctypes.c_int

    lib.aclrtMalloc.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.aclrtMalloc.restype = ctypes.c_int
    lib.aclrtFree.argtypes = [ctypes.c_void_p]
    lib.aclrtFree.restype = ctypes.c_int
    lib.aclrtMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.aclrtMemcpy.restype = ctypes.c_int

    lib.aclCreateDataBuffer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.aclCreateDataBuffer.restype = ctypes.c_void_p
    lib.aclDestroyDataBuffer.argtypes = [ctypes.c_void_p]
    lib.aclDestroyDataBuffer.restype = None
    lib.aclGetDataBufferAddr.argtypes = [ctypes.c_void_p]
    lib.aclGetDataBufferAddr.restype = ctypes.c_void_p
    lib.aclGetDataBufferSizeV2.argtypes = [ctypes.c_void_p]
    lib.aclGetDataBufferSizeV2.restype = ctypes.c_size_t


def _check(lib: ctypes.CDLL, ret: int, name: str) -> None:
    if ret != ACL_SUCCESS:
        raise RuntimeError(f"{name} failed ret={ret}")


class AclDeviceSession:
    """Single aclInit / device / context / stream — shared by multiple OMs."""

    def __init__(self, log_fn: Callable[[str], None]) -> None:
        self._log = log_fn
        self._lib = _load_lib()
        _setup_lib(self._lib)
        self.context: int | None = None
        self.stream: int | None = None
        self._closed = False

        _check(self._lib, self._lib.aclInit(None), "aclInit")
        _check(self._lib, self._lib.aclrtSetDevice(0), "aclrtSetDevice")

        ctx = ctypes.c_void_p()
        _check(self._lib, self._lib.aclrtCreateContext(ctypes.byref(ctx), 0), "aclrtCreateContext")
        self.context = ctx.value
        _check(self._lib, self._lib.aclrtSetCurrentContext(ctx), "aclrtSetCurrentContext")

        stream = ctypes.c_void_p()
        _check(self._lib, self._lib.aclrtCreateStream(ctypes.byref(stream)), "aclrtCreateStream")
        self.stream = stream.value
        self._log("acl device session ready (ctypes)")

    @property
    def lib(self) -> ctypes.CDLL:
        return self._lib

    def close(self) -> None:
        if self._closed:
            return
        lib = self._lib
        if self.stream is not None:
            lib.aclrtDestroyStream(self.stream)
            self.stream = None
        if self.context is not None:
            lib.aclrtDestroyContext(self.context)
            self.context = None
        lib.aclrtResetDevice(0)
        lib.aclFinalize()
        self._closed = True
        self._log("acl device session closed")


class AclLoadedModel:
    """One OM loaded on a shared AclDeviceSession."""

    def __init__(
        self,
        session: AclDeviceSession,
        om_path: str,
        log_fn: Callable[[str], None],
    ) -> None:
        self._session = session
        self._log = log_fn
        self._lib = session.lib
        self.om_path = om_path
        self.model_id = ctypes.c_uint32(0)
        self.model_desc: int | None = None
        self._output_sizes: list[int] = []
        self._closed = False

        path_b = om_path.encode("utf-8")
        _check(
            self._lib,
            self._lib.aclmdlLoadFromFile(path_b, ctypes.byref(self.model_id)),
            f"aclmdlLoadFromFile({om_path})",
        )

        desc = self._lib.aclmdlCreateDesc()
        if not desc:
            raise RuntimeError("aclmdlCreateDesc returned null")
        self.model_desc = desc
        _check(
            self._lib,
            self._lib.aclmdlGetDesc(self.model_desc, self.model_id.value),
            "aclmdlGetDesc",
        )

        n_out = int(self._lib.aclmdlGetNumOutputs(self.model_desc))
        for i in range(n_out):
            size = int(self._lib.aclmdlGetOutputSizeByIndex(self.model_desc, i))
            self._output_sizes.append(size)
            self._log(f"  output[{i}] size={size}")

        self._log(
            f"model resident loaded id={self.model_id.value} outputs={n_out} backend=ctypes"
        )

    def _destroy_dataset(self, dataset: int | None) -> None:
        if not dataset:
            return
        lib = self._lib
        n = int(lib.aclmdlGetDatasetNumBuffers(dataset))
        for i in range(n):
            buf = lib.aclmdlGetDatasetBuffer(dataset, i)
            if not buf:
                continue
            dev_ptr = lib.aclGetDataBufferAddr(buf)
            lib.aclrtFree(dev_ptr)
            lib.aclDestroyDataBuffer(buf)
        lib.aclmdlDestroyDataset(dataset)

    def _build_input_dataset(self, blobs: list[bytes]) -> int:
        lib = self._lib
        dataset = lib.aclmdlCreateDataset()
        if not dataset:
            raise RuntimeError("aclmdlCreateDataset failed")
        for blob in blobs:
            host = (ctypes.c_byte * len(blob)).from_buffer_copy(blob)
            dev_ptr = ctypes.c_void_p()
            _check(
                lib,
                lib.aclrtMalloc(
                    ctypes.byref(dev_ptr),
                    len(blob),
                    ACL_MEM_MALLOC_HUGE_FIRST,
                ),
                "aclrtMalloc input",
            )
            _check(
                lib,
                lib.aclrtMemcpy(
                    dev_ptr,
                    len(blob),
                    ctypes.cast(host, ctypes.c_void_p),
                    len(blob),
                    ACL_MEMCPY_H2D,
                ),
                "aclrtMemcpy H2D",
            )
            data_buf = lib.aclCreateDataBuffer(dev_ptr, len(blob))
            if not data_buf:
                raise RuntimeError("aclCreateDataBuffer input failed")
            _check(lib, lib.aclmdlAddDatasetBuffer(dataset, data_buf), "aclmdlAddDatasetBuffer input")
        return dataset

    def _build_output_dataset(self) -> int:
        lib = self._lib
        dataset = lib.aclmdlCreateDataset()
        if not dataset:
            raise RuntimeError("aclmdlCreateDataset output failed")
        for size in self._output_sizes:
            dev_ptr = ctypes.c_void_p()
            _check(
                lib,
                lib.aclrtMalloc(
                    ctypes.byref(dev_ptr),
                    size,
                    ACL_MEM_MALLOC_HUGE_FIRST,
                ),
                "aclrtMalloc output",
            )
            data_buf = lib.aclCreateDataBuffer(dev_ptr, size)
            if not data_buf:
                raise RuntimeError("aclCreateDataBuffer output failed")
            _check(lib, lib.aclmdlAddDatasetBuffer(dataset, data_buf), "aclmdlAddDatasetBuffer output")
        return dataset

    def execute_job(self, input_dir: Path, output_dir: Path, num_inputs: int) -> None:
        import time

        blobs: list[bytes] = []
        for i in range(num_inputs):
            p = input_dir / f"{i}.bin"
            if not p.is_file():
                raise FileNotFoundError(p)
            blobs.append(p.read_bytes())

        self._log(f"  execute: read {len(blobs)} inputs, total={sum(len(b) for b in blobs)} bytes")
        in_ds = self._build_input_dataset(blobs)
        out_ds = self._build_output_dataset()
        lib = self._lib
        try:
            t0 = time.time()
            self._log("  execute: aclmdlExecute ...")
            _check(
                lib,
                lib.aclmdlExecute(self.model_id.value, in_ds, out_ds),
                "aclmdlExecute",
            )
            if self._session.stream:
                _check(lib, lib.aclrtSynchronizeStream(self._session.stream), "aclrtSynchronizeStream")
            self._log(f"  execute: aclmdlExecute done elapsed={time.time() - t0:.1f}s")
            output_dir.mkdir(parents=True, exist_ok=True)
            n = int(lib.aclmdlGetDatasetNumBuffers(out_ds))
            for idx in range(n):
                buf = lib.aclmdlGetDatasetBuffer(out_ds, idx)
                dev_ptr = lib.aclGetDataBufferAddr(buf)
                size = int(lib.aclGetDataBufferSizeV2(buf))
                host = (ctypes.c_byte * size)()
                _check(
                    lib,
                    lib.aclrtMemcpy(
                        ctypes.cast(host, ctypes.c_void_p),
                        size,
                        dev_ptr,
                        size,
                        ACL_MEMCPY_D2H,
                    ),
                    "aclrtMemcpy D2H",
                )
                (output_dir / f"{idx}.bin").write_bytes(bytes(host))
        finally:
            self._destroy_dataset(in_ds)
            self._destroy_dataset(out_ds)

    def close(self) -> None:
        if self._closed:
            return
        lib = self._lib
        if self.model_desc is not None:
            lib.aclmdlDestroyDesc(self.model_desc)
            self.model_desc = None
        if self.model_id.value:
            lib.aclmdlUnload(self.model_id.value)
            self.model_id = ctypes.c_uint32(0)
        self._closed = True
        self._log("model unloaded")


class AclResidentModelCtypes:
    """Load one OM via libascendcl.so; execute per job (standalone process)."""

    def __init__(self, om_path: str, log_fn: Callable[[str], None]) -> None:
        self._log = log_fn
        self._session = AclDeviceSession(log_fn)
        self._model = AclLoadedModel(self._session, om_path, log_fn)
        self.om_path = om_path
        self.stream = self._session.stream

    def execute_job(self, input_dir: Path, output_dir: Path, num_inputs: int) -> None:
        self._model.execute_job(input_dir, output_dir, num_inputs)

    def close(self) -> None:
        self._model.close()
        self._session.close()
        self._log("model unloaded, acl finalized (ctypes)")


def try_create(om_path: str, log_fn: Callable[[str], None]) -> AclResidentModelCtypes:
    return AclResidentModelCtypes(om_path, log_fn)


class AclPerJobRunner:
    """Worker keeps acl session alive; load OM per job, unload after execute."""

    def __init__(self, om_path: str, log_fn: Callable[[str], None]) -> None:
        self.om_path = om_path
        self._log = log_fn
        self._session = AclDeviceSession(log_fn)

    def execute_job(self, input_dir: Path, output_dir: Path, num_inputs: int) -> None:
        self._log("load OM for job ...")
        model = AclLoadedModel(self._session, self.om_path, self._log)
        try:
            model.execute_job(input_dir, output_dir, num_inputs)
        finally:
            model.close()
            self._log("OM unloaded after job")

    def close(self) -> None:
        self._session.close()


def try_create_perjob(om_path: str, log_fn: Callable[[str], None]) -> AclPerJobRunner:
    return AclPerJobRunner(om_path, log_fn)


def try_create_pool(
    workers: list[tuple[str, str]],
    log_fn: Callable[[str], None],
) -> tuple[AclDeviceSession, dict[str, AclLoadedModel]]:
    session = AclDeviceSession(log_fn)
    models: dict[str, AclLoadedModel] = {}
    for name, om_path in workers:
        log_fn(f"pool loading OM: {name} ...")
        models[name] = AclLoadedModel(
            session,
            om_path,
            lambda msg, n=name: log_fn(f"[{n}] {msg}"),
        )
    return session, models


class LazyModelPool:
    """Single acl session; load/unload OMs on demand with LRU cap."""

    def __init__(
        self,
        session: AclDeviceSession,
        workers: list[tuple[str, str]],
        log_fn: Callable[[str], None],
        max_loaded: int,
    ) -> None:
        self._session = session
        self._log = log_fn
        self._paths = {name: path for name, path in workers}
        self._max_loaded = max(1, max_loaded)
        self._loaded: dict[str, AclLoadedModel] = {}
        self._last_used: dict[str, float] = {}
        self._in_use: set[str] = set()

    def loaded_count(self) -> int:
        return len(self._loaded)

    def _worker_log(self, name: str, msg: str) -> None:
        self._log(f"[{name}] {msg}")

    def _evict_one(self) -> None:
        import time

        candidates = [
            n
            for n in self._loaded
            if n not in self._in_use
        ]
        if not candidates:
            raise RuntimeError(
                f"cannot evict: all {len(self._loaded)} loaded models in use "
                f"(max={self._max_loaded}); reduce batch concurrency or OM_RESIDENT_MAX_LOADED"
            )
        victim = min(candidates, key=lambda n: self._last_used.get(n, 0.0))
        self._log(f"pool evict OM: {victim} (loaded={len(self._loaded)} max={self._max_loaded})")
        self._loaded[victim].close()
        del self._loaded[victim]
        self._last_used.pop(victim, None)

    def ensure_loaded(self, name: str) -> AclLoadedModel:
        import time

        if name not in self._paths:
            raise KeyError(name)
        if name in self._loaded:
            self._last_used[name] = time.time()
            return self._loaded[name]

        om_path = self._paths[name]
        while len(self._loaded) >= self._max_loaded:
            self._evict_one()

        self._log(f"pool load OM: {name} ...")
        try:
            model = AclLoadedModel(
                self._session,
                om_path,
                lambda msg, n=name: self._worker_log(n, msg),
            )
        except RuntimeError as exc:
            hint = ""
            if "507032" in str(exc) or "aclmdlLoadFromFile" in str(exc):
                hint = (
                    f"; NPU HBM full (loaded={len(self._loaded)} max={self._max_loaded}). "
                    "Try OM_RESIDENT_MAX_LOADED=6 or fewer images in flight."
                )
            raise RuntimeError(f"load {name} failed: {exc}{hint}") from exc

        self._loaded[name] = model
        self._last_used[name] = time.time()
        return model

    def execute_job(
        self,
        name: str,
        input_dir: Path,
        output_dir: Path,
        num_inputs: int,
    ) -> None:
        self._in_use.add(name)
        try:
            model = self.ensure_loaded(name)
            model.execute_job(input_dir, output_dir, num_inputs)
        finally:
            self._in_use.discard(name)

    def close(self) -> None:
        for model in list(self._loaded.values()):
            model.close()
        self._loaded.clear()
        self._session.close()


def try_create_lazy_pool(
    workers: list[tuple[str, str]],
    log_fn: Callable[[str], None],
    max_loaded: int,
) -> LazyModelPool:
    session = AclDeviceSession(log_fn)
    log_fn(
        f"lazy pool session ready (max_loaded={max_loaded}, stages={len(workers)})"
    )
    return LazyModelPool(session, workers, log_fn, max_loaded)
