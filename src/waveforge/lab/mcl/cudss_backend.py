"""Optional FP64 GPU LU with leased, reusable symbolic analysis.

Only released contexts enter the bounded cache. A factor retained by autograd
therefore cannot be overwritten by another solve with the same sparsity pattern.
Compensated refinement and acceptance checks remain in Factorization.
"""

import atexit
import ctypes as c
import hashlib
import os
import threading
from collections import OrderedDict
from importlib.metadata import distribution
from pathlib import Path

import numpy as np
import torch

_lock = threading.RLock()
_library = None
_dll_directories = []
_pool = OrderedDict()
_serial = 0
_stats = {"analyses": 0, "analysis_reuses": 0}
_MAX_IDLE = 4


def _load_library():
    global _library
    with _lock:
        if _library is not None:
            return _library
        major = torch.version.cuda.split(".")[0]
        package = distribution(f"nvidia-cudss-cu{major}")
        if not package.version.startswith("1.0."):
            raise ImportError("this binding requires the validated cuDSS 1.0 API")
        if os.name == "nt":
            folder = Path(package.locate_file(f"nvidia/cu{major}/bin"))
            for directory in (Path(torch.__file__).parent / "lib", folder):
                _dll_directories.append(os.add_dll_directory(str(directory)))
            library = c.WinDLL(str(folder / "cudss64_1.dll"))
        else:
            candidates = [
                p for p in package.files if Path(str(p)).name.startswith("libcudss.so")
            ]
            if not candidates:
                raise ImportError(
                    "cuDSS shared library not found in matching CUDA package"
                )
            library = c.CDLL(str(package.locate_file(candidates[0])))
        pointer = c.c_void_p
        signatures = {
            "cudssCreate": [c.POINTER(pointer)],
            "cudssConfigCreate": [c.POINTER(pointer)],
            "cudssDataCreate": [pointer, c.POINTER(pointer)],
            "cudssSetStream": [pointer, pointer],
            "cudssConfigSet": [pointer, c.c_int, pointer, c.c_size_t],
            "cudssMatrixCreateDn": [c.POINTER(pointer)]
            + [c.c_int64] * 3
            + [pointer, c.c_int, c.c_int],
            "cudssMatrixCreateCsr": [c.POINTER(pointer)]
            + [c.c_int64] * 3
            + [pointer] * 4
            + [c.c_int] * 6,
            "cudssExecute": [pointer, c.c_int] + [pointer] * 5,
            "cudssMatrixDestroy": [pointer],
            "cudssDataDestroy": [pointer, pointer],
            "cudssConfigDestroy": [pointer],
            "cudssDestroy": [pointer],
        }
        for name, signature in signatures.items():
            function = getattr(library, name)
            function.argtypes, function.restype = signature, c.c_int
        _library = library
        return library


class _Context:
    def __init__(self, matrix, nrhs, device):
        self.library = _load_library()
        self.device = device
        self.handle, self.config, self.data = (c.c_void_p() for _ in range(3))
        self.a, self.b, self.x = (c.c_void_p() for _ in range(3))
        self.closed = False
        try:
            with torch.cuda.device(device):
                self.call("cudssCreate", c.byref(self.handle))
                self.stream()
                self.call("cudssConfigCreate", c.byref(self.config))
                self.call("cudssDataCreate", self.handle, c.byref(self.data))
                # No factor downcasting. Outer FP64 compensated corrections are
                # identical to the SciPy path, so native IR is unnecessary.
                for parameter, value in ((5, 0), (14, 1), (28, 0)):
                    self.configure(parameter, value)
                self.n = matrix.shape[0]
                self.offsets = torch.tensor(
                    matrix.indptr, dtype=torch.int32, device=device
                )
                self.columns = torch.tensor(
                    matrix.indices, dtype=torch.int32, device=device
                )
                self.values = torch.tensor(
                    matrix.data, dtype=torch.float64, device=device
                )
                self.rhs = torch.empty(
                    (nrhs, self.n), dtype=torch.float64, device=device
                ).T
                self.solution = torch.empty_like(self.rhs)
                self.host_rhs = torch.empty(
                    (nrhs, self.n), dtype=torch.float64, pin_memory=True
                ).T
                self.host_solution = torch.empty_like(self.host_rhs, pin_memory=True)
                self.call(
                    "cudssMatrixCreateCsr",
                    c.byref(self.a),
                    self.n,
                    self.n,
                    matrix.nnz,
                    self.offsets.data_ptr(),
                    None,
                    self.columns.data_ptr(),
                    self.values.data_ptr(),
                    10,
                    10,
                    1,
                    0,
                    0,
                    0,
                )
                for descriptor, values in ((self.b, self.rhs), (self.x, self.solution)):
                    self.call(
                        "cudssMatrixCreateDn",
                        c.byref(descriptor),
                        self.n,
                        nrhs,
                        self.n,
                        values.data_ptr(),
                        1,
                        0,
                    )
                self.execute(3)
                self.execute(4)
        except BaseException:
            self.close()
            raise

    def call(self, name, *args):
        status = getattr(self.library, name)(*args)
        if status:
            raise RuntimeError(f"{name} failed with cuDSS status {status}")

    def configure(self, parameter, value):
        value = c.c_int(value)
        self.call(
            "cudssConfigSet", self.config, parameter, c.byref(value), c.sizeof(value)
        )

    def stream(self):
        self.call(
            "cudssSetStream",
            self.handle,
            c.c_void_p(torch.cuda.current_stream(self.device).cuda_stream),
        )

    def execute(self, phase):
        self.call(
            "cudssExecute",
            self.handle,
            phase,
            self.config,
            self.data,
            self.a,
            self.x,
            self.b,
        )

    def refactor(self, matrix):
        with torch.cuda.device(self.device):
            self.stream()
            self.values.copy_(torch.from_numpy(matrix.data))
            self.execute(8)

    def solve(self, rhs, transpose):
        with torch.cuda.device(self.device):
            self.stream()
            self.configure(4, int(transpose))
            np.copyto(self.host_rhs.numpy(), rhs)
            self.rhs.copy_(self.host_rhs, non_blocking=True)
            self.execute(1008)
            self.host_solution.copy_(self.solution, non_blocking=True)
            torch.cuda.current_stream(self.device).synchronize()
            return self.host_solution.numpy().copy()

    def close(self):
        if self.closed:
            return
        self.closed = True
        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device)
            for descriptor in (self.x, self.b, self.a):
                if descriptor.value:
                    self.call("cudssMatrixDestroy", descriptor)
            if self.data.value:
                self.call("cudssDataDestroy", self.handle, self.data)
            if self.config.value:
                self.call("cudssConfigDestroy", self.config)
            if self.handle.value:
                self.call("cudssDestroy", self.handle)


def _release(key, context):
    global _serial
    with _lock:
        _serial += 1
        _pool[_serial] = (key, context)
        while len(_pool) > _MAX_IDLE:
            _, (_, old) = _pool.popitem(last=False)
            old.close()


def clear_pool():
    """Free idle GPU factors; live factors remain independently owned."""
    with _lock:
        while _pool:
            _, (_, context) = _pool.popitem(last=False)
            context.close()


def pool_info():
    with _lock:
        return dict(_stats, idle_contexts=len(_pool))


class CudssLU:
    """Small SuperLU-compatible interface; real FP64, vector/multiple RHS."""

    def __init__(self, matrix):
        matrix = matrix.tocsr(copy=True)
        matrix.sum_duplicates()
        matrix.sort_indices()
        if (
            matrix.shape[0] != matrix.shape[1]
            or matrix.dtype != np.float64
            or not np.isfinite(matrix.data).all()
            or max(matrix.shape[0], matrix.nnz) > np.iinfo(np.int32).max
        ):
            raise ValueError(
                "cuDSS requires square, finite real FP64, int32-indexable CSR"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("cuDSS requires an available CUDA GPU")
        self.matrix = matrix
        self.device = torch.device("cuda", torch.cuda.current_device())
        digest = hashlib.sha256(
            matrix.indptr.tobytes() + matrix.indices.tobytes()
        ).digest()
        self.key = (self.device.index, matrix.shape, digest)
        self.contexts = {}
        self.lock = threading.RLock()
        self.closed = False

    def solve(self, rhs, trans="N"):
        if trans not in ("N", "T", "H"):
            raise ValueError("cuDSS transpose must be N, T or H")
        rhs = np.asarray(rhs)
        vector = rhs.ndim == 1
        if vector:
            rhs = rhs[:, None]
        if (
            rhs.ndim != 2
            or rhs.shape[0] != self.matrix.shape[0]
            or rhs.shape[1] < 1
            or rhs.dtype != np.float64
            or not np.isfinite(rhs).all()
        ):
            raise ValueError("finite compatible FP64 right hand side required")
        with self.lock:
            if self.closed:
                raise RuntimeError("GPU factor already closed")
            nrhs = rhs.shape[1]
            key = (*self.key, nrhs)
            context = self.contexts.get(nrhs)
            try:
                if context is None:
                    with _lock:
                        for token, (cached_key, cached) in list(_pool.items()):
                            if cached_key == key:
                                del _pool[token]
                                context = cached
                                break
                    if context is None:
                        context = _Context(self.matrix, nrhs, self.device)
                        with _lock:
                            _stats["analyses"] += 1
                    else:
                        context.refactor(self.matrix)
                        with _lock:
                            _stats["analysis_reuses"] += 1
                    self.contexts[nrhs] = context
                answer = context.solve(rhs, trans != "N")
            except BaseException:
                self.contexts.pop(nrhs, None)
                if context is not None:
                    context.close()
                raise
        return answer[:, 0] if vector else answer

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                for nrhs, context in self.contexts.items():
                    _release((*self.key, nrhs), context)
                self.contexts.clear()

    def __del__(self):
        if hasattr(self, "contexts"):
            self.close()


atexit.register(clear_pool)
