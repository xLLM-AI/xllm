# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Isolated-compile runtime for the MiniMax-H3 Ref2VA RainFusion pair.

The frozen ``_tilelang`` DSL kernels compile in a short-lived subprocess so the
service process never imports TileLang's TVM next to the CANN runtime TVM. This
module owns only the package-private cache, the subprocess launch and the typed
ctypes calls into the cached shared libraries.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

from xllm_kernel.ops.attention.rainfusion import plan
from xllm_kernel.ops.attention.rainfusion.records import build_reorder_records

_CACHE_SUBDIR = "xllm_rainfusion"
_MANIFEST_FILE = "manifest.json"
_SELECTOR_LIBRARY = "selector.so"
_ATTENTION_LIBRARY = "attention.so"
_KERNEL_SYMBOL = "call"
_SCALAR_COUNT = 8
_RECIPE_VERSION = "rainfusion-isolated-jit-v1"

_SELECTOR_POINTER_ARGS = (
    "q_gm_handle",
    "k_gm_handle",
    "v_gm_handle",
    "records_gm_handle",
    "plan_ids_gm_handle",
    "plan_count_gm_handle",
    "rq_gm_handle",
    "rk_gm_handle",
    "rv_gm_handle",
    "q_pool_gm_handle",
    "k_pool_gm_handle",
    "ring_gm_handle",
)
_SELECTOR_SCALAR_ARGS = (
    "rt_total_blocks",
    "rt_topk_r",
    "rt_record_count",
    "rt_records_per_worker",
    "rt_tail_rows",
    "rt_ff_blocks",
    "rt_vfb",
    "rt_num_sq",
)
_ATTENTION_POINTER_ARGS = (
    "reordered_q_handle",
    "reordered_k_handle",
    "reordered_v_handle",
    "plan_ids_handle",
    "plan_count_handle",
    "reorder_records_flat_handle",
    "output_handle",
    "workspace_s_handle",
    "workspace_p_handle",
    "workspace_o_handle",
    "packed_k_handle",
    "packed_v_handle",
    "reordered_output_handle",
)
_ATTENTION_SCALAR_ARGS = (
    "rt_valid_rows",
    "rt_num_blocks",
    "rt_kv_tail_rows",
    "rt_dense_prefix_blocks",
    "rt_sparse_end_block",
    "rt_record_count",
    "rt_records_per_worker",
    "rt_pack_row_tiles",
)
_KERNEL_ABI = {
    "selector": (_SELECTOR_LIBRARY, _SELECTOR_POINTER_ARGS, _SELECTOR_SCALAR_ARGS),
    "attention": (_ATTENTION_LIBRARY, _ATTENTION_POINTER_ARGS, _ATTENTION_SCALAR_ARGS),
}
_TOOLCHAIN_ENV = (
    "ASCEND_HOME_PATH",
    "ASCEND_HOME",
    "TL_ROOT",
    "TL_RUN_MODE",
    "TL_CCE_AUTO_SYNC",
    "TL_CCE_OPT_LEVEL",
    "TL_PTO_DEBUG",
    "TL_TEMPLATE_PATH",
    "TVM_IMPORT_PYTHON_PATH",
    "TVM_LIBRARY_PATH",
    "ASCEND_RT_VISIBLE_DEVICES",
    "TORCH_DEVICE_BACKEND_AUTOLOAD",
)


@dataclass(frozen=True)
class RainfusionWorkspace:
    """Per-request scratch and selector outputs, allocated once per request."""

    reorder_records: torch.Tensor
    plan_ids: torch.Tensor
    plan_count: torch.Tensor
    reordered_q: torch.Tensor
    reordered_k: torch.Tensor
    reordered_v: torch.Tensor
    q_pool: torch.Tensor
    k_pool: torch.Tensor
    score_ring: torch.Tensor
    workspace_s: torch.Tensor
    workspace_p: torch.Tensor
    workspace_o: torch.Tensor
    packed_k: torch.Tensor
    packed_v: torch.Tensor
    reordered_output: torch.Tensor
    derived: plan.MinimaxH3RainfusionDerived
    record_count: int
    records_padded: int


class _NpuKernel:
    """One cached RainFusion shared library bound to its verified C ABI."""

    def __init__(self, library: Path, pointer_args: tuple[str, ...]) -> None:
        self._lib = ctypes.CDLL(str(library), mode=ctypes.RTLD_LOCAL)
        self._function = getattr(self._lib, _KERNEL_SYMBOL)
        self._function.argtypes = (
            [ctypes.c_void_p] * len(pointer_args) + [ctypes.c_int32] * _SCALAR_COUNT + [ctypes.c_void_p]
        )
        self._function.restype = None
        self._pointer_count = len(pointer_args)

    def run(self, tensors: Sequence[torch.Tensor], scalars: Sequence[int]) -> None:
        """Launch one kernel on the current stream of the tensors' device."""
        if len(tensors) != self._pointer_count:
            raise ValueError(f"RainFusion kernel expects {self._pointer_count} tensors, got {len(tensors)}")
        if len(scalars) != _SCALAR_COUNT:
            raise ValueError(f"RainFusion kernel expects {_SCALAR_COUNT} scalars, got {len(scalars)}")
        device = tensors[0].device
        with torch.npu.device(device):
            stream = torch.npu.current_stream(device)
            self._function(
                *(ctypes.c_void_p(tensor.data_ptr()) for tensor in tensors),
                *(ctypes.c_int32(value) for value in scalars),
                ctypes.c_void_p(int(stream.npu_stream)),
            )


@dataclass(frozen=True)
class RainfusionKernels:
    """The two compiled production specializations for one NPU device."""

    selector: _NpuKernel
    attention: _NpuKernel

    def run_selector(
        self,
        geometry: plan.MinimaxH3RainfusionGeometry,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        workspace: RainfusionWorkspace,
    ) -> None:
        """Write the rf_v2 plans and the reordered Q/K/V triple."""
        derived = workspace.derived
        self.selector.run(
            (
                query,
                key,
                value,
                workspace.reorder_records.view(-1),
                workspace.plan_ids,
                workspace.plan_count,
                workspace.reordered_q,
                workspace.reordered_k,
                workspace.reordered_v,
                workspace.q_pool,
                workspace.k_pool,
                workspace.score_ring,
            ),
            (
                derived.total_blocks,
                geometry.topk_r,
                workspace.record_count,
                _records_per_worker(workspace.record_count),
                derived.tail_rows,
                derived.ff_blocks,
                derived.video_full_blocks,
                derived.num_sq,
            ),
        )

    def run_attention(
        self,
        geometry: plan.MinimaxH3RainfusionGeometry,
        workspace: RainfusionWorkspace,
        output: torch.Tensor,
    ) -> None:
        """Run the mixed dense/sparse attention and the inverse reorder."""
        derived = workspace.derived
        self.attention.run(
            (
                workspace.reordered_q,
                workspace.reordered_k,
                workspace.reordered_v,
                workspace.plan_ids,
                workspace.plan_count,
                workspace.reorder_records.view(-1),
                output,
                workspace.workspace_s,
                workspace.workspace_p,
                workspace.workspace_o,
                workspace.packed_k,
                workspace.packed_v,
                workspace.reordered_output,
            ),
            (
                geometry.valid_rows,
                derived.total_blocks,
                derived.tail_rows,
                derived.ff_blocks,
                derived.video_full_blocks,
                workspace.record_count,
                _records_per_worker(workspace.record_count),
                plan.ceildiv(geometry.valid_rows, plan.KV_PACK_ROWS),
            ),
        )


def _records_per_worker(record_count: int) -> int:
    return plan.ceildiv(record_count, plan.WORKERS)


def specialization_key() -> str:
    """Return the fixed capacity specialization compiled by both kernels."""
    return f"{plan.SELECTOR_VARIANT_KEY}|{plan.ATTENTION_VARIANT_KEY}"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recipe_sources() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    sources = {
        "compile": root / "_compile.py",
        "plan": root / "plan.py",
        "selector_dsl": root / "_tilelang" / "minimax_h3_ref2va_rf_v2_selector.py",
        "attention_dsl": root / "_tilelang" / "minimax_h3_rainfusion_attention.py",
    }
    return {name: _file_sha256(path) for name, path in sources.items()}


def _tilelang_version() -> str:
    try:
        return version("tilelang")
    except PackageNotFoundError:
        return "unavailable"


def _toolchain_identity() -> dict[str, object]:
    env = _child_env()
    bisheng = shutil.which("bisheng", path=env.get("PATH"))
    compiler_version = ""
    if bisheng is not None:
        completed = subprocess.run([bisheng, "--version"], capture_output=True, text=True)
        compiler_version = (completed.stdout or completed.stderr).strip()
    return {
        "bisheng": bisheng,
        "version": compiler_version,
        "env": {name: env.get(name) for name in _TOOLCHAIN_ENV},
    }


def _cache_identity() -> dict[str, object]:
    return {
        "recipe": _RECIPE_VERSION,
        "target": "ascendc",
        "platform": "A3",
        "tilelang": _tilelang_version(),
        "toolchain": _toolchain_identity(),
        "sources": _recipe_sources(),
    }


def _cache_key(identity: dict[str, object] | None = None) -> str:
    payload = json.dumps(_cache_identity() if identity is None else identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_root() -> Path:
    default = os.path.expanduser("~/.tilelang/cache")
    return Path(os.environ.get("TILELANG_CACHE_DIR", default)).expanduser().absolute() / _CACHE_SUBDIR


def _python_candidates() -> list[str]:
    # sys.executable is the xllm host when Python is embedded; sysconfig's
    # installed bindir and PATH provide the same supported interpreter.
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    candidates: list[str] = []
    for executable in (sys.executable, getattr(sys, "_base_executable", None)):
        if executable:
            candidates.append(executable)
    bindir = sysconfig.get_config_var("BINDIR")
    if bindir:
        candidates.extend(str(Path(bindir) / name) for name in (f"python{py_version}", "python3", "python"))
    for name in (f"python{py_version}", "python3", "python"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    return list(dict.fromkeys(candidates))


def _python_executable() -> str:
    expected = f"{sys.version_info.major}.{sys.version_info.minor}"
    for candidate in _python_candidates():
        path = Path(candidate)
        if not path.name.startswith("python") or not path.is_file() or not os.access(path, os.X_OK):
            continue
        probe = subprocess.run(
            [str(path), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0 and probe.stdout.strip() == expected:
            return str(path)
    raise RuntimeError("could not locate the Python interpreter for the RainFusion isolated compiler")


def _package_root() -> Path:
    import xllm_kernel

    return Path(xllm_kernel.__file__).resolve().parent.parent


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    package_root = str(_package_root())
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = package_root if not pythonpath else f"{package_root}{os.pathsep}{pythonpath}"
    env["ASCEND_RT_VISIBLE_DEVICES"] = ""
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    env.pop("TVM_IMPORT_PYTHON_PATH", None)
    env.pop("TVM_LIBRARY_PATH", None)
    return env


def _run_compiler(destination: Path) -> None:
    argv = [
        _python_executable(),
        "-B",
        "-m",
        "xllm_kernel.ops.attention.rainfusion._compile",
        "--output-dir",
        str(destination),
    ]
    completed = subprocess.run(
        argv,
        cwd=_package_root(),
        env=_child_env(),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"RainFusion isolated TileLang compile failed (exit {completed.returncode}):\n{detail}")
    for filename in (_SELECTOR_LIBRARY, _ATTENTION_LIBRARY):
        library = destination / filename
        if not library.is_file() or library.stat().st_size == 0:
            raise RuntimeError(f"RainFusion isolated compiler did not produce {filename}")


def _write_manifest(directory: Path, key: str, identity: dict[str, object]) -> None:
    kernels: dict[str, object] = {}
    for name, (filename, pointer_args, scalar_args) in _KERNEL_ABI.items():
        path = directory / filename
        kernels[name] = {
            "library": filename,
            "sha256": _file_sha256(path),
            "bytes": path.stat().st_size,
            "symbol": _KERNEL_SYMBOL,
            "pointer_args": list(pointer_args),
            "scalar_args": list(scalar_args),
        }
    manifest = {"key": key, "identity": identity, "kernels": kernels}
    temporary = directory / f".{_MANIFEST_FILE}"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, directory / _MANIFEST_FILE)


def _validated_pair(directory: Path, key: str, identity: dict[str, object]) -> tuple[Path, Path] | None:
    manifest_path = directory / _MANIFEST_FILE
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    if manifest.get("key") != key or manifest.get("identity") != identity:
        return None
    kernels = manifest.get("kernels")
    if not isinstance(kernels, dict):
        return None
    paths: list[Path] = []
    for name, (filename, pointer_args, scalar_args) in _KERNEL_ABI.items():
        entry = kernels.get(name)
        if not isinstance(entry, dict):
            return None
        path = directory / filename
        if not path.is_file():
            return None
        if entry.get("library") != filename or entry.get("symbol") != _KERNEL_SYMBOL:
            return None
        if entry.get("pointer_args") != list(pointer_args) or entry.get("scalar_args") != list(scalar_args):
            return None
        if (
            not entry.get("bytes")
            or entry.get("sha256") != _file_sha256(path)
            or entry.get("bytes") != path.stat().st_size
        ):
            return None
        paths.append(path)
    return paths[0], paths[1]


def _ensure_compiled_pair() -> tuple[Path, Path]:
    identity = _cache_identity()
    key = _cache_key(identity)
    root = _cache_root()
    root.mkdir(parents=True, exist_ok=True)
    kernel_dir = root / key
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            paths = _validated_pair(kernel_dir, key, identity)
            if paths is not None:
                return paths
            if kernel_dir.exists():
                shutil.rmtree(kernel_dir)
            temporary = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=root))
            try:
                _run_compiler(temporary)
                _write_manifest(temporary, key, identity)
                if _validated_pair(temporary, key, identity) is None:
                    raise RuntimeError("RainFusion isolated compiler produced an incomplete kernel pair")
                os.rename(temporary, kernel_dir)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    paths = _validated_pair(kernel_dir, key, identity)
    if paths is None:
        raise RuntimeError("RainFusion kernel cache publish is incomplete")
    return paths


def allocate_workspace(
    geometry: plan.MinimaxH3RainfusionGeometry,
    device: torch.device,
) -> RainfusionWorkspace:
    """Allocate the per-request buffers at the qualified physical extents."""
    derived = plan.derive(geometry)
    capacity = plan.QUALIFIED_CAPACITY
    rows = geometry.total_rows
    stack_width = plan.STACK_BLOCKS * plan.BLOCK
    bf16 = torch.bfloat16
    fp32 = torch.float32
    int32 = torch.int32
    return RainfusionWorkspace(
        reorder_records=build_reorder_records(geometry, device),
        plan_ids=torch.empty((capacity.num_blocks, plan.HEADS, capacity.plan_width), dtype=int32, device=device),
        plan_count=torch.empty((capacity.num_blocks, plan.PLAN_ALIGN), dtype=int32, device=device),
        reordered_q=torch.empty((rows, plan.ROW_WIDTH), dtype=bf16, device=device),
        reordered_k=torch.empty((rows, plan.ROW_WIDTH), dtype=bf16, device=device),
        reordered_v=torch.empty((rows, plan.ROW_WIDTH), dtype=bf16, device=device),
        q_pool=torch.empty((plan.HEADS, capacity.q_pool_pad, plan.HEAD_DIM), dtype=bf16, device=device),
        k_pool=torch.empty((plan.HEADS, capacity.k_pool_pad, plan.HEAD_DIM), dtype=bf16, device=device),
        score_ring=torch.empty(
            (plan.NUM_CORES, plan.RING_SLOTS, plan.M_TILE, capacity.k_pool_pad),
            dtype=fp32,
            device=device,
        ),
        workspace_s=torch.empty(
            (plan.TASK_SLOTS, plan.NUM_CORES, plan.BLOCK, stack_width),
            dtype=fp32,
            device=device,
        ),
        workspace_p=torch.empty(
            (plan.TASK_SLOTS, plan.NUM_CORES, plan.BLOCK, stack_width),
            dtype=bf16,
            device=device,
        ),
        workspace_o=torch.empty(
            (plan.TASK_SLOTS, plan.NUM_CORES, plan.BLOCK, plan.HEAD_DIM),
            dtype=fp32,
            device=device,
        ),
        packed_k=torch.empty((plan.HEADS, capacity.total_rows, plan.HEAD_DIM), dtype=bf16, device=device),
        packed_v=torch.empty((plan.HEADS, capacity.total_rows, plan.HEAD_DIM), dtype=bf16, device=device),
        reordered_output=torch.empty((rows, plan.ROW_WIDTH), dtype=bf16, device=device),
        derived=derived,
        record_count=plan.record_count(geometry),
        records_padded=plan.records_padded(geometry),
    )


@cache
def prepare_kernels(device_type: str, device_index: int | None) -> RainfusionKernels:
    """Prepare the cached shared libraries and load their typed callables once."""
    if device_type == "privateuseone":
        device_type = "npu"
    elif device_type != "npu":
        raise ValueError("MiniMax-H3 RainFusion TileLang kernels require an NPU device")

    selector_library, attention_library = _ensure_compiled_pair()
    return RainfusionKernels(
        selector=_NpuKernel(selector_library, _SELECTOR_POINTER_ARGS),
        attention=_NpuKernel(attention_library, _ATTENTION_POINTER_ARGS),
    )


def prepare_for_device(device: torch.device) -> RainfusionKernels:
    """Return the cached specialization pair for this torch NPU device."""
    if device.type not in ("npu", "privateuseone"):
        raise ValueError("MiniMax-H3 RainFusion TileLang kernels require an NPU device")
    return prepare_kernels("npu", device.index)


__all__ = [
    "RainfusionKernels",
    "RainfusionWorkspace",
    "allocate_workspace",
    "prepare_for_device",
    "prepare_kernels",
    "specialization_key",
]
