import importlib.util
import os
import sys
import sysconfig
from types import ModuleType
from typing import Any


def _get_python_version_tag() -> str:
    # returns "310", "311", ...
    return sysconfig.get_python_version().replace(".", "")


def _find_export_so_path() -> str:
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    source_root = os.path.dirname(pkg_dir)
    if os.path.isfile(os.path.join(source_root, "setup.py")) and os.path.isfile(
        os.path.join(source_root, "CMakeLists.txt")
    ):
        # A source checkout owns its setuptools build. Never use an old manual
        # copy in the source package or an extension from another installation.
        build_lib = f"lib.{sysconfig.get_platform()}-{sys.implementation.cache_tag}"
        export_path = os.path.join(
            source_root, "build", build_lib, "xllm", f"xllm_export{sysconfig.get_config_var('EXT_SUFFIX')}"
        )
        if os.path.isfile(export_path):
            return export_path
        raise ImportError(
            f"cannot find xllm_export shared library for source checkout {source_root!r}. "
            f"Expected: {export_path!r}. Build the extension with this Python interpreter."
        )

    pyver = _get_python_version_tag()

    # Preferred, exact tags we build for today.
    candidates = [
        os.path.join(pkg_dir, f"xllm_export.cpython-{pyver}-x86_64-linux-gnu.so"),
        os.path.join(pkg_dir, f"xllm_export.cpython-{pyver}-aarch64-linux-gnu.so"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)

    # Fallback: accept any xllm_export*.so that got packaged (tag may differ).
    for fname in os.listdir(pkg_dir):
        if fname.startswith("xllm_export") and fname.endswith(".so"):
            return os.path.abspath(os.path.join(pkg_dir, fname))

    raise ImportError(f"cannot find xllm_export shared library under {pkg_dir!r}. Expected one of: {candidates!r}")


def _load_xllm_export() -> ModuleType:
    loaded_module = sys.modules.get("xllm_export")
    if loaded_module is not None:
        return loaded_module

    export_so_path = _find_export_so_path()

    from xllm._npu_runtime import configure_npu_runtime, preload_npu_ops

    bundled_library = configure_npu_runtime(export_so_path)

    # Initialize torch_npu on the main thread before loading its shared
    # libraries or letting the C++ engine spawn worker threads. It is absent
    # on non-NPU platforms. Both offline entry points need this ordering.
    if importlib.util.find_spec("torch_npu") is not None:
        import torch_npu  # noqa: F401

    if bundled_library is not None:
        preload_npu_ops(bundled_library)

    spec = importlib.util.spec_from_file_location("xllm_export", export_so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to create import spec for xllm_export: {export_so_path}")

    # Make `import xllm_export` work for submodules (pybind/*) by loading and
    # registering it before importing any modules that depend on it.
    module = importlib.util.module_from_spec(spec)
    sys.modules["xllm_export"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("xllm_export", None)
        raise

    # Export the host interpreter so triton_jit's C++ subprocess (MLU JIT
    # compile / signature dump) shells out via this venv's python instead of a
    # build-machine venv path baked into the binary. Set once xllm_export is
    # loaded, before any pybind submodule import or model construction, so the
    # value is in place well before the first kernel compile. popenv reads it
    # at subprocess time; subprocess inherits the parent env.
    os.environ.setdefault("XLLM_TRITON_JIT_PYTHON", sys.executable)
    return module


_PUBLIC_NAMES = {
    "ArgumentParser",
    "Embedding",
    "LLM",
    "LLMMaster",
    "VLM",
    "VLMMaster",
    "Options",
    "SamplingParams",
    "PoolingParams",
    "RequestParams",
    "RequestOutput",
    "Usage",
    "SequenceOutput",
    "Status",
    "StatusCode",
    "MMType",
    "MMData",
    "xllm_export",
}

_PUBLIC_API_LOADED = False


def _load_public_api() -> None:
    global _PUBLIC_API_LOADED
    if _PUBLIC_API_LOADED:
        return

    xllm_export = _load_xllm_export()

    from xllm.pybind.args import ArgumentParser
    from xllm.pybind.embedding import Embedding
    from xllm.pybind.llm import LLM
    from xllm.pybind.params import PoolingParams, SamplingParams

    try:
        from xllm.pybind.vlm import VLM
    except Exception:
        VLM = None

    globals().update(
        {
            "ArgumentParser": ArgumentParser,
            "Embedding": Embedding,
            "LLM": LLM,
            "LLMMaster": xllm_export.LLMMaster,
            "VLM": VLM,
            "VLMMaster": xllm_export.VLMMaster,
            "Options": xllm_export.Options,
            "SamplingParams": SamplingParams,
            "PoolingParams": PoolingParams,
            "RequestParams": xllm_export.RequestParams,
            "RequestOutput": xllm_export.RequestOutput,
            "Usage": xllm_export.Usage,
            "SequenceOutput": xllm_export.SequenceOutput,
            "Status": xllm_export.Status,
            "StatusCode": xllm_export.StatusCode,
            "MMType": xllm_export.MMType,
            "MMData": xllm_export.MMData,
            "xllm_export": xllm_export,
        }
    )
    _PUBLIC_API_LOADED = True


def __getattr__(name: str) -> Any:
    if name == "xllm_export":
        module = _load_xllm_export()
        globals()[name] = module
        return module
    if name in _PUBLIC_NAMES:
        _load_public_api()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _PUBLIC_NAMES)


__all__ = [
    "ArgumentParser",
    "Embedding",
    "LLM",
    "LLMMaster",
    "VLM",
    "VLMMaster",
    "Options",
    "SamplingParams",
    "PoolingParams",
    "RequestParams",
    "RequestOutput",
    "Usage",
    "SequenceOutput",
    "Status",
    "StatusCode",
]
