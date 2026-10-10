# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Optional, ahead-of-time PyTorch bridge; ordinary installs stay pure Python."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path

from setuptools import setup

_ROOT = Path(__file__).resolve().parent
_GVIRT_REVISION = "4a6dc3102d5778928e4d21116bc4d4694d51e584"


def _package_path(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"Building the xlite bridge requires installed {name}")
    return Path(spec.origin).parent


def _extension_options() -> dict:
    if os.environ.get("XLLM_KERNEL_BUILD_XLITE", "0") != "1":
        return {}

    import torch
    import torch_npu
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    npu = _package_path("torch_npu")
    xlite = _package_path("xlite")
    if (xlite / ".xlite_git_head").read_text().strip() != _GVIRT_REVISION:
        raise RuntimeError(f"The xlite bridge requires GVirt {_GVIRT_REVISION}")
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    libraries = ["libxlite_kernels_bf16_npu.so", "libxlite_kernels_f16_npu.so"]
    manifest = {
        "schema_version": 1,
        "gvirt_revision": _GVIRT_REVISION,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "source_sha256": hashlib.sha256((_ROOT / "csrc/xlite/rms_norm.cpp").read_bytes()).hexdigest(),
        "libraries": {name: hashlib.sha256((xlite / "lib" / name).read_bytes()).hexdigest() for name in libraries},
    }

    class XliteBuildExtension(BuildExtension):
        def run(self) -> None:
            super().run()
            destination = Path(self.get_ext_fullpath("xllm_kernel._xlite"))
            destination.with_name("_xlite_build.json").write_text(json.dumps(manifest, indent=2) + "\n")

    library_dirs = [str(npu / "lib"), str(xlite / "lib"), str(cann / "lib64")]
    return {
        "ext_modules": [
            CppExtension(
                "xllm_kernel._xlite",
                sources=["csrc/xlite/rms_norm.cpp"],
                include_dirs=[str(npu / "include"), str(cann / "include")],
                library_dirs=library_dirs,
                libraries=["torch_npu", "ascendcl", "xlite_kernels_bf16_npu", "xlite_kernels_f16_npu"],
                runtime_library_dirs=["$ORIGIN/../torch_npu/lib", "$ORIGIN/../xlite/lib", *library_dirs],
                extra_compile_args=["-O2", "-std=c++17"],
            )
        ],
        "cmdclass": {"build_ext": XliteBuildExtension},
    }


setup(
    options={
        "build": {"build_base": "build/xlite" if os.environ.get("XLLM_KERNEL_BUILD_XLITE") == "1" else "build/python"}
    },
    **_extension_options(),
)
