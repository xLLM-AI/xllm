# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Build Python kernels and the optional NPU ATB projection binding."""

import os
from importlib.util import find_spec
from pathlib import Path

from setuptools import setup

extensions = []
commands = {}
if (npu_spec := find_spec("torch_npu")) is not None:
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    npu_root = Path(npu_spec.origin).parent
    ascend_root = Path(os.environ.get("NPU_HOME_PATH") or os.environ["ASCEND_HOME_PATH"])
    atb_root = Path(os.environ["ATB_HOME_PATH"])
    extensions.append(
        CppExtension(
            "xllm_kernel._atb",
            sources=["csrc/atb/matmul_ein_sum.cpp"],
            include_dirs=[
                str(Path(__file__).resolve().parent),
                str(npu_root / "include"),
                str(ascend_root / "include"),
                str(atb_root / "include"),
            ],
            library_dirs=[str(npu_root / "lib"), str(ascend_root / "lib64"), str(atb_root / "lib")],
            libraries=["torch_npu", "atb", "glog"],
            extra_compile_args=["-O2", "-fvisibility=hidden", "-fvisibility-inlines-hidden"],
        )
    )
    commands["build_ext"] = BuildExtension.with_options(no_python_abi_suffix=True)

setup(ext_modules=extensions, cmdclass=commands)
