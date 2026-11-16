# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Build the independent package, optionally including its native runtime."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class NativeBuild(build_ext):
    def build_extension(self, ext: Extension) -> None:
        source = Path(__file__).resolve().parent
        build = Path(self.build_temp).resolve() / "native"
        output = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        build.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(build),
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output}",
                f"-DXLLM_KERNEL_USE_NPU={os.environ.get('XLLM_KERNEL_USE_NPU', 'OFF')}",
            ],
            check=True,
        )
        subprocess.run(["cmake", "--build", str(build), "--parallel", os.environ.get("MAX_JOBS", "4")], check=True)

        subprocess.run(["cmake", "--install", str(build), "--prefix", str(output)], check=True)

    def get_ext_filename(self, ext_name: str) -> str:
        return ext_name.replace(".", os.sep) + ".so"


setup(
    ext_modules=[Extension("xllm_kernel.libxllm_kernel", sources=[])]
    if os.environ.get("XLLM_KERNEL_BUILD_NATIVE") == "1"
    else [],
    cmdclass={"build_ext": NativeBuild},
)
