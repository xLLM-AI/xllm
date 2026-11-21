# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Build the independent PackedPageIndices host extension with Ninja."""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="xllm-packed-page-indices",
    version="0.1.0",
    py_modules=["solution", "kernels"],
    ext_modules=[
        CppExtension(
            "_packed_page_indices_host",
            [str(Path(__file__).with_name("host.cpp"))],
            extra_compile_args=["-O2", "-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
