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

"""Build the thin PyTorch binding against a private CANN operator installation."""

from __future__ import annotations

import os
from pathlib import Path

import torch_npu
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

root = Path(__file__).resolve().parent
cann = Path(os.environ["ASCEND_HOME_PATH"])
private = Path(os.environ["T5_OPP_ROOT"])
headers = list(private.rglob("aclnn_t5_split_qkv_append_cache.h"))
libs = list(private.rglob("libcust_opapi.so"))
if len(headers) != 1 or len(libs) != 1:
    raise RuntimeError("Expected exactly one task ACLNN header/library in T5_OPP_ROOT")
npu = Path(torch_npu.__file__).resolve().parent
direct = root / "build/direct/lib"
if not (direct / "libt5_direct.so").is_file():
    raise RuntimeError("Build the direct launcher first; expected build/direct/lib/libt5_direct.so")
setup(
    name="t5-qkv-append-cache",
    version="0.1.0",
    ext_modules=[
        CppExtension(
            "_t5_qkv",
            [str(root / "csrc/torch_binding.cpp")],
            include_dirs=[
                str(root),
                str(headers[0].parent),
                str(cann / "include"),
                str(npu.parent),
                str(npu / "include"),
            ],
            library_dirs=[str(libs[0].parent), str(cann / "lib64"), str(npu / "lib"), str(direct)],
            libraries=["cust_opapi", "ascendcl", "nnopbase", "torch_npu", "t5_direct"],
            runtime_library_dirs=[str(libs[0].parent), str(cann / "lib64"), str(npu / "lib"), str(direct)],
            extra_compile_args=["-O3", "-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
