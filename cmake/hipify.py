#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import re
import shutil

from torch.utils.hipify.hipify_python import hipify


def norm(path: str) -> str:
    return path.replace("\\", "/")


def map_output_relpath(source_abs: str, project_dir: str) -> str:
    """
    Map:
      <project_dir>/cuda/foo.cu   -> dcu/foo.hip
      <project_dir>/cuda/bar.h    -> dcu/bar.h
      <project_dir>/cuda/baz.cuh  -> dcu/baz.cuh
    """
    rel_path = norm(os.path.relpath(source_abs, project_dir))

    if rel_path.startswith("cuda/"):
        rel_path = rel_path.replace("cuda/", "dcu/", 1)
    else:
        rel_path = rel_path.replace("/cuda/", "/dcu/")

    stem, ext = os.path.splitext(rel_path)
    if ext == ".cu":
      rel_path = stem + ".hip"

    return rel_path


def rewrite_generated_includes(text: str) -> str:
    text = re.sub(
        r'"(?:\.\./)+(?:cuda|hip)/(?:cuda|hip)_ops_api\.h"',
        '"../cuda/cuda_ops_api.h"',
        text,
    )
    text = text.replace('"hip_ops_api.h"', '"../cuda/cuda_ops_api.h"')
    text = text.replace('"cuda_ops_api.h"', '"../cuda/cuda_ops_api.h"')

    # Shared helper headers stay under the generated cuda include tree.
    text = re.sub(r'"(?:\.\./)+(?:cuda|hip)/utils\.h"', '"../cuda/utils.h"', text)

    text = re.sub(r'"(?:\.\./)+(?:cuda|hip)/type_convert\.cuh"', '"type_convert.cuh"', text)
    text = re.sub(r'"(?:\.\./)+(?:cuda|hip)/device_utils\.cuh"', '"device_utils.cuh"', text)
    text = re.sub(r'"(?:\.\./)+(?:cuda|hip)/topk_last_dim\.cuh"', '"topk_last_dim.cuh"', text)
    text = re.sub(r'"(?:\.\./)+(?:cuda|hip)/fp8_quant_utils\.cuh"', '"fp8_quant_utils.cuh"', text)

    text = re.sub(r'"(?:\.\./)+(?:cuda|hip)/moe/', '"moe/', text)

    return text


def copy_if_different(src: str, dst: str) -> None:
    if os.path.exists(dst):
        with open(src, "rb") as fsrc, open(dst, "rb") as fdst:
            if fsrc.read() == fdst.read():
                return

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def write_text_if_different(dst: str, text: str) -> None:
    data = text.encode("utf-8")

    if os.path.exists(dst):
        with open(dst, "rb") as f:
            if f.read() == data:
                return

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    tmp = dst + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)

    os.replace(tmp, dst)


def copy_dcu_headers(src_project_dir: str, final_output_dir: str) -> None:
    dst_dcu_dir = os.path.join(final_output_dir, "dcu")
    os.makedirs(dst_dcu_dir, exist_ok=True)

    src = os.path.join(src_project_dir, "dcu", "dcu_ops_api.h")
    dst = os.path.join(dst_dcu_dir, "dcu_ops_api.h")
    if os.path.exists(src):
        copy_if_different(src, dst)

    dst_cuda_dir = os.path.join(final_output_dir, "cuda")
    os.makedirs(dst_cuda_dir, exist_ok=True)

    for h in [
        "cuda_ops_api.h",
        "utils.h",
    ]:
        src = os.path.join(src_project_dir, "cuda", h)
        dst = os.path.join(dst_cuda_dir, h)
        if os.path.exists(src):
            copy_if_different(src, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--project_dir", required=True,
                        help="Source root containing cuda/ and dcu/")
    parser.add_argument("-o", "--output_dir", required=True,
                        help="Final build output root, e.g. <build>/xllm/core/kernels")
    parser.add_argument("sources", nargs="*", default=[],
                        help="Files to hipify")

    args = parser.parse_args()

    src_project_dir = os.path.abspath(args.project_dir)
    final_output_dir = os.path.abspath(args.output_dir)

    staging_dir = os.path.join(final_output_dir, "_hipify_stage")
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)

    shutil.copytree(
        src_project_dir,
        staging_dir,
        ignore=shutil.ignore_patterns("hip"),
    )

    staged_sources = []
    for s in args.sources:
        s_abs = os.path.abspath(s)
        rel = os.path.relpath(s_abs, src_project_dir)
        staged_sources.append(os.path.join(staging_dir, rel))

    includes = [os.path.join(staging_dir, "*")]

    hipify_result = hipify(
        project_directory=staging_dir,
        output_directory=staging_dir,
        header_include_dirs=[],
        includes=includes,
        extra_files=staged_sources,
        show_detailed=True,
        is_pytorch_extension=True,
        hipify_extra_files_only=True,
    )

    copy_dcu_headers(src_project_dir, final_output_dir)

    final_outputs = []

    for orig_source_abs, staged_source_abs in zip(
        map(os.path.abspath, args.sources), staged_sources
    ):
        hipified_abs = (
            hipify_result[staged_source_abs].hipified_path
            if (
                staged_source_abs in hipify_result
                and hipify_result[staged_source_abs].hipified_path is not None
            )
            else staged_source_abs
        )

        if not os.path.exists(hipified_abs):
            raise FileNotFoundError(
                f"Hipified file not found for {orig_source_abs}: {hipified_abs}"
            )

        rel_out = map_output_relpath(orig_source_abs, src_project_dir)
        final_path = os.path.abspath(os.path.join(final_output_dir, rel_out))
        os.makedirs(os.path.dirname(final_path), exist_ok=True)

        with open(hipified_abs, "r", encoding="utf-8") as f:
            text = f.read()

        text = rewrite_generated_includes(text)

        write_text_if_different(final_path, text)

        final_outputs.append(final_path)

    print("\n".join(final_outputs))
