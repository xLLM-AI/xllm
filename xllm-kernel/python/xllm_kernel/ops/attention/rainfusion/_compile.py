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

"""Isolated TileLang compiler for the MiniMax-H3 RainFusion kernel pair.

This module is executed only by the launcher's compiler subprocess. Keeping the
TileLang and TVM imports here prevents the service process from importing
TileLang's TVM alongside the CANN runtime TVM used by the audio VAE.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tvm.tir import PrimFunc

_SELECTOR_LIBRARY = "selector.so"
_ATTENTION_LIBRARY = "attention.so"


def _selector_program() -> PrimFunc:
    """Build the selector PrimFunc from the pinned DSL wrapper."""
    from xllm_kernel.ops.attention.rainfusion import plan
    from xllm_kernel.ops.attention.rainfusion._tilelang.minimax_h3_ref2va_rf_v2_selector import (
        minimax_h3_ref2va_rf_v2_selector_rfa,
    )

    capacity = plan.QUALIFIED_CAPACITY
    return minimax_h3_ref2va_rf_v2_selector_rfa.__wrapped__(
        total_rows=plan.PRODUCTION_TOTAL_ROWS,
        valid_rows=plan.PRODUCTION_VALID_ROWS,
        video_start=plan.PRODUCTION_VIDEO_START,
        grid_t=plan.PRODUCTION_GRID_T,
        grid_h=plan.PRODUCTION_GRID_H,
        grid_w=plan.PRODUCTION_GRID_W,
        record_count=plan.PRODUCTION_RECORD_COUNT,
        topk_r=plan.PRODUCTION_TOPK_R,
        cap_total_blocks=capacity.num_blocks,
        cap_record_count=capacity.record_count,
        cap_topk_r=capacity.topk_r,
        cap_total_rows=capacity.total_rows,
    )


def _attention_program() -> PrimFunc:
    """Build the attention PrimFunc from the pinned DSL wrapper."""
    from xllm_kernel.ops.attention.rainfusion import plan
    from xllm_kernel.ops.attention.rainfusion._tilelang.minimax_h3_rainfusion_attention import (
        minimax_h3_rainfusion_attention,
    )

    capacity = plan.QUALIFIED_CAPACITY
    return minimax_h3_rainfusion_attention.__wrapped__(
        total_rows=plan.PRODUCTION_TOTAL_ROWS,
        valid_rows=plan.PRODUCTION_VALID_ROWS,
        num_blocks=plan.PRODUCTION_NUM_BLOCKS,
        record_count=plan.PRODUCTION_RECORD_COUNT,
        records_padded=plan.PRODUCTION_RECORDS_PADDED,
        dense_prefix_blocks=plan.PRODUCTION_DENSE_PREFIX_BLOCKS,
        sparse_end_block=plan.PRODUCTION_SPARSE_END_BLOCK,
        cap_total_rows=capacity.total_rows,
        cap_num_blocks=capacity.num_blocks,
        cap_record_count=capacity.record_count,
        cap_records_padded=capacity.records_padded,
        dtype=plan.PRODUCTION_DTYPE,
        output_order="original",
        task_limit=0,
    )


def _kernel_library_path(kernel: object) -> Path:
    """Return the shared library generated for one compiled JIT kernel."""
    adapter = getattr(kernel, "adapter", None)
    libpath = getattr(adapter, "libpath", None)
    if libpath:
        return Path(libpath)
    lib_name = getattr(getattr(adapter, "lib", None), "_name", None)
    if lib_name:
        return Path(lib_name)
    raise RuntimeError("isolated TileLang compile produced no shared library path")


def _compile_into(output_dir: Path) -> None:
    """Compile both production specializations into ``output_dir``."""
    # TileLang's Cython adapter builds torch.device("npu"); this side-effect
    # import registers the alias without initializing or allocating a device.
    import tilelang
    import torch_npu  # noqa: F401

    from xllm_kernel.ops.attention.rainfusion._tilelang.minimax_h3_rainfusion_attention import (
        PASS_CONFIGS as ATTENTION_PASS_CONFIGS,
    )
    from xllm_kernel.ops.attention.rainfusion._tilelang.minimax_h3_ref2va_rf_v2_selector import (
        PASS_CONFIGS as SELECTOR_PASS_CONFIGS,
    )

    # The launcher owns the published cache; TileLang must not rewrite its own
    # shared disk cache while other workers prepare the same cold cache.
    tilelang.disable_cache()
    selector = tilelang.compile(
        _selector_program(),
        execution_backend="cython",
        target="ascendc",
        platform="A3",
        pass_configs=SELECTOR_PASS_CONFIGS,
    )
    attention = tilelang.compile(
        _attention_program(),
        execution_backend="cython",
        target="ascendc",
        platform="A3",
        pass_configs=ATTENTION_PASS_CONFIGS,
    )
    shutil.copyfile(_kernel_library_path(selector), output_dir / _SELECTOR_LIBRARY)
    shutil.copyfile(_kernel_library_path(attention), output_dir / _ATTENTION_LIBRARY)


def main() -> None:
    """Run the isolated compile requested by the RainFusion launcher."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    _compile_into(args.output_dir)


if __name__ == "__main__":
    main()
