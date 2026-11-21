# Copyright 2026 The xLLM Authors. All Rights Reserved.
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
"""AOT descriptor for the frozen final MiniMax-H3 RainFusion attention.

Launch 2 of the two-launch H3 RainFusion hot path. Consumes the selector's
reordered Q/K/V plus its direct INT32 plans, runs the mixed dense/sparse
attention, and applies the record-driven inverse reorder so the returned output
is back in the original packed local-U8 row order.

``output_order`` and ``task_limit`` are build-time constants of the single
production specialization, not runtime dispatch axes, so they appear in the
specialization but deliberately not in ``DISPATCH_SCHEMA``.
"""

import tilelang

from xllm.python.kernels_npu.tilelang import (
    h3_rainfusion_attention as kernel_impl,
)
from xllm.python.kernels_npu.tilelang.h3_rainfusion_attention import (
    h3_rainfusion_attention,
)

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (kernel_impl,)

# Reference logical geometry for the capacity specialization.
PRODUCTION_TOTAL_ROWS = 60160
PRODUCTION_VALID_ROWS = 60132
PRODUCTION_NUM_BLOCKS = 470
# record_count is the active record extent and drives the inverse epilogue's
# per-worker iteration bound; records_padded is the physical [N,4] table extent
# and must stay divisible by the 48 AIV workers. They are not interchangeable.
PRODUCTION_RECORD_COUNT = 5557
PRODUCTION_RECORDS_PADDED = 5568
# Query blocks [0,8) are the video first frame and [291,470) are the physical
# video tail plus the dense context rows; both select all 470 KV blocks. Query
# blocks [8,291) are the sparse region.
PRODUCTION_DENSE_PREFIX_BLOCKS = 8
PRODUCTION_SPARSE_END_BLOCK = 291
PRODUCTION_DTYPE = "bfloat16"
# The production path returns the original packed row order, so the inverse
# record epilogue is always compiled in.
PRODUCTION_OUTPUT_ORDER = "original"
# 0 compiles every (query block, head) task; a nonzero prefix is a JIT-only
# debug facility and must never reach an AOT specialization.
PRODUCTION_TASK_LIMIT = 0

PRODUCTION_VARIANT_KEY = "cap_ctr61952_cnb483_crc5568_crp5568_bf16_abi2"

# Capacity axes for the attention capacity ABI. Physical buffers, strides and GM
# extents come from these; the logical geometry is passed as runtime scalars.
PRODUCTION_CAP_TOTAL_ROWS = 61952
PRODUCTION_CAP_NUM_BLOCKS = 483
PRODUCTION_CAP_RECORD_COUNT = 5568
PRODUCTION_CAP_RECORDS_PADDED = 5568


@register_kernel
class H3RainfusionAttentionKernel(TilelangKernel):
    KERNEL_NAME = "h3_rainfusion_attention"
    DISPATCH_SCHEMA = [
        DispatchField("total_rows", "int32"),
        DispatchField("valid_rows", "int32"),
        DispatchField("num_blocks", "int32"),
        DispatchField("record_count", "int32"),
        DispatchField("records_padded", "int32"),
        DispatchField("dense_prefix_blocks", "int32"),
        DispatchField("sparse_end_block", "int32"),
        DispatchField("dtype", "dtype"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": PRODUCTION_VARIANT_KEY,
            "total_rows": PRODUCTION_TOTAL_ROWS,
            "valid_rows": PRODUCTION_VALID_ROWS,
            "num_blocks": PRODUCTION_NUM_BLOCKS,
            "record_count": PRODUCTION_RECORD_COUNT,
            "records_padded": PRODUCTION_RECORDS_PADDED,
            "dense_prefix_blocks": PRODUCTION_DENSE_PREFIX_BLOCKS,
            "sparse_end_block": PRODUCTION_SPARSE_END_BLOCK,
            "dtype": PRODUCTION_DTYPE,
            "output_order": PRODUCTION_OUTPUT_ORDER,
            "task_limit": PRODUCTION_TASK_LIMIT,
            "cap_total_rows": PRODUCTION_CAP_TOTAL_ROWS,
            "cap_num_blocks": PRODUCTION_CAP_NUM_BLOCKS,
            "cap_record_count": PRODUCTION_CAP_RECORD_COUNT,
            "cap_records_padded": PRODUCTION_CAP_RECORDS_PADDED,
        }
    ]

    @staticmethod
    def generate_source(
        total_rows: int,
        valid_rows: int,
        num_blocks: int,
        record_count: int,
        records_padded: int,
        dense_prefix_blocks: int,
        sparse_end_block: int,
        dtype: str,
        output_order: str,
        task_limit: int,
        cap_total_rows: int = 0,
        cap_num_blocks: int = 0,
        cap_record_count: int = 0,
        cap_records_padded: int = 0,
    ) -> str:
        # The frozen kernel carries its own PASS_CONFIGS through @tilelang.jit.
        # Never lower it under xLLM's shared DEFAULT_ASCEND_PASS_CONFIGS: the
        # three-slot S/P/O ring, L0C slot rotation and the two-slot K preload
        # schedule all depend on the manual flag discipline that AUTO_SYNC /
        # AUTO_CV_SYNC / AUTO_CV_COMBINE / MEMORY_PLANNING would rewrite.
        if output_order != PRODUCTION_OUTPUT_ORDER:
            raise ValueError(f"H3 RainFusion attention AOT only builds output_order={PRODUCTION_OUTPUT_ORDER}")
        if task_limit != 0:
            raise ValueError("H3 RainFusion attention AOT must compile every task; task_limit must be 0")
        tilelang.disable_cache()
        kernel = h3_rainfusion_attention(
            total_rows=total_rows,
            valid_rows=valid_rows,
            num_blocks=num_blocks,
            record_count=record_count,
            records_padded=records_padded,
            dense_prefix_blocks=dense_prefix_blocks,
            sparse_end_block=sparse_end_block,
            cap_total_rows=cap_total_rows,
            cap_num_blocks=cap_num_blocks,
            cap_record_count=cap_record_count,
            cap_records_padded=cap_records_padded,
            dtype=dtype,
            output_order=output_order,
            task_limit=task_limit,
        )
        return kernel.get_kernel_source()
