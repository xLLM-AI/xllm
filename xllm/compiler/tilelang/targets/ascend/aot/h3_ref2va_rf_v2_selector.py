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
"""AOT descriptor for the frozen MiniMax-H3 Ref2VA rf_v2 direct-INT32 selector.

Launch 1 of the two-launch H3 RainFusion hot path. Consumes the original
local-U8 packed TND Q/K/V plus the constant reorder record table and emits the
direct INT32 plans and the rf_v2 reordered Q/K/V triple consumed by
``h3_rainfusion_attention``.

The qualified production geometry is the only compiled specialization; any
other geometry must select a fallback before launch.
"""

import tilelang

from xllm.python.kernels_npu.tilelang import (
    h3_ref2va_rf_v2_selector as kernel_impl,
)
from xllm.python.kernels_npu.tilelang.h3_ref2va_rf_v2_selector import (
    h3_ref2va_rf_v2_selector_rfa,
)

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (kernel_impl,)

# Reference logical geometry for the capacity specialization.
# total_rows is block aligned; valid_rows is the packed used length; the target
# video span starts at video_start and covers grid_t*grid_h*grid_w rows.
PRODUCTION_TOTAL_ROWS = 60160
PRODUCTION_VALID_ROWS = 60132
PRODUCTION_VIDEO_START = 22836
PRODUCTION_GRID_T = 37
PRODUCTION_GRID_H = 24
PRODUCTION_GRID_W = 42
# Active records; the kernel derives records_padded = 48 * ceildiv(count, 48)
# = 5568 from this value, so it must stay 5557 and never the padded extent.
PRODUCTION_RECORD_COUNT = 5557
PRODUCTION_TOPK_R = 94

# The variant key carries an explicit ABI VERSION. V4 (17 parameters: 12 GM handles,
# 4 runtime scalars, stream) and V5 (21 parameters: 12 GM handles, 8 runtime scalars,
# stream) would otherwise share one key while being incompatible C function types,
# which is a mis-link hazard. abi2 == the 8-scalar runtime geometry ABI.
PRODUCTION_ABI_VERSION = 2
PRODUCTION_VARIANT_KEY = "cap_ctr61952_ctb483_crc5568_ctk97_abi2"

# S2 capacity axes. Physical buffers, strides and pool extents are sized from these;
# the logical geometry above is the ACTIVE extent the kernel walks. One capacity object
# therefore serves every geometry that fits, including both sides of the Sort 96->97
# boundary (1440x736 -> 476 blocks / K96 and 1472x736 -> 483 blocks / K97).
PRODUCTION_CAP_TOTAL_ROWS = 61952
PRODUCTION_CAP_TOTAL_BLOCKS = 483
PRODUCTION_CAP_RECORD_COUNT = 5568
PRODUCTION_CAP_TOPK_R = 97


@register_kernel
class H3Ref2vaRfV2SelectorKernel(TilelangKernel):
    KERNEL_NAME = "h3_ref2va_rf_v2_selector"
    DISPATCH_SCHEMA = [
        DispatchField("total_rows", "int32"),
        DispatchField("valid_rows", "int32"),
        DispatchField("video_start", "int32"),
        DispatchField("grid_t", "int32"),
        DispatchField("grid_h", "int32"),
        DispatchField("grid_w", "int32"),
        DispatchField("record_count", "int32"),
        DispatchField("topk_r", "int32"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": PRODUCTION_VARIANT_KEY,
            "total_rows": PRODUCTION_TOTAL_ROWS,
            "valid_rows": PRODUCTION_VALID_ROWS,
            "video_start": PRODUCTION_VIDEO_START,
            "grid_t": PRODUCTION_GRID_T,
            "grid_h": PRODUCTION_GRID_H,
            "grid_w": PRODUCTION_GRID_W,
            "record_count": PRODUCTION_RECORD_COUNT,
            "topk_r": PRODUCTION_TOPK_R,
            "cap_total_blocks": PRODUCTION_CAP_TOTAL_BLOCKS,
            "cap_record_count": PRODUCTION_CAP_RECORD_COUNT,
            "cap_topk_r": PRODUCTION_CAP_TOPK_R,
            "cap_total_rows": PRODUCTION_CAP_TOTAL_ROWS,
        }
    ]

    @staticmethod
    def generate_source(
        total_rows: int,
        valid_rows: int,
        video_start: int,
        grid_t: int,
        grid_h: int,
        grid_w: int,
        record_count: int,
        topk_r: int,
        cap_total_blocks: int = 0,
        cap_record_count: int = 0,
        cap_topk_r: int = 0,
        cap_total_rows: int = 0,
    ) -> str:
        # The frozen kernel carries its own PASS_CONFIGS through @tilelang.jit.
        # Never lower it under xLLM's shared DEFAULT_ASCEND_PASS_CONFIGS: the
        # kernel disables AUTO_SYNC / AUTO_CV_SYNC / AUTO_CV_COMBINE /
        # MEMORY_PLANNING and relies on its manual flag discipline plus the
        # hand-planned UB/L1 address map.
        tilelang.disable_cache()
        kernel = h3_ref2va_rf_v2_selector_rfa(
            total_rows=total_rows,
            valid_rows=valid_rows,
            video_start=video_start,
            grid_t=grid_t,
            grid_h=grid_h,
            grid_w=grid_w,
            record_count=record_count,
            topk_r=topk_r,
            cap_total_blocks=cap_total_blocks,
            cap_record_count=cap_record_count,
            cap_topk_r=cap_topk_r,
            cap_total_rows=cap_total_rows,
        )
        return kernel.get_kernel_source()
