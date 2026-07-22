#!/usr/bin/env python3

# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tilelang
import tilelang.language as T

from .utils import DEFAULT_ASCEND_PASS_CONFIGS
from ....common.spec import DispatchField, TilelangKernel, register_kernel

SUPPORTED_SPEC_WIDTHS = (4, 5, 6)
PADDED_WIDTH = 8


def build_spec_verify_metadata_update_kernel(
    spec_width: int,
    block_table_width: int,
):
    if spec_width not in SUPPORTED_SPEC_WIDTHS:
        raise ValueError(f"unsupported speculative verify width: {spec_width}")

    @T.prim_func
    def spec_verify_metadata_update(
        positions: T.Tensor((spec_width,), "int32"),
        kv_seq_lens: T.Tensor((1,), "int32"),
        new_cache_slots: T.Tensor((spec_width,), "int32"),
        block_table: T.Tensor((block_table_width,), "int32"),
        linear_state_index: T.Tensor((1,), "int32"),
        num_accepted: T.Tensor((1,), "int32"),
        persistent_position_0: T.Tensor((PADDED_WIDTH,), "int32"),
        persistent_position_1: T.Tensor((PADDED_WIDTH,), "int32"),
        persistent_position_2: T.Tensor((PADDED_WIDTH,), "int32"),
        persistent_q_seq_lens: T.Tensor((1,), "int32"),
        persistent_kv_seq_lens: T.Tensor((1,), "int32"),
        persistent_new_cache_slots: T.Tensor((PADDED_WIDTH,), "int32"),
        persistent_block_table: T.Tensor((block_table_width,), "int32"),
        persistent_linear_state_index: T.Tensor((1,), "int32"),
        persistent_num_accepted: T.Tensor((1,), "int32"),
        persistent_q_cu_seq_lens: T.Tensor((2,), "int32"),
        persistent_expanded_kv_seq_lens: T.Tensor(
            (PADDED_WIDTH,), "int32"
        ),
        persistent_expanded_block_table_0: T.Tensor(
            (block_table_width,), "int32"
        ),
        persistent_expanded_block_table_1: T.Tensor(
            (block_table_width,), "int32"
        ),
        persistent_expanded_block_table_2: T.Tensor(
            (block_table_width,), "int32"
        ),
        persistent_expanded_block_table_3: T.Tensor(
            (block_table_width,), "int32"
        ),
        persistent_expanded_block_table_4: T.Tensor(
            (block_table_width,), "int32"
        ),
        persistent_expanded_block_table_5: T.Tensor(
            (block_table_width,), "int32"
        ),
    ):
        with T.Kernel(1, is_npu=True):
            with T.Scope("V"):
                positions_ub = T.alloc_ub((PADDED_WIDTH,), "int32")
                slots_ub = T.alloc_ub((PADDED_WIDTH,), "int32")
                block_table_ub = T.alloc_ub(
                    (block_table_width,), "int32"
                )
                q_seq_lens_ub = T.alloc_ub((1,), "int32")
                kv_seq_lens_ub = T.alloc_ub((1,), "int32")
                linear_state_ub = T.alloc_ub((1,), "int32")
                num_accepted_ub = T.alloc_ub((1,), "int32")
                q_cu_seq_lens_ub = T.alloc_ub((2,), "int32")
                expanded_kv_seq_lens_ub = T.alloc_ub(
                    (PADDED_WIDTH,), "int32"
                )

                T.copy(positions[0], positions_ub[0:spec_width])
                T.copy(new_cache_slots[0], slots_ub[0:spec_width])
                T.copy(block_table[0], block_table_ub)
                T.copy(kv_seq_lens[0], kv_seq_lens_ub)
                T.copy(linear_state_index[0], linear_state_ub)
                T.copy(num_accepted[0], num_accepted_ub)

                for i in range(spec_width, PADDED_WIDTH):
                    positions_ub[i] = 0
                    slots_ub[i] = 0

                q_seq_lens_ub[0] = spec_width
                q_cu_seq_lens_ub[0] = 0
                q_cu_seq_lens_ub[1] = spec_width
                for i in T.serial(spec_width):
                    expanded_kv_seq_lens_ub[i] = (
                        kv_seq_lens_ub[0] - spec_width + i + 1
                    )
                for i in range(spec_width, PADDED_WIDTH):
                    expanded_kv_seq_lens_ub[i] = 0

                T.copy(positions_ub, persistent_position_0[0])
                T.copy(positions_ub, persistent_position_1[0])
                T.copy(positions_ub, persistent_position_2[0])
                T.copy(q_seq_lens_ub, persistent_q_seq_lens[0])
                T.copy(kv_seq_lens_ub, persistent_kv_seq_lens[0])
                T.copy(slots_ub, persistent_new_cache_slots[0])
                T.copy(block_table_ub, persistent_block_table[0])
                T.copy(linear_state_ub, persistent_linear_state_index[0])
                T.copy(num_accepted_ub, persistent_num_accepted[0])
                T.copy(q_cu_seq_lens_ub, persistent_q_cu_seq_lens[0])
                T.copy(
                    expanded_kv_seq_lens_ub,
                    persistent_expanded_kv_seq_lens[0],
                )
                T.copy(
                    block_table_ub,
                    persistent_expanded_block_table_0[0],
                )
                T.copy(
                    block_table_ub,
                    persistent_expanded_block_table_1[0],
                )
                T.copy(
                    block_table_ub,
                    persistent_expanded_block_table_2[0],
                )
                T.copy(
                    block_table_ub,
                    persistent_expanded_block_table_3[0],
                )
                T.copy(
                    block_table_ub,
                    persistent_expanded_block_table_4[0],
                )
                T.copy(
                    block_table_ub,
                    persistent_expanded_block_table_5[0],
                )

    return spec_verify_metadata_update


@register_kernel
class SpecVerifyMetadataUpdateKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("spec_width", "int32"),
        DispatchField("block_table_width", "int32"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": f"w{spec_width}_bt64",
            "spec_width": spec_width,
            "block_table_width": 64,
        }
        for spec_width in SUPPORTED_SPEC_WIDTHS
    ]

    @staticmethod
    def generate_source(spec_width: int, block_table_width: int) -> str:
        tilelang.disable_cache()
        kernel = build_spec_verify_metadata_update_kernel(
            spec_width=spec_width,
            block_table_width=block_table_width,
        )
        with tilelang.tvm.transform.PassContext(
            opt_level=3, config=DEFAULT_ASCEND_PASS_CONFIGS
        ):
            lowered = tilelang.engine.lower(kernel)
        return lowered.kernel_source
