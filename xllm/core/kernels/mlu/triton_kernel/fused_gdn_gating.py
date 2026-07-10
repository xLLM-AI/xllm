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

# This kernel is adapted from vLLM's FLA Triton ops:
# https://github.com/vllm-project/vllm/tree/v0.18.0/vllm/model_executor/layers/fla/ops
# Upstream license: Apache License, Version 2.0.
# Modified for xLLM MLU TMO integration.

import triton
import triton.language as tl


@triton.jit
def tmo_fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
    core_num: tl.constexpr,
    batch,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    num_block_per = batch // core_num
    num_block_rem = batch % core_num

    core_deal_num_size = num_block_per + (i_b < num_block_rem)
    core_deal_start = num_block_per * i_b + min(num_block_rem, i_b)

    for core_loop in range(0, core_deal_num_size):
        i_b_start = core_deal_start + core_loop
        head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
        off = i_b_start * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
        mask = head_off < NUM_HEADS
        blk_A_log = tl.load(A_log + head_off, mask=mask)
        blk_a = tl.load(a + off, mask=mask)
        blk_b = tl.load(b + off, mask=mask)
        blk_bias = tl.load(dt_bias + head_off, mask=mask)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
        softplus_x = tl.where(
            beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
        )
        blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
        tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
        # compute beta_output = sigmoid(b)
        blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
        tl.store(
            beta_output + off, blk_beta_output.to(beta_output.dtype.element_ty), mask=mask
        )
