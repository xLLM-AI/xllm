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

"""Tests for x_attention_v2 JIT direct-launch path.

Calls the host entry function `x_attention_v2` (non-jit) from the kernel .py,
which internally computes tiling and launches `x_attention_v2_kernel` via JIT.
Golden cache is stored under golden_cache/x_attention/.
"""

import copy
import os
import random
from dataclasses import dataclass

import pytest
import torch
from ml_dtypes import bfloat16

torch_npu = pytest.importorskip("torch_npu")

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_CACHE_DIR = os.path.join(WORKSPACE, "golden_cache", "x_attention")
torch.manual_seed(1)

_DTYPE_NAME = {torch.bfloat16: "bf16", torch.float16: "fp16"}


def _golden_cache_key(
    dtype: torch.dtype,
    num_head: int,
    kv_heads: int,
    request_num: int,
    beam_size: int,
    kv_seqlen: int,
    unshared_seqlen: int,
) -> str:
    dtype_name = _DTYPE_NAME.get(dtype, str(dtype))
    return f"{dtype_name}_{num_head}_{kv_heads}_{request_num}_{beam_size}_{kv_seqlen}_{unshared_seqlen}"


def _save_golden(key: str, data_dict: dict) -> None:
    os.makedirs(GOLDEN_CACHE_DIR, exist_ok=True)
    torch.save(data_dict, os.path.join(GOLDEN_CACHE_DIR, f"{key}.pt"))


def _load_golden(key: str) -> dict | None:
    cache_path = os.path.join(GOLDEN_CACHE_DIR, f"{key}.pt")
    if not os.path.exists(cache_path):
        return None
    return torch.load(cache_path, weights_only=False)


def _gen_seqlen(max_q_seqlen: int, max_kv_seqlen: int, is_varied_len: int, batch: int) -> tuple[list[int], list[int]]:
    q_seqlen_list = []
    kv_seqlen_list = []
    if is_varied_len == 0:
        q_seqlen_list = [max_q_seqlen] * batch
        kv_seqlen_list = [max_kv_seqlen] * batch
    else:
        for _ in range(batch):
            q_seq = random.randint(1, max_q_seqlen)
            kv_seq = random.randint(1, max_kv_seqlen)
            q_seqlen_list.append(q_seq)
            kv_seqlen_list.append(kv_seq)
    return q_seqlen_list, kv_seqlen_list


class TestXAttentionV2:
    @dataclass
    class AttentionInputs:
        query: torch.Tensor
        key_cache: torch.Tensor
        value_cache: torch.Tensor
        unshared_k: torch.Tensor
        unshared_v: torch.Tensor
        block_tables: list
        unshared_block_tables: list
        q_seqlen_list: list
        k_seqlen_list: list
        global_mask: torch.Tensor | None
        mask_type: int
        shape_param: "TestXAttentionV2.GenDataParams"

    @dataclass
    class GenDataParams:
        q_seqlen_list: list
        k_seqlen_list: list
        beam_size: int
        unshared_kvlen: int
        num_heads: int
        kv_heads: int
        head_size: int
        num_blocks: int
        block_size: int
        mask_type: int
        dtype: torch.dtype
        shared_kv_type: int
        unshared_kv_type: int

    @classmethod
    def group_matmul(
        cls,
        head: int,
        kv_head: int,
        left: torch.Tensor,
        right: torch.Tensor,
        right_row: int | None = None,
        right_col: int | None = None,
    ) -> torch.Tensor:
        group_num = head // kv_head
        score = None
        for i in range(kv_head):
            if right_row is None:
                current_right = right[i : (i + 1), :, :]
            else:
                current_right = right[i : (i + 1), :right_row, :right_col]
            left_group = left[i * group_num : (i + 1) * group_num, :, :]
            group_score = torch.matmul(left_group.to(torch.float32), current_right.to(torch.float32))
            score = group_score if score is None else torch.cat((score, group_score), dim=0)
        return score

    @classmethod
    def softmax_numpy(cls, sim: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row_max = torch.max(sim, dim=-1, keepdim=True).values
        sim_sub = sim - row_max
        sim_sub = torch.exp(sim_sub)
        row_sum = torch.sum(sim_sub, dim=-1, keepdim=True)
        soft_res = sim_sub
        return soft_res, row_max, row_sum

    def ref_masked_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query = query.permute(1, 0, 2)
        key = key.permute(1, 2, 0)
        sim_high = self.group_matmul(query.shape[0], key.shape[0], query, key)
        sim_high = sim_high * scale
        p_high, gm, gl = self.softmax_numpy(sim_high)
        p = p_high.to(query.dtype)
        p_high = p_high.to(torch.float32)
        value = value.permute(1, 0, 2)
        out_high = self.group_matmul(query.shape[0], key.shape[0], p_high, value)
        out = self.group_matmul(query.shape[0], key.shape[0], p, value)
        out_high = out_high.permute(1, 0, 2)
        out = out.permute(1, 0, 2)
        out = out.to(query.dtype)
        return out, out_high, gm, gl

    def ref_single_query_unshared_kv_attention(
        self,
        attention_inputs: "TestXAttentionV2.AttentionInputs",
        output: torch.Tensor,
        true_out: torch.Tensor,
        unshared_gl: torch.Tensor,
        unshared_gm: torch.Tensor,
    ) -> None:
        num_heads = attention_inputs.shape_param.num_heads
        kv_heads = attention_inputs.shape_param.kv_heads
        head_size = attention_inputs.shape_param.head_size
        beam_size = attention_inputs.shape_param.beam_size
        request_num = len(attention_inputs.q_seqlen_list)
        batch = beam_size * request_num
        decode_step = attention_inputs.shape_param.unshared_kvlen
        unshared_kv_type = attention_inputs.shape_param.unshared_kv_type
        max_decode_step = (
            attention_inputs.unshared_k.shape[-2]
            if len(attention_inputs.unshared_k.shape) == 5
            else attention_inputs.unshared_k.shape[2]
        )
        scale = 1.0 / (head_size**0.5)
        if unshared_kv_type == 0:
            assert attention_inputs.query.shape == (batch, num_heads, head_size)
            assert attention_inputs.unshared_k.shape == (batch, kv_heads, max_decode_step, head_size)
            assert attention_inputs.unshared_v.shape == (batch, kv_heads, max_decode_step, head_size)
            for i in range(batch):
                q = attention_inputs.query[i : i + 1, :, :]
                k = attention_inputs.unshared_k[i, :, :, :]
                v = attention_inputs.unshared_v[i, :, :, :]
                q_t = q.permute(1, 0, 2)
                k_t = k.permute(0, 2, 1)
                sim = self.group_matmul(num_heads, kv_heads, q_t, k_t, head_size, decode_step)
                sim = sim * scale
                p, gm, gl = self.softmax_numpy(sim)
                gm = gm.permute(1, 0, 2)
                gl = gl.permute(1, 0, 2)
                p_high = p.to(torch.float32)
                out_high = self.group_matmul(num_heads, kv_heads, p_high, v, decode_step, head_size)
                out_high = out_high.permute(1, 0, 2)
                p_low = p.to(attention_inputs.query.dtype)
                out_low = self.group_matmul(num_heads, kv_heads, p_low, v, decode_step, head_size)
                out_low = out_low.permute(1, 0, 2)
                out_low = out_low.to(attention_inputs.query.dtype)
                output[i : i + 1, :, :] = out_low
                true_out[i : i + 1, :, :] = out_high
                unshared_gm[i, :, :] = gm[:, :, :]
                unshared_gl[i, :, :] = gl[:, :, :]
        else:
            assert attention_inputs.query.shape == (batch, num_heads, head_size)
            assert len(attention_inputs.unshared_k.shape) == 5
            assert attention_inputs.unshared_k.shape == attention_inputs.unshared_v.shape
            max_request_num = attention_inputs.unshared_k.shape[0]
            for req_idx in range(request_num):
                if (
                    attention_inputs.unshared_block_tables is not None
                    and len(attention_inputs.unshared_block_tables) > req_idx
                ):
                    cache_idx = attention_inputs.unshared_block_tables[req_idx][0]
                else:
                    cache_idx = req_idx
                for beam_idx in range(beam_size):
                    i = req_idx * beam_size + beam_idx
                    q = attention_inputs.query[i : i + 1, :, :]
                    k = attention_inputs.unshared_k[cache_idx, beam_idx, :, :, :]
                    v = attention_inputs.unshared_v[cache_idx, beam_idx, :, :, :]
                    q_t = q.permute(1, 0, 2)
                    k_t = k.permute(0, 2, 1)
                    sim = self.group_matmul(num_heads, kv_heads, q_t, k_t, head_size, decode_step)
                    sim = sim * scale
                    p, gm, gl = self.softmax_numpy(sim)
                    gm = gm.permute(1, 0, 2)
                    gl = gl.permute(1, 0, 2)
                    p_high = p.to(torch.float32)
                    out_high = self.group_matmul(num_heads, kv_heads, p_high, v, decode_step, head_size)
                    out_high = out_high.permute(1, 0, 2)
                    p_low = p.to(attention_inputs.query.dtype)
                    out_low = self.group_matmul(num_heads, kv_heads, p_low, v, decode_step, head_size)
                    out_low = out_low.permute(1, 0, 2)
                    out_low = out_low.to(attention_inputs.query.dtype)
                    output[i : i + 1, :, :] = out_low
                    true_out[i : i + 1, :, :] = out_high
                    unshared_gm[i, :, :] = gm[:, :, :]
                    unshared_gl[i, :, :] = gl[:, :, :]

    def ref_single_query_shared_kv_attention(
        self,
        attention_inputs: "TestXAttentionV2.AttentionInputs",
        output: torch.Tensor,
        true_out: torch.Tensor,
        shared_gl: torch.Tensor,
        shared_gm: torch.Tensor,
    ) -> None:
        num_heads = attention_inputs.shape_param.num_heads
        kv_heads = attention_inputs.shape_param.kv_heads
        head_size_qk = attention_inputs.shape_param.head_size
        head_size_vo = attention_inputs.shape_param.head_size
        block_size = attention_inputs.shape_param.block_size
        beam_size = attention_inputs.shape_param.beam_size
        request_num = len(attention_inputs.shape_param.q_seqlen_list)
        shared_kv_type = attention_inputs.shape_param.shared_kv_type
        cu_seqlen = 0
        kv_seqlen_now = 0
        layout = "TND"
        for i in range(request_num):
            q_seqlen = int(beam_size)
            k_seqlen = int(attention_inputs.k_seqlen_list[i])
            if layout == "TND":
                q = attention_inputs.query[cu_seqlen : (cu_seqlen + q_seqlen), :, :]
            elif layout == "BSND":
                q = attention_inputs.query[i, :, :, :]
            keys = []
            values = []
            if shared_kv_type == 1:
                block_table = attention_inputs.block_tables[i]
                for j in range(k_seqlen):
                    block_number = int(block_table[j // block_size])
                    block_offset = j % block_size
                    k = attention_inputs.key_cache[block_number, block_offset, :, :]
                    k = k.reshape(kv_heads, head_size_qk)
                    keys.append(k)
                    v = attention_inputs.value_cache[block_number, block_offset, :, :]
                    v = v.reshape(kv_heads, head_size_vo)
                    values.append(v)
            else:
                for j in range(k_seqlen):
                    k = attention_inputs.key_cache[kv_seqlen_now + j, :, :]
                    keys.append(k)
                    v = attention_inputs.value_cache[kv_seqlen_now + j, :, :]
                    values.append(v)
            keys = torch.stack(keys, axis=0)
            values = torch.stack(values, axis=0)
            scale = 1.0 / (head_size_qk**0.5)
            mask = None
            out, out_high, gm, gl = self.ref_masked_attention(q, keys, values, scale, mask)
            out = out.reshape(-1, num_heads, head_size_vo)
            out_high = out_high.reshape(-1, num_heads, head_size_vo)
            gm = gm.permute(1, 0, 2)
            gl = gl.permute(1, 0, 2)
            output[cu_seqlen : cu_seqlen + q_seqlen, :, :] = out
            true_out[cu_seqlen : cu_seqlen + q_seqlen, :, :] = out_high
            shared_gl[cu_seqlen : cu_seqlen + q_seqlen, :, :] = gl
            shared_gm[cu_seqlen : cu_seqlen + q_seqlen, :, :] = gm
            cu_seqlen += q_seqlen
            kv_seqlen_now += k_seqlen

    def call_jit_op(
        self,
        attention_inputs: "TestXAttentionV2.AttentionInputs",
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        unshared_k: torch.Tensor,
        unshared_v: torch.Tensor,
        block_tables: list | None,
        unshared_block_tables: list | None,
        actual_shared_kvlen: list,
        decode_step: int,
    ) -> torch.Tensor:
        shared_kv_type = attention_inputs.shape_param.shared_kv_type
        unshared_kv_type = attention_inputs.shape_param.unshared_kv_type
        q = query.npu()
        k = key_cache.npu()
        v = value_cache.npu()
        unshared_k = unshared_k.npu()
        unshared_v = unshared_v.npu()
        if shared_kv_type == 1:
            block_tables = torch.tensor(copy.deepcopy(block_tables), dtype=torch.int32).npu()
        else:
            block_tables = None
        if unshared_kv_type == 1:
            unshared_block_tables = torch.tensor(copy.deepcopy(unshared_block_tables), dtype=torch.int32).npu()
        else:
            unshared_block_tables = None
        actual_shared_kvlen = torch.tensor(actual_shared_kvlen, dtype=torch.int32).npu()
        decode_step_tensor = torch.tensor([decode_step], dtype=torch.int32).npu()

        from xllm.python.kernels_npu.pypto.x_attention_v2 import x_attention_v2 as jit_x_attention_v2

        attn_out = torch.empty_like(q)
        jit_x_attention_v2(
            q,
            k,
            v,
            unshared_k.contiguous(),
            unshared_v.contiguous(),
            actual_shared_kvlen,
            decode_step_tensor,
            attn_out,
            unshared_block_table=unshared_block_tables,
            shared_block_table=block_tables,
        )
        return attn_out

    def calc_data(self, gen_data_params: "TestXAttentionV2.GenDataParams") -> None:
        cache_key = _golden_cache_key(
            gen_data_params.dtype,
            gen_data_params.num_heads,
            gen_data_params.kv_heads,
            len(gen_data_params.q_seqlen_list),
            gen_data_params.beam_size,
            gen_data_params.k_seqlen_list[0],
            gen_data_params.unshared_kvlen,
        )
        cached = _load_golden(cache_key)
        if cached is not None:
            query = cached["query"]
            key_cache = cached["key_cache"]
            value_cache = cached["value_cache"]
            unshared_key = cached["unshared_key"]
            unshared_value = cached["unshared_value"]
            block_tables = cached["block_tables"]
            unshared_block_tables = cached["unshared_block_tables"]
            actual_shared_kvlen = cached["actual_shared_kvlen"]
            decode_step = cached["decode_step"]
            final_true_out = cached["final_true_out"]

            attention_inputs = self.AttentionInputs(
                query,
                key_cache,
                value_cache,
                unshared_key,
                unshared_value,
                block_tables,
                unshared_block_tables,
                gen_data_params.q_seqlen_list,
                gen_data_params.k_seqlen_list,
                None,
                gen_data_params.mask_type,
                gen_data_params,
            )
            npu_res = self.call_jit_op(
                attention_inputs,
                query,
                key_cache,
                value_cache,
                unshared_key,
                unshared_value,
                block_tables,
                unshared_block_tables,
                actual_shared_kvlen,
                decode_step,
            )
            golden_res = final_true_out
            npu_res = npu_res.cpu().float()
            if gen_data_params.dtype == torch.bfloat16:
                assert torch.allclose(npu_res, golden_res, atol=0.01, rtol=0.01)
            else:
                assert torch.allclose(npu_res, golden_res, atol=0.001, rtol=0.001)
            return

        head_size_qk = gen_data_params.head_size
        head_size_vo = gen_data_params.head_size
        q_min_range = -1.0
        q_max_range = 1.0
        kv_min_range = -1.0
        kv_max_range = 1.0
        beam_size = gen_data_params.beam_size
        request_num = len(gen_data_params.k_seqlen_list)
        decode_step = gen_data_params.unshared_kvlen
        max_decode_step = gen_data_params.unshared_kvlen

        num_tokens = sum(gen_data_params.q_seqlen_list) * beam_size
        num_shared_kv = sum(gen_data_params.k_seqlen_list)

        batch_size = request_num * beam_size
        torch_dtype = gen_data_params.dtype
        query = torch.empty((num_tokens, gen_data_params.num_heads, head_size_qk), dtype=torch_dtype).uniform_(
            q_min_range, q_max_range
        )
        max_k_seqlen = max(gen_data_params.k_seqlen_list)
        block_tables = []
        key_cache = None
        value_cache = None

        if gen_data_params.shared_kv_type == 1:
            key_cache = torch.empty(
                (gen_data_params.num_blocks, gen_data_params.block_size, gen_data_params.kv_heads, head_size_qk),
                dtype=torch_dtype,
            ).uniform_(kv_min_range, kv_max_range)
            value_cache = torch.empty(
                (gen_data_params.num_blocks, gen_data_params.block_size, gen_data_params.kv_heads, head_size_vo),
                dtype=torch_dtype,
            ).uniform_(kv_min_range, kv_max_range)
            max_num_blocks_per_seq = (max_k_seqlen + gen_data_params.block_size - 1) // gen_data_params.block_size
            for i in range(request_num):
                block_table = [max_num_blocks_per_seq * i + j for j in range(max_num_blocks_per_seq)]
                block_tables.append(block_table)
        else:
            key_cache = torch.empty(
                (num_shared_kv, gen_data_params.kv_heads, head_size_qk), dtype=torch_dtype
            ).uniform_(kv_min_range, kv_max_range)
            value_cache = torch.empty(
                (num_shared_kv, gen_data_params.kv_heads, head_size_vo), dtype=torch_dtype
            ).uniform_(kv_min_range, kv_max_range)
            block_tables = None

        unshared_key = None
        unshared_value = None
        unshared_block_tables = None

        if gen_data_params.unshared_kv_type == 0:
            unshared_key = torch.empty(
                (batch_size, gen_data_params.kv_heads, max_decode_step, head_size_qk), dtype=torch_dtype
            ).uniform_(kv_min_range, kv_max_range)
            unshared_value = torch.empty(
                (batch_size, gen_data_params.kv_heads, max_decode_step, head_size_vo), dtype=torch_dtype
            ).uniform_(kv_min_range, kv_max_range)
        else:
            max_request_num = request_num
            unshared_key = torch.empty(
                (max_request_num, beam_size, gen_data_params.kv_heads, max_decode_step, head_size_qk), dtype=torch_dtype
            ).uniform_(kv_min_range, kv_max_range)
            unshared_value = torch.empty(
                (max_request_num, beam_size, gen_data_params.kv_heads, max_decode_step, head_size_vo), dtype=torch_dtype
            ).uniform_(kv_min_range, kv_max_range)
            unshared_block_tables = []
            for i in range(request_num):
                unshared_block_tables.append([request_num - 1 - i])

        shape_out = (num_tokens, gen_data_params.num_heads, head_size_vo)
        sum_max_shape_out = (num_tokens, gen_data_params.num_heads, 1)
        shared_ref_out = torch.zeros(shape_out, dtype=torch_dtype)
        shared_true_out = torch.zeros(shape_out, dtype=torch.float32)
        shared_gl = torch.zeros(sum_max_shape_out, dtype=torch.float32)
        shared_gm = torch.zeros(sum_max_shape_out, dtype=torch.float32)

        unshared_ref_out = torch.zeros(shape_out, dtype=torch_dtype)
        unshared_true_out = torch.zeros(shape_out, dtype=torch.float32)
        unshared_gl = torch.zeros(sum_max_shape_out, dtype=torch.float32)
        unshared_gm = torch.zeros(sum_max_shape_out, dtype=torch.float32)

        attention_inputs = self.AttentionInputs(
            query,
            key_cache,
            value_cache,
            unshared_key,
            unshared_value,
            block_tables,
            unshared_block_tables,
            gen_data_params.q_seqlen_list,
            gen_data_params.k_seqlen_list,
            None,
            gen_data_params.mask_type,
            gen_data_params,
        )

        self.ref_single_query_shared_kv_attention(
            attention_inputs, shared_ref_out, shared_true_out, shared_gl, shared_gm
        )
        self.ref_single_query_unshared_kv_attention(
            attention_inputs, unshared_ref_out, unshared_true_out, unshared_gl, unshared_gm
        )

        gm = torch.maximum(shared_gm, unshared_gm)
        update_shared_expgm = torch.exp(shared_gm - gm)
        update_unshared_expgm = torch.exp(unshared_gm - gm)
        gl = shared_gl * update_shared_expgm + unshared_gl * update_unshared_expgm
        tmp_shared_true = shared_true_out * update_shared_expgm
        tmp_unshared_true = unshared_true_out * update_unshared_expgm
        tmp_add = tmp_shared_true + tmp_unshared_true
        final_true_out = tmp_add / gl

        actual_shared_kvlen = gen_data_params.k_seqlen_list

        _save_golden(
            cache_key,
            {
                "query": query,
                "key_cache": key_cache,
                "value_cache": value_cache,
                "unshared_key": unshared_key,
                "unshared_value": unshared_value,
                "block_tables": block_tables,
                "unshared_block_tables": unshared_block_tables,
                "actual_shared_kvlen": actual_shared_kvlen,
                "decode_step": decode_step,
                "final_true_out": final_true_out,
            },
        )

        npu_res = self.call_jit_op(
            attention_inputs,
            query,
            key_cache,
            value_cache,
            unshared_key,
            unshared_value,
            block_tables,
            unshared_block_tables,
            actual_shared_kvlen,
            decode_step,
        )
        golden_res = final_true_out
        npu_res = npu_res.cpu().float()
        if gen_data_params.dtype == torch.bfloat16:
            assert torch.allclose(npu_res, golden_res, atol=0.01, rtol=0.01)
        else:
            assert torch.allclose(npu_res, golden_res, atol=0.001, rtol=0.001)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_head, kv_heads", [(16, 8), (32, 8), (16, 4)])
@pytest.mark.parametrize("request_num", [1, 6])
@pytest.mark.parametrize("beam_size", [128, 256, 512])
@pytest.mark.parametrize("kv_seqlen", [128, 256, 512, 1024])
@pytest.mark.parametrize("unshared_seqlen", [2, 4])
def test_x_attention_v2_jit_npu(
    dtype: torch.dtype,
    num_head: int,
    kv_heads: int,
    request_num: int,
    beam_size: int,
    kv_seqlen: int,
    unshared_seqlen: int,
) -> None:
    try:
        device_id = int(os.environ.get("ASCEND_DEVICE_ID", 0))
        torch_npu.npu.set_device(device_id)
    except Exception as e:
        pytest.skip(f"NPU device not available: {e}")

    q_seqlen = 1
    embedding_size = 128
    block_size = 128
    is_varied_len = 0
    mask_type = 0
    shared_kv_type = 0
    unshared_kv_type = 1

    q_seqlen_list, kv_seqlen_list = _gen_seqlen(q_seqlen, kv_seqlen, is_varied_len, request_num)
    max_kv_seqlen = max(kv_seqlen_list)
    num_blocks = request_num * ((max_kv_seqlen + block_size - 1) // block_size)

    test_obj = TestXAttentionV2()
    gen_data_params = test_obj.GenDataParams(
        q_seqlen_list,
        kv_seqlen_list,
        beam_size,
        unshared_seqlen,
        num_head,
        kv_heads,
        embedding_size,
        num_blocks,
        block_size,
        mask_type,
        dtype,
        shared_kv_type,
        unshared_kv_type,
    )
    test_obj.calc_data(gen_data_params)
