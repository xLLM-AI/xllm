# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Triton normalization kernels, launchers and startup registration."""

# Triton visits annotation expressions; quoted runtime types must stay quoted.
# ruff: noqa: UP037

from __future__ import annotations

from importlib.util import find_spec
from typing import Any

import torch

from xllm_kernel.ops.normalization import (
    FUSED_EH_NORM_CONTRACT,
    FUSED_QK_NORM_ROPE_CONTRACT,
    L2_NORM_CONTRACT,
    RMS_NORM_SIGMOID_GATED_CONTRACT,
    TRITON_JIT_REQUIRES,
)
from xllm_kernel.platform import PlatformEnum, current_platform
from xllm_kernel.registry import register_kernel

if current_platform.is_npu():
    # JIT-only preparation does not require the host's compiled L2 schema;
    # compiled L2 preparation does not require the optional Python JIT SDKs.
    if hasattr(torch.ops.xllm_ops, "l2_norm"):
        _l2_norm = torch.ops.xllm_ops.l2_norm

        @register_kernel(
            name="npu.triton.l2_norm.aot",
            contract=L2_NORM_CONTRACT,
            platform=PlatformEnum.NPU,
            execution_modes=("eager", "aclgraph"),
            input_domain="Nonempty BF16 NPU tensor; L2 normalization over the last dimension with epsilon added before rsqrt.",
            solution="triton",
            variant="aot",
            binding="xllm_ops",
            priority=10,
        )
        def l2_norm(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
            """Normalize the last dimension of ``value`` to unit L2 norm.

            Args:
                value: Tensor whose last dimension is normalized.
                eps: Added to the squared norm before the reciprocal square root.

            Returns:
                A tensor with the shape and dtype of ``value``.
            """
            return _l2_norm(value, eps)

    if all(find_spec(dependency) is not None for dependency in TRITON_JIT_REQUIRES):
        import triton
        import triton.language as tl

        from xllm_kernel.utils.triton import (
            extract_slice,
            get_element,
            get_vectorcore_num,
            insert_slice,
        )

        @triton.jit
        def _rms_norm_sigmoid_gated_kernel(
            value_ptr: "Any",
            gate_ptr: "Any",
            weight_ptr: "Any",
            output_ptr: "Any",
            num_rows: "Any",
            stride_value_row: "Any",
            stride_gate_row: "Any",
            stride_output_row: "Any",
            eps: "Any",
            FEATURE_DIM: tl.constexpr,
            BLOCK_FEATURE: tl.constexpr,
            ROWS_PER_BLOCK: tl.constexpr,
        ) -> None:
            """Apply one independent RMS reduction to each row, then sigmoid-gate."""
            row_start = tl.program_id(0) * ROWS_PER_BLOCK
            rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
            features = tl.arange(0, BLOCK_FEATURE)
            row_mask = rows[:, None] < num_rows
            feature_mask = features[None, :] < FEATURE_DIM
            mask = row_mask & feature_mask

            value_offsets = rows[:, None] * stride_value_row + features[None, :]
            values = tl.load(value_ptr + value_offsets, mask=mask, other=0.0).to(tl.float32)
            square_sum = tl.sum(tl.where(mask, values * values, 0.0), axis=1)
            inverse_rms = 1.0 / tl.sqrt(square_sum / FEATURE_DIM + eps)

            weights = tl.load(
                weight_ptr + features,
                mask=features < FEATURE_DIM,
                other=0.0,
            ).to(tl.float32)
            normalized = values * inverse_rms[:, None] * weights[None, :]

            gate_offsets = rows[:, None] * stride_gate_row + features[None, :]
            gates = tl.load(gate_ptr + gate_offsets, mask=mask, other=0.0).to(tl.float32)
            output = normalized * tl.sigmoid(gates)

            output_offsets = rows[:, None] * stride_output_row + features[None, :]
            tl.store(output_ptr + output_offsets, output, mask=mask)

        @register_kernel(
            name="npu.triton.rms_norm_sigmoid_gated",
            contract=RMS_NORM_SIGMOID_GATED_CONTRACT,
            platform=PlatformEnum.NPU,
            execution_modes=("eager", "aclgraph"),
            input_domain="Matching NPU value/gate shape and dtype, matching 1D weight, feature row below 64 KiB.",
            solution="triton",
            binding="triton.jit",
            priority=10,
        )
        def rms_norm_sigmoid_gated(
            value: torch.Tensor,
            gate: torch.Tensor,
            weight: torch.Tensor,
            eps: float,
        ) -> torch.Tensor:
            """Compute ``RMSNorm(value) * sigmoid(gate)`` in one NPU kernel.

            GLM-5.3-Flash uses this operation on KDA heads with a 128-element feature
            dimension. Processing a fixed number of rows per Triton program keeps the
            reduction tree for each row independent of the decode batch size.
            """
            if value.shape != gate.shape:
                raise ValueError("value and gate must have identical shapes")
            if value.dtype != gate.dtype:
                raise ValueError("value and gate must have identical dtypes")

            original_shape = value.shape
            value_2d = value.reshape(-1, value.shape[-1])
            if value_2d.stride(-1) != 1:
                value_2d = value_2d.contiguous()
            num_rows, feature_dim = value_2d.shape
            block_feature = min(65536 // value.element_size(), triton.next_power_of_2(feature_dim))
            if feature_dim > block_feature:
                raise RuntimeError("RMSNorm does not support feature dimensions >= 64 KiB")
            gate_2d = gate.reshape(-1, gate.shape[-1])
            if gate_2d.stride(-1) != 1:
                gate_2d = gate_2d.contiguous()
            weight = weight.contiguous()
            if weight.shape != (feature_dim,):
                raise ValueError("weight must match the final input dimension")

            rows_per_block = 32
            output = torch.empty_like(value_2d)
            grid = (triton.cdiv(num_rows, rows_per_block),)
            _rms_norm_sigmoid_gated_kernel[grid](
                value_2d,
                gate_2d,
                weight,
                output,
                num_rows,
                value_2d.stride(0),
                gate_2d.stride(0),
                output.stride(0),
                eps,
                FEATURE_DIM=feature_dim,
                BLOCK_FEATURE=block_feature,
                ROWS_PER_BLOCK=rows_per_block,
                num_warps=4,
            )
            return output.reshape(original_shape)

        @triton.jit
        def _fused_eh_norm_kernel(
            embed_ptr: "Any",
            carried_ptr: "Any",
            enorm_weight_ptr: "Any",
            hnorm_weight_ptr: "Any",
            output_ptr: "Any",
            num_rows: "Any",
            stride_embed_row: "Any",
            stride_carried_row: "Any",
            stride_output_row: "Any",
            eps: "Any",
            FEATURE_DIM: tl.constexpr,
            BLOCK_FEATURE: tl.constexpr,
            ROWS_PER_BLOCK: tl.constexpr,
        ) -> None:
            """Per-row RMSNorm of embed and carried, written side by side into [.., 2H]."""
            row_start = tl.program_id(0) * ROWS_PER_BLOCK
            rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
            features = tl.arange(0, BLOCK_FEATURE)
            row_mask = rows[:, None] < num_rows
            feature_mask = features[None, :] < FEATURE_DIM
            mask = row_mask & feature_mask

            embed = tl.load(embed_ptr + rows[:, None] * stride_embed_row + features[None, :], mask=mask, other=0.0).to(
                tl.float32
            )
            inv_e = 1.0 / tl.sqrt(tl.sum(tl.where(mask, embed * embed, 0.0), axis=1) / FEATURE_DIM + eps)
            enorm_w = tl.load(enorm_weight_ptr + features, mask=features < FEATURE_DIM, other=0.0).to(tl.float32)
            out_e = embed * inv_e[:, None] * enorm_w[None, :]
            tl.store(output_ptr + rows[:, None] * stride_output_row + features[None, :], out_e, mask=mask)

            carried = tl.load(
                carried_ptr + rows[:, None] * stride_carried_row + features[None, :], mask=mask, other=0.0
            ).to(tl.float32)
            inv_h = 1.0 / tl.sqrt(tl.sum(tl.where(mask, carried * carried, 0.0), axis=1) / FEATURE_DIM + eps)
            hnorm_w = tl.load(hnorm_weight_ptr + features, mask=features < FEATURE_DIM, other=0.0).to(tl.float32)
            out_h = carried * inv_h[:, None] * hnorm_w[None, :]
            tl.store(
                output_ptr + rows[:, None] * stride_output_row + FEATURE_DIM + features[None, :],
                out_h,
                mask=mask,
            )

        @register_kernel(
            name="npu.triton.fused_eh_norm",
            contract=FUSED_EH_NORM_CONTRACT,
            platform=PlatformEnum.NPU,
            execution_modes=("eager", "aclgraph"),
            input_domain="Matching NPU embedding/carried shape and dtype; final feature row below 64 KiB; concatenate normalized inputs.",
            solution="triton",
            binding="triton.jit",
            priority=10,
        )
        def fused_eh_norm(
            embed: torch.Tensor,
            carried: torch.Tensor,
            enorm_weight: torch.Tensor,
            hnorm_weight: torch.Tensor,
            eps: float,
        ) -> torch.Tensor:
            """Compute ``cat(RMSNorm(embed, enorm_w), RMSNorm(carried, hnorm_w))`` in one kernel.

            ``embed`` and ``carried`` share shape ``[..., H]``; the result is
            ``[..., 2H]`` (embed norm first, carried norm second) ready for ``eh_proj``.
            """
            if embed.shape != carried.shape:
                raise ValueError("embed and carried must have identical shapes")
            if embed.dtype != carried.dtype:
                raise ValueError("embed and carried must have identical dtypes")

            original_shape = embed.shape
            feature_dim = embed.shape[-1]
            embed_2d = embed.reshape(-1, feature_dim)
            carried_2d = carried.reshape(-1, feature_dim)
            if embed_2d.stride(-1) != 1:
                embed_2d = embed_2d.contiguous()
            if carried_2d.stride(-1) != 1:
                carried_2d = carried_2d.contiguous()
            enorm_weight = enorm_weight.contiguous()
            hnorm_weight = hnorm_weight.contiguous()
            if enorm_weight.shape != (feature_dim,) or hnorm_weight.shape != (feature_dim,):
                raise ValueError("norm weights must match the final input dimension")

            block_feature = min(65536 // embed.element_size(), triton.next_power_of_2(feature_dim))
            if feature_dim > block_feature:
                raise RuntimeError("fused eh-norm does not support feature dimensions >= 64 KiB")

            num_rows = embed_2d.shape[0]
            rows_per_block = max(1, min(32, 4096 // block_feature))
            output = torch.empty(num_rows, 2 * feature_dim, dtype=embed.dtype, device=embed.device)
            grid = (triton.cdiv(num_rows, rows_per_block),)
            _fused_eh_norm_kernel[grid](
                embed_2d,
                carried_2d,
                enorm_weight,
                hnorm_weight,
                output,
                num_rows,
                embed_2d.stride(0),
                carried_2d.stride(0),
                output.stride(0),
                eps,
                FEATURE_DIM=feature_dim,
                BLOCK_FEATURE=block_feature,
                ROWS_PER_BLOCK=rows_per_block,
                num_warps=4,
            )
            return output.reshape(*original_shape[:-1], 2 * feature_dim)

        @triton.jit
        def _split_qkv_rmsnorm_rope_kernel(
            input_gm_ptr: "Any",
            q_gm_ptr: "Any",
            k_gm_ptr: "Any",
            v_gm_ptr: "Any",
            q_weight_ptr: "Any",
            k_weight_ptr: "Any",
            batch_size: "Any",
            q_hidden_size: tl.constexpr,
            kv_hidden_size: tl.constexpr,
            total_hidden_size: tl.constexpr,
            eps: tl.constexpr,
            HEAD_DIM: tl.constexpr,
            ROPE_DIM: tl.constexpr,
            HALF_ROPE_DIM: tl.constexpr,
            num_vectorcore: tl.constexpr,
            batch_size_per_iter_per_vec: tl.constexpr,
            qk_head_nums_per_iter_per_vec: tl.constexpr,
            q_head_num: tl.constexpr,
            kv_head_num: tl.constexpr,
            qk_head_num_sum: tl.constexpr,
            v_batch_size_per_iter_per_vec: tl.constexpr,
            positions_gm_ptr: "Any",
            cos_sin_cache_gm_ptr: "Any",
        ) -> None:
            row_pid = tl.program_id(0)

            q_weight_values = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM))
            k_weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM))

            batch_size_per_vec = tl.cdiv(batch_size, num_vectorcore)
            iter_num_per_vec = tl.cdiv(batch_size_per_vec, batch_size_per_iter_per_vec)
            v_iter_num_per_vec = tl.cdiv(batch_size_per_vec, v_batch_size_per_iter_per_vec)
            input_batch_offset = row_pid * batch_size_per_vec
            mblk_idx = tl.arange(0, batch_size_per_iter_per_vec) + input_batch_offset
            nblk_idx = tl.arange(0, q_hidden_size + kv_hidden_size)
            nmask = nblk_idx < total_hidden_size

            input_batch_offset_end = min(input_batch_offset + batch_size_per_vec, batch_size)

            pos_indices = input_batch_offset + tl.arange(0, batch_size_per_iter_per_vec)
            output_q_nblk_idx = tl.arange(0, q_hidden_size)
            output_q_nmask = output_q_nblk_idx < q_hidden_size
            output_kv_nblk_idx = tl.arange(0, kv_hidden_size)
            output_kv_nmask = output_kv_nblk_idx < kv_hidden_size
            sin_cos_range = tl.arange(0, ROPE_DIM)
            cos_sin_cache_offset = cos_sin_cache_gm_ptr + sin_cos_range

            for iter in tl.range(iter_num_per_vec):
                pos_offset = iter * batch_size_per_iter_per_vec
                x = tl.load(
                    positions_gm_ptr + pos_indices + pos_offset,
                    mask=(pos_indices + pos_offset) < input_batch_offset_end,
                )
                mmask = (mblk_idx + pos_offset) < input_batch_offset_end
                mask = (mmask[:, None]) & (nmask[None, :])
                idx = (mblk_idx + pos_offset)[:, None] * total_hidden_size + nblk_idx[None, :]
                values_tmp1 = tl.load(input_gm_ptr + idx, mask=mask).reshape(qk_head_nums_per_iter_per_vec, HEAD_DIM)

                values_tmp3 = tl.zeros((batch_size_per_iter_per_vec, ROPE_DIM), dtype=tl.bfloat16)
                for i in tl.range(batch_size_per_iter_per_vec):
                    pos = get_element(x, (i,))
                    values_tmp3 = insert_slice(
                        values_tmp3.reshape(batch_size_per_iter_per_vec, ROPE_DIM),
                        tl.load(pos * ROPE_DIM + cos_sin_cache_offset[:, None]).reshape(1, ROPE_DIM),
                        offsets=(i, 0),
                        sizes=(1, ROPE_DIM),
                        strides=(1, 1),
                    )
                values_tmp3 = values_tmp3.reshape(batch_size_per_iter_per_vec, 1, ROPE_DIM)
                cos = extract_slice(
                    values_tmp3,
                    offsets=(0, 0, 0),
                    sizes=(batch_size_per_iter_per_vec, 1, HALF_ROPE_DIM),
                    strides=(1, 1, 1),
                )
                sin = extract_slice(
                    values_tmp3,
                    offsets=(0, 0, HALF_ROPE_DIM),
                    sizes=(batch_size_per_iter_per_vec, 1, HALF_ROPE_DIM),
                    strides=(1, 1, 1),
                )

                normalized_values = values_tmp1.to(tl.float32)
                normalized_values = normalized_values * normalized_values
                normalized_values = tl.sum(normalized_values, axis=1) / HEAD_DIM
                normalized_values = 1 / tl.sqrt(normalized_values + eps).reshape(qk_head_nums_per_iter_per_vec, 1)
                normalized_values = values_tmp1 * normalized_values

                # Q: norm * weight + rope
                normalized_values_tmp = extract_slice(
                    normalized_values.reshape(batch_size_per_iter_per_vec, qk_head_num_sum, HEAD_DIM),
                    offsets=(0, 0, 0),
                    sizes=(batch_size_per_iter_per_vec, q_head_num, HEAD_DIM),
                    strides=(1, 1, 1),
                )
                normalized_values_tmp = (normalized_values_tmp * q_weight_values).to(tl.bfloat16)

                values_tmp = tl.zeros((batch_size_per_iter_per_vec, q_head_num, ROPE_DIM), dtype=tl.bfloat16)
                x1 = extract_slice(
                    normalized_values_tmp,
                    offsets=(0, 0, 0),
                    sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
                    strides=(1, 1, 1),
                )
                x2 = extract_slice(
                    normalized_values_tmp,
                    offsets=(0, 0, HALF_ROPE_DIM),
                    sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
                    strides=(1, 1, 1),
                )
                values_tmp = insert_slice(
                    values_tmp,
                    x1 * cos - x2 * sin,
                    offsets=(0, 0, 0),
                    sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
                    strides=(1, 1, 1),
                )
                values_tmp = insert_slice(
                    values_tmp,
                    x2 * cos + x1 * sin,
                    offsets=(0, 0, HALF_ROPE_DIM),
                    sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
                    strides=(1, 1, 1),
                )
                q_output_idx = output_q_nblk_idx[None, :] + (mblk_idx + pos_offset)[:, None] * q_hidden_size
                q_mask = (mmask[:, None]) & (output_q_nmask[None, :])
                tl.store(
                    q_gm_ptr + q_output_idx,
                    values_tmp.reshape(batch_size_per_iter_per_vec, q_hidden_size),
                    mask=q_mask,
                )

                # K: norm * weight + rope
                normalized_values_tmp1 = extract_slice(
                    normalized_values.reshape(batch_size_per_iter_per_vec, qk_head_num_sum, HEAD_DIM),
                    offsets=(0, q_head_num, 0),
                    sizes=(batch_size_per_iter_per_vec, kv_head_num, HEAD_DIM),
                    strides=(1, 1, 1),
                )
                normalized_values_tmp1 = (normalized_values_tmp1 * k_weight_values).to(tl.bfloat16)

                values_tmp2 = tl.zeros((batch_size_per_iter_per_vec, kv_head_num, ROPE_DIM), dtype=tl.bfloat16)
                x1 = extract_slice(
                    normalized_values_tmp1,
                    offsets=(0, 0, 0),
                    sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
                    strides=(1, 1, 1),
                )
                x2 = extract_slice(
                    normalized_values_tmp1,
                    offsets=(0, 0, HALF_ROPE_DIM),
                    sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
                    strides=(1, 1, 1),
                )
                values_tmp2 = insert_slice(
                    values_tmp2,
                    x1 * cos - x2 * sin,
                    offsets=(0, 0, 0),
                    sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
                    strides=(1, 1, 1),
                )
                values_tmp2 = insert_slice(
                    values_tmp2,
                    x2 * cos + x1 * sin,
                    offsets=(0, 0, HALF_ROPE_DIM),
                    sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
                    strides=(1, 1, 1),
                )
                kv_output_idx = output_kv_nblk_idx[None, :] + (mblk_idx + pos_offset)[:, None] * kv_hidden_size
                kv_mask = (mmask[:, None]) & (output_kv_nmask[None, :])
                tl.store(
                    k_gm_ptr + kv_output_idx,
                    values_tmp2.reshape(batch_size_per_iter_per_vec, kv_hidden_size),
                    mask=kv_mask,
                )

            # V: just copy
            mblk_idx = tl.arange(0, v_batch_size_per_iter_per_vec) + input_batch_offset
            nblk_idx = tl.arange(q_hidden_size + kv_hidden_size, total_hidden_size)
            nmask = nblk_idx < total_hidden_size
            out_nblk_idx = tl.arange(0, kv_hidden_size)
            out_nmask = out_nblk_idx < kv_hidden_size

            for _ in tl.range(v_iter_num_per_vec):
                mmask = mblk_idx < input_batch_offset_end
                mask = (mmask[:, None]) & (nmask[None, :])
                idx = mblk_idx[:, None] * total_hidden_size + nblk_idx[None, :]
                values = tl.load(input_gm_ptr + idx, mask=mask)
                out_idx = mblk_idx[:, None] * kv_hidden_size + out_nblk_idx[None, :]
                out_mask = (mmask[:, None]) & (out_nmask[None, :])
                tl.store(v_gm_ptr + out_idx, values, mask=out_mask)
                mblk_idx += v_batch_size_per_iter_per_vec

        def _split_qkv_rmsnorm_rope(
            qkv: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            positions: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            q_hidden_size: int,
            kv_hidden_size: int,
            head_dim: int,
            eps: float,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            num_vectorcore = get_vectorcore_num()
            rope_dim = cos_sin_cache.shape[-1]
            batch_size = qkv.shape[0]
            total_hidden_size = q_hidden_size + kv_hidden_size * 2

            q_output = torch.empty(batch_size, q_hidden_size, device=qkv.device, dtype=qkv.dtype)
            k_output = torch.empty(batch_size, kv_hidden_size, device=qkv.device, dtype=qkv.dtype)
            v_output = torch.empty(batch_size, kv_hidden_size, device=qkv.device, dtype=qkv.dtype)

            q_head_num = q_hidden_size // head_dim
            kv_head_num = kv_hidden_size // head_dim

            UB_SIZE = 87040
            factor = 5 * q_hidden_size + 3 * kv_hidden_size + rope_dim * 2 + q_head_num * rope_dim // 2
            batch_size_per_iter_per_vec = max(1, int(UB_SIZE / qkv.element_size()) // factor)
            qk_head_num_sum = q_head_num + kv_head_num
            qk_head_nums_per_iter_per_vec = batch_size_per_iter_per_vec * qk_head_num_sum
            v_batch_size_per_iter_per_vec = int(UB_SIZE / torch.bfloat16.itemsize // (kv_hidden_size + 1))

            grid = (num_vectorcore, 1, 1)
            _split_qkv_rmsnorm_rope_kernel[grid](
                qkv,
                q_output,
                k_output,
                v_output,
                q_weight,
                k_weight,
                batch_size,
                q_hidden_size,
                kv_hidden_size,
                total_hidden_size,
                eps,
                head_dim,
                rope_dim,
                rope_dim // 2,
                num_vectorcore,
                batch_size_per_iter_per_vec,
                qk_head_nums_per_iter_per_vec,
                q_head_num,
                kv_head_num,
                qk_head_num_sum,
                v_batch_size_per_iter_per_vec,
                positions,
                cos_sin_cache,
            )
            return q_output, k_output, v_output

        @register_kernel(
            name="npu.triton.fused_qk_norm_rope",
            contract=FUSED_QK_NORM_ROPE_CONTRACT,
            platform=PlatformEnum.NPU,
            execution_modes=("eager", "aclgraph"),
            input_domain="Packed NPU QKV, per-head Q/K weights and split-layout position-indexed cosine/sine table.",
            solution="triton",
            binding="triton.jit",
            priority=10,
        )
        def fused_qk_norm_rope(
            qkv: torch.Tensor,
            *,
            num_heads_q: int,
            num_heads_k: int,
            num_heads_v: int,
            head_dim: int,
            eps: float,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            position_ids: torch.Tensor,
            cos: torch.Tensor | None = None,
            sin: torch.Tensor | None = None,
            interleaved: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Apply per-head QK-RMSNorm and RoPE to a packed QKV projection.

            Args:
                qkv: Packed projection of shape ``[num_tokens, q_size + 2 * kv_size]``.
                num_heads_q: Query heads on this rank.
                num_heads_k: Key heads on this rank.
                num_heads_v: Value heads on this rank; derived from ``qkv`` here.
                head_dim: Size of one head.
                eps: RMSNorm epsilon shared by the query and key norms.
                q_weight: Query RMSNorm weight of shape ``[head_dim]``.
                k_weight: Key RMSNorm weight of shape ``[head_dim]``.
                cos_sin_cache: Interleaved cosine/sine table indexed by position.
                position_ids: Position of every token.
                cos: Unused on NPU; the kernel reads ``cos_sin_cache`` directly.
                sin: Unused on NPU; the kernel reads ``cos_sin_cache`` directly.
                interleaved: Unused on NPU; the kernel always uses the split layout.

            Returns:
                Query, key and value, each of shape ``[num_tokens, heads * head_dim]``.
                They may be views into one fused buffer, so a caller must not write
                into them in place.
            """
            del num_heads_v, cos, sin, interleaved
            return _split_qkv_rmsnorm_rope(
                qkv,
                cos_sin_cache,
                position_ids,
                q_weight,
                k_weight,
                num_heads_q * head_dim,
                num_heads_k * head_dim,
                head_dim,
                eps,
            )
