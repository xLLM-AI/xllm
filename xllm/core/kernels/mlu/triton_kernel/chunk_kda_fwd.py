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

"""MLU Triton kernels for packed variable-length chunk KDA."""

import triton
import triton.language as tl


@triton.jit
def _inverse_unit_lower_16(lower: tl.tensor, B0: tl.constexpr) -> tl.tensor:
    """Invert I+lower for one strictly-lower 16x16 tile."""
    offsets = tl.arange(0, B0)
    below_diagonal = offsets[:, None] > offsets[None, :]
    diagonal = offsets[:, None] == offsets[None, :]
    power_one = -tl.where(below_diagonal, lower, 0.0)
    sum_two = power_one + diagonal
    power_two = tl.dot(power_one, power_one, allow_tf32=False)
    sum_four = sum_two + tl.dot(power_two, sum_two, allow_tf32=False)
    power_four = tl.dot(power_two, power_two, allow_tf32=False)
    sum_eight = sum_four + tl.dot(power_four, sum_four, allow_tf32=False)
    power_eight = tl.dot(power_four, power_four, allow_tf32=False)
    return sum_eight + tl.dot(power_eight, sum_eight, allow_tf32=False)


@triton.jit(do_not_specialize=["CHUNK_BASE", "SLOTS"])
def tmo_chunk_kda_gate_kernel(
    log_gate: tl.tensor,
    gate_cumsum: tl.tensor,
    q: tl.tensor,
    k: tl.tensor,
    q_exp: tl.tensor,
    k_exp: tl.tensor,
    k_inv_exp: tl.tensor,
    cu_seqlens: tl.tensor,
    chunk_indices: tl.tensor,
    CHUNK_BASE: tl.int64,
    SLOTS: tl.int64,
    H: tl.constexpr,
    BT: tl.constexpr,
    D: tl.constexpr,
    BK: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    program_count = tl.num_programs(0)
    key_blocks: tl.constexpr = triton.cdiv(D, BK)
    total_jobs = SLOTS * H * key_blocks
    row_offsets = tl.arange(0, BT)
    key_offsets = tl.arange(0, BK)

    for physical_job in range(pid, total_jobs, program_count):
        if BT == 16:
            flat_job = total_jobs - 1 - physical_job
        else:
            flat_job = (physical_job + 8) % total_jobs
        key_block = flat_job % key_blocks
        head = (flat_job // key_blocks) % H
        slot = flat_job // (key_blocks * H)
        metadata_slot = CHUNK_BASE + slot
        sequence = tl.load(chunk_indices + metadata_slot * 2).to(tl.int32)
        local_chunk = tl.load(chunk_indices + metadata_slot * 2 + 1).to(tl.int32)
        sequence_begin = tl.load(cu_seqlens + sequence).to(tl.int32)
        sequence_end = tl.load(cu_seqlens + sequence + 1).to(tl.int32)
        chunk_start = local_chunk * BT
        valid_rows = tl.minimum(BT, sequence_end - sequence_begin - chunk_start)

        rows = row_offsets
        dimensions = key_block * BK + key_offsets
        row_mask = rows < valid_rows
        dimension_mask = dimensions < D
        input_offsets = (sequence_begin + chunk_start + rows[:, None]) * H * D + head * D + dimensions[None, :]
        gate_values = tl.load(
            log_gate + input_offsets,
            mask=row_mask[:, None] & dimension_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        cumulative_values = tl.cumsum(gate_values, axis=0)
        output_offsets = (slot * H + head) * BT * D + rows[:, None] * D + dimensions[None, :]
        output_mask = row_mask[:, None] & dimension_mask[None, :]
        last_cumulative = tl.sum(
            tl.where(
                rows[:, None] == valid_rows - 1,
                cumulative_values,
                0.0,
            ),
            axis=0,
        )
        gate_last_offsets = (slot * H + head) * D + dimensions
        tl.store(
            gate_cumsum + gate_last_offsets,
            last_cumulative,
            mask=dimension_mask,
        )
        q_values = tl.load(
            q + input_offsets,
            mask=output_mask,
            other=0.0,
        ).to(tl.float32)
        k_values = tl.load(
            k + input_offsets,
            mask=output_mask,
            other=0.0,
        ).to(tl.float32)
        if USE_QK_L2NORM:
            q_inverse_norm = tl.rsqrt(tl.sum(q_values * q_values, axis=1) + 1.0e-6)
            k_inverse_norm = tl.rsqrt(tl.sum(k_values * k_values, axis=1) + 1.0e-6)
            q_values *= q_inverse_norm[:, None]
            k_values *= k_inverse_norm[:, None]
        q_values *= 0.08838834764831845
        # Keep normalized operands and the log-space cumulative gate separate.
        # Materializing exp(-cumulative_values) overflows for the public
        # log_gate range [-5, 0] because one 64-token chunk can reach -320.
        tl.store(q_exp + output_offsets, q_values, mask=output_mask)
        tl.store(k_exp + output_offsets, k_values, mask=output_mask)
        tl.store(k_inv_exp + output_offsets, cumulative_values, mask=output_mask)


@triton.jit(do_not_specialize=["CHUNK_BASE", "SLOTS"])
def tmo_chunk_kda_kkt_kernel(
    q_exp: tl.tensor,
    k_exp: tl.tensor,
    k_inv_exp: tl.tensor,
    beta: tl.tensor,
    lower: tl.tensor,
    aq: tl.tensor,
    cu_seqlens: tl.tensor,
    chunk_indices: tl.tensor,
    CHUNK_BASE: tl.int64,
    SLOTS: tl.int64,
    H: tl.constexpr,
    BT: tl.constexpr,
    D: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    program_count = tl.num_programs(0)
    chunk_blocks: tl.constexpr = triton.cdiv(BT, BC)
    block_pairs: tl.constexpr = chunk_blocks * (chunk_blocks + 1) // 2
    total_jobs = SLOTS * H * block_pairs
    block_offsets = tl.arange(0, BC)

    for physical_job in range(pid, total_jobs, program_count):
        if BT == 64:
            flat_job = (physical_job + 8) % total_jobs
        else:
            flat_job = physical_job
        pair = flat_job % block_pairs
        if BC == 32:
            slot = (flat_job // block_pairs) % SLOTS
            head = flat_job // (block_pairs * SLOTS)
            row_block = tl.where(pair == 0, 0, 1)
            column_block = tl.where(pair == 2, 1, 0)
        elif BT == BC:
            head = (flat_job // block_pairs) % H
            slot = flat_job // (block_pairs * H)
            row_block = 0
            column_block = 0
        else:
            head = (flat_job // block_pairs) % H
            slot = flat_job // (block_pairs * H)
            linear_pair = pair
            linear_pair += tl.where(pair >= 1, 3, 0)
            linear_pair += tl.where(pair >= 3, 2, 0)
            linear_pair += tl.where(pair >= 6, 1, 0)
            row_block = linear_pair // chunk_blocks
            column_block = linear_pair % chunk_blocks
        row_start = row_block * BC
        column_start = column_block * BC
        metadata_slot = CHUNK_BASE + slot
        sequence = tl.load(chunk_indices + metadata_slot * 2).to(tl.int32)
        local_chunk = tl.load(chunk_indices + metadata_slot * 2 + 1).to(tl.int32)
        sequence_begin = tl.load(cu_seqlens + sequence).to(tl.int32)
        sequence_end = tl.load(cu_seqlens + sequence + 1).to(tl.int32)
        chunk_start = local_chunk * BT
        valid_rows = tl.minimum(BT, sequence_end - sequence_begin - chunk_start)

        rows = row_start + block_offsets
        columns = column_start + block_offsets
        row_mask = rows < valid_rows
        column_mask = columns < valid_rows
        lower_values = tl.zeros((BC, BC), dtype=tl.float32)
        aq_values = tl.zeros((BC, BC), dtype=tl.float32)

        if True:
            workspace_base = (slot * H + head) * BT * D
            for key_start in range(0, D, BK):
                keys = key_start + tl.arange(0, BK)
                key_mask = keys < D
                row_offsets = workspace_base + rows[:, None] * D + keys[None, :]
                column_offsets = workspace_base + columns[:, None] * D + keys[None, :]
                q_rows = tl.load(
                    q_exp + row_offsets,
                    mask=row_mask[:, None] & key_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                k_rows = tl.load(
                    k_exp + row_offsets,
                    mask=row_mask[:, None] & key_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                k_columns = tl.load(
                    k_exp + column_offsets,
                    mask=column_mask[:, None] & key_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                gate_rows = tl.load(
                    k_inv_exp + row_offsets,
                    mask=row_mask[:, None] & key_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                gate_columns = tl.load(
                    k_inv_exp + column_offsets,
                    mask=column_mask[:, None] & key_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                anchor_offsets = workspace_base + row_start * D + keys
                anchor_gate = tl.load(
                    k_inv_exp + anchor_offsets,
                    mask=(row_start < valid_rows) & key_mask,
                    other=0.0,
                ).to(tl.float32)
                row_decay = tl.extra.mlu.libdevice.fast_expf(gate_rows - anchor_gate[None, :])
                column_decay = tl.extra.mlu.libdevice.fast_expf(anchor_gate[:, None] - tl.trans(gate_columns))
                q_rows *= row_decay
                k_rows *= row_decay
                k_columns = tl.trans(tl.trans(k_columns) * column_decay)
                aq_values += tl.dot(q_rows, tl.trans(k_columns), allow_tf32=False)
                lower_values += tl.dot(
                    k_rows,
                    tl.trans(k_columns),
                    allow_tf32=False,
                )

            global_rows = sequence_begin + chunk_start + rows
            beta_values = tl.load(
                beta + global_rows * H + head,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            lower_values *= beta_values[:, None]
            valid_matrix = row_mask[:, None] & column_mask[None, :]
            lower_values = tl.where(
                valid_matrix & (rows[:, None] > columns[None, :]),
                lower_values,
                0.0,
            )
            aq_values = tl.where(
                valid_matrix & (rows[:, None] >= columns[None, :]),
                aq_values,
                0.0,
            )

        matrix_offsets = (slot * H + head) * BT * BT + rows[:, None] * BT + columns[None, :]
        tl.store(lower + matrix_offsets, lower_values)
        tl.store(aq + matrix_offsets, aq_values)


@triton.jit(do_not_specialize=["SLOTS"])
def tmo_chunk_kda_inverse_kernel(
    lower_inverse: tl.tensor,
    SLOTS: tl.int64,
    H: tl.constexpr,
    BT: tl.constexpr,
    B0: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    program_count = tl.num_programs(0)
    total_jobs = SLOTS * H
    block_offsets = tl.arange(0, B0)

    for physical_job in range(pid, total_jobs, program_count):
        if BT == 64:
            flat_job = (physical_job + 6) % total_jobs
        else:
            flat_job = physical_job
        head = flat_job % H
        slot = flat_job // H
        matrix_base = (slot * H + head) * BT * BT

        offsets_11 = matrix_base + block_offsets[:, None] * BT + block_offsets[None, :]
        inverse_11 = _inverse_unit_lower_16(
            tl.load(lower_inverse + offsets_11).to(tl.float32),
            B0,
        )
        tl.store(lower_inverse + offsets_11, inverse_11)

        if BT != B0:
            offsets_22 = matrix_base + (B0 + block_offsets[:, None]) * BT + B0 + block_offsets[None, :]
            offsets_33 = matrix_base + (2 * B0 + block_offsets[:, None]) * BT + 2 * B0 + block_offsets[None, :]
            offsets_44 = matrix_base + (3 * B0 + block_offsets[:, None]) * BT + 3 * B0 + block_offsets[None, :]
            inverse_22 = _inverse_unit_lower_16(tl.load(lower_inverse + offsets_22).to(tl.float32), B0)
            inverse_33 = _inverse_unit_lower_16(tl.load(lower_inverse + offsets_33).to(tl.float32), B0)
            inverse_44 = _inverse_unit_lower_16(tl.load(lower_inverse + offsets_44).to(tl.float32), B0)

            offsets_21 = matrix_base + (B0 + block_offsets[:, None]) * BT + block_offsets[None, :]
            offsets_31 = matrix_base + (2 * B0 + block_offsets[:, None]) * BT + block_offsets[None, :]
            offsets_32 = matrix_base + (2 * B0 + block_offsets[:, None]) * BT + B0 + block_offsets[None, :]
            offsets_41 = matrix_base + (3 * B0 + block_offsets[:, None]) * BT + block_offsets[None, :]
            offsets_42 = matrix_base + (3 * B0 + block_offsets[:, None]) * BT + B0 + block_offsets[None, :]
            offsets_43 = matrix_base + (3 * B0 + block_offsets[:, None]) * BT + 2 * B0 + block_offsets[None, :]
            lower_21 = tl.load(lower_inverse + offsets_21).to(tl.float32)
            lower_31 = tl.load(lower_inverse + offsets_31).to(tl.float32)
            lower_32 = tl.load(lower_inverse + offsets_32).to(tl.float32)
            lower_41 = tl.load(lower_inverse + offsets_41).to(tl.float32)
            lower_42 = tl.load(lower_inverse + offsets_42).to(tl.float32)
            lower_43 = tl.load(lower_inverse + offsets_43).to(tl.float32)

            inverse_21 = -tl.dot(
                tl.dot(inverse_22, lower_21, allow_tf32=False),
                inverse_11,
                allow_tf32=False,
            )
            inverse_32 = -tl.dot(
                tl.dot(inverse_33, lower_32, allow_tf32=False),
                inverse_22,
                allow_tf32=False,
            )
            inverse_43 = -tl.dot(
                tl.dot(inverse_44, lower_43, allow_tf32=False),
                inverse_33,
                allow_tf32=False,
            )
            inverse_31 = -tl.dot(
                inverse_33,
                tl.dot(lower_31, inverse_11, allow_tf32=False) + tl.dot(lower_32, inverse_21, allow_tf32=False),
                allow_tf32=False,
            )
            inverse_42 = -tl.dot(
                inverse_44,
                tl.dot(lower_42, inverse_22, allow_tf32=False) + tl.dot(lower_43, inverse_32, allow_tf32=False),
                allow_tf32=False,
            )
            inverse_41 = -tl.dot(
                inverse_44,
                tl.dot(lower_41, inverse_11, allow_tf32=False)
                + tl.dot(lower_42, inverse_21, allow_tf32=False)
                + tl.dot(lower_43, inverse_31, allow_tf32=False),
                allow_tf32=False,
            )

            tl.store(lower_inverse + offsets_22, inverse_22)
            tl.store(lower_inverse + offsets_33, inverse_33)
            tl.store(lower_inverse + offsets_44, inverse_44)
            tl.store(lower_inverse + offsets_21, inverse_21)
            tl.store(lower_inverse + offsets_31, inverse_31)
            tl.store(lower_inverse + offsets_32, inverse_32)
            tl.store(lower_inverse + offsets_41, inverse_41)
            tl.store(lower_inverse + offsets_42, inverse_42)
            tl.store(lower_inverse + offsets_43, inverse_43)


@triton.jit(do_not_specialize=["CHUNK_BASE", "SLOTS"])
def tmo_chunk_kda_wu_kernel(
    k_normalized: tl.tensor,
    q_scaled: tl.tensor,
    v: tl.tensor,
    gate_cumsum: tl.tensor,
    beta: tl.tensor,
    inverse: tl.tensor,
    w: tl.tensor,
    u: tl.tensor,
    qg: tl.tensor,
    kg: tl.tensor,
    cu_seqlens: tl.tensor,
    chunk_indices: tl.tensor,
    CHUNK_BASE: tl.int64,
    SLOTS: tl.int64,
    H: tl.constexpr,
    BT: tl.constexpr,
    D: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    program_count = tl.num_programs(0)
    total_jobs = SLOTS * H
    row_offsets = tl.arange(0, BT)
    matrix_offsets = row_offsets[:, None] * BT + row_offsets[None, :]

    for physical_job in range(pid, total_jobs, program_count):
        if BT == 64:
            flat_job = (physical_job + 7) % total_jobs
        else:
            flat_job = physical_job
        head = flat_job % H
        slot = flat_job // H
        metadata_slot = CHUNK_BASE + slot
        sequence = tl.load(chunk_indices + metadata_slot * 2).to(tl.int32)
        local_chunk = tl.load(chunk_indices + metadata_slot * 2 + 1).to(tl.int32)
        sequence_begin = tl.load(cu_seqlens + sequence).to(tl.int32)
        sequence_end = tl.load(cu_seqlens + sequence + 1).to(tl.int32)
        chunk_start = local_chunk * BT
        valid_rows = tl.minimum(BT, sequence_end - sequence_begin - chunk_start)
        row_mask = row_offsets < valid_rows
        global_tokens = sequence_begin + chunk_start + row_offsets
        beta_values = tl.load(
            beta + global_tokens * H + head,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        matrix_base = (slot * H + head) * BT * BT
        inverse_values = tl.load(inverse + matrix_base + matrix_offsets).to(tl.float32)
        inverse_values = tl.where(
            row_offsets[:, None] >= row_offsets[None, :],
            inverse_values,
            0.0,
        )
        workspace_base = (slot * H + head) * BT * D

        for value_start in range(0, D, BV):
            value_offsets = value_start + tl.arange(0, BV)
            value_mask = value_offsets < D
            input_offsets = global_tokens[:, None] * H * D + head * D + value_offsets[None, :]
            value_values = tl.load(
                v + input_offsets,
                mask=row_mask[:, None] & value_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            u_seed = value_values * beta_values[:, None]
            u_values = tl.dot(inverse_values, u_seed, allow_tf32=False)
            output_offsets = workspace_base + row_offsets[:, None] * D + value_offsets[None, :]
            tl.store(
                u + output_offsets,
                tl.where(row_mask[:, None] & value_mask[None, :], u_values, 0.0),
            )

        last_row = valid_rows - 1
        for key_start in range(0, D, BK):
            key_offsets = key_start + tl.arange(0, BK)
            key_mask = key_offsets < D
            workspace_offsets = workspace_base + row_offsets[:, None] * D + key_offsets[None, :]
            normalized_key = tl.load(
                w + workspace_offsets,
                mask=row_mask[:, None] & key_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            gate_values = tl.load(
                kg + workspace_offsets,
                mask=row_mask[:, None] & key_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            last_gate = tl.load(
                gate_cumsum + (slot * H + head) * D + key_offsets,
                mask=key_mask,
                other=0.0,
            ).to(tl.float32)
            positive_decay = tl.extra.mlu.libdevice.fast_expf(gate_values)
            relative_last_decay = tl.extra.mlu.libdevice.fast_expf(last_gate[None, :] - gate_values)
            w_seed = normalized_key * positive_decay * beta_values[:, None]
            w_values = tl.dot(inverse_values, w_seed, allow_tf32=False)
            kg_values = normalized_key * relative_last_decay
            normalized_q = tl.load(
                qg + workspace_offsets,
                mask=row_mask[:, None] & key_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            qg_values = normalized_q * positive_decay
            output_mask = row_mask[:, None] & key_mask[None, :]
            tl.store(
                qg + workspace_offsets,
                tl.where(output_mask, qg_values, 0.0),
            )
            tl.store(w + workspace_offsets, tl.where(output_mask, w_values, 0.0))
            tl.store(kg + workspace_offsets, tl.where(output_mask, kg_values, 0.0))


@triton.jit(do_not_specialize=["CHUNK_BASE", "SLOTS", "N"])
def tmo_chunk_kda_state_kernel(
    w: tl.tensor,
    u: tl.tensor,
    qg: tl.tensor,
    kg: tl.tensor,
    aq: tl.tensor,
    gate_cumsum: tl.tensor,
    input_state: tl.tensor,
    final_state: tl.tensor,
    output: tl.tensor,
    cu_seqlens: tl.tensor,
    chunk_indices: tl.tensor,
    CHUNK_BASE: tl.int64,
    SLOTS: tl.int64,
    N: tl.int64,
    H: tl.constexpr,
    BT: tl.constexpr,
    D: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    program_count = tl.num_programs(0)
    value_blocks: tl.constexpr = triton.cdiv(D, BV)
    total_jobs = N * H * value_blocks
    row_offsets = tl.arange(0, BT)
    key_offsets = tl.arange(0, BK)
    value_offsets = tl.arange(0, BV)

    for physical_job in range(pid, total_jobs, program_count):
        if BT == 64:
            flat_job = (physical_job + 5) % total_jobs
        else:
            flat_job = physical_job
        value_block = flat_job % value_blocks
        head = (flat_job // value_blocks) % H
        sequence = flat_job // (value_blocks * H)
        values = value_block * BV + value_offsets
        value_mask = values < D
        state_base = (sequence * H + head) * D * D
        state_one_offsets = state_base + values[:, None] * D + key_offsets[None, :]
        state_two_offsets = state_base + values[:, None] * D + BK + key_offsets[None, :]
        if BT == 64:
            state_two = tl.load(
                input_state + state_two_offsets,
                mask=value_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            state_one = tl.load(
                input_state + state_one_offsets,
                mask=value_mask[:, None],
                other=0.0,
            ).to(tl.float32)
        else:
            state_one = tl.load(
                input_state + state_one_offsets,
                mask=value_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            state_two = tl.load(
                input_state + state_two_offsets,
                mask=value_mask[:, None],
                other=0.0,
            ).to(tl.float32)

        for slot in range(0, SLOTS):
            metadata_slot = CHUNK_BASE + slot
            chunk_sequence = tl.load(chunk_indices + metadata_slot * 2).to(tl.int32)
            if chunk_sequence == sequence:
                local_chunk = tl.load(chunk_indices + metadata_slot * 2 + 1).to(tl.int32)
                sequence_begin = tl.load(cu_seqlens + sequence).to(tl.int32)
                sequence_end = tl.load(cu_seqlens + sequence + 1).to(tl.int32)
                chunk_start = local_chunk * BT
                valid_rows = tl.minimum(
                    BT,
                    sequence_end - sequence_begin - chunk_start,
                )
                row_mask = row_offsets < valid_rows
                workspace_base = (slot * H + head) * BT * D
                w_one_offsets = workspace_base + row_offsets[:, None] * D + key_offsets[None, :]
                w_two_offsets = w_one_offsets + BK
                w_one = tl.load(
                    w + w_one_offsets,
                    mask=row_mask[:, None],
                    other=0.0,
                ).to(tl.float32)
                w_two = tl.load(
                    w + w_two_offsets,
                    mask=row_mask[:, None],
                    other=0.0,
                ).to(tl.float32)
                u_offsets = workspace_base + row_offsets[:, None] * D + values[None, :]
                u_values = tl.load(
                    u + u_offsets,
                    mask=row_mask[:, None] & value_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                new_values = u_values - tl.dot(
                    w_one,
                    tl.trans(state_one),
                    allow_tf32=False,
                )
                new_values -= tl.dot(
                    w_two,
                    tl.trans(state_two),
                    allow_tf32=False,
                )

                qg_one = tl.load(
                    qg + w_one_offsets,
                    mask=row_mask[:, None],
                    other=0.0,
                ).to(tl.float32)
                qg_two = tl.load(
                    qg + w_two_offsets,
                    mask=row_mask[:, None],
                    other=0.0,
                ).to(tl.float32)
                output_values = tl.dot(
                    qg_one,
                    tl.trans(state_one),
                    allow_tf32=False,
                )
                output_values += tl.dot(
                    qg_two,
                    tl.trans(state_two),
                    allow_tf32=False,
                )
                aq_base = (slot * H + head) * BT * BT
                aq_offsets = aq_base + row_offsets[:, None] * BT + row_offsets[None, :]
                aq_values = tl.load(aq + aq_offsets).to(tl.float32)
                aq_values = tl.where(
                    row_offsets[:, None] >= row_offsets[None, :],
                    aq_values,
                    0.0,
                )
                output_values += tl.dot(
                    aq_values,
                    new_values,
                    allow_tf32=False,
                )
                global_tokens = sequence_begin + chunk_start + row_offsets
                output_offsets = global_tokens[:, None] * H * D + head * D + values[None, :]
                tl.store(
                    output + output_offsets,
                    output_values,
                    mask=row_mask[:, None] & value_mask[None, :],
                )

                last_row = valid_rows - 1
                last_gate_one = tl.load(
                    gate_cumsum + (slot * H + head) * D + key_offsets,
                ).to(tl.float32)
                last_gate_two = tl.load(
                    gate_cumsum + (slot * H + head) * D + BK + key_offsets,
                ).to(tl.float32)
                state_one *= tl.extra.mlu.libdevice.fast_expf(last_gate_one)[None, :]
                state_two *= tl.extra.mlu.libdevice.fast_expf(last_gate_two)[None, :]
                kg_one = tl.load(
                    kg + w_one_offsets,
                    mask=row_mask[:, None],
                    other=0.0,
                ).to(tl.float32)
                kg_two = tl.load(
                    kg + w_two_offsets,
                    mask=row_mask[:, None],
                    other=0.0,
                ).to(tl.float32)
                state_one += tl.trans(tl.dot(tl.trans(kg_one), new_values, allow_tf32=False))
                state_two += tl.trans(tl.dot(tl.trans(kg_two), new_values, allow_tf32=False))

        tl.store(
            final_state + state_one_offsets,
            state_one,
            mask=value_mask[:, None],
        )
        tl.store(
            final_state + state_two_offsets,
            state_two,
            mask=value_mask[:, None],
        )
