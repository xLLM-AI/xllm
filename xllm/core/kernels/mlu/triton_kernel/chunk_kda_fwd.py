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

"""Four-stage packed KDA: gate preparation, local products, solve, state update."""

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
    v: tl.tensor,
    beta: tl.tensor,
    w: tl.tensor,
    u: tl.tensor,
    qg: tl.tensor,
    kg: tl.tensor,
    log_gate: tl.tensor,
    q: tl.tensor,
    k: tl.tensor,
    normalized_q: tl.tensor,
    normalized_k: tl.tensor,
    cumulative_gate: tl.tensor,
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

    for flat_job in range(pid, total_jobs, program_count):
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
        # Keep cumulative gates in log space: exp(-cumsum) overflows at BT64.
        positive_decay = tl.extra.mlu.libdevice.fast_expf(cumulative_values)
        tl.store(cumulative_gate + output_offsets, cumulative_values, mask=output_mask)
        q_values = tl.load(q + input_offsets, mask=output_mask, other=0.0).to(tl.float32)
        if USE_QK_L2NORM:
            q_inverse_norm = tl.rsqrt(tl.sum(q_values * q_values, axis=1) + 1.0e-6)
            q_values *= q_inverse_norm[:, None]
        scaled_q = q_values * 0.08838834764831845
        tl.store(normalized_q + output_offsets, scaled_q, mask=output_mask)
        tl.store(
            qg + output_offsets,
            tl.where(output_mask, scaled_q * positive_decay, 0.0),
        )
        beta_values = tl.load(
            beta + (sequence_begin + chunk_start + rows) * H + head,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        k_values = tl.load(k + input_offsets, mask=output_mask, other=0.0).to(tl.float32)
        if USE_QK_L2NORM:
            k_inverse_norm = tl.rsqrt(tl.sum(k_values * k_values, axis=1) + 1.0e-6)
            k_values *= k_inverse_norm[:, None]
        tl.store(normalized_k + output_offsets, k_values, mask=output_mask)
        tl.store(
            w + output_offsets,
            tl.where(output_mask, k_values * positive_decay * beta_values[:, None], 0.0),
        )
        relative_decay = tl.extra.mlu.libdevice.fast_expf(last_cumulative[None, :] - cumulative_values)
        tl.store(
            kg + output_offsets,
            tl.where(output_mask, k_values * relative_decay, 0.0),
        )
        v_values = tl.load(v + input_offsets, mask=output_mask, other=0.0).to(tl.float32)
        u_output_offsets = (slot * H + head) * BT * D + dimensions[None, :] * BT + rows[:, None]
        tl.store(
            u + u_output_offsets,
            tl.where(output_mask, v_values * beta_values[:, None], 0.0),
        )


@triton.jit(do_not_specialize=["CHUNK_BASE", "SLOTS"])
def tmo_chunk_kda_kkt_kernel(
    normalized_q: tl.tensor,
    normalized_k: tl.tensor,
    cumulative_gate: tl.tensor,
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
    BN: tl.constexpr,
    BK: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    program_count = tl.num_programs(0)
    total_jobs = SLOTS * H
    columns = tl.arange(0, BN)
    keys = tl.arange(0, BK)
    block_offsets = tl.arange(0, BC)
    for flat_job in range(pid, total_jobs, program_count):
        head = flat_job % H
        slot = flat_job // H
        metadata_slot = CHUNK_BASE + slot
        sequence = tl.load(chunk_indices + metadata_slot * 2).to(tl.int32)
        local_chunk = tl.load(chunk_indices + metadata_slot * 2 + 1).to(tl.int32)
        sequence_begin = tl.load(cu_seqlens + sequence).to(tl.int32)
        sequence_end = tl.load(cu_seqlens + sequence + 1).to(tl.int32)
        chunk_start = local_chunk * BT
        valid_rows = tl.minimum(BT, sequence_end - sequence_begin - chunk_start)
        workspace_base = (slot * H + head) * BT * D
        column_offsets = workspace_base + columns[:, None] * D + keys[None, :]
        k_columns = tl.load(normalized_k + column_offsets, mask=(columns < valid_rows)[:, None], other=0.0).to(
            tl.float32
        )
        gate_columns = tl.load(cumulative_gate + column_offsets, mask=(columns < valid_rows)[:, None], other=0.0).to(
            tl.float32
        )
        for row_block in range(0, BT // BC):
            row_start = row_block * BC
            rows = row_start + block_offsets
            row_mask = rows < valid_rows
            column_mask = (columns < valid_rows) & (columns < row_start + BC)
            row_offsets = workspace_base + rows[:, None] * D + keys[None, :]
            q_rows = tl.load(normalized_q + row_offsets, mask=row_mask[:, None], other=0.0).to(tl.float32)
            k_rows = tl.load(normalized_k + row_offsets, mask=row_mask[:, None], other=0.0).to(tl.float32)
            gate_rows = tl.load(cumulative_gate + row_offsets, mask=row_mask[:, None], other=0.0).to(tl.float32)
            # A 16-row anchor bounds positive exponent differences by 75
            # for the supported log_gate range [-5, 0], including BT64.
            anchor = tl.load(
                cumulative_gate + workspace_base + row_start * D + keys,
                mask=row_start < valid_rows,
                other=0.0,
            ).to(tl.float32)
            row_decay = tl.extra.mlu.libdevice.fast_expf(gate_rows - anchor[None, :])
            safe_column_gate = tl.where(
                (column_mask & (row_start < valid_rows))[None, :],
                tl.trans(gate_columns),
                0.0,
            )
            column_decay = tl.extra.mlu.libdevice.fast_expf(anchor[:, None] - safe_column_gate)
            weighted_columns = tl.where(column_mask[None, :], tl.trans(k_columns) * column_decay, 0.0)
            aq_values = tl.dot(q_rows * row_decay, weighted_columns, allow_tf32=False)
            lower_values = tl.dot(k_rows * row_decay, weighted_columns, allow_tf32=False)
            beta_values = tl.load(
                beta + (sequence_begin + chunk_start + rows) * H + head,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            lower_values *= beta_values[:, None]
            valid_matrix = row_mask[:, None] & column_mask[None, :]
            lower_values = tl.where(valid_matrix & (rows[:, None] > columns[None, :]), lower_values, 0.0)
            aq_values = tl.where(valid_matrix & (rows[:, None] >= columns[None, :]), aq_values, 0.0)
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

    for flat_job in range(pid, total_jobs, program_count):
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


@triton.jit(do_not_specialize=["CHUNK_BASE", "SLOTS", "N"])
def tmo_chunk_kda_state_kernel(
    inverse: tl.tensor,
    w: tl.tensor,
    u: tl.tensor,
    qg: tl.tensor,
    kg: tl.tensor,
    aq: tl.tensor,
    cumulative_gate: tl.tensor,
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

    for flat_job in range(pid, total_jobs, program_count):
        value_block = flat_job % value_blocks
        head = (flat_job // value_blocks) % H
        sequence = flat_job // (value_blocks * H)
        values = value_block * BV + value_offsets
        value_mask = values < D
        state_base = (sequence * H + head) * D * D
        state_one_offsets = state_base + values[:, None] * D + key_offsets[None, :]
        state_one = tl.load(
            input_state + state_one_offsets,
            mask=value_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        sequence_begin = tl.load(cu_seqlens + sequence).to(tl.int32)
        sequence_end = tl.load(cu_seqlens + sequence + 1).to(tl.int32)
        for slot in range(0, SLOTS):
            metadata_slot = CHUNK_BASE + slot
            chunk_sequence = tl.load(chunk_indices + metadata_slot * 2).to(tl.int32)
            if chunk_sequence == sequence:
                local_chunk = tl.load(chunk_indices + metadata_slot * 2 + 1).to(tl.int32)
                chunk_start = local_chunk * BT
                valid_rows = tl.minimum(
                    BT,
                    sequence_end - sequence_begin - chunk_start,
                )
                row_mask = row_offsets < valid_rows
                workspace_base = (slot * H + head) * BT * D
                w_one_offsets = workspace_base + row_offsets[:, None] * D + key_offsets[None, :]
                w_one = tl.load(
                    w + w_one_offsets,
                    mask=row_mask[:, None],
                    other=0.0,
                ).to(tl.float32)
                u_offsets = workspace_base + values[:, None] * BT + row_offsets[None, :]
                u_values = tl.load(
                    u + u_offsets,
                    mask=value_mask[:, None] & row_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                new_values = u_values - tl.dot(
                    state_one,
                    tl.trans(w_one),
                    allow_tf32=False,
                )

                matrix_base = (slot * H + head) * BT * BT
                matrix_offsets = matrix_base + row_offsets[:, None] * BT + row_offsets[None, :]
                causal_mask = row_offsets[:, None] >= row_offsets[None, :]
                inverse_values = tl.load(inverse + matrix_offsets).to(tl.float32)
                inverse_values = tl.where(causal_mask, inverse_values, 0.0)
                new_values = tl.dot(new_values, tl.trans(inverse_values), allow_tf32=False)

                qg_one = tl.load(
                    qg + w_one_offsets,
                    mask=row_mask[:, None],
                    other=0.0,
                ).to(tl.float32)
                output_values = tl.dot(
                    state_one,
                    tl.trans(qg_one),
                    allow_tf32=False,
                )
                aq_values = tl.load(aq + matrix_offsets).to(tl.float32)
                aq_values = tl.where(causal_mask, aq_values, 0.0)
                output_values += tl.dot(
                    new_values,
                    tl.trans(aq_values),
                    allow_tf32=False,
                )
                global_tokens = sequence_begin + chunk_start + row_offsets
                output_offsets = global_tokens[:, None] * H * D + head * D + values[None, :]
                tl.store(
                    output + output_offsets,
                    tl.trans(output_values),
                    mask=row_mask[:, None] & value_mask[None, :],
                )

                last_gate_one = tl.load(
                    cumulative_gate + workspace_base + (valid_rows - 1) * D + key_offsets,
                    mask=valid_rows > 0,
                    other=0.0,
                ).to(tl.float32)
                state_one *= tl.extra.mlu.libdevice.fast_expf(last_gate_one)[None, :]
                kg_one = tl.load(
                    kg + w_one_offsets,
                    mask=row_mask[:, None],
                    other=0.0,
                ).to(tl.float32)
                state_one += tl.dot(new_values, kg_one, allow_tf32=False)

        tl.store(
            final_state + state_one_offsets,
            state_one,
            mask=value_mask[:, None],
        )
