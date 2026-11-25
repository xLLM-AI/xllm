# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Head geometry and gather-table preparation shared by the mRoPE backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

SUPPORTED_HEAD_SPECS = ((256, 64),)
COMPILE_MAX_TOKENS = 100_000_000
VEC_NUM = 2

# All Qwen3.5/3.6 model original head configurations:
#   Model              num_attention_heads  num_key_value_heads
#   Qwen3.5-0.8B/2B          8                    2
#   Qwen3.5-4B/9B           16                    4
#   Qwen3.5-27B             24                    4
#   Qwen3.5-35B-A3B         16                    2
#   Qwen3.5-122B/397B       32                    2
MODEL_HEAD_CONFIGS: tuple[tuple[int, int], ...] = (
    (8, 2),
    (16, 4),
    (24, 4),
    (16, 2),
    (32, 2),
)

SUPPORTED_TP_SIZES = (1, 2, 4, 8, 16)


def compute_tp_split_head_configs(
    model_head_configs: tuple[tuple[int, int], ...],
    tp_sizes: tuple[int, ...],
) -> list[tuple[int, int]]:
    """Compute deduplicated (q_heads, kv_heads) after TP split.

    For each (total_q, total_kv) and each tp:
      - q_heads = total_q / tp  (skip if not divisible)
      - kv_heads = total_kv / tp if total_kv >= tp,
                   else 1 if tp % total_kv == 0
    """
    seen: set[tuple[int, int]] = set()
    result: list[tuple[int, int]] = []
    for total_q, total_kv in model_head_configs:
        for tp in tp_sizes:
            if total_q % tp != 0:
                continue
            q_heads = total_q // tp
            if total_kv >= tp:
                if total_kv % tp != 0:
                    continue
                kv_heads = total_kv // tp
            else:
                if tp % total_kv != 0:
                    continue
                kv_heads = 1
            pair = (q_heads, kv_heads)
            if pair not in seen:
                seen.add(pair)
                result.append(pair)
    result.sort(key=lambda p: (-p[0], -p[1]))
    return result


# Deduplicated TP-split shapes:
#   0.8B/2B   (8,2):  tp1=(8,2)  tp2=(4,1)  tp4=(2,1)
#   4B/9B    (16,4):  tp1=(16,4) tp2=(8,2)  tp4=(4,1)  tp8=(2,1)
#   27B      (24,4):  tp1=(24,4) tp2=(12,2) tp4=(6,1)  tp8=(3,1)
#   35B-A3B  (16,2):  tp1=(16,2) tp2=(8,1)  tp4=(4,1)  tp8=(2,1)
#   122B/397B(32,2):  tp1=(32,2) tp2=(16,1) tp4=(8,1)  tp8=(4,1) tp16=(2,1)
ALL_HEAD_CONFIGS = compute_tp_split_head_configs(MODEL_HEAD_CONFIGS, SUPPORTED_TP_SIZES)


def _validate_mrope_section(
    *,
    rope_dim: int,
    mrope_section: tuple[int, int, int],
) -> None:
    if len(mrope_section) != 3:
        raise ValueError(f"mrope_section must have 3 items [t, h, w], got {mrope_section}")
    if any(section < 0 for section in mrope_section):
        raise ValueError(f"mrope_section values must be >= 0, got {mrope_section}")
    if sum(mrope_section) != rope_dim // 2:
        raise ValueError(
            f"sum(mrope_section) must equal rope_dim // 2, got mrope_section={mrope_section}, rope_dim={rope_dim}"
        )


def build_mrope_gather_pattern_merged(
    *,
    half_rope_dim: int,
    rope_dim: int,
    mrope_section: tuple[int, int, int],
    is_interleaved: bool,
    gather_pad_dim: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build uint32 gather byte-offset pattern for merged cos+sin gather.

    Source buffer layout (bf16):
        [t_cos(hrd) | t_sin(hrd) | h_cos(hrd) | h_sin(hrd) | w_cos(hrd) | w_sin(hrd)]
        = 3 * rope_dim elements total.

    Output layout after gather:
        [assembled_cos(hrd) | assembled_sin(hrd)] = rope_dim elements.
    """
    import torch

    t_len, h_len, w_len = mrope_section
    axis_id = [0] * half_rope_dim
    if is_interleaved:
        for i in range(half_rope_dim):
            if i % 3 == 1 and i <= 3 * h_len:
                axis_id[i] = 1
            elif i % 3 == 2 and i <= 3 * w_len:
                axis_id[i] = 2
    else:
        t_end = t_len
        h_end = t_end + h_len
        for i in range(half_rope_dim):
            if i >= h_end:
                axis_id[i] = 2
            elif i >= t_end:
                axis_id[i] = 1

    elem_bytes = 2  # bfloat16
    pattern = torch.zeros(gather_pad_dim, dtype=torch.int32, device="cpu")
    for i in range(half_rope_dim):
        # cos part: output[i] reads from axis_id[i]'s cos segment position i
        pattern[i] = (axis_id[i] * rope_dim + i) * elem_bytes
        # sin part: output[half_rope_dim + i] reads from axis_id[i]'s sin segment
        pattern[half_rope_dim + i] = (axis_id[i] * rope_dim + half_rope_dim + i) * elem_bytes
    return pattern.to(device).view(torch.uint32)
