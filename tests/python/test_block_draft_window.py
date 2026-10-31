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

"""Independent global-position checks for compact block-draft page tables."""

from types import SimpleNamespace

import pytest
import torch

from xllm.python.attention.block_draft_window import (
    block_draft_model_window_context,
    block_draft_window_context,
    prepare_windowed_block_table,
    windowed_block_page_count,
)
from xllm.python.layers.attention import Attention


def _attention(policy: str, window: int = 2048) -> Attention:
    return Attention(
        num_heads=4,
        num_kv_heads=1,
        head_dim=64,
        scale=0.125,
        sliding_window=window,
        layer_id=0,
        causal=policy == "dspark_causal",
        fia_sparse_mode=4 if policy == "dflash2" else 0,
        fia_pre_tokens=window - 1,
        fia_next_tokens=7,
        fia_use_attention_mask=True,
    )


@pytest.mark.parametrize("width", [1, 7, 8, 17])
@pytest.mark.parametrize("context", [0, 31, 2047, 2048])
def test_window_page_capacity_covers_every_start_residue(width: int, context: int) -> None:
    pages = windowed_block_page_count(context, width, 128, 1024)
    for residue in range(128):
        assert context + width + residue <= pages * 128
    assert windowed_block_page_count(context, width, 128, 1) == 1
    assert windowed_block_page_count(2048, 8, 128, 1024) == 18


@pytest.mark.parametrize("policy", ["dspark", "dspark_causal", "dflash2"])
@pytest.mark.parametrize("width", [7, 8])
@pytest.mark.parametrize("boundary", [128, 2048, 4096, 16384, 32768, 65536])
def test_window_pages_and_mask_preserve_global_visible_keys(policy: str, width: int, boundary: int) -> None:
    pytest.importorskip("torch_npu")
    from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend

    lengths = [boundary - 1, boundary + 1]
    source_pages = (max(lengths) + 127) // 128 + 2
    source = torch.arange(2 * source_pages, dtype=torch.int32).flip(0).view(2, source_pages)
    for row, length in enumerate(lengths):
        source[row, (length + 127) // 128 :].fill_(-1)
    original = source.clone()
    device_lengths = torch.tensor(lengths, dtype=torch.int32)
    layer = _attention(policy)
    context = block_draft_window_context(layer)
    assert context is not None
    window = prepare_windowed_block_table(source, device_lengths, width, 128, context)
    backend = NpuPagedAttentionBackend(4, 1, 64, 0.125, 2048, False, torch.device("cpu"), torch.float16)
    prepared = SimpleNamespace(
        device_kv_lengths=device_lengths,
        query_lengths=[width, width],
        kv_capacity=source_pages * 128,
    )
    mask = backend._block_attention_mask(prepared, layer, window.key_positions)
    assert window.capacity <= 2304
    assert mask.shape == (2, 1, width, window.capacity)
    assert bool((window.block_table >= 0).all())
    torch.testing.assert_close(source, original)
    for row, length in enumerate(lengths):
        global_keys = window.key_positions[row, 0, 0]
        first_page = int(global_keys[0]) // 128
        valid_pages = (length + 127) // 128 - first_page
        torch.testing.assert_close(
            window.block_table[row, :valid_pages], source[row, first_page : first_page + valid_pages]
        )
        anchor = length - width
        for query in range(width):
            # This reference is expressed only in global token positions. It
            # does not use the production page-selection or mask predicates.
            if policy == "dflash2":
                lower = max(0, anchor + query - 2047)
                upper = min(length, anchor + query + 8)
            else:
                lower = max(0, anchor - 2048)
                upper = anchor + query + 1 if policy == "dspark_causal" else length
            expected = torch.arange(lower, upper, dtype=global_keys.dtype)
            actual = global_keys[~mask[row, 0, query]]
            torch.testing.assert_close(actual, expected)


def test_window_page_selection_never_reads_lengths_back_to_host(monkeypatch: pytest.MonkeyPatch) -> None:
    source = torch.arange(520, dtype=torch.int32).view(2, 260)
    lengths = torch.tensor([32769, 7], dtype=torch.int32)

    def reject_readback(*args: object, **kwargs: object) -> None:
        raise AssertionError("window page selection must remain on Device")

    with monkeypatch.context() as guard:
        for name in ("item", "tolist", "cpu"):
            guard.setattr(torch.Tensor, name, reject_readback)
        window = prepare_windowed_block_table(source, lengths, 7, 128, 2048)
    assert window.block_table.shape == (2, 18)
    torch.testing.assert_close(lengths, torch.tensor([32769, 7], dtype=torch.int32))
    assert int(window.key_positions[1, 0, 0, 0]) == 0


def test_window_graph_policy_requires_every_attention_layer_to_be_windowed() -> None:
    model = torch.nn.ModuleList([_attention("dspark"), _attention("dflash2", 4096)])
    assert block_draft_model_window_context(model) == 4095
    model.append(Attention(4, 1, 64, 0.125, 0, 2))
    assert block_draft_model_window_context(model) is None
    assert block_draft_model_window_context(torch.nn.Identity()) is None


def test_window_padding_uses_live_page_after_prefix_blocks_are_retired() -> None:
    source = torch.zeros((1, 64), dtype=torch.int32)
    source[0, 15:32] = torch.arange(11, 28, dtype=torch.int32)
    original = source.clone()
    window = prepare_windowed_block_table(source, torch.tensor([4096], dtype=torch.int32), 7, 128, 2048)
    assert window.block_table.shape == (1, 18)
    torch.testing.assert_close(window.block_table[0, :17], source[0, 15:32])
    assert int(window.block_table[0, -1]) == 11
    assert bool((window.block_table > 0).all())
    torch.testing.assert_close(source, original)
