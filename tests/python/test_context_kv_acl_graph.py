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

"""Context graph startup admission, padding and real-device cache parity."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from xllm.python.attention.backend import LayerCache
from xllm.python.model_executor.runners.context_kv_acl_graph import ContextKvAclGraphRunner


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> ContextKvAclGraphRunner:
    model = SimpleNamespace(
        model=SimpleNamespace(context_projection_weight=torch.ones(2, 4)),
        write_context_kv=Mock(side_effect=lambda hidden, *_: hidden[:, :2].clone()),
    )
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_stream=Mock(return_value=Mock()),
            Stream=Mock(return_value=Mock()),
            stream=Mock(side_effect=lambda *args: nullcontext()),
            NPUGraph=Mock(side_effect=Mock),
            graph=Mock(side_effect=lambda *args, **kwargs: nullcontext()),
        ),
        raising=False,
    )
    monkeypatch.setattr("xllm.python.model_executor.runners.context_kv_acl_graph._get_graph_pool", lambda: (0, 1))
    result = ContextKvAclGraphRunner(model, torch.device("cpu"))
    result.bind_kv_caches([LayerCache(torch.full((1, 16, 1, 2), 42.0), torch.full((1, 16, 1, 2), 43.0))])
    return result


def test_startup_captures_all_prefix_buckets_once_without_writing_cache(runner: ContextKvAclGraphRunner) -> None:
    runner.warmup(5)
    calls = runner.model.write_context_kv.call_count
    runner.warmup(5)
    assert runner._buckets == [1, 2, 4, 8]
    assert runner.model.write_context_kv.call_count == calls
    for call in runner.model.write_context_kv.call_args_list:
        assert torch.all(call.args[2] == -1)
        assert call.args[4] is None


def test_capture_uses_private_stream_and_orders_initialization_and_replay(runner: ContextKvAclGraphRunner) -> None:
    caller_stream = torch.npu.current_stream()
    runner.warmup(4)
    capture_stream = runner._capture_stream
    assert capture_stream is not caller_stream
    torch.npu.Stream.assert_called_once_with(device=runner.device)
    capture_stream.wait_stream.assert_called_once_with(caller_stream)
    caller_stream.wait_stream.assert_called_once_with(capture_stream)
    torch.npu.stream.assert_called_once_with(capture_stream)
    assert all(call.kwargs["stream"] is capture_stream for call in torch.npu.graph.call_args_list)
    runner.warmup(8)
    torch.npu.Stream.assert_called_once()
    assert capture_stream.wait_stream.call_count == 2
    assert caller_stream.wait_stream.call_count == 2


def test_replay_changes_positions_and_masks_stale_tail_without_recapture(runner: ContextKvAclGraphRunner) -> None:
    runner.warmup(4)
    entry = runner._graphs[4]
    runner.execute(torch.ones(4, 4), torch.arange(4), torch.arange(4, dtype=torch.int32))
    capture_calls = torch.npu.graph.call_count
    forward_calls = runner.model.write_context_kv.call_count
    hidden = torch.full((3, 4), 7.0)
    output = runner.execute(hidden, torch.tensor([8, 9, 10]), torch.tensor([7, -1, 9], dtype=torch.int32))
    torch.testing.assert_close(entry.hidden[:3], hidden)
    torch.testing.assert_close(entry.positions, torch.tensor([8, 9, 10, 3], dtype=torch.int32))
    torch.testing.assert_close(entry.slots, torch.tensor([7, -1, 9, -1], dtype=torch.int32))
    assert torch.all(entry.hidden[3:] == 1)
    assert output.shape == (3, 2)
    assert entry.graph.replay.call_count == 2
    assert torch.npu.graph.call_count == capture_calls
    assert runner.model.write_context_kv.call_count == forward_calls


def test_missing_startup_capture_is_not_a_runtime_fallback(runner: ContextKvAclGraphRunner) -> None:
    with pytest.raises(RuntimeError, match="startup warmup"):
        runner.execute(torch.ones(1, 4), torch.ones(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32))
    torch.npu.graph.assert_not_called()
    runner.model.write_context_kv.assert_not_called()


def test_declared_out_of_capacity_path_uses_eager_without_capture(runner: ContextKvAclGraphRunner) -> None:
    runner.warmup(2)
    capture_calls = torch.npu.graph.call_count
    hidden = torch.ones(3, 4)
    output = runner.execute(hidden, torch.arange(3), torch.arange(3, dtype=torch.int32))
    assert output.shape == (3, 2)
    assert torch.npu.graph.call_count == capture_calls
    assert runner.replays == 0


@pytest.mark.parametrize("change", ["width", "rows", "dtype", "slot_dtype"])
def test_invalid_inputs_fail_before_replay(runner: ContextKvAclGraphRunner, change: str) -> None:
    runner.warmup(4)
    hidden, positions, slots = torch.ones(3, 4), torch.arange(3), torch.arange(3, dtype=torch.int32)
    if change == "width":
        hidden = torch.ones(3, 5)
    elif change == "rows":
        positions = torch.arange(2)
    elif change == "dtype":
        hidden = hidden.double()
    else:
        slots = slots.long()
    with pytest.raises(ValueError):
        runner.execute(hidden, positions, slots)
    assert runner.replays == 0


def test_cache_rebinding_is_rejected(runner: ContextKvAclGraphRunner) -> None:
    old_key, old_value = runner._kv_caches[0]
    runner.bind_kv_caches([LayerCache(old_key, old_value)])
    with pytest.raises(ValueError, match="bindings changed"):
        runner.bind_kv_caches([LayerCache(old_key.clone(), old_value)])


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires Ascend NPU")
@pytest.mark.parametrize("default_stream", [True, False], ids=["default_startup", "nondefault_startup"])
def test_real_context_graph_matches_eager_and_preserves_masked_slots(
    default_stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xllm.python.models.qwen3_dspark import Qwen3DSparkForCausalLM

    monkeypatch.setattr("xllm.python.model_executor.runners.acl_graph._GRAPH_POOL", None)
    torch.manual_seed(23)
    model = Qwen3DSparkForCausalLM(
        {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "intermediate_size": 128,
            "max_position_embeddings": 32,
            "vocab_size": 8,
            "markov_rank": 2,
            "dtype": "bfloat16",
            "device": "npu:0",
        }
    )
    with torch.no_grad():
        model.model.fc.load_weight(torch.randn(64, 128), tp_rank=0)
        for parameter in model.parameters():
            parameter.normal_()
        model.model._build_context_kv_buffers()
    caches = [
        (
            torch.full((1, 128, 1, 64), 42.0, dtype=torch.bfloat16, device="npu:0"),
            torch.full((1, 128, 1, 64), 43.0, dtype=torch.bfloat16, device="npu:0"),
        )
        for _ in range(2)
    ]
    graph_runner = ContextKvAclGraphRunner(model, torch.device("npu:0"))
    graph_runner.bind_kv_caches([LayerCache(*cache) for cache in caches])
    # Service startup uses the default stream; serving can use a compute stream.
    # Both callers must initialize and replay without capturing on that stream.
    stream = torch.npu.default_stream("npu:0") if default_stream else torch.npu.Stream(device="npu:0")
    if stream != torch.npu.current_stream():
        stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        graph_runner.warmup(4)
        assert graph_runner._capture_stream != stream
        assert torch.npu.current_stream() == stream
        for keys, values in caches:
            assert torch.all(keys == 42).item()
            assert torch.all(values == 43).item()
        for iteration, rows in enumerate((4, 3, 4, 1, 3, 2)):
            hidden = torch.randn(rows, 128, dtype=torch.bfloat16, device="npu:0")
            positions = torch.arange(iteration, iteration + rows, dtype=torch.int32, device="npu:0")
            slots = torch.arange(2, rows + 2, dtype=torch.int32, device="npu:0")
            slots[-1] = -1
            reference_caches = [(key.clone(), value.clone()) for key, value in caches]
            reference = model.write_context_kv(hidden, positions, slots, reference_caches, None)
            actual = graph_runner.execute(hidden, positions, slots)
            torch.npu.synchronize()
            torch.testing.assert_close(actual, reference)
            for actual_pair, expected_pair in zip(caches, reference_caches):
                for actual_cache, expected_cache in zip(actual_pair, expected_pair):
                    torch.testing.assert_close(actual_cache, expected_cache)
    assert graph_runner.replays == 6
