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
# ==============================================================================

import json
import signal
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from xllm.pybind import embedding, llm, utils, vlm
from xllm.pybind.args import ArgumentParser


@pytest.mark.parametrize(
    "arguments,expected",
    [
        ([], True),
        (["--enable_graph_double_buffer"], True),
        (["--enable_graph_double_buffer", "false"], False),
        (["--enable_graph_double_buffer", "true"], True),
    ],
)
def test_graph_double_buffer_cli(arguments: list[str], expected: bool) -> None:
    assert ArgumentParser().parser.parse_args(arguments).enable_graph_double_buffer is expected


@pytest.mark.parametrize(
    "arguments,expected",
    [([], False), (["--disable_graph_warmup"], True), (["--disable_graph_warmup", "false"], False)],
)
def test_disable_graph_warmup_cli(arguments: list[str], expected: bool) -> None:
    assert ArgumentParser().parser.parse_args(arguments).disable_graph_warmup is expected


@pytest.mark.parametrize("arguments,expected", [([], 16), (["--acl_graph_decode_batch_size_limit", "3"], 3)])
def test_graph_decode_limit_cli(arguments: list[str], expected: int) -> None:
    assert ArgumentParser().parser.parse_args(arguments).acl_graph_decode_batch_size_limit == expected


@pytest.mark.parametrize(
    "arguments,expected",
    [
        ([], 0),
        (["--block_draft_graph_warmup_max_kv_len", "-1"], -1),
        (["--block_draft_graph_warmup_max_kv_len", "4096"], 4096),
    ],
)
def test_block_draft_graph_warmup_limit_cli(arguments: list[str], expected: int) -> None:
    assert ArgumentParser().parser.parse_args(arguments).block_draft_graph_warmup_max_kv_len == expected


@pytest.mark.parametrize("warmup_limit", [-1, 0, 4096])
@pytest.mark.parametrize(
    "api_module,api_name,backend",
    [(llm, "LLM", "llm"), (llm, "LLM", "vlm"), (vlm, "VLM", "vlm"), (embedding, "Embedding", "llm")],
)
def test_constructor_keeps_graph_options_per_instance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    api_module: ModuleType,
    api_name: str,
    backend: str,
    warmup_limit: int,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen2_vl" if backend == "vlm" else "qwen3"}), encoding="utf-8"
    )
    master = MagicMock()
    monkeypatch.setattr(api_module, "VLMMaster" if backend == "vlm" else "LLMMaster", master)
    monkeypatch.setattr(signal, "signal", MagicMock())
    monkeypatch.setattr(utils, "get_free_port", lambda: 26001)
    monkeypatch.setattr(utils.xllm_export, "get_model_backend", lambda _: backend)
    monkeypatch.setattr(utils.xllm_export, "configure_cpp_chat_template", MagicMock())
    constructor = getattr(api_module, api_name)

    single_buffer = constructor(
        model=str(tmp_path),
        enable_graph=True,
        enable_graph_double_buffer=False,
        acl_graph_decode_batch_size_limit=3,
        disable_graph_warmup=True,
        block_draft_graph_warmup_max_kv_len=warmup_limit,
    )
    single_buffer_options = master.call_args.args[0]
    double_buffer = constructor(model=str(tmp_path), enable_graph=True)
    double_buffer_options = master.call_args.args[0]

    assert single_buffer.master is master.return_value
    assert double_buffer.master is master.return_value
    assert isinstance(single_buffer_options, api_module.Options)
    assert isinstance(double_buffer_options, api_module.Options)
    assert single_buffer_options is not double_buffer_options
    assert single_buffer_options.enable_graph is True
    assert double_buffer_options.enable_graph is True
    assert single_buffer_options.enable_graph_double_buffer is False
    assert double_buffer_options.enable_graph_double_buffer is True
    assert single_buffer_options.acl_graph_decode_batch_size_limit == 3
    assert double_buffer_options.acl_graph_decode_batch_size_limit == 16
    assert single_buffer_options.disable_graph_warmup is True
    assert double_buffer_options.disable_graph_warmup is False
    assert single_buffer_options.block_draft_graph_warmup_max_kv_len == warmup_limit
    assert double_buffer_options.block_draft_graph_warmup_max_kv_len == 0
