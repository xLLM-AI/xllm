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


def test_context_parallel_cli() -> None:
    parser = ArgumentParser().parser
    assert parser.parse_args([]).cp_size == 1
    assert parser.parse_args(["--cp_size", "4"]).cp_size == 4
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--enable_prefill_sp"])
    assert error.value.code == 2


@pytest.mark.parametrize("api_module,api_name", [(llm, "LLM"), (embedding, "Embedding"), (vlm, "VLM")])
def test_constructor_context_parallel_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, api_module: ModuleType, api_name: str
) -> None:
    is_vlm = api_module is vlm
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen2_vl" if is_vlm else "qwen3"}), encoding="utf-8"
    )
    master = MagicMock()
    monkeypatch.setattr(api_module, "VLMMaster" if is_vlm else "LLMMaster", master)
    monkeypatch.setattr(signal, "signal", MagicMock())
    monkeypatch.setattr(utils, "get_free_port", lambda: 26001)
    monkeypatch.setattr(utils.xllm_export, "get_model_backend", lambda _: "vlm" if is_vlm else "llm")
    monkeypatch.setattr(utils.xllm_export, "configure_cpp_chat_template", MagicMock())
    constructor = getattr(api_module, api_name)
    for kwargs, expected in (({}, 1), ({"cp_size": 4}, 4)):
        master.reset_mock()
        instance = constructor(model=str(tmp_path), **kwargs)
        master.assert_called_once()
        options = master.call_args.args[0]
        assert isinstance(options, api_module.Options)
        assert (options.model_path, options.cp_size) == (str(tmp_path), expected)
        assert instance.master is master.return_value
    master.reset_mock()
    with pytest.raises(TypeError, match="Unexpected keyword arguments: enable_prefill_sp"):
        constructor(model=str(tmp_path), enable_prefill_sp=True)
    master.assert_not_called()
