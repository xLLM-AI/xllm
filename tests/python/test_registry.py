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

from unittest.mock import Mock

import pytest

from xllm.python import registry


def test_unsupported_dspark_platform_fails_before_import(monkeypatch: pytest.MonkeyPatch) -> None:
    import_model = Mock()
    monkeypatch.setattr(registry.current_platform, "device_type", lambda: "cuda")
    monkeypatch.setattr(registry, "import_module", import_model)

    with pytest.raises(NotImplementedError, match="DSparkDraftModel.*cuda"):
        registry.get_model_class("DSparkDraftModel")

    import_model.assert_not_called()


def test_dspark_model_is_registered_for_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry.current_platform, "device_type", lambda: "npu")

    model_class = registry.get_model_class("DSparkDraftModel")

    assert model_class.__name__ == "Qwen3DSparkForCausalLM"
