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


def test_unsupported_model_fails_before_import(monkeypatch: pytest.MonkeyPatch) -> None:
    import_model = Mock()
    monkeypatch.setattr(registry.current_platform, "device_type", lambda: "cuda")
    monkeypatch.setattr(registry, "import_module", import_model)

    with pytest.raises(NotImplementedError, match="qwen3_vl.*cuda"):
        registry.get_model_class("qwen3_vl")

    import_model.assert_not_called()


def test_glm_mtp_checkpoint_names_are_registered() -> None:
    expected = ("xllm.python.models.glm5_2_mtp", "Glm52MtpForCausalLM")

    assert registry._REGISTRY["GlmMoeDsaMtpForCausalLM"] == expected
    assert registry._REGISTRY["glm_moe_dsa_mtp"] == expected



def test_glm5_next_text_and_vl_registry_entries_are_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry.current_platform, "device_type", lambda: "npu")

    text_cls = registry.get_model_class("glm5_next")
    vl_cls = registry.get_model_class("glm5_next_vl")

    assert text_cls.__name__ == "Glm5NextForCausalLM"
    assert vl_cls.__name__ == "Glm5NextVLModel"
