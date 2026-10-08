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

from types import SimpleNamespace

import pytest

from xllm.python.models import deepseek_v4
from xllm.python.models.deepseek_v4 import (
    _find_checkpoint_key,
    _require_checkpoint_key,
    _resolve_mlp_projection_names,
)


class _KeySet:
    def __init__(self, *keys: str) -> None:
        self.keys = set(keys)

    def has(self, name: str) -> bool:
        return name in self.keys


class _StopLoading(Exception):
    pass


def test_checkpoint_key_resolution_preserves_candidate_order() -> None:
    loader = _KeySet("model.head.weight", "head.weight")

    assert _find_checkpoint_key(loader, ("lm_head.weight", "model.head.weight", "head.weight")) == "model.head.weight"


def test_required_checkpoint_key_reports_all_supported_names() -> None:
    with pytest.raises(KeyError, match=r"lm_head\.weight, head\.weight"):
        _require_checkpoint_key(_KeySet(), ("lm_head.weight", "head.weight"), "output head")


def test_dsv4_loader_uses_shared_model_prefix_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_prefixes: tuple[str, ...] | None = None

    class _RecordingLoader:
        def __init__(
            self,
            model: object,
            state_dicts: object,
            tp_size: int,
            tp_rank: int,
            src_prefixes: tuple[str, ...] = ("",),
            name_aliases: dict[str, tuple[str, ...]] | None = None,
        ) -> None:
            del model, state_dicts, tp_size, tp_rank, name_aliases
            nonlocal captured_prefixes
            captured_prefixes = src_prefixes
            raise _StopLoading

    monkeypatch.setattr(deepseek_v4, "W8A8WeightLoader", _RecordingLoader)
    model = SimpleNamespace(cfg=SimpleNamespace(tp_size=1, tp_rank=0))

    with pytest.raises(_StopLoading):
        deepseek_v4.DeepseekV4ForCausalLM.load_weights(model, [], 0, 1)

    assert captured_prefixes == ("", "model.")


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        (("ffn.gate_proj.weight", "ffn.up_proj.weight", "ffn.down_proj.weight"), ("gate_proj", "up_proj", "down_proj")),
        (("ffn.w1.weight", "ffn.w3.weight", "ffn.w2.weight"), ("w1", "w3", "w2")),
    ],
)
def test_mlp_projection_resolution_accepts_native_and_legacy_names(
    keys: tuple[str, ...],
    expected: tuple[str, str, str],
) -> None:
    assert _resolve_mlp_projection_names(_KeySet(*keys), "ffn.") == expected
