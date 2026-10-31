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

from __future__ import annotations

import types
import warnings

import pytest

from xllm.python.kernels_npu import _custom_op


@pytest.mark.parametrize(
    "xllm_ops",
    [types.SimpleNamespace(rms_norm=object()), types.SimpleNamespace()],
    ids=["stale_binary", "no_xllm_runtime"],
)
def test_register_fake_missing_operator(monkeypatch: pytest.MonkeyPatch, xllm_ops: object) -> None:
    monkeypatch.setattr(_custom_op.torch, "ops", types.SimpleNamespace(xllm_ops=xllm_ops))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _custom_op.register_fake("xllm_ops::definitely_missing_op_xyz", lambda tensor: tensor)

    if hasattr(xllm_ops, "rms_norm"):
        assert len(caught) == 1
        message = str(caught[0].message)
        assert "xllm_ops::definitely_missing_op_xyz" in message
        assert "not registered" in message
    else:
        assert caught == []
