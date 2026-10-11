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

import pytest

from xllm.pybind.params import SamplingParams


def test_vllm_defaults_and_logprobs() -> None:
    params = SamplingParams()
    assert (params.temperature, params.top_p, params.top_k, params.min_p) == (1, 1, 0, 0)
    assert params.max_tokens == 16
    assert params.seed is None
    assert params.logprobs is None
    assert params.stop == []
    assert params.stop_token_ids == []
    for value, expected in ((None, None), (False, 0), (True, 1), (0, 0), (5, 5), (-1, -1)):
        params = SamplingParams(logprobs=value)
        assert params.logprobs == expected
        assert params.to_request_params().logprobs == (expected is not None)


def test_normalizes_greedy_sampling_and_optional_values() -> None:
    params = SamplingParams(temperature=0, top_k=4, top_p=0.5, min_p=0.5)
    assert (params.temperature, params.top_k, params.top_p, params.min_p) == (0, 0, 1, 0)
    assert SamplingParams(temperature=0.001).temperature == pytest.approx(0.01)
    params = SamplingParams(max_tokens=None, seed=-1, stop="END", bad_words=None, logit_bias=None)
    assert params.max_tokens is None
    assert params.to_request_params().max_tokens == 0
    assert params.seed is None
    assert params.stop == ["END"]
    assert params.bad_words == []
    assert params.logit_bias == {}
    assert SamplingParams(stop=None, stop_token_ids=None).stop_token_ids == []
    assert SamplingParams(stop_token_ids=[2, 2, 3]).stop_token_ids == [2, 2, 3]
    assert SamplingParams(bad_words=["bad", "bad"]).bad_words == ["bad", "bad"]


def test_preserves_sampling_constraints() -> None:
    params = SamplingParams(
        seed=42, min_tokens=3, min_p=0.2, logit_bias={"2": 150, 3: -150}, allowed_token_ids=[2, 3], bad_words=["bad"]
    )
    assert params.seed == 42
    assert params.min_tokens == 3
    assert params.min_p == pytest.approx(0.2)
    assert params.logit_bias == {2: 100, 3: -100}
    assert params.allowed_token_ids == [2, 3]
    assert params.bad_words == ["bad"]


@pytest.mark.parametrize("temperature", [2.01, 3.0, 100.0, 1e30])
def test_accepts_finite_temperatures_above_two(temperature: float) -> None:
    params = SamplingParams(temperature=temperature)
    assert params.temperature == pytest.approx(temperature)
    assert params.to_request_params().temperature == pytest.approx(temperature)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": -0.01},
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"temperature": 0, "n": 2},
        {"top_p": 0},
        {"top_k": -2},
        {"min_p": 1.1},
        {"repetition_penalty": 0},
        {"max_tokens": 0},
        {"max_tokens": 2, "min_tokens": 3},
        {"allowed_token_ids": []},
        {"bad_words": [""]},
        {"stop": ""},
        {"logit_bias": {1: float("nan")}},
        {"logit_bias": {1: float("inf")}},
        {"logprobs": -2},
    ],
)
def test_rejects_invalid_sampling_parameters(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)
