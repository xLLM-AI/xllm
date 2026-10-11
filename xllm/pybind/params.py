# Copyright 2025-2026 The xLLM Authors.
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

import math
from typing import Any, Optional, Union

from xllm_export import RequestParams


class _RequestParamsProxy:
    def __init__(self, **kwargs: Any) -> None:
        object.__setattr__(self, "_request_params", RequestParams())
        object.__setattr__(self, "_explicit_fields", set())
        for key, value in kwargs.items():
            self._set_field(key, value)

    def _set_field(self, key: str, value: Any) -> None:
        if not hasattr(self._request_params, key):
            raise TypeError(f"Unexpected parameter: {key}")
        if key == "logprobs" and isinstance(value, int) and not isinstance(value, bool):
            self._request_params.logprobs = True
            self._request_params.top_logprobs = value
            self._explicit_fields.add(key)
            return
        setattr(self._request_params, key, value)
        self._explicit_fields.add(key)

    def __getattr__(self, key: str) -> Any:
        return getattr(self._request_params, key)

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "_request_params":
            object.__setattr__(self, key, value)
            return
        self._set_field(key, value)

    def to_request_params(self) -> RequestParams:
        return self._request_params

    def explicit_fields(self) -> set[str]:
        return set(self._explicit_fields)


class SamplingParams(_RequestParamsProxy):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._request_params.max_tokens = 16
        self._request_params.stop = []
        self._request_params.stop_token_ids = []
        for key, value in kwargs.items():
            self._set_field(key, value)

        self._request_params.validate()
        if 0 < self.temperature < 0.01:
            self.temperature = 0.01
        if self.temperature == 0:
            self.top_p = 1.0
            self.top_k = 0
            self.min_p = 0.0

    def __getattr__(self, key: str) -> Any:
        if key == "logprobs":
            return self._request_params.top_logprobs if self._request_params.logprobs else None
        if key == "max_tokens" and self._request_params.max_tokens == 0:
            return None
        return super().__getattr__(key)

    def _set_field(self, key: str, value: Any) -> None:
        if key == "stop" and isinstance(value, str):
            value = [value]
        if key in {"stop", "stop_token_ids", "bad_words"} and value is None:
            value = []
        if key == "max_tokens" and value is None:
            value = 0
        elif key == "max_tokens" and value < 1:
            raise ValueError("max_tokens must be at least 1 or None")
        if key == "seed" and value == -1:
            value = None
        if (
            key in {"temperature", "top_p", "presence_penalty", "frequency_penalty", "repetition_penalty", "n"}
            and value is None
        ):
            value = {
                "temperature": 1.0,
                "top_p": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "repetition_penalty": 1.0,
                "n": 1,
            }[key]
        if key == "logprobs" and value is None:
            self._request_params.logprobs = False
            self._request_params.top_logprobs = 0
            self._explicit_fields.add(key)
            return
        if key == "logprobs" and isinstance(value, bool):
            value = int(value)
        if key == "logit_bias" and value is not None:
            if any(not math.isfinite(bias) for bias in value.values()):
                raise ValueError("logit_bias values must be finite")
            value = {int(token): min(100.0, max(-100.0, bias)) for token, bias in value.items()}
        if key == "logit_bias" and value is None:
            value = {}
        super()._set_field(key, value)


class BeamSearchParams(SamplingParams):
    def __init__(self, beam_width: int = 1, max_tokens: int = 16, **kwargs: Any) -> None:
        kwargs.setdefault("temperature", 0.0)
        super().__init__(beam_width=beam_width, max_tokens=max_tokens, **kwargs)


class PoolingParams(_RequestParamsProxy):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.is_embeddings = True


ParamLike = Union[RequestParams, _RequestParamsProxy]
ParamsLike = Optional[ParamLike | list[ParamLike]]


def to_request_params(
    params: ParamLike | None,
    default_cls: type[_RequestParamsProxy] = SamplingParams,
) -> RequestParams:
    if params is None:
        return default_cls().to_request_params()
    if isinstance(params, RequestParams):
        return params
    if isinstance(params, _RequestParamsProxy):
        return params.to_request_params()
    raise TypeError(
        "Unsupported params type. Expected RequestParams, SamplingParams, BeamSearchParams, or PoolingParams."
    )


def to_request_params_list(
    params: ParamsLike,
    default_cls: type[_RequestParamsProxy] = SamplingParams,
) -> list[RequestParams]:
    if params is None:
        return [default_cls().to_request_params()]
    if isinstance(params, list):
        if len(params) == 0:
            return [default_cls().to_request_params()]
        return [to_request_params(item, default_cls=default_cls) for item in params]
    return [to_request_params(params, default_cls=default_cls)]
