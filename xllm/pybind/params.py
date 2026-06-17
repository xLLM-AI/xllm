from typing import Any, List, Optional, Set, Type, Union

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

    def explicit_fields(self) -> Set[str]:
        return set(self._explicit_fields)


class SamplingParams(_RequestParamsProxy):
    pass


class BeamSearchParams(SamplingParams):
    def __init__(self,
                 beam_width: int = 1,
                 max_tokens: int = 16,
                 **kwargs: Any) -> None:
        super().__init__(beam_width=beam_width, max_tokens=max_tokens, **kwargs)


class PoolingParams(_RequestParamsProxy):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.is_embeddings = True


ParamLike = Union[RequestParams, _RequestParamsProxy]
ParamsLike = Optional[Union[ParamLike, List[ParamLike]]]


def to_request_params(
    params: Optional[ParamLike],
    default_cls: Type[_RequestParamsProxy] = SamplingParams,
) -> RequestParams:
    if params is None:
        return default_cls().to_request_params()
    if isinstance(params, RequestParams):
        return params
    if isinstance(params, _RequestParamsProxy):
        return params.to_request_params()
    raise TypeError(
        "Unsupported params type. Expected RequestParams, SamplingParams, "
        "BeamSearchParams, or PoolingParams."
    )


def to_request_params_list(
    params: ParamsLike,
    default_cls: Type[_RequestParamsProxy] = SamplingParams,
) -> List[RequestParams]:
    if params is None:
        return [default_cls().to_request_params()]
    if isinstance(params, list):
        if len(params) == 0:
            return [default_cls().to_request_params()]
        return [to_request_params(item, default_cls=default_cls) for item in params]
    return [to_request_params(params, default_cls=default_cls)]
