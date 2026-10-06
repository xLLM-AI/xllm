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

"""Responses checks against a running brpc fixture or model service.

Pytest uses the normal tests/python/conftest.py entry point. The standalone CLI
is an HTTP-only client for an externally owned service; it neither loads native
extensions nor starts services. Configured endpoint failures are not skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx
    import openai


@dataclass(frozen=True)
class _Settings:
    base_url: str
    model: str
    fixture: bool = False
    limit_one: bool = False
    expect_reasoning: bool = False
    max_output_tokens: int = 4096
    evidence_dir: Path | None = None


_TOOLS = [
    {
        "type": "function",
        "name": "weather",
        "description": "Get the current weather in a city. Use this instead of guessing the weather.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": False,
    }
]
_TERMINALS = {"response.completed", "response.incomplete", "response.failed"}
_RESPONSE_FIELDS = {
    "id",
    "object",
    "created_at",
    "status",
    "error",
    "incomplete_details",
    "model",
    "instructions",
    "output",
    "usage",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "temperature",
    "top_p",
    "max_output_tokens",
    "text",
    "reasoning",
    "metadata",
    "store",
    "background",
    "previous_response_id",
    "truncation",
    "access_programs",
    "completed_at",
}


def _environment_settings() -> _Settings:
    base_url = os.environ.get("XLLM_RESPONSES_BASE_URL")
    if not base_url:
        import pytest

        pytest.skip("Set XLLM_RESPONSES_BASE_URL to an externally owned Responses service")
    model = os.environ.get("XLLM_RESPONSES_MODEL")
    if not model:
        raise ValueError("XLLM_RESPONSES_MODEL is required when an endpoint is configured")
    fixture = os.environ.get("XLLM_RESPONSES_FIXTURE", "0") == "1"
    evidence = os.environ.get("XLLM_RESPONSES_EVIDENCE_DIR")
    return _Settings(
        base_url=base_url.rstrip("/"),
        model=model,
        fixture=fixture,
        limit_one=os.environ.get("XLLM_RESPONSES_LIMIT_ONE", "0") == "1",
        expect_reasoning=os.environ.get("XLLM_RESPONSES_EXPECT_REASONING", "0") == "1",
        max_output_tokens=int(os.environ.get("XLLM_RESPONSES_MAX_OUTPUT_TOKENS", "4096")),
        evidence_dir=Path(evidence) if evidence else None,
    )


@contextmanager
def _clients(settings: _Settings) -> Iterator[tuple[httpx.Client, openai.OpenAI]]:
    import httpx
    import openai

    with (
        httpx.Client(timeout=180, trust_env=False) as http_client,
        openai.OpenAI(
            base_url=settings.base_url,
            api_key="responses-validation",
            max_retries=0,
            http_client=http_client,
        ) as sdk,
    ):
        yield http_client, sdk


def _prompt(settings: _Settings, case: str) -> str:
    if settings.fixture:
        return f"__responses_fixture_{case}__"
    prompts = {
        "text": "Reply briefly in one sentence and include the word hello.",
        "json": 'Return only the JSON object {"ok":true}, without Markdown or additional text.',
        "tool": "Use weather to get the current weather in Paris. Call weather with city Paris; do not guess.",
        "incomplete": "Write every integer from 1 to 10000, one per line, without abbreviating or stopping early.",
        "hold": "Write a long detailed numbered list of 10000 separate facts. Do not abbreviate or stop early.",
        "reasoning": "What is 7 times 8? Think carefully, then give a short answer.",
    }
    return prompts[case]


def _payload(settings: _Settings, case: str, **fields: Any) -> dict[str, Any]:
    return {
        "model": settings.model,
        "input": _prompt(settings, case),
        "max_output_tokens": settings.max_output_tokens,
        "temperature": 0,
        "store": False,
        **fields,
    }


def _record(settings: _Settings, label: str, value: Any) -> None:
    if settings.evidence_dir is None:
        return
    settings.evidence_dir.mkdir(parents=True, exist_ok=True)
    with (settings.evidence_dir / f"{label}.json").open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")


def _dump(value: Any) -> dict[str, Any]:
    return value.model_dump(mode="json", exclude_unset=True)


def _nonnegative_integer(value: Any) -> None:
    assert type(value) is int and value >= 0, value


def _assert_usage(usage: dict[str, Any]) -> None:
    assert set(usage) >= {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "input_tokens_details",
        "output_tokens_details",
    }
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        _nonnegative_integer(usage[field])
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
    details = usage["input_tokens_details"]
    assert set(details) >= {"cached_tokens", "cache_write_tokens"}
    for field in ("cached_tokens", "cache_write_tokens"):
        _nonnegative_integer(details[field])
    assert details["cached_tokens"] <= usage["input_tokens"]
    assert details["cache_write_tokens"] == 0
    reasoning_tokens = usage["output_tokens_details"]["reasoning_tokens"]
    _nonnegative_integer(reasoning_tokens)
    assert reasoning_tokens == 0


def _output_text(response: dict[str, Any]) -> str:
    return "".join(
        part["text"]
        for item in response["output"]
        if item["type"] == "message"
        for part in item["content"]
        if part["type"] == "output_text"
    )


def _assert_response(response: dict[str, Any], status: str = "completed", *, sdk: bool = False) -> None:
    assert set(response) >= _RESPONSE_FIELDS, _RESPONSE_FIELDS - set(response)
    assert response["object"] == "response" and "choices" not in response
    assert response["id"].startswith("resp_") and len(response["id"]) > 5
    for field in ("created_at", "completed_at"):
        timestamp = response[field]
        if field == "completed_at" and status != "completed":
            assert timestamp is None
        elif sdk and type(timestamp) is float:
            # SDK Response models declare timestamps as float; raw JSON remains integer.
            assert math.isfinite(timestamp) and timestamp >= 0 and timestamp.is_integer(), timestamp
        else:
            _nonnegative_integer(timestamp)
    assert response["status"] == status
    assert response["store"] is False and response["background"] is False
    assert response["previous_response_id"] is None
    assert response["parallel_tool_calls"] is True
    assert response["truncation"] == "disabled"
    assert response["text"]["format"]["type"] in {"text", "json_object"}
    if status in {"completed", "incomplete"}:
        assert response["error"] is None
        _assert_usage(response["usage"])
    if status == "incomplete":
        assert response["incomplete_details"] == {"reason": "max_output_tokens"}
    elif status == "failed":
        assert set(response["error"]) >= {"code", "message"} and response["error"]["message"]
    else:
        assert response["incomplete_details"] is None
    ids: set[str] = set()
    call_ids: set[str] = set()
    for item in response["output"]:
        assert item["id"] not in ids
        ids.add(item["id"])
        assert item["status"] == ("completed" if status == "completed" else "incomplete")
        if item["type"] == "message":
            assert item["id"].startswith("msg_") and item["role"] == "assistant"
            assert item["content"]
            for part in item["content"]:
                assert part["type"] == "output_text" and type(part["text"]) is str
                assert type(part["annotations"]) is list and type(part["logprobs"]) is list
                part["text"].encode("utf-8", errors="strict")
        elif item["type"] == "function_call":
            assert item["id"].startswith("fc_")
            assert item["call_id"].startswith("call_") and item["call_id"] not in call_ids
            call_ids.add(item["call_id"])
            assert item["name"] and type(item["arguments"]) is str
            if status == "completed":
                assert type(json.loads(item["arguments"])) is dict
        elif item["type"] == "reasoning":
            assert item["id"].startswith("rs_")
            assert item["summary"] == [], "Raw reasoning is not a summary"
            assert item["content"]
            for part in item["content"]:
                assert part["type"] == "reasoning_text" and type(part["text"]) is str
        else:
            raise AssertionError(f"Unsupported output item: {item['type']}")


def _assert_events(events: list[dict[str, Any]], status: str = "completed", *, sdk: bool = False) -> dict[str, Any]:
    assert [event["type"] for event in events[:2]] == ["response.created", "response.in_progress"]
    assert [event["sequence_number"] for event in events] == list(range(len(events)))
    assert all(type(event["sequence_number"]) is int and "choices" not in event for event in events)
    terminals = [event for event in events if event["type"] in _TERMINALS]
    assert len(terminals) == 1 and events[-1] is terminals[0]
    assert terminals[0]["type"] == f"response.{status}"
    initial = events[0]["response"]
    assert initial["status"] == "in_progress" and initial["output"] == [] and initial["usage"] is None
    assert events[1]["response"]["id"] == initial["id"]
    items: list[dict[str, Any]] = []
    parts: dict[int, list[dict[str, Any]]] = {}
    deltas: dict[tuple[int, int], str] = {}
    arguments: dict[int, str] = {}
    done_items: dict[int, dict[str, Any]] = {}
    done_text: set[tuple[int, int]] = set()
    done_arguments: set[int] = set()
    for event in events[2:-1]:
        kind = event["type"]
        index = event["output_index"]
        assert type(index) is int and index >= 0
        if kind == "response.output_item.added":
            assert index == len(items), "Output indexes must be contiguous"
            item = event["item"]
            assert item["status"] == "in_progress"
            assert item["id"] not in {existing["id"] for existing in items}
            if item["type"] == "function_call":
                assert item["arguments"] == ""
                arguments[index] = ""
            else:
                assert item["content"] == []
            items.append(item)
            continue
        assert index < len(items)
        item = items[index]
        if kind == "response.output_item.done":
            assert index not in done_items
            assert event["item"]["id"] == item["id"] and event["item"]["type"] == item["type"]
            done_items[index] = event["item"]
            if item["type"] == "function_call":
                assert index in done_arguments
                assert event["item"]["arguments"] == arguments[index]
                assert event["item"]["name"] == item["name"]
                assert event["item"]["call_id"] == item["call_id"]
            else:
                for content_index, part in enumerate(event["item"]["content"]):
                    assert (index, content_index) in done_text
                    assert part["text"] == deltas[(index, content_index)]
            continue
        assert event["item_id"] == item["id"]
        if kind.startswith("response.function_call_arguments."):
            assert item["type"] == "function_call"
            if kind.endswith(".delta"):
                assert index not in done_arguments
                arguments[index] += event["delta"]
            else:
                assert kind.endswith(".done") and index not in done_arguments
                assert event["arguments"] == arguments[index]
                if "name" in event:
                    assert event["name"] == item["name"]
                done_arguments.add(index)
            continue
        content_index = event["content_index"]
        assert type(content_index) is int and content_index >= 0
        key = (index, content_index)
        if kind == "response.content_part.added":
            assert item["type"] == "message"
            content_parts = parts.setdefault(index, [])
            assert content_index == len(content_parts)
            assert event["part"]["type"] == "output_text" and event["part"]["text"] == ""
            content_parts.append(event["part"])
        elif kind in {"response.output_text.delta", "response.reasoning_text.delta"}:
            if kind == "response.output_text.delta":
                assert item["type"] == "message" and content_index < len(parts[index])
                assert type(event["logprobs"]) is list
            else:
                assert item["type"] == "reasoning" and content_index == 0
            assert key not in done_text and type(event["delta"]) is str
            event["delta"].encode("utf-8", errors="strict")
            deltas[key] = deltas.get(key, "") + event["delta"]
        elif kind in {"response.output_text.done", "response.reasoning_text.done"}:
            assert key not in done_text and event["text"] == deltas[key]
            if kind == "response.output_text.done":
                assert type(event["logprobs"]) is list
            done_text.add(key)
        elif kind == "response.content_part.done":
            assert key in done_text and event["part"]["text"] == deltas[key]
        else:
            raise AssertionError(f"Unexpected SSE event: {kind}")
    final = terminals[0]["response"]
    _assert_response(final, status, sdk=sdk)
    assert final["id"] == initial["id"] and final["model"] == initial["model"]
    assert len(final["output"]) == len(items)
    for index, item in enumerate(final["output"]):
        assert item["id"] == items[index]["id"]
        if status != "failed":
            assert item == done_items[index]
        if item["type"] == "function_call":
            assert item["arguments"] == arguments[index]
        else:
            for content_index, part in enumerate(item["content"]):
                assert part["text"] == deltas[(index, content_index)]
    return final


def _assert_accumulator(events: list[dict[str, Any]], final: dict[str, Any]) -> None:
    # SDK helpers may enrich or synthesize events. Wire ordering is checked
    # independently by raw SSE and responses.create(stream=True).
    _assert_response(final, sdk=True)
    terminals = [event for event in events if event["type"] in _TERMINALS]
    assert len(terminals) == 1 and terminals[0]["type"] == "response.completed"
    assert final == terminals[0]["response"]
    for output_index, item in enumerate(final["output"]):
        if item["type"] == "function_call":
            delta_type = "response.function_call_arguments.delta"
            value = item["arguments"]
        else:
            delta_type = "response.output_text.delta" if item["type"] == "message" else "response.reasoning_text.delta"
            value = "".join(part["text"] for part in item["content"])
        deltas = [event for event in events if event["type"] == delta_type and event["output_index"] == output_index]
        assert all(event["item_id"] == item["id"] for event in deltas)
        assert "".join(event["delta"] for event in deltas) == value


def _parse_sse(body: bytes) -> list[dict[str, Any]]:
    text = body.decode("utf-8", errors="strict")
    assert "[DONE]" not in text and text.endswith(("\n\n", "\r\n\r\n"))
    events: list[dict[str, Any]] = []
    event_name: str | None = None
    data: list[str] = []
    for line in text.splitlines() + [""]:
        if not line:
            if data:
                event = json.loads("\n".join(data))
                assert event_name == event["type"], "SSE name and JSON type must agree"
                events.append(event)
            event_name, data = None, []
        elif line.startswith("event: "):
            assert event_name is None
            event_name = line[7:]
        elif line.startswith("data: "):
            data.append(line[6:])
        else:
            assert line.startswith(":"), line
    return events


def _post_raw(
    settings: _Settings,
    client: httpx.Client,
    label: str,
    payload: dict[str, Any],
    status: str = "completed",
) -> dict[str, Any]:
    response = client.post(f"{settings.base_url}/responses", json=payload)
    _record(settings, label, {"status": response.status_code, "headers": dict(response.headers), "body": response.text})
    assert response.status_code == 200, response.text
    if payload.get("stream", False):
        assert response.headers["content-type"].startswith("text/event-stream")
        return _assert_events(_parse_sse(response.content), status)
    assert response.headers["content-type"].startswith("application/json")
    parsed = response.json()
    _assert_response(parsed, status)
    return parsed


def _assert_error(response: httpx.Response, code: int, param: str | None = None) -> None:
    assert response.status_code == code, response.text
    assert response.headers["content-type"].startswith("application/json")
    assert "retry-after" not in response.headers
    body = response.json()
    assert set(body) == {"error"}
    error = body["error"]
    assert set(error) == {"message", "type", "param", "code"}
    assert type(error["message"]) is str and error["message"]
    expected_type = {400: "BadRequestError", 404: "NotFoundError", 429: "RateLimitError", 500: "InternalServerError"}[
        code
    ]
    assert error["type"] == expected_type and error["code"] == code
    assert error["param"] == param


def _assert_text_result(settings: _Settings, response: dict[str, Any]) -> None:
    assert _output_text(response).strip()
    if settings.fixture:
        assert _output_text(response) == "hello world"
    if settings.expect_reasoning:
        reasoning = [item for item in response["output"] if item["type"] == "reasoning"]
        assert reasoning and any(part["text"] for item in reasoning for part in item["content"])


def _check_text(settings: _Settings) -> None:
    with _clients(settings) as (http_client, sdk):
        raw = _post_raw(settings, http_client, "raw-text", _payload(settings, "text"))
        _assert_text_result(settings, raw)
        omitted_store = _payload(settings, "text")
        omitted_store.pop("store")
        omitted_store.pop("model")
        stateless = _post_raw(settings, http_client, "raw-stateless-defaults", omitted_store)
        assert stateless["store"] is False and stateless["model"] == settings.model
        response = sdk.responses.create(**_payload(settings, "text"), stream=False)
        parsed = _dump(response)
        _record(settings, "sdk-create-text", parsed)
        _assert_response(parsed, sdk=True)
        _assert_text_result(settings, parsed)
        assert response.output_text == _output_text(parsed)
        history = [
            {"role": "user", "content": _prompt(settings, "text")},
            *parsed["output"],
            {"role": "user", "content": "Now reply briefly again."},
        ]
        followup = sdk.responses.create(**_payload(settings, "text", input=history))
        _record(settings, "sdk-text-history", _dump(followup))
        _assert_response(_dump(followup), sdk=True)
        assert followup.output_text.strip()


def _check_json_object(settings: _Settings) -> None:
    payload = _payload(
        settings,
        "json",
        text={"format": {"type": "json_object"}},
        instructions='Return only the JSON object {"ok":true}, without Markdown or additional text.',
    )
    with _clients(settings) as (http_client, sdk):
        for streaming in (False, True):
            request = {**payload, "stream": streaming}
            raw = _post_raw(settings, http_client, f"raw-json-object-{streaming}", request)
            if streaming:
                with sdk.responses.create(**request) as stream:
                    events = [_dump(event) for event in stream]
                _record(settings, "sdk-json-object-stream", events)
                parsed = _assert_events(events, sdk=True)
            else:
                response = sdk.responses.create(**request)
                parsed = _dump(response)
                _record(settings, "sdk-json-object", parsed)
                _assert_response(parsed, sdk=True)
                assert response.output_text == _output_text(parsed)
            for result in (raw, parsed):
                assert result["text"]["format"] == {"type": "json_object"}
                assert json.loads(_output_text(result)) == {"ok": True}


def _check_stream(settings: _Settings) -> None:
    with _clients(settings) as (http_client, sdk):
        raw = _post_raw(settings, http_client, "raw-stream-text", _payload(settings, "text", stream=True))
        _assert_text_result(settings, raw)
        with sdk.responses.create(**_payload(settings, "text"), stream=True) as stream:
            events = [_dump(event) for event in stream]
        _record(settings, "sdk-create-stream-text", events)
        _assert_text_result(settings, _assert_events(events, sdk=True))
        with sdk.responses.stream(**_payload(settings, "text")) as stream:
            helper_events = [_dump(event) for event in stream]
            final = stream.get_final_response()
        _record(settings, "sdk-accumulator-text", {"events": helper_events, "final": _dump(final)})
        _assert_accumulator(helper_events, _dump(final))
        _assert_text_result(settings, _dump(final))
        assert final.output_text == _output_text(_dump(final))
        incomplete = _payload(settings, "incomplete", max_output_tokens=16)
        for streaming in (False, True):
            raw = _post_raw(
                settings, http_client, f"raw-incomplete-{streaming}", {**incomplete, "stream": streaming}, "incomplete"
            )
            assert 0 < raw["usage"]["output_tokens"] <= 16
            if streaming:
                with sdk.responses.create(**incomplete, stream=True) as stream:
                    events = [_dump(event) for event in stream]
                _record(settings, "sdk-incomplete-stream", events)
                _assert_events(events, "incomplete", sdk=True)
            else:
                result = _dump(sdk.responses.create(**incomplete, stream=False))
                _record(settings, "sdk-incomplete", result)
                _assert_response(result, "incomplete", sdk=True)


def _assert_weather_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    calls = [item for item in response["output"] if item["type"] == "function_call"]
    assert calls, "The model must produce a real client-side function call for this check"
    for call in calls:
        assert call["name"] == "weather"
        assert json.loads(call["arguments"]) == {"city": "Paris"}
    return calls


def _check_functions(settings: _Settings) -> None:
    payload = _payload(settings, "tool", tools=_TOOLS, tool_choice="auto")
    with _clients(settings) as (http_client, sdk):
        for streaming in (False, True):
            raw = _post_raw(settings, http_client, f"raw-tools-{streaming}", {**payload, "stream": streaming})
            _assert_weather_calls(raw)
        result = _dump(sdk.responses.create(**payload, stream=False))
        _record(settings, "sdk-tools", result)
        _assert_response(result, sdk=True)
        _assert_weather_calls(result)
        with sdk.responses.create(**payload, stream=True) as stream:
            events = [_dump(event) for event in stream]
        _record(settings, "sdk-tools-stream", events)
        _assert_weather_calls(_assert_events(events, sdk=True))
        with sdk.responses.stream(**payload) as stream:
            helper_events = [_dump(event) for event in stream]
            response = stream.get_final_response()
        final = _dump(response)
        _record(settings, "sdk-tools-accumulator", {"events": helper_events, "final": final})
        _assert_accumulator(helper_events, final)
        calls = _assert_weather_calls(final)
        # Accumulator models add client-only parsed fields, not Responses input fields.
        history = [
            {"role": "user", "content": _prompt(settings, "tool")},
            *[
                item.model_dump(
                    mode="json",
                    exclude_unset=True,
                    exclude={"parsed_arguments": True, "content": {"__all__": {"parsed"}}},
                )
                for item in response.output
            ],
        ]
        sdk_history = [
            history[0],
            *[
                item if item.type == "function_call" else replay
                for item, replay in zip(response.output, history[1:], strict=True)
            ],
        ]
        for call in calls:
            # This is a client-supplied result, not execution of a server-side tool.
            result = {
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": json.dumps({"city": "Paris", "condition": "sunny"}),
            }
            history.append(result)
            sdk_history.append(result)
        followup = _payload(settings, "tool", input=history, tools=_TOOLS, tool_choice="none")
        for streaming in (False, True):
            raw = _post_raw(
                settings, http_client, f"raw-function-history-{streaming}", {**followup, "stream": streaming}
            )
            assert _output_text(raw).strip()
            assert not any(item["type"] == "function_call" for item in raw["output"])
        response = sdk.responses.create(**{**followup, "input": sdk_history}, stream=False)
        _record(settings, "sdk-function-history", _dump(response))
        _assert_response(_dump(response), sdk=True)
        assert response.output_text.strip()
        if settings.fixture:
            assert response.output_text == "It is sunny in Paris."


def _check_errors(settings: _Settings) -> None:
    import openai

    invalid = [
        ({"store": True}, "store"),
        ({"previous_response_id": "resp_prior"}, "previous_response_id"),
        ({"background": True}, "background"),
        ({"max_output_tokens": 15}, "max_output_tokens"),
        ({"max_tokens": 16}, "max_tokens"),
        ({"tool_choice": "required"}, "tool_choice"),
        ({"parallel_tool_calls": False}, "parallel_tool_calls"),
        ({"tools": [{"type": "function", "name": "weather", "strict": True}]}, "tools[0].strict"),
        ({"tools": [{"type": "web_search"}]}, "tools[0].type"),
        ({"reasoning": {"summary": "auto"}}, "reasoning.summary"),
        ({"text": {"format": {"type": "json_schema"}}}, "text.format.type"),
        ({"input": [{"role": "user", "content": [{"type": "input_image"}]}]}, "input[0].content[0].type"),
    ]
    with _clients(settings) as (http_client, sdk):
        for streaming in (False, True):
            for index, (fields, param) in enumerate(invalid):
                payload = _payload(settings, "text", stream=streaming, **fields)
                raw = http_client.post(f"{settings.base_url}/responses", json=payload)
                _record(
                    settings,
                    f"raw-invalid-{streaming}-{index}",
                    {"status": raw.status_code, "headers": dict(raw.headers), "body": raw.text},
                )
                _assert_error(raw, 400, param)
                try:
                    sdk.responses.create(**_payload(settings, "text"), stream=streaming, extra_body=fields)
                except openai.BadRequestError as error:
                    _assert_error(error.response, 400, param)
                else:
                    raise AssertionError("Official SDK did not raise BadRequestError")
            missing = _payload(settings, "text", model="__responses_nonexistent_model__", stream=streaming)
            raw = http_client.post(f"{settings.base_url}/responses", json=missing)
            _assert_error(raw, 404, "model")
            _record(settings, f"raw-missing-model-{streaming}", {"status": raw.status_code, "body": raw.text})
            try:
                sdk.responses.create(**missing)
            except openai.NotFoundError as error:
                _assert_error(error.response, 404, "model")
            else:
                raise AssertionError("Official SDK did not raise NotFoundError")
            # Discoverable syntax errors stay JSON even for a requested stream.
            malformed = http_client.post(
                f"{settings.base_url}/responses",
                content='{"stream":true,',
                headers={"Content-Type": "application/json"},
            )
            _assert_error(malformed, 400)


def _wait_recovery(settings: _Settings, client: httpx.Client, label: str) -> None:
    deadline = time.monotonic() + 90
    attempts: list[dict[str, Any]] = []
    while True:
        response = client.post(f"{settings.base_url}/responses", json=_payload(settings, "text"))
        attempts.append({"status": response.status_code, "body": response.text})
        if response.status_code == 200:
            _record(settings, label, attempts)
            _assert_response(response.json())
            return
        _assert_error(response, 429)
        assert time.monotonic() < deadline, "The accepted request's slot was not recovered after disconnect"
        time.sleep(0.05)


def _check_limit_one(settings: _Settings) -> None:
    import openai

    assert settings.limit_one, "The owning runner must configure max_concurrent_requests=1"
    with _clients(settings) as (http_client, sdk):
        payload = _payload(settings, "hold", stream=True)
        with http_client.stream("POST", f"{settings.base_url}/responses", json=payload) as held:
            assert held.status_code == 200
            assert held.headers["content-type"].startswith("text/event-stream")
            for line in held.iter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    if event["type"].endswith(".delta"):
                        _record(settings, "held-first-delta", event)
                        break
            else:
                raise AssertionError("The held request ended without a first output delta")
            for streaming in (False, True):
                rejected = _payload(settings, "text", stream=streaming)
                raw = http_client.post(f"{settings.base_url}/responses", json=rejected)
                _record(
                    settings,
                    f"raw-limit-{streaming}",
                    {"status": raw.status_code, "headers": dict(raw.headers), "body": raw.text},
                )
                _assert_error(raw, 429)
                try:
                    sdk.responses.create(**rejected)
                except openai.RateLimitError as error:
                    _assert_error(error.response, 429)
                else:
                    raise AssertionError("Official SDK did not raise RateLimitError")
        _wait_recovery(settings, http_client, "disconnect-after-output-recovery")
    if settings.fixture:
        asyncio.run(_disconnect_before_output(settings))
        with _clients(settings) as (http_client, _sdk):
            _wait_recovery(settings, http_client, "disconnect-before-output-recovery")


async def _disconnect_before_output(settings: _Settings) -> None:
    import httpx

    async with httpx.AsyncClient(timeout=90, trust_env=False) as client:
        pending = asyncio.create_task(
            client.post(f"{settings.base_url}/responses", json=_payload(settings, "hold_before", stream=True))
        )
        attempts: list[dict[str, Any]] = []
        deadline = time.monotonic() + 30
        try:
            while True:
                assert not pending.done(), "Fixture must hold before the first response byte"
                pressure = await client.post(f"{settings.base_url}/responses", json=_payload(settings, "text"))
                attempts.append({"status": pressure.status_code, "body": pressure.text})
                if pressure.status_code == 429:
                    _assert_error(pressure, 429)
                    _record(settings, "before-output-admission-barrier", attempts)
                    break
                assert pressure.status_code == 200, pressure.text
                _assert_response(pressure.json())
                assert time.monotonic() < deadline, "The fixture did not acquire the held request's slot"
                await asyncio.sleep(0.01)
        finally:
            pending.cancel()
            with suppress(asyncio.CancelledError):
                await pending


def _check_fixture_outcomes(settings: _Settings) -> None:
    import openai

    assert settings.fixture
    with _clients(settings) as (http_client, sdk):
        for streaming in (False, True):
            reasoning = _post_raw(
                settings, http_client, f"raw-reasoning-{streaming}", _payload(settings, "reasoning", stream=streaming)
            )
            raw_trace = [item for item in reasoning["output"] if item["type"] == "reasoning"]
            assert len(raw_trace) == 1 and raw_trace[0]["summary"] == []
            assert raw_trace[0]["content"] == [{"type": "reasoning_text", "text": "why"}]
            assert _output_text(reasoning) == "answer"
            assert reasoning["usage"]["output_tokens_details"]["reasoning_tokens"] == 0
            utf8 = _post_raw(
                settings, http_client, f"raw-utf8-{streaming}", _payload(settings, "utf8", stream=streaming)
            )
            assert _output_text(utf8) == "你好 👋"
            failed_payload = _payload(settings, "failed", stream=streaming)
            _post_raw(settings, http_client, f"raw-failed-{streaming}", failed_payload, "failed")
            if streaming:
                with sdk.responses.create(**failed_payload) as stream:
                    events = [_dump(event) for event in stream]
                _record(settings, "sdk-failed-stream", events)
                _assert_events(events, "failed", sdk=True)
            else:
                failed = _dump(sdk.responses.create(**failed_payload))
                _record(settings, "sdk-failed", failed)
                _assert_response(failed, "failed", sdk=True)
            early_payload = _payload(settings, "preflight_failed", stream=streaming)
            raw = http_client.post(f"{settings.base_url}/responses", json=early_payload)
            _record(
                settings,
                f"raw-resource-exhausted-{streaming}",
                {"status": raw.status_code, "headers": dict(raw.headers), "body": raw.text},
            )
            _assert_error(raw, 500)
            try:
                sdk.responses.create(**early_payload)
            except openai.InternalServerError as error:
                _assert_error(error.response, 500)
            else:
                raise AssertionError("Execution resource exhaustion was not InternalServerError")
        for case in ("reasoning", "utf8"):
            with sdk.responses.stream(**_payload(settings, case)) as stream:
                events = [_dump(event) for event in stream]
                final = _dump(stream.get_final_response())
            _record(settings, f"sdk-{case}-accumulator", {"events": events, "final": final})
            _assert_accumulator(events, final)
            assert _output_text(final) == ("answer" if case == "reasoning" else "你好 👋")


def test_openai_responses_text_and_history() -> None:
    _check_text(_environment_settings())


def test_openai_responses_json_object() -> None:
    _check_json_object(_environment_settings())


def test_openai_responses_raw_and_sdk_stream() -> None:
    _check_stream(_environment_settings())


def test_openai_responses_function_tools_and_history() -> None:
    _check_functions(_environment_settings())


def test_openai_responses_http_and_sdk_errors() -> None:
    _check_errors(_environment_settings())


def test_openai_responses_admission_and_disconnect() -> None:
    settings = _environment_settings()
    if not settings.limit_one:
        import pytest

        pytest.skip("The external runner must select XLLM_RESPONSES_LIMIT_ONE=1 and configure limit one")
    _check_limit_one(settings)


def test_openai_responses_fixture_reasoning_utf8_and_failures() -> None:
    settings = _environment_settings()
    if not settings.fixture:
        import pytest

        pytest.skip("Deterministic parser/failure cases require XLLM_RESPONSES_FIXTURE=1")
    _check_fixture_outcomes(settings)


def _main() -> None:
    import openai

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="OpenAI base URL, including /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", action="store_true", help="Use deterministic brpc fixture selectors")
    parser.add_argument("--limit-one", action="store_true", help="Require limit-one admission/disconnect checks")
    parser.add_argument("--expect-reasoning", action="store_true", help="Require genuine raw reasoning text")
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args()
    if args.max_output_tokens < 16:
        parser.error("--max-output-tokens must be at least 16")
    settings = _Settings(
        base_url=args.base_url.rstrip("/"),
        model=args.model,
        fixture=args.fixture,
        limit_one=args.limit_one,
        expect_reasoning=args.expect_reasoning,
        max_output_tokens=args.max_output_tokens,
        evidence_dir=args.evidence_dir,
    )
    checks = [_check_text, _check_json_object, _check_stream, _check_functions, _check_errors]
    if settings.fixture:
        checks.append(_check_fixture_outcomes)
    if settings.limit_one:
        checks.append(_check_limit_one)
    completed: list[str] = []
    for check in checks:
        check(settings)
        completed.append(check.__name__)
    report = {
        "verdict": "PASS",
        "evidence_kind": "sdk-fixture" if settings.fixture else "sdk-model",
        "openai_version": openai.__version__,
        "checks": completed,
    }
    _record(settings, "result", report)
    print(json.dumps(report))


if __name__ == "__main__":
    _main()
