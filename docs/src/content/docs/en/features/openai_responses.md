---
title: "OpenAI Responses API"
sidebar:
  order: 90
---
<!-- Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0 -->

xLLM exposes **`POST /v1/responses`** through its native API service. This endpoint uses Responses objects and named SSE events, not Chat Completions `choices` or `chat.completion.chunk`. It shares the existing model selection, templates, request factory, scheduler, and admission limiter; it does not call `/v1/chat/completions` internally.

Start a model using the [launch guide](/en/getting_started/launch_xllm/). Replace `loaded-model` below with the served model name. The singular `/v1/response` endpoint is not provided.

## Text and client-carried history

```bash
curl http://127.0.0.1:9977/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "loaded-model",
    "input": "Explain prefix caching in one sentence.",
    "instructions": "Be concise.",
    "max_output_tokens": 128,
    "temperature": 0,
    "store": false
  }'
```

With the official OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:9977/v1", api_key="EMPTY")
response = client.responses.create(
    model="loaded-model",
    input="Explain prefix caching in one sentence.",
    instructions="Be concise.",
    max_output_tokens=128,
    store=False,
)
print(response.status)
print(response.output_text)

# Carry the conversation explicitly: xLLM does not retain response objects.
history = [{"role": "user", "content": "Explain prefix caching in one sentence."}]
history.extend(item.model_dump(exclude_unset=True) for item in response.output)
history.append({"role": "user", "content": "When is it useful?"})
next_response = client.responses.create(
    model="loaded-model",
    instructions="Be concise.",
    input=history,
    max_output_tokens=128,
    store=False,
)
```

`input` is required for this stateless endpoint. It accepts a string or a nonempty array of textual messages and supported history items. Messages accept `user`, `assistant`, `system`, and `developer` roles; developer messages use the system-instruction priority of the existing template. Message content may be a string or an array of `input_text` parts. Assistant history may contain `output_text` parts with empty annotations. Text parts are concatenated in order. `instructions`, when supplied, is prepended as a system message for the current request; carry it again when needed on later requests.

**Storage profile:** `store=false` is supported. Omitted or null `store` also means **false** in xLLM, unlike the hosted API's storage default. `store=true`, non-null `previous_response_id`, and `background=true` are rejected. IDs identify the response and its output items, but are not storage handles. Retrieval, deletion, background execution, and server-side conversation continuation are not provided.

## Streaming

```python
with client.responses.create(
    model="loaded-model",
    input="Say hello briefly.",
    max_output_tokens=128,
    store=False,
    stream=True,
) as stream:
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
        elif event.type in {"response.completed", "response.incomplete", "response.failed"}:
            print("\n", event.response.status)
```

The SDK's completed-response accumulator can also be used:

```python
with client.responses.stream(
    model="loaded-model",
    input="Say hello briefly.",
    max_output_tokens=128,
    store=False,
) as stream:
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
    response = stream.get_final_response()
```

Check the installed SDK's behavior for non-completed terminals. For example, OpenAI Python 2.54.0's accumulator only makes `get_final_response()` available after `response.completed`; use `responses.create(stream=True)` and inspect the terminal event to handle incomplete or failed generations without that assumption.

Streaming uses `Content-Type: text/event-stream`. Each `event:` name equals the JSON `type`; `sequence_number` starts at zero and increases by one. The lifecycle is:

1. `response.created`, then `response.in_progress`.
2. `response.output_item.added` with a stable item ID and contiguous `output_index`.
3. For text, `response.content_part.added`, then `response.output_text.delta`; for function calls, `response.function_call_arguments.delta`. Raw reasoning uses `response.reasoning_text.delta` on a reasoning item, not a reasoning summary.
4. On successful or length-limited generation, matching text/argument done events, content-part done events where applicable, and `response.output_item.done`.
5. Exactly one terminal event containing the full final Response: `response.completed`, `response.incomplete`, or `response.failed`.

There is no Chat-style `[DONE]` sentinel. A disconnected client cannot receive a terminal event and must not interpret EOF as completion. Client cancellation releases request ownership through the normal request lifecycle; xLLM does not invent a `response.cancelled` SSE event or expose a cancellation endpoint for stored responses.

## Client-side function tools

Function tools use the **flat Responses form**, not Chat's nested `function` object. They require a supported tool parser configured for the selected model. This Responses profile supports `qwen25`, `qwen3_coder`, `glm45`, `glm47`, and `glm5`, including aliases resolving to those formats; other formats are rejected before generation. Responses reuses the model tool parsers and their value conversions. Supported streaming parsers consume available names, arguments, and complete calls in the same increment, then use an explicit end-of-stream operation to release pending output. Recognized unfinished tool data fails a normally completed response; length-limited or failed generation retains partial output without inventing missing values. Complete semantic arguments do not require every outer wrapper to be present at EOF. The API validates returned call names, indexes, and completed argument JSON objects. This is not full model-syntax or strict end-tag validation, and does not guarantee arbitrary malformed input or all-model chunk-boundary consistency. These checks are distinct from strict JSON Schema enforcement, which is unsupported. xLLM emits calls; **the client supplies results**, and no function is executed by the server.

```python
import json

question = "Use weather to get the current weather in Paris."
tools = [{
    "type": "function",
    "name": "weather",
    "description": "Get the current weather in a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
    "strict": False,
}]
response = client.responses.create(
    model="loaded-model",
    input=question,
    tools=tools,
    tool_choice="auto",
    max_output_tokens=512,
    store=False,
)
history = [{"role": "user", "content": question}]
history.extend(item.model_dump(exclude_unset=True) for item in response.output)
for item in response.output:
    if item.type == "function_call":
        arguments = json.loads(item.arguments)
        # Obtain the result in your application; this example supplies a value.
        result = {"city": arguments["city"], "condition": "sunny"}
        history.append({
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": json.dumps(result),
        })
answer = client.responses.create(
    model="loaded-model",
    input=history,
    tools=tools,
    tool_choice="none",
    max_output_tokens=512,
    store=False,
)
print(answer.output_text)
```

`auto` allows the model to choose text or one or more calls; it does not force a call. `none` disables new calls while preserving replayed history. Every historical call must have a matching `function_call_output`, identified by its unique `call_id`. History arguments must encode a JSON object; results may be text or `input_text` parts. Out-of-order results after the corresponding calls are allowed. Malformed client-supplied history arguments and unmatched or duplicated results are request errors, not repaired values. Generated calls are subject to the existing model parser's behavior; the API checks the parsed result rather than reconstructing discarded model syntax.

This profile uses non-strict function schemas: omitted or null `strict` normalizes to false, and `strict=true` is rejected. `tool_choice="required"`, named forced choices, and `parallel_tool_calls=false` are rejected because the current backend cannot enforce them. Omitted, null, or true `parallel_tool_calls` allows multiple calls. Built-in tools such as web search, code interpreter, file search, computer use, and MCP are unsupported.

## Reasoning, structured output, and usage

With a configured, supported reasoning parser, the endpoint preserves the model's **raw reasoning** in `reasoning` output items with `summary=[]` and `content` containing `reasoning_text`. The same items can be replayed before the corresponding assistant message or function call. Raw traces are not summaries and are not represented as fabricated encrypted content. The endpoint rejects non-null `reasoning.effort` and `reasoning.summary`: the existing engine cannot enforce effort levels or generate summaries. Empty reasoning controls do not enable a different model or parser.

`text.format={"type":"text"}` is the default. `text.format={"type":"json_object"}` uses the existing constrained decoder when `enable_json_object_output` is enabled for the service. It cannot be combined with active function tools. JSON Schema output, strict schemas, and format controls that the engine cannot enforce are rejected rather than passed through as promises.

Final usage contains:

- `input_tokens`, `output_tokens`, and `total_tokens` from generation accounting.
- `input_tokens_details.cached_tokens`: the high-water count of actual prompt positions reused from the prefix cache.
- `input_tokens_details.cache_write_tokens`: always `0`, matching SGLang's Responses serialization. This profile does not separately report cache-write usage. The value is not an actual write count and does not mean that no internal cache writes occurred.
- `output_tokens_details.reasoning_tokens`: always integer `0` for compatibility. This profile does not separately report reasoning-token usage. Zero does not mean that the model performed no reasoning or emitted no raw reasoning text; it does not alter actual output/total usage or the generation budget.

Cache-hit input is reported by `cached_tokens`; uncached input can be derived as `input_tokens - cached_tokens`. Internal cache publication is not a separate reported usage category. The fixed cache-write value does not change caching behavior. Responses still rejects disaggregated/distributed serving paths, external KV storage, and host offload. Their text-state propagation or cache-hit accounting is outside the supported profile; omitting separate reasoning-token accounting does not expand those capabilities.

## Supported request profile

| Field | Behavior |
|---|---|
| `model` | Omitted selects the loaded default model; an explicit value must be a nonempty string naming a loaded LLM. |
| `input` | Required string or nonempty array of supported textual/history items. |
| `instructions` | Text or null; prepended to the current request only. |
| `stream` | Omitted/null/false returns JSON; true returns named SSE. |
| `max_output_tokens` | Integer at least 16; maps to the existing `max_tokens` generation budget, including reasoning. Omitted/null uses the engine default of 5120. Existing model/context limits still apply. |
| `temperature`, `top_p` | Finite numbers in `[0,2]` and `[0,1]`; omitted/null defaults to 1. |
| `tools`, `tool_choice` | Client function tools only, with `auto` or `none`; tool-parser capability checked before admission. |
| `parallel_tool_calls` | Omitted/null/true; false is unsupported. |
| `text.format` | `text`, or service-enabled `json_object`; no `json_schema`. |
| `reasoning` | Omitted/null/empty controls; non-null effort/summary unsupported. Raw output depends on the configured parser. |
| `metadata` | Up to 16 string pairs; keys up to 64 and values up to 512 bytes in this profile. Omitted/null becomes `{}`. |
| `store`, `background` | Omitted/null/false only. |
| `previous_response_id` | Omitted/null only; replay history explicitly. |
| `truncation` | Omitted/null/`disabled` only; no automatic history truncation. |
| `include` | Omitted/null/empty list only. |

Unknown fields, Chat-only parameters such as `messages`, `n`, and `max_tokens`, media inputs, hosted annotations, item references, assistant phase controls, and unsupported item/tool types produce an attributed HTTP 400. Unsupported controls are not silently ignored.

## Statuses and HTTP errors

| Situation | Result |
|---|---|
| Normal EOS/stop or completed function call | Response `status="completed"`. |
| Generation reaches its output-token limit | `status="incomplete"`, `incomplete_details.reason="max_output_tokens"`. |
| Generation fails after output starts | Response `status="failed"` and a non-null error; streaming ends with `response.failed`, not completed. |
| Client disconnect/cancellation | Request is cancelled and its admitted slot released; no fabricated successful result. |
| Invalid or unsupported request/capability | HTTP 400 JSON before SSE commitment. |
| Missing model | HTTP 404 JSON. |
| Concurrent-request admission rejected | HTTP 429 JSON in both stream modes; the official SDK raises `RateLimitError`. |
| Queue/execution resource exhaustion before output | Server error, not concurrency HTTP 429. |

Pre-output failures use the shared OpenAI error envelope with `message`, `type`, `param`, and `code`. A stream request that is rejected still receives `application/json`, not HTTP 200 or an SSE error substitute. xLLM does not invent `Retry-After`. Only accepted requests own a limiter slot, and request lifecycle cleanup releases that slot exactly once.

## Validation entry points

Dedicated native targets cover request conversion, output/SSE serialization, and real brpc transport: `openai_responses_request_test`, `openai_responses_output_test`, and `openai_responses_protocol_test`. Its official SDK case is excluded from ordinary runs with the standard `DISABLED_` prefix and shares the `XLLM_ENABLE_SDK_TESTS` CMake option (default `OFF`) with the other protocols. Explicitly run `OpenAIResponsesProtocolTest.DISABLED_OfficialSdkCompatibility` with `--gtest_also_run_disabled_tests`; it uses CMake's selected `Python3_EXECUTABLE`, which must already have `openai` and `httpx` installed. Missing packages fail explicit validation; the test does not install them or switch interpreters.

The Python SDK CTest target is also disabled in ordinary runs. Run pytest explicitly from the checkout root inside the required container, using the normal native initialization in `tests/python/conftest.py`:

```bash
XLLM_RESPONSES_BASE_URL=http://127.0.0.1:9977/v1 \
XLLM_RESPONSES_MODEL=loaded-model \
python -m pytest tests/python/test_openai_responses_protocol.py
```

The same module provides an HTTP-only client for an existing externally owned service:

```bash
python tests/python/test_openai_responses_protocol.py \
  --base-url http://127.0.0.1:9977/v1 --model loaded-model \
  --expect-reasoning --evidence-dir /path/to/fresh-results
```

For ordinary model validation, the owning runner configures `max_concurrent_requests=16` and the client sends requests serially, without `--fixture` or `--limit-one`. `--expect-reasoning` additionally requires genuine raw reasoning text. Admission/disconnect checks are separate: `--limit-one` requires the owning runner to configure `max_concurrent_requests=1`; the client does not change service configuration or start another runner. `--fixture` selects deterministic brpc fixture prompts, including reasoning, split UTF-8, and failures. It does not enable admission/disconnect checks; those require the separate explicit `--limit-one` opt-in. Without that flag, requests go to the real model and function-call behavior must actually occur. Fixture and model evidence are recorded separately. These commands describe validation, not a claim that a particular model or environment has passed.
