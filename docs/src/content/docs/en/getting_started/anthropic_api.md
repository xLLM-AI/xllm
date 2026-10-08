---
title: "Anthropic-Compatible API"
---

xLLM supports the Anthropic Messages request format for locally served LLMs. Use cURL, the Anthropic Python SDK, or a client configured for an Anthropic-compatible server.

| Endpoint | Method | Usage |
| --- | --- | --- |
| `/v1/messages` | POST | Generate messages, optionally with streaming, reasoning, and tool use. |
| `/v1/messages/count_tokens` | POST | Count input tokens without generating an answer. |

These routes are registered with the LLM service. They use the same loaded model, chat template, and reasoning/tool parsers as the [OpenAI-compatible APIs](/en/getting_started/openai_api/). No separate API-enabling flag is needed.

## Launch a server

Start an LLM using the [launch guide](/en/getting_started/launch_xllm/). You can reuse the single-NPU Qwen3-8B deployment in the [OpenAI API guide](/en/getting_started/openai_api/#launch-a-server), including `--reasoning_parser=qwen3` and `--tool_call_parser=qwen3`. Select the matching parsers when serving another model.

After model loading, confirm its ID:

```bash
export HOST=127.0.0.1
export PORT=18000
export MODEL=Qwen3-8B

curl -sS "http://${HOST}:${PORT}/v1/models"
```

xLLM validates `model`. Use an ID returned by `/v1/models`, such as the value set with `--model_id`. An arbitrary Claude model name is not an alias for your local model.

## Send a message

### Using cURL

Provide a positive `max_tokens`, a model ID, and a non-empty `messages` list. Put the system prompt in the top-level `system` field. The following Qwen3 example disables thinking for a short answer:

```bash
curl -sS "http://${HOST}:${PORT}/v1/messages" \
  -H 'Content-Type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -H 'x-api-key: EMPTY' \
  -d "{
    \"model\": \"${MODEL}\",
    \"max_tokens\": 128,
    \"system\": \"Give concise answers.\",
    \"messages\": [{\"role\": \"user\", \"content\": \"What is an inference engine?\"}],
    \"temperature\": 0,
    \"chat_template_kwargs\": {\"enable_thinking\": false}
  }"
```

### Using the Anthropic Python SDK

```bash
python -m pip install --upgrade anthropic
```

Use the server root as `base_url`, without `/v1`: the SDK adds `/v1/messages`. Direct xLLM access does not require an Anthropic service key, so use `EMPTY` for the SDK. An authenticated gateway may require its own credentials. Run this setup before the later Python examples:

```python
import os
from anthropic import Anthropic

host = os.environ.get("HOST", "127.0.0.1")
port = os.environ.get("PORT", "18000")
model = os.environ.get("MODEL", "Qwen3-8B")
client = Anthropic(base_url=f"http://{host}:{port}", api_key="EMPTY")

message = client.messages.create(
    model=model,
    max_tokens=128,
    system="Give concise answers.",
    messages=[{"role": "user", "content": "What is an inference engine?"}],
    extra_body={
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
for block in message.content:
    if block.type == "text":
        print(block.text)
print(message.stop_reason, message.usage)
```

Responses contain a `content` array, which may include `text`, `thinking`, or `tool_use` blocks. Select by `type` instead of assuming the first block contains text. Usage is reported as `input_tokens` and `output_tokens`.

Pass sampling fields such as `temperature`, `top_p`, and `top_k` through `extra_body` if your SDK version does not expose them as named arguments. In raw HTTP JSON, these fields belong at the top level.

## System prompts and conversation history

`system` accepts a string or a list of text blocks. `messages` contains the user/assistant history, with string content or content-block arrays. Send the history on each request; the endpoint does not store your conversation for you.

```python
message = client.messages.create(
    model=model,
    max_tokens=128,
    system=[{"type": "text", "text": "Explain technical ideas simply."}],
    messages=[
        {"role": "user", "content": "What does batching mean?"},
        {"role": "assistant", "content": "Processing several requests together."},
        {"role": "user", "content": "Why can it improve throughput?"},
    ],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
for block in message.content:
    if block.type == "text":
        print(block.text)
```

## Streaming

The SDK's streaming helper yields text and can assemble the final message, including its usage and stop reason:

```python
with client.messages.stream(
    model=model,
    max_tokens=128,
    messages=[{"role": "user", "content": "Explain batching in two sentences."}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
    final_message = stream.get_final_message()
print("\n", final_message.stop_reason, final_message.usage)
```

For raw HTTP, set `"stream": true` and use `curl -N`. The response uses Anthropic SSE events: `message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, and `message_stop`. Text deltas use `text_delta`; tool arguments use `input_json_delta`. A stream finishes with `message_stop`, rather than OpenAI's `[DONE]` marker.

## Tool use

Configure the model's tool parser on the server, then provide Anthropic-style tool definitions with `name`, `description`, and `input_schema`. xLLM returns `tool_use` blocks; your application runs the tools.

```python
messages = [{"role": "user", "content": "What is the weather in Shanghai?"}]
tools = [{
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}]
message = client.messages.create(
    model=model,
    max_tokens=256,
    messages=messages,
    tools=tools,
    tool_choice={"type": "auto"},
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
tool_uses = [block for block in message.content if block.type == "tool_use"]
for block in tool_uses:
    print(block.id, block.name, block.input)
```

To send results back, append the assistant's content and a user message containing `tool_result` blocks. Each `tool_use_id` must match its original call. The next example uses an explicitly simulated result; replace it with your application's tool output.

```python
if tool_uses:
    messages.append({
        "role": "assistant",
        "content": [block.model_dump(exclude_none=True) for block in message.content],
    })
    messages.append({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": "Simulated result for this example: sunny, 20 degrees Celsius.",
        } for block in tool_uses],
    })
    answer = client.messages.create(
        model=model,
        max_tokens=256,
        messages=messages,
        tools=tools,
        tool_choice={"type": "none"},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    for block in answer.content:
        if block.type == "text":
            print(block.text)
```

## Thinking output

For the Qwen3 example, keep `--reasoning_parser=qwen3` enabled and pass `chat_template_kwargs.enable_thinking` through `extra_body`. The HTTP JSON field is `chat_template_kwargs` at the top level. Allow enough tokens for reasoning and the final answer.

```python
message = client.messages.create(
    model=model,
    max_tokens=4096,
    messages=[{"role": "user", "content": "What is 2 + 2? Give a brief answer."}],
    extra_body={
        "temperature": 0.6,
        "chat_template_kwargs": {"enable_thinking": True},
    },
)
for block in message.content:
    if block.type == "thinking":
        print("Thinking:", block.thinking)
    elif block.type == "text":
        print("Answer:", block.text)
```

Template options depend on the model. The Anthropic `thinking.budget_tokens` control is not implemented as a reasoning-token budget in xLLM; use the model's template options and the total `max_tokens` limit.

## Count input tokens

The count endpoint renders the chat template and tokenizes the input without generation. Include the same `system`, `messages`, `tools`, and template options as the generation request so the count describes that prompt. `max_tokens` is not required.

```python
count = client.messages.count_tokens(
    model=model,
    system="Give concise answers.",
    messages=[{"role": "user", "content": "What is an inference engine?"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(count.input_tokens)
```

## Connect Claude Code

After installing Claude Code and starting a tool-capable xLLM service, set the server root and map its model aliases to your served model ID. These variables are documented in the [Claude Code environment reference](https://code.claude.com/docs/en/env-vars).

```bash
export HOST=127.0.0.1
export PORT=18000
export MODEL=Qwen3-8B
export ANTHROPIC_BASE_URL="http://${HOST}:${PORT}"
export ANTHROPIC_AUTH_TOKEN=EMPTY
export ANTHROPIC_DEFAULT_HAIKU_MODEL="${MODEL}"
export ANTHROPIC_DEFAULT_SONNET_MODEL="${MODEL}"
export ANTHROPIC_DEFAULT_OPUS_MODEL="${MODEL}"
export CLAUDE_CODE_ATTRIBUTION_HEADER=0

claude --model "${MODEL}"
```

`CLAUDE_CODE_ATTRIBUTION_HEADER=0` omits the client attribution block; the effect on prefix reuse depends on the client version and server cache configuration. Do not add a `[1m]` suffix unless that exact ID is served: xLLM checks model names, and a client-side suffix does not enlarge the model's context window. Client features that send unsupported content or constrained-output requests remain subject to the limits below. This connection recipe is not an end-to-end validation of every Claude Code workflow.

## Parameters and compatibility limits

| Parameter | Usage |
| --- | --- |
| `model` | An available model ID; required for generation and token counting. |
| `max_tokens` | Required positive output limit for generation. |
| `system` | System prompt string or text-block list. |
| `temperature`, `top_p`, `top_k` | Sampling controls; use non-negative temperature, `top_p` in `(0, 1]`, and `top_k=0` to disable top-k filtering. |
| `stop_sequences` | List of stop strings. |
| `stream` | Enable Anthropic SSE output. |
| `tools`, `tool_choice` | Function definitions and choice (`auto`, `none`, `any`, or `tool` with a name); output quality depends on the model/parser. |
| `chat_template_kwargs` | xLLM extension for model-specific template options. |

- The Messages API currently serves LLMs. Image inputs and image/tool-reference content in tool results are rejected. For image chat, use the [OpenAI vision API](/en/getting_started/openai_api_vision/).
- Non-empty `output_config.format.schema` constraints and `kv_transfer_params` are unsupported and return HTTP 400. Accepted tool schemas do not imply constrained JSON generation.
- `stop_reason="end_turn"` means generation ended normally, `max_tokens` indicates a length limit, and `tool_use` indicates a tool call.
- For a model-related HTTP 404, check the model ID with `/v1/models`. For HTTP 501, check that this is an LLM deployment. For request failures, inspect `error.message`; malformed requests and unsupported backend options may have different error envelopes.
