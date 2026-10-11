---
title: "OpenAI-Compatible APIs"
---

xLLM exposes OpenAI-compatible HTTP endpoints for local models. You can send requests with cURL or the OpenAI Python client by pointing `base_url` to your xLLM server. This guide covers text generation; see [Vision](/en/getting_started/openai_api_vision/) and [Embeddings](/en/getting_started/openai_api_embeddings/) for other model tasks.

| Endpoint | Method | Usage |
| --- | --- | --- |
| `/v1/models` | GET | List available model IDs. |
| `/v1/chat/completions` | POST | Generate a reply from messages; supports streaming and image inputs with a VLM. |
| `/v1/completions` | POST | Continue a text prompt or token IDs; supports streaming and batched prompts. |
| `/v1/embeddings` | POST | Encode text or token IDs with an embedding service. |

Compatibility applies to the endpoints and options documented here. `/v1/responses` is not implemented.

## Launch a server

Install xLLM and prepare the model weights following the [Quick Start](/en/getting_started/quick_start/). The following example starts Qwen3-8B on one Ascend NPU using the Python/PyTorch model implementation and ACL Graph for decoding. Replace the model path and choose a model that fits the device memory. For other hardware or multiple devices, use the corresponding [launch instructions](/en/getting_started/launch_xllm/).

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export HCCL_IF_BASE_PORT=43432

xllm --model=/path/to/Qwen3-8B \
  --model_id=Qwen3-8B \
  --model_impl=python \
  --npu_kernel_backend=TORCH \
  --enable_graph=true \
  --python_graph_backend=aclgraph \
  --host=127.0.0.1 \
  --port=18000 \
  --master_node_addr=127.0.0.1:9748 \
  --nnodes=1 \
  --node_rank=0 \
  --block_size=128 \
  --max_memory_utilization=0.86 \
  --enable_prefix_cache=false \
  --enable_chunked_prefill=true \
  --enable_schedule_overlap=true \
  --enable_shm=false \
  --reasoning_parser=qwen3 \
  --tool_call_parser=qwen3
```

Wait for model loading to finish, then run the client examples in another terminal. The example binds to loopback. For remote clients, set `--host` to a reachable server address and use that address as `HOST` below.

`--model` is the weights path; `--model_id` is the name used in requests. If `--model_id` is omitted, xLLM uses the final directory name of the model path. Query `/v1/models` to confirm the accepted ID:

```bash
export HOST=127.0.0.1
export PORT=18000
export MODEL=Qwen3-8B

curl -sS "http://${HOST}:${PORT}/v1/models"
```

## Set up the Python client

```bash
python -m pip install --upgrade openai
```

Run this setup before the Python examples on this page. The direct xLLM endpoint does not require an OpenAI service key; `EMPTY` satisfies the SDK's key requirement. If you access xLLM through an authenticated gateway, use its credentials.

```python
import os
from openai import OpenAI

host = os.environ.get("HOST", "127.0.0.1")
port = os.environ.get("PORT", "18000")
model = os.environ.get("MODEL", "Qwen3-8B")
client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key="EMPTY")

print([item.id for item in client.models.list().data])
```

## Chat completions

Send conversation history in `messages`. xLLM renders it with the model's chat template; you do not need to insert chat delimiters yourself. The Qwen3 examples disable thinking to keep these short replies within the output budget. Template options are model-specific.

### Using cURL

```bash
curl -sS "http://${HOST}:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"Give concise answers.\"},
      {\"role\": \"user\", \"content\": \"What is an inference engine?\"}
    ],
    \"temperature\": 0,
    \"max_tokens\": 128,
    \"chat_template_kwargs\": {\"enable_thinking\": false}
  }"
```

### Using the OpenAI Python client

```python
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "Give concise answers."},
        {"role": "user", "content": "What is an inference engine?"},
    ],
    temperature=0,
    max_tokens=128,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(response.choices[0].message.content)
print(response.usage)
```

For multiple turns, append the assistant reply and the next user message to `messages`, then send the history again. Read generated text from `choices[].message.content`; token counts are in `usage.prompt_tokens`, `usage.completion_tokens`, and `usage.total_tokens`.

### Streaming

Set `stream=True` to receive incremental replies. `stream_options={"include_usage": True}` adds a final usage chunk whose `choices` array is empty, so check it before indexing. These options require `stream=True`.

```python
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Explain batching in two sentences."}],
    max_tokens=128,
    temperature=0,
    stream=True,
    stream_options={"include_usage": True},
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
for chunk in stream:
    if chunk.usage is not None:
        print("\nUsage:", chunk.usage)
    if not chunk.choices:
        continue
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
print()
```

With cURL, add `-N` to disable output buffering and `"stream": true` to the JSON body. The HTTP response uses Server-Sent Events (SSE): each event starts with `data:`, and a completed stream ends with `data: [DONE]`.

### Thinking and reasoning

The Qwen3 launch command above enables `--reasoning_parser=qwen3`. Set `enable_thinking` in `chat_template_kwargs` to control Qwen3's thinking mode, and allow enough output tokens for both reasoning and the answer. Other model families require their own template options and parser; see the [CLI reference](/en/cli_reference/).

On xLLM's HTTP API, parsed reasoning is returned as `message.reasoning` (or `delta.reasoning` while streaming). Access this extension with `getattr` when using the SDK:

```python
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "What is 2 + 2? Give a brief answer."}],
    max_tokens=4096,
    temperature=0.6,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)
message = response.choices[0].message
print("Reasoning:", getattr(message, "reasoning", None))
print("Answer:", message.content)
```

### Tool calling

Use a model with a tool-capable chat template and a matching `--tool_call_parser`, such as the Qwen3 setup above. Define function schemas in `tools`. With `tool_choice="auto"`, the model can return a normal answer or `message.tool_calls`.

```python
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "What is the weather in Shanghai?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }],
    tool_choice="auto",
    max_tokens=256,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
message = response.choices[0].message
for tool_call in message.tool_calls or []:
    print(tool_call.id, tool_call.function.name, tool_call.function.arguments)
if message.content:
    print(message.content)
```

xLLM returns the requested function name and JSON argument string; your application executes the function. To continue the conversation, append the assistant message containing `tool_calls`, then a `role="tool"` message with the matching `tool_call_id` and the result in `content`, and call chat completions again.

## Completions

The completions endpoint accepts `prompt` and returns `choices[].text`. It does not apply a chat template. Use it for raw continuation, or supply a prompt already formatted for your model.

```python
response = client.completions.create(
    model=model,
    prompt="An inference engine is",
    temperature=0,
    max_tokens=64,
)
print(response.choices[0].text)
```

For streaming, add `stream=True` and read `chunk.choices[0].text`. If you also request usage, handle empty `choices` as in the chat example.

`prompt` can be a string, a list of strings, a list of token IDs, or a list of token-ID lists. Batched results are ordered by prompt and then by choice; `usage` covers the entire request. Token IDs must come from the served model's tokenizer.

```python
response = client.completions.create(
    model=model,
    prompt=["The capital of France is", "The capital of Japan is"],
    temperature=0,
    max_tokens=16,
    n=1,
)
for choice in response.choices:
    print(choice.index, choice.text)
```

## Common parameters and extensions

| Parameter | xLLM behavior |
| --- | --- |
| `max_tokens` | Positive output-token limit. Completions and Python `SamplingParams` default to 16. Omitted chat limits and Python `max_tokens=None` use the remaining model context. |
| `max_completion_tokens` | Chat-only alias; takes precedence over `max_tokens` when both are present. |
| `temperature` | Finite, non-negative value, default `1.0`; values above `2` are accepted, matching vLLM `0.23.0`. Positive values below `0.01` are raised to `0.01`; `0` selects greedy decoding and disables `top_k`, `top_p`, and `min_p`. |
| `top_p` | Sampling cutoff in `(0, 1]`. |
| `n` | Number of returned candidates; must be `1` with `temperature=0`. |
| `stop` | A string or list of non-empty strings. |
| `presence_penalty`, `frequency_penalty` | Values in `[-2, 2]`. |
| `logprobs` | Boolean for chat; integer count for completions and Python. Chat also uses `top_logprobs`. Counts of `0` return the sampled token, and `-1` returns the full vocabulary. Values are computed before penalties, biases, temperature, and filtering. |
| `top_k` | Extension: integer cutoff; `-1` or `0` disables the cutoff. |
| `repetition_penalty` | Extension: positive repetition penalty. |
| `min_p` | Relative probability cutoff in `[0, 1]`, default `0`; applied after temperature and before top-k/top-p. |
| `seed` | Optional signed 64-bit seed; `-1` means unset. Seeded streams are independent of batch position. Identical tokens across different engines or hardware are not guaranteed. |
| `min_tokens` | Non-negative minimum output length, at most `max_tokens`. Blocks EOS and stop-token IDs until the minimum is reached. |
| `logit_bias` | Token-ID-to-bias mapping; values are clipped to `[-100, 100]`. |
| `allowed_token_ids` | Non-empty list of allowed token IDs. |
| `bad_words` | List of non-empty strings; prevents completing their token sequences in generated text. |
| `chat_template_kwargs` | Chat extension: template options supported by the model. |

Pass extensions through the SDK's `extra_body`. In raw HTTP JSON, place them at the top level, without an `extra_body` wrapper:

```python
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Suggest a name for a coding assistant."}],
    temperature=0.7,
    top_p=0.9,
    max_tokens=64,
    extra_body={
        "top_k": 20,
        "repetition_penalty": 1.05,
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
print(response.choices[0].message.content)
```

For xLLM's `beam_width` extension, see [Online Service](/en/getting_started/online_service/). Leave `best_of` unset for streaming examples; xLLM streams only when `best_of` equals `n`.

## Compatibility limits and errors

- `seed`, `min_tokens`, `logit_bias`, `allowed_token_ids`, `bad_words`, and full-vocabulary logprobs require `enable_task_pipeline=false`. `seed`, `min_tokens`, and `bad_words` also require `num_speculative_tokens=0`; `bad_words` requires `enable_schedule_overlap=false`. REC endpoints reject the new seed and token-constraint controls.
- Non-null `prompt_logprobs`, `logprob_token_ids`, `structured_outputs`, and `prompt_embeds` remain unsupported and return HTTP 400. This interface does not implement all engine-level fields of vLLM `0.23.0` `SamplingParams`.
- Frequency and presence penalties count generated tokens only; repetition penalties include prompt tokens. Explicit `stop_token_ids` remain active when `ignore_eos=true`.
- `response_format={"type": "json_schema"}` is unsupported. Chat `json_object` requires the server's JSON-output configuration; it is not JSON Schema support.
- Embedding requests do not support `dimensions`; see the [Embedding guide](/en/getting_started/openai_api_embeddings/).
- HTTP 400 indicates an invalid or unsupported request. Inspect the JSON `error.message` and `error.param`. HTTP 404 for a model means its name does not match an available model; query `/v1/models`.
- A successful generation with `finish_reason="length"` reached the output or context limit. Increase the output budget or shorten the input as appropriate.
