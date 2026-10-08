---
title: "Anthropic 兼容接口"
---

xLLM 支持通过 Anthropic Messages 请求格式访问本地部署的 LLM，可以使用 cURL、Anthropic Python SDK 或可配置 Anthropic 兼容服务地址的客户端。

| 接口 | 方法 | 用途 |
| --- | --- | --- |
| `/v1/messages` | POST | 生成消息，可使用流式输出、推理输出和工具调用。 |
| `/v1/messages/count_tokens` | POST | 统计输入 token，不生成回答。 |

这些路由随 LLM 服务注册，与 [OpenAI 兼容接口](/zh/getting_started/openai_api/)共用已加载的模型、聊天模板和推理/工具解析器，无需额外的接口启用开关。

## 启动服务

按照[启动说明](/zh/getting_started/launch_xllm/)部署 LLM。可以直接复用 [OpenAI 接口文档](/zh/getting_started/openai_api/#启动服务)中的单 NPU Qwen3-8B 服务，其中已配置 `--reasoning_parser=qwen3` 和 `--tool_call_parser=qwen3`；使用其他模型时，请选择匹配的解析器。

模型加载完成后，先确认模型 ID：

```bash
export HOST=127.0.0.1
export PORT=18000
export MODEL=Qwen3-8B

curl -sS "http://${HOST}:${PORT}/v1/models"
```

xLLM 会校验 `model`。请使用 `/v1/models` 返回的 ID，例如启动时的 `--model_id`。任意 Claude 模型名不会自动映射到本地模型。

## 发送消息

### 使用 cURL

请求需要提供正整数 `max_tokens`、模型 ID 和非空的 `messages` 列表。系统提示词放在顶层 `system` 字段。本例为 Qwen3 关闭思考模式，以生成简短回答：

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

### 使用 Anthropic Python SDK

```bash
python -m pip install --upgrade anthropic
```

`base_url` 使用服务根地址，不加 `/v1`，SDK 会追加 `/v1/messages`。直接访问 xLLM 无需 Anthropic 平台密钥，可用 `EMPTY` 满足 SDK 配置要求；带鉴权的网关可能需要自己的凭据。后续 Python 示例均沿用这里的初始化：

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

响应的 `content` 是内容块数组，可能包含 `text`、`thinking` 或 `tool_use`。请按 `type` 读取，不要假定首个内容块一定是文本。用量统计使用 `input_tokens` 和 `output_tokens`。

如果 SDK 版本未提供 `temperature`、`top_p`、`top_k` 等采样参数的具名参数，可通过 `extra_body` 传入。直接发送 HTTP JSON 时，将这些字段放在请求体顶层。

## 系统提示词与多轮对话

`system` 支持字符串或文本块列表。`messages` 包含 user/assistant 对话历史，内容可以是字符串或内容块数组。每次请求都应发送历史消息，接口不会替客户端保存对话。

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

## 流式输出

SDK 的流式辅助方法可以逐段返回文本，并合并得到包含用量和停止原因的最终消息：

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

直接使用 HTTP 时，设置 `"stream": true`，并使用 `curl -N`。响应采用 Anthropic SSE 事件：`message_start`、`content_block_start`、`content_block_delta`、`content_block_stop`、`message_delta` 和 `message_stop`。文本增量类型为 `text_delta`，工具参数增量类型为 `input_json_delta`。流以 `message_stop` 结束，而不是 OpenAI 的 `[DONE]` 标记。

## 工具调用

先在服务端配置匹配模型的工具解析器，再传入包含 `name`、`description` 和 `input_schema` 的 Anthropic 格式工具定义。xLLM 返回 `tool_use` 内容块，由应用执行工具。

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

返回工具结果时，追加 assistant 的内容和包含 `tool_result` 的 user 消息。每个 `tool_use_id` 必须匹配原始调用。下面使用明确标注的模拟结果，请替换成应用实际执行工具得到的输出。

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

## 推理输出

使用 Qwen3 时，保持 `--reasoning_parser=qwen3` 配置，并通过 `extra_body` 传入 `chat_template_kwargs.enable_thinking`。对应的原始 HTTP JSON 字段为顶层的 `chat_template_kwargs`。请为推理过程和最终回答预留足够的 token。

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

聊天模板选项取决于模型。xLLM 未将 Anthropic 的 `thinking.budget_tokens` 实现为独立推理 token 预算，请使用模型对应的模板选项和总输出上限 `max_tokens`。

## 统计输入 token

计数接口渲染聊天模板并对输入分词，不执行生成。请使用与生成请求一致的 `system`、`messages`、`tools` 和模板选项，使结果对应同一个 prompt。该接口无需 `max_tokens`。

```python
count = client.messages.count_tokens(
    model=model,
    system="Give concise answers.",
    messages=[{"role": "user", "content": "What is an inference engine?"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(count.input_tokens)
```

## 连接 Claude Code

安装 Claude Code 并启动具备工具调用能力的 xLLM 服务后，设置服务根地址，并将模型别名映射到实际提供服务的模型 ID。这些变量可查阅 [Claude Code 环境变量参考](https://code.claude.com/docs/en/env-vars)。

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

`CLAUDE_CODE_ATTRIBUTION_HEADER=0` 用于去掉客户端归属信息块，对前缀复用的影响取决于客户端版本和服务端缓存配置。不要随意增加 `[1m]` 后缀：xLLM 会检查模型名，客户端后缀也不会扩大模型上下文；只有实际提供该完整 ID 时才能使用。客户端若发送不支持的内容或受约束输出请求，仍受下面的限制。本配置说明不代表所有 Claude Code 工作流都经过了端到端验证。

## 参数与兼容性限制

| 参数 | 用途 |
| --- | --- |
| `model` | 可用的模型 ID，生成和计数接口均必需。 |
| `max_tokens` | 生成接口必需，为正整数输出上限。 |
| `system` | 系统提示词字符串或文本块列表。 |
| `temperature`、`top_p`、`top_k` | 采样控制；temperature 使用非负值，`top_p` 范围为 `(0, 1]`，`top_k=0` 关闭 top-k 截断。 |
| `stop_sequences` | 停止字符串列表。 |
| `stream` | 启用 Anthropic SSE 输出。 |
| `tools`、`tool_choice` | 函数定义与选择方式（`auto`、`none`、`any` 或带名称的 `tool`）；输出质量取决于模型及解析器。 |
| `chat_template_kwargs` | xLLM 扩展参数，传入模型特定的模板选项。 |

- Messages API 当前用于 LLM，不支持图像输入，也不支持工具结果中的图像或工具引用内容。图像对话请使用 [OpenAI 视觉接口](/zh/getting_started/openai_api_vision/)。
- 不支持非空的 `output_config.format.schema` 约束和 `kv_transfer_params`，会返回 HTTP 400。接受工具 schema 不代表具有受约束 JSON 生成能力。
- `stop_reason="end_turn"` 表示正常结束，`max_tokens` 表示达到长度上限，`tool_use` 表示工具调用。
- 模型相关的 HTTP 404 请通过 `/v1/models` 检查模型 ID；HTTP 501 请检查是否为 LLM 部署。请求失败时请读取 `error.message`，格式错误和后端不支持选项的错误响应结构可能不同。
