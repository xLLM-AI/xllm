---
title: "OpenAI 兼容接口"
---

xLLM 为本地模型提供 OpenAI 兼容 HTTP 接口。将 OpenAI Python 客户端的 `base_url` 指向 xLLM 服务后，即可使用 SDK 或 cURL 发送请求。本页介绍文本生成，其他模型任务请参阅[视觉输入](/zh/getting_started/openai_api_vision/)和 [Embedding](/zh/getting_started/openai_api_embeddings/)。

| 接口 | 方法 | 用途 |
| --- | --- | --- |
| `/v1/models` | GET | 列出可用的模型 ID。 |
| `/v1/chat/completions` | POST | 根据消息生成回复；支持流式输出，VLM 服务还支持图像输入。 |
| `/v1/completions` | POST | 续写文本或 token ID 输入；支持流式输出和批量 prompt。 |
| `/v1/embeddings` | POST | 使用 Embedding 服务编码文本或 token ID。 |

兼容范围以本文介绍的接口和选项为准。目前未实现 `/v1/responses`。

## 启动服务

先按照[快速开始](/zh/getting_started/quick_start/)安装 xLLM 并准备模型权重。下面使用单张昇腾 NPU，通过 Python/PyTorch 模型实现和 ACL Graph 解码运行 Qwen3-8B；请替换模型路径，并确保设备显存足够。其他硬件或多卡部署请参考对应的[启动说明](/zh/getting_started/launch_xllm/)。

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

等待模型加载完成，再在另一个终端运行客户端示例。上述命令监听回环地址；从其他机器访问时，请将 `--host` 设为可访问的服务地址，并将下面的 `HOST` 改为该地址。

`--model` 指定权重路径，`--model_id` 指定请求使用的模型名。省略 `--model_id` 时，xLLM 使用模型路径的最后一级目录名。可以通过 `/v1/models` 确认可用 ID：

```bash
export HOST=127.0.0.1
export PORT=18000
export MODEL=Qwen3-8B

curl -sS "http://${HOST}:${PORT}/v1/models"
```

## 配置 Python 客户端

```bash
python -m pip install --upgrade openai
```

运行本页 Python 示例前，先执行以下初始化代码。直接访问 xLLM 无需 OpenAI 平台密钥，`EMPTY` 用于满足 SDK 的密钥配置要求；如果通过带鉴权的网关访问，请使用网关要求的凭据。

```python
import os
from openai import OpenAI

host = os.environ.get("HOST", "127.0.0.1")
port = os.environ.get("PORT", "18000")
model = os.environ.get("MODEL", "Qwen3-8B")
client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key="EMPTY")

print([item.id for item in client.models.list().data])
```

## 对话生成

通过 `messages` 传入对话历史。xLLM 使用模型的聊天模板构建输入，无需手动拼接对话分隔符。这里为 Qwen3 关闭思考模式，使简短回复适合示例的输出长度；聊天模板选项取决于具体模型。

### 使用 cURL

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

### 使用 OpenAI Python 客户端

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

多轮对话时，将上一轮 assistant 回复和新的 user 消息追加到 `messages`，再发送完整历史。生成文本位于 `choices[].message.content`，token 用量位于 `usage.prompt_tokens`、`usage.completion_tokens` 和 `usage.total_tokens`。

### 流式输出

设置 `stream=True` 可增量接收回复。`stream_options={"include_usage": True}` 会额外返回最后一个用量统计 chunk，其 `choices` 为空数组，因此读取下标前需要检查。该选项只能用于流式请求。

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

使用 cURL 时，增加 `-N` 关闭输出缓冲，并在 JSON 请求体中加入 `"stream": true`。HTTP 响应采用 Server-Sent Events（SSE），每个事件以 `data:` 开头，完成的流以 `data: [DONE]` 结束。

### 思考与推理输出

前面的 Qwen3 启动命令已设置 `--reasoning_parser=qwen3`。通过 `chat_template_kwargs` 中的 `enable_thinking` 控制 Qwen3 思考模式，并为推理过程和最终回答预留足够的输出 token。其他模型需要匹配各自的模板选项和解析器，参见 [CLI 参考](/zh/cli_reference/)。

xLLM 的 HTTP 接口将解析出的推理内容放在 `message.reasoning` 中，流式输出则使用 `delta.reasoning`。使用 SDK 时，可通过 `getattr` 读取这个扩展字段：

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

### 工具调用

工具调用需要支持工具的模型聊天模板，以及匹配的 `--tool_call_parser`，例如前面的 Qwen3 配置。在 `tools` 中声明函数结构；设置 `tool_choice="auto"` 时，模型可以返回普通回答或 `message.tool_calls`。

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

xLLM 返回函数名和 JSON 参数字符串，由应用执行函数。继续对话时，先追加包含 `tool_calls` 的 assistant 消息，再追加 `role="tool"` 的结果消息，其中 `tool_call_id` 对应调用 ID，`content` 为工具结果，随后再次请求对话接口。

## 文本补全

文本补全接口接收 `prompt`，通过 `choices[].text` 返回续写文本。该接口不应用聊天模板，适用于原始文本续写，或已经按模型要求格式化的 prompt。

```python
response = client.completions.create(
    model=model,
    prompt="An inference engine is",
    temperature=0,
    max_tokens=64,
)
print(response.choices[0].text)
```

流式补全可增加 `stream=True`，通过 `chunk.choices[0].text` 读取文本；同时请求用量统计时，需像流式对话示例一样处理空 `choices`。

`prompt` 支持字符串、字符串列表、token ID 列表或 token ID 列表的列表。批量结果先按 prompt 顺序、再按候选顺序排列，`usage` 为整个请求的统计。token ID 必须使用所服务模型的 tokenizer 生成。

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

## 常用参数与扩展

| 参数 | xLLM 行为 |
| --- | --- |
| `max_tokens` | 正整数，限制输出 token 数。文本补全和 Python `SamplingParams` 默认值为 16；对话接口省略该值或 Python 设置 `max_tokens=None` 时使用剩余模型上下文长度。 |
| `max_completion_tokens` | 仅用于对话；与 `max_tokens` 同时提供时优先使用该值。 |
| `temperature` | 非负有限数，默认 `1.0`；与 vLLM `0.23.0` 一致，允许大于 `2`。小于 `0.01` 的正数会提高到 `0.01`；`0` 表示贪心解码，并关闭 `top_k`、`top_p` 和 `min_p`。 |
| `top_p` | 采样截断阈值，范围为 `(0, 1]`。 |
| `n` | 返回候选数；`temperature=0` 时必须为 `1`。 |
| `stop` | 字符串或由非空字符串组成的列表。 |
| `presence_penalty`、`frequency_penalty` | 范围为 `[-2, 2]`。 |
| `logprobs` | 对话接口使用布尔值，补全接口和 Python 使用整数数量；对话还可设置 `top_logprobs`。数量为 `0` 时仅返回采样 token，为 `-1` 时返回整个词表。概率在惩罚、偏置、温度和截断处理前计算。 |
| `top_k` | 扩展参数：整数采样截断阈值，`-1` 或 `0` 表示不截断。 |
| `repetition_penalty` | 扩展参数：正数，控制重复惩罚。 |
| `min_p` | 相对概率阈值，范围 `[0, 1]`，默认 `0`；在温度处理后、top-k/top-p 前生效。 |
| `seed` | 可选的有符号 64 位随机种子，`-1` 表示不设置。随机流不受批次位置影响；不同引擎或硬件不保证产生相同 token。 |
| `min_tokens` | 最少输出 token 数，非负且不超过 `max_tokens`；达到该长度前禁止生成 EOS 和停止 token。 |
| `logit_bias` | token ID 到偏置的映射，值截断到 `[-100, 100]`。 |
| `allowed_token_ids` | 非空的允许生成 token ID 列表。 |
| `bad_words` | 非空字符串列表，禁止在生成文本中完成对应 token 序列。 |
| `chat_template_kwargs` | 对话扩展参数：传入模型支持的聊天模板选项。 |

SDK 通过 `extra_body` 传入扩展参数；直接发送 HTTP JSON 时，将这些字段放在请求体顶层，无需 `extra_body` 包装：

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

xLLM 的 `beam_width` 扩展请参阅[在线服务](/zh/getting_started/online_service/)。流式示例无需设置 `best_of`；只有 `best_of` 等于 `n` 时，xLLM 才采用流式输出。

## 兼容性限制与错误处理

- `seed`、`min_tokens`、`logit_bias`、`allowed_token_ids`、`bad_words` 和全词表 logprobs 要求 `enable_task_pipeline=false`。`seed`、`min_tokens` 和 `bad_words` 还要求 `num_speculative_tokens=0`；`bad_words` 要求 `enable_schedule_overlap=false`。REC 接口会拒绝新增的随机种子和 token 约束参数。
- 非空的 `prompt_logprobs`、`logprob_token_ids`、`structured_outputs` 和 `prompt_embeds` 仍不受支持，会返回 HTTP 400。此接口没有实现 vLLM `0.23.0` `SamplingParams` 的全部引擎层字段。
- 频率惩罚和存在惩罚仅统计生成 token，重复惩罚同时考虑提示词。`ignore_eos=true` 时显式指定的 `stop_token_ids` 仍然生效。
- 不支持 `response_format={"type": "json_schema"}`。对话接口的 `json_object` 需要服务端 JSON 输出配置，不能视为支持 JSON Schema。
- Embedding 请求不支持 `dimensions`，详见 [Embedding 使用说明](/zh/getting_started/openai_api_embeddings/)。
- HTTP 400 表示参数无效或不受支持，请查看 JSON 中的 `error.message` 和 `error.param`。模型相关的 HTTP 404 表示模型名不匹配，请查询 `/v1/models`。
- 成功生成时若 `finish_reason="length"`，表示达到输出或上下文长度限制，可按需增加输出预算或缩短输入。
