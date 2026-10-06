---
title: "OpenAI Responses API"
sidebar:
  order: 90
---
<!-- Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0 -->

xLLM 原生 API 服务提供 **`POST /v1/responses`**。该接口返回 Responses 对象和命名 SSE 事件，不使用 Chat Completions 的 `choices` 或 `chat.completion.chunk`。实现复用现有模型选择、模板、请求工厂、调度器和并发准入限制，不在内部调用 `/v1/chat/completions`。

先按照[启动文档](/zh/getting_started/launch_xllm/)启动模型，再将示例中的 `loaded-model` 替换为服务中的模型名称。接口路径是复数 `/v1/responses`，不提供 `/v1/response`。

## 文本与客户端携带的历史

```bash
curl http://127.0.0.1:9977/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "loaded-model",
    "input": "用一句话解释前缀缓存。",
    "instructions": "回答要简洁。",
    "max_output_tokens": 128,
    "temperature": 0,
    "store": false
  }'
```

使用官方 OpenAI Python SDK：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:9977/v1", api_key="EMPTY")
response = client.responses.create(
    model="loaded-model",
    input="用一句话解释前缀缓存。",
    instructions="回答要简洁。",
    max_output_tokens=128,
    store=False,
)
print(response.status)
print(response.output_text)

# 客户端显式携带历史；xLLM 不保存 Response 对象。
history = [{"role": "user", "content": "用一句话解释前缀缓存。"}]
history.extend(item.model_dump(exclude_unset=True) for item in response.output)
history.append({"role": "user", "content": "它在什么情况下有用？"})
next_response = client.responses.create(
    model="loaded-model",
    instructions="回答要简洁。",
    input=history,
    max_output_tokens=128,
    store=False,
)
```

本无状态接口要求提供 `input`，其值可以是字符串，也可以是由文本消息及受支持历史条目组成的非空数组。消息角色支持 `user`、`assistant`、`system` 和 `developer`；现有模板将 developer 消息按 system 指令优先级处理。消息内容可以是字符串或 `input_text` 数组。assistant 历史可使用带空 annotations 的 `output_text`。多个文本片段按原顺序直接拼接。`instructions` 会作为 system 消息加到当前请求前面；后续请求如仍需该指令，应再次携带。

**存储范围：**支持 `store=false`。xLLM 中省略或传 null 的 `store` 同样表示 **false**，这与托管 API 的存储默认值不同。`store=true`、非 null 的 `previous_response_id` 和 `background=true` 均报错。Response 和条目 ID 仅用于标识结果，不是可检索的存储句柄。当前不提供结果检索、删除、后台任务或服务端会话续接。

## 流式返回

```python
with client.responses.create(
    model="loaded-model",
    input="简短地问好。",
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

对于正常完成的响应，也可使用 SDK 累积器：

```python
with client.responses.stream(
    model="loaded-model",
    input="简短地问好。",
    max_output_tokens=128,
    store=False,
) as stream:
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
    response = stream.get_final_response()
```

非 completed 终态的处理应以已安装 SDK 的行为为准。例如 OpenAI Python 2.54.0 的累积器仅在 `response.completed` 后提供 `get_final_response()`。需要处理 incomplete 或 failed 时，可使用 `responses.create(stream=True)` 读取终态事件，不能假设该辅助方法适用于所有终态。

流式响应使用 `Content-Type: text/event-stream`。每个 `event:` 名称与 JSON 的 `type` 一致；`sequence_number` 从 0 开始逐一递增。生命周期如下：

1. `response.created`，随后 `response.in_progress`。
2. `response.output_item.added`，携带稳定 ID 和从零连续递增的 `output_index`。
3. 文本先发送 `response.content_part.added`，再发送 `response.output_text.delta`；函数参数使用 `response.function_call_arguments.delta`。原始推理使用 reasoning 条目上的 `response.reasoning_text.delta`，不伪装成摘要。
4. 正常完成或达到长度上限时，发送对应的文本/参数 done 事件、适用的 content-part done 事件和 `response.output_item.done`。
5. 仅一个终态事件携带完整最终 Response：`response.completed`、`response.incomplete` 或 `response.failed`。

没有 Chat 风格的 `[DONE]`。客户端断开后无法接收终态，不能把 EOF 当成成功完成。客户端取消沿用正常请求生命周期释放所有权；xLLM 不伪造 `response.cancelled` SSE，也不提供存储型响应取消接口。

## 客户端函数工具

工具采用 **Responses 扁平格式**，不是 Chat 中嵌套的 `function` 对象。所选模型需要配置受支持的工具解析器。本版本支持 `qwen25`、`qwen3_coder`、`glm45`、`glm47`、`glm5` 及解析到这些格式的别名；其他格式在生成前拒绝。Responses 复用模型工具解析器及其值转换。受支持的流式解析器在同一次增量调用中消费已具备解析条件的名称、参数和完整调用，生成结束后通过显式结束操作释放待输出内容。正常完成时，已识别但仍缺数据的工具状态会报错；长度上限或生成失败则保留部分输出，不补造缺失值。语义完整的参数在 EOF 时不要求所有外层包装都存在。API 校验返回的调用名称、索引及已完成参数是否为 JSON 对象；这些处理不是完整模型语法或严格结束标签校验，也不保证任意畸形输入或所有模型格式的任意分片一致性。这些检查不等于严格 JSON Schema 约束，后者仍不支持。xLLM 只产生调用，**由客户端提供结果**，服务端不执行函数。

```python
import json

question = "请使用 weather 查询巴黎的当前天气。"
tools = [{
    "type": "function",
    "name": "weather",
    "description": "查询指定城市的当前天气。",
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
        # 在应用中取得结果；本示例由客户端直接提供一个值。
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

`auto` 允许模型返回文本或一个/多个调用，但不强制产生调用。`none` 禁止新的调用，同时保留回放历史。每个历史函数调用都必须对应唯一 `call_id` 的 `function_call_output`。历史调用的参数必须编码为 JSON 对象，结果可为文本或 `input_text` 数组。对应调用之后的多个结果允许交换顺序。客户端提供的畸形历史参数、缺失结果或重复结果会作为请求错误报出，不会被修补成默认值。生成调用沿用已有模型解析器的行为；API 校验解析结果，不重新构造解析器已丢弃的模型语法。

本接口只提供非严格函数 schema：省略或 null 的 `strict` 归一为 false，`strict=true` 报错。后端无法强制执行的 `tool_choice="required"`、按名称强制调用、`parallel_tool_calls=false` 均报错。省略、null 或 true 的 `parallel_tool_calls` 允许多个调用。内置 web search、code interpreter、file search、computer use、MCP 等工具不受支持。

## 推理、结构化输出与 usage

配置受支持的推理解析器后，接口保留模型的**原始推理**：`reasoning` 输出条目的 `summary=[]`，`content` 包含 `reasoning_text`。这些条目可回放到对应 assistant 消息或函数调用前。原始推理不是摘要，也不会编码成伪造的 encrypted content。非 null 的 `reasoning.effort` 和 `reasoning.summary` 会被拒绝，因为现有引擎不能强制推理强度或生成摘要。空推理控制不意味着切换模型或启用另一个解析器。

`text.format={"type":"text"}` 为默认值。服务开启 `enable_json_object_output` 后，`text.format={"type":"json_object"}` 使用已有的约束解码器；不能与活跃函数工具组合。JSON Schema、strict schema 及其他引擎无法保证的格式控制会被明确拒绝。

最终 usage 包含：

- `input_tokens`、`output_tokens`、`total_tokens`：生成路径的实际计数。
- `input_tokens_details.cached_tokens`：实际复用前缀缓存的提示词位置数的高水位。
- `input_tokens_details.cache_write_tokens`：固定为 `0`，与 SGLang 的 Responses 序列化行为一致。本接口不单独报告缓存写入用量；该值不是实际写入计数，也不表示内部没有发生缓存写入。
- `output_tokens_details.reasoning_tokens`：固定为整数 `0`，用于兼容。本接口不单独报告推理 token 用量；`0` 不表示模型没有实际推理或没有输出原始推理文本，也不会改变实际 output/total 用量或生成预算。

缓存命中输入由 `cached_tokens` 表示，未命中输入可由 `input_tokens - cached_tokens` 得到；内部缓存发布不作为单独的用量类别报告。固定的缓存写入字段值不会改变缓存行为。Responses 仍拒绝分离式/分布式服务、外部 KV 存储和 host offload。这些配置的文本状态传递或缓存命中计数不在受支持范围内；不单独报告推理 token 用量不会扩大这些能力。

## 支持的请求范围

| 字段 | 行为 |
|---|---|
| `model` | 省略时选择已加载的默认模型；显式提供时须为已加载 LLM 的非空名称字符串。 |
| `input` | 必填字符串或受支持文本/历史条目的非空数组。 |
| `instructions` | 文本或 null；仅前置到当前请求。 |
| `stream` | 省略/null/false 返回 JSON；true 返回命名 SSE。 |
| `max_output_tokens` | 至少 16 的整数，映射到现有 `max_tokens` 生成预算，包含推理。省略/null 使用引擎默认值 5120；仍受模型和上下文限制。 |
| `temperature`、`top_p` | 有限数值，范围分别为 `[0,2]`、`[0,1]`；省略/null 默认 1。 |
| `tools`、`tool_choice` | 仅客户端函数，选择 `auto` 或 `none`；准入前检查所选模型解析能力。 |
| `parallel_tool_calls` | 省略/null/true；不支持 false。 |
| `text.format` | `text` 或服务已开启的 `json_object`；不支持 `json_schema`。 |
| `reasoning` | 省略/null/空控制；不支持非 null 的 effort/summary。原始输出取决于已配置解析器。 |
| `metadata` | 至多 16 个字符串键值对；本接口键长至多 64 字节、值长至多 512 字节。省略/null 归一为 `{}`。 |
| `store`、`background` | 仅省略/null/false。 |
| `previous_response_id` | 仅省略/null；历史需显式回放。 |
| `truncation` | 仅省略/null/`disabled`；不自动截断历史。 |
| `include` | 仅省略/null/空列表。 |

未知字段、`messages`/`n`/`max_tokens` 等 Chat 参数、媒体输入、托管工具 annotations、item reference、assistant phase 控制及不支持的条目/工具类型均返回可定位参数的 HTTP 400，不会静默忽略。

## 终态与 HTTP 错误

| 情况 | 返回 |
|---|---|
| 正常 EOS/stop 或完成的函数调用 | Response `status="completed"`。 |
| 达到输出 token 上限 | `status="incomplete"`，`incomplete_details.reason="max_output_tokens"`。 |
| 已开始输出后生成失败 | Response `status="failed"`，error 非 null；流式以 `response.failed` 结束，不冒充 completed。 |
| 客户端断开/取消 | 取消请求并释放已取得的并发名额；不伪造成功结果。 |
| 无效/不支持的请求或能力 | SSE 提交前返回 HTTP 400 JSON。 |
| 模型不存在 | HTTP 404 JSON。 |
| 并发准入被拒绝 | 两种 stream 模式都返回 HTTP 429 JSON；官方 SDK 抛出 `RateLimitError`。 |
| 输出前队列/执行资源耗尽 | 服务端错误，与并发准入 HTTP 429 区分。 |

输出前错误使用共享 OpenAI envelope，包含 `message`、`type`、`param`、`code`。请求了 stream 但被拒绝时仍返回 `application/json`，不是 HTTP 200 或替代 SSE 错误。xLLM 不伪造 `Retry-After`。只有接受的请求拥有 limiter 名额，正常请求生命周期负责恰好一次释放。

## 验证入口

专用原生目标覆盖请求转换、输出/SSE 序列化和真实 brpc 传输：`openai_responses_request_test`、`openai_responses_output_test`、`openai_responses_protocol_test`。官方 SDK 用例通过标准 `DISABLED_` 前缀排除在普通运行之外，并与其他协议共用默认 `OFF` 的 `XLLM_ENABLE_SDK_TESTS` CMake 选项；显式使用 `--gtest_also_run_disabled_tests` 运行 `OpenAIResponsesProtocolTest.DISABLED_OfficialSdkCompatibility`。该用例使用 CMake 已选定的 `Python3_EXECUTABLE`，要求预先安装 `openai`、`httpx`。缺少包时显式验证失败，不自动安装或切换解释器。

Python SDK 的 CTest 目标同样默认禁用。应在所需容器内、checkout 根目录显式运行 pytest，原生初始化由 `tests/python/conftest.py` 统一负责：

```bash
XLLM_RESPONSES_BASE_URL=http://127.0.0.1:9977/v1 \
XLLM_RESPONSES_MODEL=loaded-model \
python -m pytest tests/python/test_openai_responses_protocol.py
```

同一模块还提供面向已启动、由外部拥有生命周期的服务的纯 HTTP 客户端：

```bash
python tests/python/test_openai_responses_protocol.py \
  --base-url http://127.0.0.1:9977/v1 --model loaded-model \
  --expect-reasoning --evidence-dir /path/to/fresh-results
```

普通真实模型验证由服务 runner 配置 `max_concurrent_requests=16`，客户端串行请求，不使用 `--fixture` 或 `--limit-one`。`--expect-reasoning` 额外要求真实原始推理文本。准入/断开验证单独执行：`--limit-one` 要求服务 runner 已将 `max_concurrent_requests` 配置为 1；客户端不会修改服务配置或另建 runner。`--fixture` 选择确定性的 brpc fixture 输入，包括推理、跨字节 UTF-8 和失败；它不启用准入/断开专项，这些检查只由独立、显式的 `--limit-one` 选项启用。不使用该选项时请求由真实模型处理，函数调用必须实际产生。Fixture 和真实模型证据分开记录。上述命令是验证入口说明，不代表某个模型或环境已经通过测试。
