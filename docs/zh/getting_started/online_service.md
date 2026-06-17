# 在线服务
先按照[xllm启动文档](launch_xllm.md)启动xllm服务。下面给出LLM和VLM的客户端调用示例，需要根据实际情况修改其中的参数。

## LLM 客户端调用
### HTTP 调用

chat模式：
```bash
curl http://localhost:9977/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2-7B-Instruct",
    "max_tokens": 10,
    "temperature": 0,
    "stream": true,
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "hello xllm"
      }
    ]
  }'
```

completions模式：
```bash
curl http://127.0.0.1:9977/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2-7B-Instruct",
    "prompt": "hello xllm",
    "max_tokens": 10,
    "temperature": 0,
    "stream": true
  }'
```

Beam Search：

将 `beam_width` 设置为大于 `1` 的值即可开启 LLM Beam Search。`/v1/chat/completions` 和 `/v1/completions` 均支持该参数。beam-search top-k 候选数量在 chat 请求中使用 `top_logprobs` 配置，在 completions 请求中使用数值型 `logprobs` 字段配置。如果这些字段未设置，xLLM 会使用 `beam_width` 作为 top logprob 数量。如果希望每个 beam 考虑更多候选 token，可以将候选数量设置为大于 `beam_width` 的值。这里的 top-k 不同于采样截断参数 `top_k`。`best_of` 不是 Beam Search 开关，本文档也不使用 `num_return_sequences` 来控制 LLM 返回的 beam 数。

chat模式：
```bash
curl http://localhost:9977/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2-7B-Instruct",
    "max_tokens": 20,
    "temperature": 0,
    "stream": false,
    "beam_width": 2,
    "logprobs": true,
    "top_logprobs": 4,
    "messages": [
      {
        "role": "user",
        "content": "请简短介绍 xLLM。"
      }
    ]
  }'
```

completions模式：
```bash
curl http://127.0.0.1:9977/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2-7B-Instruct",
    "prompt": "请简短介绍 xLLM。",
    "max_tokens": 20,
    "temperature": 0,
    "stream": false,
    "beam_width": 2,
    "logprobs": 4
  }'
```

sample模式：
```bash
curl http://127.0.0.1:9977/v1/sample \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2-7B-Instruct",
    "prompt": "问题：<emb_0> 是否命中。结论：<emb_0>",
    "selector": {
      "type": "literal",
      "value": "<emb_0>"
    },
    "logprobs": 5,
    "request_id": "sample-demo-001"
  }'
```

典型响应：
```json
{
  "id": "sample-demo-001",
  "object": "sample_completion",
  "created": 1773369600,
  "model": "Qwen2-7B-Instruct",
  "choices": [
    {
      "index": 0,
      "text": "True",
      "logprobs": {
        "tokens": ["True", "False"],
        "token_ids": [3456, 7890],
        "token_logprobs": [-0.12, -2.31]
      },
      "finish_reason": "selector_match"
    },
    {
      "index": 1,
      "text": "",
      "logprobs": {
        "tokens": [],
        "token_ids": [],
        "token_logprobs": []
      },
      "finish_reason": "empty_logprobs"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 2,
    "total_tokens": 22
  }
}
```

`/v1/sample` 使用说明：

- 仅支持 `--backend=llm`，当前不支持 VLM/DiT/Rec。
- `selector.type` 当前固定为 `literal`，`selector.value` 按 prompt 文本顺序全文匹配。
- `logprobs` 默认值为 `5`，允许范围为 `[1, 5]`。
- `choices[i].index` 即该命中的 `sample_id`，与 prompt 中命中顺序一一对应。
- selector 无命中时返回 `200` 且 `choices=[]`；某命中位点无可用 logprobs 时返回 `finish_reason="empty_logprobs"`。
- 服务日志只记录 `request_id`、`sample_id`、`match_count`、`model` 等摘要字段，不记录完整 prompt。

`/v1/sample` 常见错误语义：

- 缺少 `model/prompt/selector/selector.value`、`selector.type != literal` 或 `logprobs` 越界时返回 `INVALID_ARGUMENT`。
- 模型不存在或后端不是 `llm` 时返回 `UNKNOWN`。
- 并发达到上限时返回 `RESOURCE_EXHAUSTED`。
- 模型处于 sleep 状态时返回 `UNAVAILABLE`。

### Python调用
```python
import requests
import json

url = f"http://localhost:9977/v1/chat/completions"
messages = [
    {'role': 'user', 'content': "列出三个国家和他的首都。"}
]

request_data = {
    "model": "Qwen2-7B-Instruct",
    "messages": messages,
    "stream": False, 
    "temperature": 0.6, 
    "max_tokens": 2048, 
}

response = requests.post(url, json=request_data)
if response.status_code != 200:
    print(response.status_code, response.text)
else:
    ans = json.loads(response.text)["choices"]
    print(ans[0]['message'])
```


## VLM 客户端调用
### HTTP API

```python
import base64
import requests

api_url = "http://localhost:12345/v1/chat/completions"
image_url = ""

def encode_image(url: str) -> str:
    with requests.get(url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode("utf-8")

    return result

image_base64 = encode_image(image_url)
payload = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "介绍下这张图片"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
            ],
        }
    ],
    "model": "Qwen2.5-VL-7B-Instruct",
    "max_completion_tokens": 128,
}

response = requests.post(
    api_url,
    json=payload,
    headers={"Content-Type": "application/json"}
)
print(response.json())
```


### OpenAI API
```python
from openai import OpenAI
import base64
import requests

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:12345/v1"
image_url = ""

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def encode_image(url: str) -> str:
    with requests.get(url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode("utf-8")

    return result

image_base64 = encode_image(image_url)
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "介绍下这张图片"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
            ],
        }
    ],
    model="Qwen2.5-VL-7B-Instruct",
    max_completion_tokens=128,
)

result = chat_completion.choices[0].message.content
print("Chat completion output:", result)
```
