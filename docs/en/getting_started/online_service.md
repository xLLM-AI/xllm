# Online Service

First, start the xllm service according to the [xllm launch documentation](launch_xllm.md). Below are examples of client calls for LLM and VLM. Please modify the parameters according to your actual situation.

## LLM Client Calls
### HTTP Call

Chat mode:
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

Completions mode:
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

Beam Search:

Set `beam_width` to a value greater than `1` to enable LLM Beam Search. This parameter is supported by both `/v1/chat/completions` and `/v1/completions`. The beam-search top-k candidate count is configured with `top_logprobs` in chat requests and with the numeric `logprobs` field in completion requests. When these fields are omitted, xLLM uses `beam_width` as the top logprob count. Set the candidate count to a value greater than `beam_width` when you want each beam to consider more candidate tokens. This is different from the sampling cutoff parameter `top_k`. `best_of` is not the Beam Search switch, and this LLM API guide does not use `num_return_sequences` to control the returned beams.

Chat mode:
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
        "content": "Write a short introduction to xLLM."
      }
    ]
  }'
```

Completions mode:
```bash
curl http://127.0.0.1:9977/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2-7B-Instruct",
    "prompt": "Write a short introduction to xLLM.",
    "max_tokens": 20,
    "temperature": 0,
    "stream": false,
    "beam_width": 2,
    "logprobs": 4
  }'
```

Sample mode:
```bash
curl http://127.0.0.1:9977/v1/sample \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2-7B-Instruct",
    "prompt": "Question: <emb_0> matched or not. Conclusion: <emb_0>",
    "selector": {
      "type": "literal",
      "value": "<emb_0>"
    },
    "logprobs": 5,
    "request_id": "sample-demo-001"
  }'
```

Typical response:
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

`/v1/sample` notes:

- Only `--backend=llm` is supported. VLM/DiT/Rec are not supported yet.
- `selector.type` is currently fixed to `literal`. `selector.value` is matched against prompt text in full and in order.
- `logprobs` defaults to `5`, with an allowed range of `[1, 5]`.
- `choices[i].index` is the matched `sample_id`, corresponding one-to-one with the matched order in prompt.
- If no selector match is found, the service returns `200` with `choices=[]`. If a matched position has no available logprobs, it returns `finish_reason="empty_logprobs"`.
- Service logs only summary fields such as `request_id`, `sample_id`, `match_count`, and `model`, and do not log the full prompt.

`/v1/sample` common error semantics:

- Missing `model/prompt/selector/selector.value`, `selector.type != literal`, or out-of-range `logprobs` returns `INVALID_ARGUMENT`.
- If the model does not exist or the backend is not `llm`, it returns `UNKNOWN`.
- When concurrency reaches the upper limit, it returns `RESOURCE_EXHAUSTED`.
- When the model is in sleep state, it returns `UNAVAILABLE`.

### Python Call
```python
import requests
import json

url = f"http://localhost:9977/v1/chat/completions"
messages = [
    {'role': 'user', 'content': "List three countries and their capitals."}
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


## VLM Client Calls
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
                {"type": "text", "text": "Describe this image"},
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
                {"type": "text", "text": "Describe this image"},
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
