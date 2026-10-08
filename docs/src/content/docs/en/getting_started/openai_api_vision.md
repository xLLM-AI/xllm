---
title: "OpenAI APIs - Vision"
---

Use `/v1/chat/completions` with a vision-language model to ask questions about images. The request contains text and `image_url` parts in a message's `content` array. For general client setup and streaming, see [OpenAI-Compatible APIs](/en/getting_started/openai_api/).

## Launch a VLM server

Prepare xLLM and Qwen3-VL-8B-Instruct. This example uses the Python/PyTorch implementation with ACL Graph decoding on one Ascend NPU; adjust the device configuration and model size following the [launch guide](/en/getting_started/launch_xllm/). The VLM example disables prefix caching and chunked prefill, as described in [Multimodal Support](/en/features/multimodal/).

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export HCCL_IF_BASE_PORT=43432

xllm --model=/path/to/Qwen3-VL-8B-Instruct \
  --model_id=Qwen3-VL-8B-Instruct \
  --backend=vlm \
  --model_impl=python \
  --npu_kernel_backend=TORCH \
  --enable_graph=true \
  --python_graph_backend=aclgraph \
  --host=127.0.0.1 \
  --port=18001 \
  --master_node_addr=127.0.0.1:9748 \
  --nnodes=1 \
  --node_rank=0 \
  --block_size=128 \
  --max_memory_utilization=0.86 \
  --enable_prefix_cache=false \
  --enable_chunked_prefill=false \
  --enable_schedule_overlap=false \
  --enable_shm=false
```

Run this as a separate deployment from the text-generation example. If servers run concurrently, allocate separate devices, master addresses, and communication ports. Wait for model loading before sending requests.

## Use an image URL

Use a direct HTTP image URL accessible from the xLLM server. The current downloader does not configure HTTPS/TLS or follow redirects. For an HTTPS-only source, download the image on the client and use the Base64 example below. The downloader connects directly, so a URL reachable only through your shell's HTTP proxy may still fail.

For a reproducible local example, run the following in another terminal on the same host (and in the same container, if applicable) as xLLM. Leave this image server running while you send requests:

```bash
mkdir -p vision-images
curl -fL http://images.cocodataset.org/val2017/000000039769.jpg \
  -o vision-images/photo.jpg
python3 -m http.server 18003 --bind 127.0.0.1 --directory vision-images
```

The loopback URL below is resolved by the xLLM server. For a remote image server, replace `IMAGE_URL` with its directly reachable HTTP address.

```bash
export HOST=127.0.0.1
export PORT=18001
export MODEL=Qwen3-VL-8B-Instruct
export IMAGE_URL=http://127.0.0.1:18003/photo.jpg

curl -sS "http://${HOST}:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"text\", \"text\": \"Describe this image.\"},
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"${IMAGE_URL}\"}}
      ]
    }],
    \"max_tokens\": 128,
    \"temperature\": 0
  }"
```

Install the client with `python -m pip install --upgrade openai`, then run:

```python
import os
from openai import OpenAI

host = os.environ.get("HOST", "127.0.0.1")
port = os.environ.get("PORT", "18001")
model = os.environ.get("MODEL", "Qwen3-VL-8B-Instruct")
image_url = os.environ.get(
    "IMAGE_URL",
    "http://127.0.0.1:18003/photo.jpg",
)
client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model=model,
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    }],
    max_tokens=128,
    temperature=0,
)
print(response.choices[0].message.content)
```

## Use a local image with Base64

Reuse `client` and `model` from the previous Python example. Encode an image on the client as a data URL; its MIME type must match the file. This example expects a JPEG named `photo.jpg`, no larger than 1024 × 1024 pixels. Larger images produce larger preprocessed tensors and can exceed the default 64 MiB internal RPC message limit; reduce the image resolution before sending it.

```python
import base64
from pathlib import Path

encoded = base64.b64encode(Path("photo.jpg").read_bytes()).decode("ascii")
response = client.chat.completions.create(
    model=model,
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this photo."},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{encoded}",
            }},
        ],
    }],
    max_tokens=128,
)
print(response.choices[0].message.content)
```

## Multiple images

For models that support multiple images, include several `image_url` parts in the same message. Reuse the client setup above and provide two local JPEG files, each no larger than 1024 × 1024 pixels:

```python
import base64
from pathlib import Path

content = [{"type": "text", "text": "Compare these two photos."}]
for filename in ["photo_1.jpg", "photo_2.jpg"]:
    encoded = base64.b64encode(Path(filename).read_bytes()).decode("ascii")
    content.append({"type": "image_url", "image_url": {
        "url": f"data:image/jpeg;base64,{encoded}",
    }})
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": content}],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

The server's `--limit_image_per_prompt` controls the maximum image count (default: `8`). Image count, resolution, and the model's context limit also constrain requests. See [Supported Models](/en/supported_models/) for model and hardware availability.
