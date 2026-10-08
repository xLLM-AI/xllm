---
title: "OpenAI 接口：视觉输入"
---

使用视觉语言模型（VLM）的 `/v1/chat/completions` 接口可以对图像进行提问。在消息的 `content` 数组中传入文本和 `image_url` 内容块。通用客户端配置及流式输出请参阅 [OpenAI 兼容接口](/zh/getting_started/openai_api/)。

## 启动 VLM 服务

准备 xLLM 和 Qwen3-VL-8B-Instruct。以下示例在一张昇腾 NPU 上使用 Python/PyTorch 实现，并通过 ACL Graph 执行解码，请按[启动文档](/zh/getting_started/launch_xllm/)调整设备配置和模型大小。根据[多模态支持说明](/zh/features/multimodal/)，本例关闭前缀缓存和分块预填充。

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

请将此示例作为独立部署运行。如果同时运行多个服务，需要分别分配设备、master 地址和通信端口。等待模型加载完成后再发送请求。

## 使用图像 URL

使用 xLLM 服务端可直接访问的 HTTP 图像地址。当前下载器未配置 HTTPS/TLS，也不跟随重定向。对于仅提供 HTTPS 的图像源，请先在客户端下载，再使用下方的 Base64 示例。 下载器采用直连方式，因此仅通过终端 HTTP 代理可访问的 URL 仍可能失败。

为便于本地复现，在 xLLM 所在主机（使用容器时应在同一容器）的另一个终端运行以下命令，并在发送请求期间保持图片服务运行：

```bash
mkdir -p vision-images
curl -fL http://images.cocodataset.org/val2017/000000039769.jpg \
  -o vision-images/photo.jpg
python3 -m http.server 18003 --bind 127.0.0.1 --directory vision-images
```

下方回环地址由 xLLM 服务端解析。如需使用远程图片服务，请将 `IMAGE_URL` 替换为服务端可直接访问的 HTTP 地址。

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

通过 `python -m pip install --upgrade openai` 安装客户端，再运行：

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

## 使用 Base64 发送本地图像

沿用上一个 Python 示例中的 `client` 和 `model`，将客户端本地图像编码为 data URL。MIME 类型需要与文件格式一致；本例使用名为 `photo.jpg`、尺寸不超过 1024 × 1024 像素的 JPEG 文件。大图预处理后的张量可能超过内部 RPC 默认的 64 MiB 消息大小限制，请先降低图像分辨率再发送。

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

## 多图输入

对于支持多图输入的模型，可以在同一条消息中传入多个 `image_url` 内容块。沿用前面的客户端配置，并准备两张尺寸均不超过 1024 × 1024 像素的本地 JPEG 图像：

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

服务端通过 `--limit_image_per_prompt` 限制每个 prompt 的图像数量，默认值为 `8`。请求还受到图像数量、分辨率和模型上下文长度的约束。模型及硬件支持情况请参阅[模型支持列表](/zh/supported_models/)。
