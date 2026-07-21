---
title: "Flux2"
description: "CookBook recipe placeholder for Flux2 diffusion model inference"
sidebar:
  order: 3
---

This section will collect Flux2 diffusion model inference recipes for xLLM.

+ Source code: https://github.com/xLLM-AI/xllm

+ Available in China: https://gitcode.com/xLLM-AI/xllm

+ Weight download: [modelscope-FLUX.2-dev](https://www.modelscope.cn/models/black-forest-labs/FLUX.2-dev/)

## 1. Pull the Image Environment

First, download the image provided by xLLM:

```bash
# A2 x86
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-x86-cann9-20260605
# A2 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-arm-cann9-20260605
# A3 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260605
```

Then create the corresponding container:

```bash
IMAGE=quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260605
CONTAINER=mydocker

docker run \
--name $CONTAINER \
--privileged \
--network=host \
--ipc=host \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-v /export/home:/export/home \
-v /home/:/home/  \
-w /export/home \
-itd $IMAGE bash
docker exec -it $CONTAINER bash
```

## 2. Pull the Source Code and Build

Download the official repository and module dependencies:

```bash
git clone https://github.com/xLLM-AI/xllm.git
cd xllm

```

Download and install dependencies:

```bash
pip install pre-commit
pre-commit install

git submodule update --init --recursive
```

Run the build to generate the executable `build/xllm/core/server/xllm` under `build/`:

```bash
python setup.py build
```

## 3. Start the Model

### If the service is being started for the first time after the machine has rebooted, initialize the devices first

If this is skipped and the NPU has not been initialized, the xLLM process may fail to start.

```bash
python -c "import torch_npu
for i in range(16):torch_npu.npu.set_device(i)"
```

Flux2 serving in xLLM uses a two-stage mode: you need to start the text-encoder component and the DiT component separately, then trigger the full Flux2 inference pipeline via the Python embedding script. Additionally, Flux2 supports TP, SP, and dit_cache features (TaylorSeer, ResidualCache), but does not currently support the chunked prefill feature.

### 1. Start the text-encoder component
#### Environment Variables

```bash
##### 1. Configure dependency path environment variables
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

##### 2. Configure log-related environment variables
rm -rf core.*
export ASCEND_MODULE_LOG_LEVEL=ATB=0
export ASDOPS_LOG_TO_FILE=1
export ASCEND_SLOG_PRINT_TO_STDOUT=1

##### 3. Configure performance and communication-related environment variables
export HCCL_IF_BASE_PORT=43432  # HCCL communication base port
```

#### Startup Command - Start the Flux2 text-encoder component (single machine, 1 card with 2 dies, TP=2)

```bash
MODEL_PATH="/path/to/flux2/text_encoder/"          # text_encoder path
MASTER_NODE_ADDR="127.0.0.1:9748"                  # Master node address (must be consistent globally)
START_PORT=18001                                   # Service starting port
LOG_DIR="log"                                      # Log directory
NNODES=2                                           # Number of nodes (this script starts 1 process)

mkdir -p $LOG_DIR

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  LOG_FILE="$LOG_DIR/mistral_node_$i.log"
  ./build/xllm/core/server/xllm \
    --model $MODEL_PATH \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --max_memory_utilization=0.86 \
    --block_size=128 \
    --tp_size=2 \
    --communication_backend="hccl" \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=true \
    --enable_return_mm_full_embeddings=true \
    --enable_mistral_prompt_to_message=true \
    --task="embed" \
    --enable_shm=true \
    --node_rank=$i \ > $LOG_FILE 2>&1 &
done
```
### 2. Start the DiT component
#### Environment Variables

```bash
##### 1. Configure dependency path environment variables
export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/  # NPU PyTorch path
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"  # PyTorch installation path
export LIBTORCH_ROOT="$PYTORCH_INSTALL_PATH"  # LibTorch path
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH  # Add NPU library path

source /usr/local/Ascend/ascend-toolkit/set_env.sh 
source /usr/local/Ascend/nnal/atb/set_env.sh

##### 2. Configure log-related environment variables
\rm -rf core.*
\rm -rf log/dit_node_*.log

export ASDOPS_LOG_LEVEL=ERROR
export ASDOPS_LOG_TO_STDOUT=1
export ASDOPS_LOG_TO_FILE=1

##### 3. Configure performance and communication-related environment variables
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.98
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export OMP_NUM_THREADS=12
export HCCL_CONNECT_TIMEOUT=7200
export INF_NAN_MODE_ENABLE=0
export INF_NAN_MODE_FORCE_DISABLE=1
export HCCL_IF_BASE_PORT=43432  # HCCL communication base port
```

#### Startup Command - Flux2 DiT component (single machine, 1 card with 2 dies, TP=2)

```bash
MASTER_NODE_ADDR="127.0.0.1:8999"                  # Master node address (must be consistent globally)
START_PORT=18018                                   # Service starting port
LOG_DIR="log"                                      # Log directory
NNODES=2                                           # Number of nodes (this script starts 2 processes)

for (( i=0; i<2; i++ ))
do
  PORT=$((START_PORT + i))
  LOG_FILE="$LOG_DIR/dit_node_$i.log"
  ./build/xllm/core/server/xllm \
    --model="/path/to/flux2/" \
    --max_memory_utilization=0.6 \
    --backend="dit" \
    --tp_size=2 \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --port $PORT \
    --communication_backend="hccl" \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=false \
    --use_contiguous_input_buffer=false \
    --dit_debug_print=true \
    --enable-shm=true \
    --node_rank=$i > $LOG_FILE 2>&1 &
done
```
### 3. Run the Python embedding script

```bash
# -*- coding: utf-8 -*-
import sys
import json
from typing import Callable, Optional, Union
from safetensors.torch import load_file
import torch
import os
import base64
import requests
import argparse
import PIL.Image
import PIL.ImageOps
import torch
import math
import io
import numpy as np
import time

from transformers import AutoProcessor, AutoTokenizer
CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024

def load_tensor(
    image: Union[str, PIL.Image.Image],
    convert_method: Optional[Callable[[PIL.Image.Image], PIL.Image.Image]] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Load `image` (URL / local path / PIL.Image) and convert to torch.Tensor.

    Args:
        image (str or PIL.Image.Image): URL (http/https) or filesystem path, or PIL Image.
        convert_method (Callable, optional): If provided, apply a custom conversion to the PIL.Image after reading.
            If None, `.convert("RGB")` is called by default.
        device (torch.device, optional): The device to place the tensor on (e.g. torch.device('cuda')).
            If None, no device transfer is performed (defaults to CPU).
        dtype (torch.dtype, optional): The dtype of the returned tensor (e.g. torch.float32). If None, torch.float32 is used.

    Returns:
        torch.Tensor: shape [C, H, W], dtype float, values in [0,1], on the specified device (if provided).
    """
    # Read as PIL.Image
    if isinstance(image, str):
        if os.path.isfile(image):
            pil_image = PIL.Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or URL. URLs must start with `http://` or `https://`, and {image} is not a valid path."
            )
    elif isinstance(image, PIL.Image.Image):
        pil_image = image
    else:
        raise ValueError(
            "Incorrect format used for the image. Should be a URL linking to an image, a local path, or a PIL image."
        )

    # Handle EXIF orientation
    pil_image = PIL.ImageOps.exif_transpose(pil_image)

    # Custom conversion or default to RGB
    if convert_method is not None:
        pil_image = convert_method(pil_image)
    else:
        pil_image = pil_image.convert("RGB")

    # Convert to numpy then to tensor; ensure contiguous and copy memory to avoid referencing PIL buffer
    np_img = np.asarray(pil_image, dtype=np.float32)  # H x W x C, float32
    # Expand channel if grayscale
    if np_img.ndim == 2:
        np_img = np_img[:, :, None]
    if np_img.shape[2] == 4:
        # RGBA -> RGB (simply strip alpha); customize convert_method for alpha compositing
        np_img = np_img[:, :, :3]

    # Normalize to [0,1]
    np_img = np_img / 255.0

    tensor = torch.from_numpy(np_img).permute(2, 0, 1).contiguous()  # C x H x W

    # dtype & device
    target_dtype = dtype or torch.float32
    tensor = tensor.to(dtype=target_dtype)

    # Clone to ensure independent memory (optional but safe)
    return tensor.clone()

def base64_to_image(base64_string, output_path):
    """
    Save a Base64 string as an image file.
    
    Args:
        base64_string: Base64-encoded string
        output_path: Output image path (e.g. 'output.jpg', 'output.png')
    """
    try:
        # Decode Base64 string
        image_data = base64.b64decode(base64_string)
        
        # Save to file
        with open(output_path, 'wb') as f:
            f.write(image_data)
            
        print(f"Image saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        return None

def image_to_base64(img: Union[str, PIL.Image.Image]) -> str:
    """
    Convert an image file path or PIL.Image to a Base64 string.
    """
    if isinstance(img, str):
        pil_image = PIL.Image.open(img)
    elif isinstance(img, PIL.Image.Image):
        pil_image = img
    else:
        raise ValueError("img must be a file path or PIL.Image object")

    # Convert to RGB
    pil_image = PIL.ImageOps.exif_transpose(pil_image)
    pil_image = pil_image.convert("RGB")

    # Save to in-memory buffer
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    byte_data = buf.getvalue()
    b64_str = base64.b64encode(byte_data).decode("utf-8")
    return b64_str

def create_tensor(data, name, datatype="FP32"):
        """
        Construct a dictionary conforming to the proto::Tensor format (4 top-level fields).

        Args:
        data: numpy array or Python list, tensor data
        name: tensor name (corresponds to the Proto name field)
        datatype: data type (corresponds to the Proto datatype field), default FP32

        Returns:
        dict: dictionary fully matching the proto::Tensor structure
        """
        # Convert to numpy array to get shape
        if not isinstance(data, np.ndarray):
                data = np.array(data)

        # 1. Handle shape: ensure all dimensions are positive (avoid backend invalid dimension errors)
        shape = list(data.shape)
        print(shape)
        if any(dim <= 0 for dim in shape):
                raise ValueError(f"Tensor {name} has non-positive dimensions: {shape}, all must be positive")

        # 2. Handle data: flatten and store in the corresponding typed contents field
        contents = {}
        flat_data = data.flatten().tolist()
        if datatype == "FP32":
                contents["fp32_contents"] = flat_data # Corresponds to Proto TensorContents.fp32_contents
        elif datatype == "INT64":
                contents["int64_contents"] = flat_data
        elif datatype == "BOOL":
                contents["bool_contents"] = flat_data
        else:
                raise ValueError(f"Unsupported data type: {datatype}, only FP32/INT64/BOOL are supported")

        # 3. Return the 4 top-level fields required by Proto (no extra nesting)
        return {
        "name": name, # Top-level name field
        "datatype": datatype, # Top-level datatype field
        "shape": shape, # Top-level shape field
        "contents": contents # Top-level contents field
        }


def test_image_generation(pos_embed):
        """Test the image generation API (using the corrected Tensor structure)"""
        api_base = "http://127.0.0.1:18018"
        api_endpoint = f"{api_base}/v1/image/generation"
        model_name = "flux2"
        try:
                # Generate example embedding vectors (shape must match model requirements)
                pooled_prompt_embeds = np.random.rand(768).astype(np.float32) # 1D: [768]
                prompt_embeds = np.random.rand(2, 768).astype(np.float32) # 2D: [2, 768]

                ip_adapter_image_embeds = np.random.rand(1, 4, 768).astype(np.float32) # 3D: [1,4,768]
                latents = np.ones((1, 4, 32, 32), dtype=np.float32) # 4D: [1,4,32,32] (ensure all shape dims are positive)

                # 2. Construct request payload (Tensor structure corrected)
                payload = {
                "model": model_name,
                "input": {
                "prompt": "A cat holding a sign that says hello world",
                "prompt_2": "",
                "negative_prompt": " ",
                "negative_prompt_2": "",
                "prompt_embed": create_tensor(
                   pos_embed.to(torch.float32),
                   name = "prompt_embeds",
                   datatype="FP32"
                )
                },
                "parameters": {
                "size": "1024*1024",
                "num_inference_steps": 50, # Note: flux-schnell recommends 4 steps, dev recommends 50; 28 may not be optimal
                "guidance_scale": 2.5,  # Must be consistent with the Python-side setting
                "true_cfg_scale": 3.0,
                "num_images_per_prompt": 1,
                "seed": 42,
                "max_sequence_length": 2048
                },
                "user": "test_user",
                "service_request_id": f"req-{int(time.time())}"
                }
                print("python num_inference_steps:", 50)
                # 3. Send request
                headers = {"Content-Type": "application/json"}

                response = requests.post(
                url=api_endpoint,
                headers=headers,
                data=json.dumps(payload),
                timeout=60 * 5
                )

                response.raise_for_status()
                result = response.json()
                # 4. Parse response
                print(f"API response: {json.dumps(result, indent=2, ensure_ascii=False)}")
                if result.get("output") and result["output"].get("results"):
                        for idx, image_result in enumerate(result["output"]["results"]):
                                print(f"\nGenerated image {idx + 1}:")
                if image_result.get("url"):
                        print(f"URL: {image_result['url']}")
                elif image_result.get("image"):
                        print(f"Size: {image_result.get('width')}x{image_result.get('height')}")
                        base64_to_image(image_result['image'], "./result.png")
                else:
                        print(f"Generation failed: {result.get('message', 'No results returned')}")

        except requests.exceptions.RequestException as e:
                print(f"Request error: {str(e)}")
        except json.JSONDecodeError:
                print("Response format error, unable to parse as JSON")
        except Exception as e:
                print(f"Processing failed: {str(e)}")

def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height

def main(args: argparse.Namespace):
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained("/path/to/flux2/text_encoder/")
    messages = [
        {"role": "system", "content": "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object\nattribution and actions without speculation."},
        {"role": "user", "content": "A cat holding a sign that says hello world"},
    ]
    formatted_input = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )                                                                                                                                                             
    # 4. Construct payload with the formatted string as input
    payload = {                                                                                                                                                   
        "model": "text_encoder",                                                                                                                                  
        "input": formatted_input,
        "encoding_format": "float"                                                                                                                                
    }

    # Send request to the text-encoder
    raw_response = requests.post("http://127.0.0.1:18001/v1/embeddings", json=payload)
    result = raw_response.json()

    # Parse embeddings
    bytes_data = result["data"][0]["mm_embeddings"][0]["embedding"]["contents"]["bytes_contents"]
    embed_data = base64.b64decode(bytes_data)
    embed_shape = result["data"][0]["mm_embeddings"][0]["embedding"]["shape"]
    pos_embed = torch.frombuffer(bytearray(embed_data), dtype=torch.bfloat16).reshape(embed_shape)

    test_image_generation(pos_embed)
    end = time.time()
    print(f"Elapsed time: {end - start:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18001)
    parser.add_argument("--model", type=str, default="text_encoder")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    args = parser.parse_args()
    main(args)
```