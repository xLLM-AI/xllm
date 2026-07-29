---
title: "Flux2"
description: "Flux2 扩散模型推理实践指南占位页"
sidebar:
  order: 3
---

本章节用于汇总 Flux2 扩散模型在 xLLM 中的推理实践。

+ 源码地址：https://github.com/xLLM-AI/xllm

+ 国内可用: https://gitcode.com/xLLM-AI/xllm

+ 权重下载: [modelscope-FLUX.2-dev](https://www.modelscope.cn/models/black-forest-labs/FLUX.2-dev/)

## 1.拉取镜像环境

首先下载xLLM提供的镜像：

```bash
# A2 x86
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-x86-cann9-20260605
# A2 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-arm-cann9-20260605
# A3 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260605
```


然后创建对应的容器

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

## 2.拉取源码并编译

下载官方仓库与模块依赖：

```bash
git clone https://github.com/xLLM-AI/xllm.git
cd xllm 

```

下载安装依赖:

```bash
pip install pre-commit
pre-commit install

git submodule update --init --recursive
```

执行编译，在`build/`下生成可执行文件`build/xllm/core/server/xllm`：

```bash
python setup.py build
```

## 3.启动模型

### 若机器为重启后初次拉起服务，需先执行以下脚本对device进行初始化

若不执行且npu未初始化可能导致xllm进程拉起失败

```bash
python -c "import torch_npu
for i in range(16):torch_npu.npu.set_device(i)"
```

当前xLLM侧的Flux2拉起服务化是两阶段模式，需要分别拉起text-encoder组件和DiT组件服务化，再通过python的embedding脚本触发整个Flux2推理进程。另外，Flux2支持TP、SP和dit_cache特性（TaylorSeer、ResidualCache），暂时不支持chunked prefill特性。

### 1. 拉起text-encoder组件服务化
#### 环境变量

```bash
##### 1. 配置依赖路径相关环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

##### 2. 配置日志相关环境变量
rm -rf core.*
export ASCEND_MODULE_LOG_LEVEL=ATB=0
export ASDOPS_LOG_TO_FILE=1
export ASCEND_SLOG_PRINT_TO_STDOUT=1

##### 3. 配置性能、通信相关环境变量
export HCCL_IF_BASE_PORT=43432  # HCCL 通信基础端口
```

#### 启动命令 - 启动Flux2的text-encoder组件（单机1卡2die TP=2）

```bash
MODEL_PATH="/path/to/flux2/text_encoder/"          # text_encoder路径
MASTER_NODE_ADDR="127.0.0.1:9748"                  # Master 节点地址（需全局一致）
START_PORT=18001                                   # 服务起始端口
LOG_DIR="log"                                      # 日志目录
NNODES=2                                           # 节点数（当前脚本启动 1 个进程）

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
### 2. 拉起DiT组件服务化
#### 环境变量

```bash
##### 1. 配置依赖路径相关环境变量
export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/  # NPU 版 PyTorch 路径
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"  # PyTorch 安装路径
export LIBTORCH_ROOT="$PYTORCH_INSTALL_PATH"  # LibTorch 路径
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH  # 添加 NPU 库路径

source /usr/local/Ascend/ascend-toolkit/set_env.sh 
source /usr/local/Ascend/nnal/atb/set_env.sh

##### 2. 配置日志相关环境变量
\rm -rf core.*
\rm -rf log/dit_node_*.log

export ASDOPS_LOG_LEVEL=ERROR
export ASDOPS_LOG_TO_STDOUT=1
export ASDOPS_LOG_TO_FILE=1

##### 3. 配置性能、通信相关环境变量
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.98
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export OMP_NUM_THREADS=12
export HCCL_CONNECT_TIMEOUT=7200
export INF_NAN_MODE_ENABLE=0
export INF_NAN_MODE_FORCE_DISABLE=1
export HCCL_IF_BASE_PORT=43432  # HCCL 通信基础端口
```

#### 启动命令 - Flux2的DiT组件（单机 1卡2die TP=2）

```bash
MASTER_NODE_ADDR="127.0.0.1:8999"                  # Master 节点地址（需全局一致）
START_PORT=18018                                   # 服务起始端口
LOG_DIR="log"                                      # 日志目录
NNODES=2                                           # 节点数（当前脚本启动 2 个进程）

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
### 3. 执行python的embedding脚本

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
        convert_method (Callable, optional): 如果提供，会在读取后对 PIL.Image 做自定义转换并返回 PIL.Image。
            若为 None，则会默认调用 `.convert("RGB")`。
        device (torch.device, optional): 将返回 tensor 放到哪个 device（例如 torch.device('cuda')）。
            若为 None，则不做 device 转移（默认 CPU）。
        dtype (torch.dtype, optional): 返回 tensor 的 dtype（例如 torch.float32）。若为 None，则使用 torch.float32。

    Returns:
        torch.Tensor: shape [C, H, W], dtype float, values in [0,1], 在指定 device（若提供）。
    """
    # 读取为 PIL.Image
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

    # 处理 EXIF 方向
    pil_image = PIL.ImageOps.exif_transpose(pil_image)

    # 自定义转换或默认 RGB
    if convert_method is not None:
        pil_image = convert_method(pil_image)
    else:
        pil_image = pil_image.convert("RGB")

    # 转 numpy 再转 tensor；确保是 contiguous 并复制内存避免引用 PIL 缓冲
    np_img = np.asarray(pil_image, dtype=np.float32)  # H x W x C, float32
    # 若是灰度单通道，扩展通道
    if np_img.ndim == 2:
        np_img = np_img[:, :, None]
    if np_img.shape[2] == 4:
        # RGBA -> RGB（简单裁掉 alpha），若需按 alpha 合成请自定义 convert_method
        np_img = np_img[:, :, :3]

    # 归一化到 [0,1]
    np_img = np_img / 255.0

    tensor = torch.from_numpy(np_img).permute(2, 0, 1).contiguous()  # C x H x W

    # dtype & device
    target_dtype = dtype or torch.float32
    tensor = tensor.to(dtype=target_dtype)

    # clone 确保独立内存（可选，但保险）
    return tensor.clone()

def base64_to_image(base64_string, output_path):
    """
    将Base64字符串保存为图片文件
    
    Args:
        base64_string: Base64编码的字符串
        output_path: 输出图片路径（如：'output.jpg', 'output.png'）
    """
    try:
        # 解码Base64字符串
        image_data = base64.b64decode(base64_string)
        
        # 保存为文件
        with open(output_path, 'wb') as f:
            f.write(image_data)
            
        print(f"图片已保存到: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"转换失败: {e}")
        return None

def image_to_base64(img: Union[str, PIL.Image.Image]) -> str:
    """
    将图片文件路径或 PIL.Image 转成 Base64 字符串
    """
    if isinstance(img, str):
        pil_image = PIL.Image.open(img)
    elif isinstance(img, PIL.Image.Image):
        pil_image = img
    else:
        raise ValueError("img必须是文件路径或PIL.Image对象")

    # 转RGB
    pil_image = PIL.ImageOps.exif_transpose(pil_image)
    pil_image = pil_image.convert("RGB")

    # 保存到内存 buffer
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    byte_data = buf.getvalue()
    b64_str = base64.b64encode(byte_data).decode("utf-8")
    return b64_str

def create_tensor(data, name, datatype="FP32"):
        """
        构造符合proto::Tensor格式的字典（修复后：直接对应Proto的4个顶层字段）

        Args:
        data: numpy数组或Python列表，张量数据
        name: 张量名称（对应Proto的name字段）
        datatype: 数据类型（对应Proto的datatype字段），默认FP32

        Returns:
        dict: 完全匹配proto::Tensor结构的字典
        """
        # 转换为numpy数组以便获取形状
        if not isinstance(data, np.ndarray):
                data = np.array(data)

        # 1. 处理形状：确保为正整数（避免后端报无效维度错误）
        shape = list(data.shape)
        print(shape)
        if any(dim <= 0 for dim in shape):
                raise ValueError(f"张量{name}的形状包含非正整数：{shape}，需全部为正")

        # 2. 处理数据：展平后存入对应类型的contents字段
        contents = {}
        flat_data = data.flatten().tolist()
        if datatype == "FP32":
                contents["fp32_contents"] = flat_data # 对应Proto的TensorContents.fp32_contents
        elif datatype == "INT64":
                contents["int64_contents"] = flat_data
        elif datatype == "BOOL":
                contents["bool_contents"] = flat_data
        else:
                raise ValueError(f"不支持的数据类型：{datatype}，仅支持FP32/INT64/BOOL")

        # 3. 直接返回Proto要求的4个顶层字段（无多余层级）
        return {
        "name": name, # 顶层name字段
        "datatype": datatype, # 顶层datatype字段（修复报错的核心）
        "shape": shape, # 顶层shape字段（修复size[0]的核心）
        "contents": contents # 顶层contents字段
        }


def test_image_generation(pos_embed):
        """测试图像生成接口（使用修复后的Tensor结构）"""
        api_base = "http://127.0.0.1:18018"
        api_endpoint = f"{api_base}/v1/image/generation"
        model_name = "flux2"
        try:
                # 生成示例嵌入向量（形状需符合模型要求，此处保持原逻辑）
                pooled_prompt_embeds = np.random.rand(768).astype(np.float32) # 1D: [768]
                prompt_embeds = np.random.rand(2, 768).astype(np.float32) # 2D: [2, 768]

                ip_adapter_image_embeds = np.random.rand(1, 4, 768).astype(np.float32) # 3D: [1,4,768]
                latents = np.ones((1, 4, 32, 32), dtype=np.float32) # 4D: [1,4,32,32]（确保shape全部为正）

                # 2. 构造请求参数（Tensor结构已修复，其他逻辑不变
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
                "num_inference_steps": 50, # 注意：flux-schnell推荐4步，dev推荐50步，28步可能非最优
                "guidance_scale": 2.5,  # 这里需要和python侧设置一致
                "true_cfg_scale": 3.0,
                "num_images_per_prompt": 1,
                "seed": 42,
                "max_sequence_length": 2048
                },
                "user": "test_user",
                "service_request_id": f"req-{int(time.time())}"
                }
                print("python num_inference_steps:", 50)
                # 3. 发送请求（后续逻辑不变）
                headers = {"Content-Type": "application/json"}

                response = requests.post(
                url=api_endpoint,
                headers=headers,
                data=json.dumps(payload),
                timeout=60 * 5
                )

                response.raise_for_status()
                result = response.json()
                # 4. 解析响应（后续逻辑不变）
                print(f"接口响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
                # print(f"请求耗时: {time.time() - :.2f}s")
                if result.get("output") and result["output"].get("results"):
                        for idx, image_result in enumerate(result["output"]["results"]):
                                print(f"\n生成图片 {idx + 1}:")
                if image_result.get("url"):
                        print(f"URL: {image_result['url']}")
                elif image_result.get("image"):
                        print(f"尺寸: {image_result.get('width')}x{image_result.get('height')}")
                        base64_to_image(image_result['image'], "./result.png")
                else:
                        print(f"生成失败: {result.get('message', '未返回结果')}")

        except requests.exceptions.RequestException as e:
                print(f"请求异常: {str(e)}")
        except json.JSONDecodeError:
                print("响应格式错误，无法解析为JSON")
        except Exception as e:
                print(f"处理失败: {str(e)}")

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
    # 4. 构造 payload，将格式化后的字符串作为 input
    payload = {                                                                                                                                                   
        "model": "text_encoder",                                                                                                                                  
        "input": formatted_input,
        "encoding_format": "float"                                                                                                                                
    }

    # 发送给mistral的请求
    raw_response = requests.post("http://127.0.0.1:18001/v1/embeddings", json=payload)
    result = raw_response.json()

    # 解析向量
    bytes_data = result["data"][0]["mm_embeddings"][0]["embedding"]["contents"]["bytes_contents"]
    embed_data = base64.b64decode(bytes_data)
    embed_shape = result["data"][0]["mm_embeddings"][0]["embedding"]["shape"]
    pos_embed = torch.frombuffer(bytearray(embed_data), dtype=torch.bfloat16).reshape(embed_shape)

    test_image_generation(pos_embed)
    end = time.time()
    print(f"耗时: {end - start:.2f} 秒")

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