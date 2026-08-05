## 备赛赛题：基于xLLM框架的推理模型性能优化
- 模型：[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- 推理框架：[xLLM lingji-competition分支](https://github.com/jd-opensource/xllm/tree/lingji-competition)
- 要求：在保证精度的前提下（详情见xLLM精度测试），使用各种优化方法优化Qwen3.5-4B的推理性能，尽可能提高 **Output Tokens per Second** (输出Tokens/秒， TPS) ，并提供优化方法的详细说明。
- 运行以下两种规模的请求，将二者的TPS相加，以总和TPS作为最终成绩，按照总和TPS大小进行排名：
1. 单并发，输入64k，输出1k
2. 单并发，输入128k，输出1k


## xLLM 开发手册

### 环境设置与编译
下载镜像：
```bash
docker pull <待定>
```

启动容器：
```bash
docker run -it \
--ipc=host \
-u 0 \
--name xllm-npu \
--privileged \
--network=host \
--device=/dev/davinci0 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /usr/local/sbin/:/usr/local/sbin/ \
-v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
-v /var/log/npu/slog/:/var/log/npu/slog \
-v /var/log/npu/profiling/:/var/log/npu/profiling \
-v /var/log/npu/dump/:/var/log/npu/dump \
-v $HOME:$HOME \
-w $HOME \
<待定> \
/bin/bash
```

编译xLLM：

镜像默认已将 xLLM 源码放在 `/opt/xllm`，直接进入该目录编译即可：
```bash
cd /opt/xllm
python setup.py build --device npu
```

如果镜像中没有 `/opt/xllm`，再重新拉取并编译：
```bash
git clone -b lingji-competition https://github.com/jd-opensource/xllm
cd xllm

# Install pre-commit for the first time
pip install pre-commit
pre-commit install

git submodule update --init --recursive

# Build xllm in docker container
python setup.py build --device npu
```

Qwen3.5 路径会依赖 TileLang 相关 kernel。不要手动删除或绕过 TileLang 生成产物，否则可能导致编译看似成功，但推理阶段失败。

如果删除了 `build` 目录，直接重新编译即可，只是耗时会回到完整编译级别：

```bash
python setup.py build --device npu
```

### 下载模型

```bash
# 安装modelscope
pip install modelscope

# 下载模型
modelscope download --model Qwen/Qwen3.5-4B --local_dir /path/to/model/Qwen3.5-4B
```


### xLLM server 启动

仓库提供了 [`qwen35_start.sh`](qwen35_start.sh) 启动脚本，已内置竞赛基线的推荐配置（非量化、单卡、关闭 prefix cache / chunked prefill / schedule overlap / shm）。完成编译后，通过环境变量指定模型路径等参数即可启动 OpenAI 兼容服务：

```bash
MODEL_PATH=/path/to/model/Qwen3.5-4B \
MODEL_ID=Qwen3.5-4B \
PORT=18000 \
VISIBLE_DEVICES=0 \
bash qwen35_start.sh
```

常用参数（可通过环境变量覆盖）：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `MODEL_PATH` | `/path/to/model/Qwen3.5-4B` | 模型目录 |
| `MODEL_ID` | `Qwen3.5-4B` | 对外暴露的模型名 |
| `PORT` | `18000` | 服务端口 |
| `VISIBLE_DEVICES` | `0` | 可见的 NPU 卡号 |
| `DEVICE` | `0` | 使用的逻辑设备号 |

如果编译产物不在默认路径 `/opt/xllm/build/lib.linux-aarch64-cpython-311/xllm/xllm`，可通过 `XLLM_BIN` 指定实际二进制：

```bash
XLLM_BIN=/opt/xllm/build/lib.linux-aarch64-cpython-311/xllm/xllm \
MODEL_PATH=/path/to/model/Qwen3.5-4B \
PORT=18000 \
VISIBLE_DEVICES=0 \
bash qwen35_start.sh
```

可以通过 `EXTRA_ARGS` 追加其它 xLLM 参数：

```bash
EXTRA_ARGS="--max_concurrent_requests=1" bash qwen35_start.sh
```


### xLLM client使用示例
```bash
curl -s "http://127.0.0.1:18000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer <API Key>" \
    -d '{
          "model": "Qwen3.5-4B",
          "messages": [
            {"role": "system", "content": "You are a user assistant."},
            {"role": "user", "content": "介绍下北京"}
          ],
          "top_p": 0.95,
          "temperature": 0.6,
          "top_k": -1,
          "stream": false
        }'
```

### xLLM 精度测试

完成以下请求的返回结果，与参考答案一致，则认为精度合格。

```bash
curl -s "http://127.0.0.1:18000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer <API Key>" \
    -d '{
          "model": "Qwen3.5-4B",
          "messages": [
            {"role": "system", "content": "You are a user assistant."},
            {"role": "user", "content": "Susie has $200 in her piggy bank. If she puts 20% more money into her piggy bank, how much money she will have?"}
          ],
          "top_p": 0.95,
          "temperature": 0.6,
          "top_k": -1,
          "stream": false
        }'

#参考答案：240
```

```bash
curl -s "http://127.0.0.1:18000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer <API Key>" \
    -d '{
          "model": "Qwen3.5-4B",
          "messages": [
            {"role": "system", "content": "You are a user assistant."},
            {"role": "user", "content": "A church has 120 members. 40% are adults. The rest are children. How many children more children are there than adults?"}
          ],
          "top_p": 0.95,
          "temperature": 0.6,
          "top_k": -1,
          "stream": false
        }'

#参考答案：24
```

```bash
curl -s "http://127.0.0.1:18000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer <API Key>" \
    -d '{
          "model": "Qwen3.5-4B",
          "messages": [
            {"role": "system", "content": "You are a user assistant."},
            {"role": "user", "content": "Mary used 15 gallons of fuel this week. Last week she used 20% less. How much fuel did she use in total for the two weeks?"}
          ],
          "top_p": 0.95,
          "temperature": 0.6,
          "top_k": -1,
          "stream": false
        }'

#参考答案：27
```

### xLLM 性能测试
下载测试数据集：
```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json -O ShareGPT_V3_unfiltered_cleaned_split.json
```

[test_xllm.py](test_xllm.py) 脚本用于测试xLLM的性能，使用示例：
```bash
python test_xllm.py \
    --backend xllm \
    --dataset-name random \
    --random-range-ratio 1 \
    --num-prompt 1 \
    --max-concurrency 1 \
    --random-input  65536 \
    --random-output 1024 \
    --host 127.0.0.1 \
    --port 18000 \
    --dataset-path /path/to/dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
    --model /path/to/model/Qwen3.5-4B
```

## 注意事项
- 禁止使用量化模型
- 只需要对文本token进行推理，不需要对图像token进行推理