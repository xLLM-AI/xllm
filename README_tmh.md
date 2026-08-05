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
docker pull quay.io/ascend/xllm:a3-arm-dev-010-cache-mtp
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
quay.io/ascend/xllm:a3-arm-dev-010-cache-mtp \
/bin/bash
```

编译xLLM：
```bash
git clone -b lingji-competition https://github.com/jd-opensource/xllm
cd xllm

# Install pre-commit for the first time
pip install pre-commit
pre-commit install

git submodule update --init --recursive

# Build xllm in docker container
python setup.py build
```

### 下载模型

```bash
# 安装modelscope
pip install modelscope

# 下载模型
modelscope download --model Qwen/Qwen3.5-4B --local_dir /path/to/model/Qwen3.5-4B
```


### xLLM server启动脚本
```bash
#!/bin/bash
set -e

rm -rf core.*

source /usr/local/Ascend/ascend-toolkit/set_env.sh 
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export HCCL_IF_BASE_PORT=43432  # HCCL communication base port


MODEL_PATH="/path/to/model/Qwen3.5-4B"               # Model path
MASTER_NODE_ADDR="127.0.0.1:9748"                  # Master node address (must be globally consistent)
START_PORT=18000                                   # Service starting port
START_DEVICE=0                                     # Starting logical device number
LOG_DIR="log"                                      # Log directory
NNODES=1                                           # Number of nodes (current script launches 1 process)

mkdir -p $LOG_DIR

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  /path/to/xllm \
    --model $MODEL_PATH \
    --devices="npu:$DEVICE" \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --max_memory_utilization=0.6 \
    --block_size=128 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=false \
    --enable_shm=false \
    --node_rank=$i \ > $LOG_FILE 2>&1 &
done
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