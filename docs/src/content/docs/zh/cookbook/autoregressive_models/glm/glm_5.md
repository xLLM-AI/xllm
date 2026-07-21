---
title: "GLM-5 / GLM-5.1 / GLM-5.2"
sidebar:
  order: 2
---

+ 源码地址：https://github.com/xLLM-AI/xllm

+ 国内可用: https://gitcode.com/xLLM-AI/xllm

+ 权重下载:

  [modelscope-GLM-5-W8A8](https://www.modelscope.cn/models/Eco-Tech/GLM-5-W8A8-xLLM-0403/files)

  [modelscope-GLM-5.1-W8A8](https://www.modelscope.cn/models/Eco-Tech/GLM-5.1-W8A8-xLLM/files)

  [modelscope-GLM-5.1-W4A8](https://www.modelscope.cn/models/Eco-Tech/GLM-5.1-w4a8)

  [modelscope-GLM-5.2-W8A8](https://www.modelscope.cn/models/Eco-Tech/GLM-5.2-W8A8/files)

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

**注意**: A2 机器性能未进行压测。

然后创建对应的容器

```bash
sudo docker run -it --ipc=host -u 0 --privileged --name mydocker --network=host \
 -v /var/queue_schedule:/var/queue_schedule \
 -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
 -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
 -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
 -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
 -v /var/log/npu/slog/:/var/log/npu/slog \
 -v ~/.ssh:/root/.ssh  \
 -v /var/log/npu/profiling/:/var/log/npu/profiling \
 -v /var/log/npu/dump/:/var/log/npu/dump \
 -v /runtime/:/runtime/ -v /etc/hccn.conf:/etc/hccn.conf \
 -v /export/home:/export/home \
 -v /home/:/home/  \
 -w /export/home \
 quay.io/jd_xllm/xllm-ai:xllm-dev-hb-rc2-x86
```

## 2.拉取源码并编译

下载官方仓库与模块依赖：

```bash
git clone https://github.com/xLLM-AI/xllm.git
cd xllm
git checkout release/v0.10.0
git submodule update --init --recursive
```

下载安装依赖:

```bash
pip install --upgrade pre-commit
yum install numactl
```

执行编译，在`build/`下生成可执行文件`build/xllm/core/server/xllm`：

```bash
python setup.py build --device npu
```

## 3.启动模型

### 若机器为重启后初次拉起服务，需先执行以下脚本对device进行初始化

#若不执行且npu未初始化可能导致xllm进程拉起失败

```bash
python -c "import torch_npu
for i in range(16):torch_npu.npu.set_device(i)"
```

### 导出MTP权重

```bash
python tools/export_mtp.py --input-dir ${W4A8/W8A8权重目录} --output-dir ${导出MTP权重目录}
```

### 启动准备

```bash
##### 1， 配置相关环境变量
export LD_PRELOAD=/usr/lib64/libtcmalloc.so.4:$LD_PRELOAD
export HCCL_EXEC_TIMEOUT=300
export HCCL_CONNECT_TIMEOUT=300
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_BASE_PORT=2864

##### 2， 清除残留日志
rm -rf /root/ascend/log/
```

## 启动命令 - A3单机 - GLM-5.2-W8A8

```bash
XLLM_PATH="./myxllm/xllm/build/xllm/core/server/xllm"
# xllm可执行文件路径
MODEL_PATH=/path/to/GLM-5.2-W8A8/
# 模型路径（以Glm-5.2-w8a8为例）
DRAFT_MODEL_PATH=/path/to/GLM-5.2-MTP/
# 前面导出的mtp权重

MASTER_NODE_ADDR="11.87.49.110:10015"
LOCAL_HOST="11.87.49.110"
# Service Port
START_PORT=18994
START_DEVICE=0
LOG_DIR="logs"
NNODES=16

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  #可选：numactl绑核 (NUMA亲和性查询命令： npu-smi info -t topo)
  #nohup numactl -C $((DEVICE*40))-$((DEVICE*40+39)) $XLLM_PATH \
  nohup $XLLM_PATH \
    --model $MODEL_PATH \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --node_rank=$i \
    --max_memory_utilization=0.86 \
    --max_tokens_per_batch=4096 \
    --max_seqs_per_batch=16 \
    --block_size=128 \
    --enable_prefix_cache=true \
    --enable_chunked_prefill=true \
    --enable_graph=true \
    --enable_schedule_overlap=true \
    --communication_backend="hccl" \
    --graph_decode_batch_size_limit=2 \
    --draft_model=$DRAFT_MODEL_PATH \
    --num_speculative_tokens=3 \
    --ep_size=16 \
    --dp_size=2 \
    --tool_call_parser=auto \
    > $LOG_FILE 2>&1 &
done

# --max_memory_utilization   单卡最大显存占用比例
# --max_tokens_per_batch     单batch最大token数  （主要限制prefill）
# --max_seqs_per_batch       单batch最大请求数   （主要限制decode）
# --communication_backend    通信backend 可选(hccl / lccl) 此处建议hccl
# --enable_schedule_overlap  开启异步调度
# --enable_prefix_cache      开启prefix_cache
# --enable_chunked_prefill   开启chunked_prefill
# --enable_graph             开启aclgraph，需要额外显存
# --acl_graph_decode_batch_size_limit    抓图的最大bs，当前需<= 32 / (预测token数 + 1)
# --draft_model              mtp - mtp权重路径
# --num_speculative_tokens   mtp - 预测token数
```

日志出现"Brpc Server Started"表示服务成功拉起。

## 其他可选环境变量

```bash
#开启确定性计算
export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=true
export ATB_MATMUL_SHUFFLE_K_ENABLE=0

# 开启动态profiling模式
export PROFILING_MODE=dynamic
\rm -rf ~/dynamic_profiling_socket_*
```

## 启动命令 - A3双机拉起样例

### Node0 (master)

```bash
MASTER_NODE_ADDR="11.87.49.110:19990"
LOCAL_HOST="11.87.49.110"
START_PORT=15890
START_DEVICE=0
LOG_DIR="logs"
NNODES=32
LOCAL_NODES=16
export HCCL_IF_BASE_PORT=48439
unset HCCL_OP_EXPANSION_MODE

for (( i=0; i<$LOCAL_NODES; i++ ))do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i));  LOG_FILE="$LOG_DIR/node_$i.log"
  nohup numactl -C $((DEVICE*40))-$((DEVICE*40+39)) $XLLM_PATH \
    --model $MODEL_PATH \
    --host $LOCAL_HOST \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --node_rank=$i \
    --max_memory_utilization=0.85 \
    --max_tokens_per_batch=8192 \
    --max_seqs_per_batch=128 \
    --block_size=128 \
    --enable_prefix_cache=true \
    --enable_chunked_prefill=true \
    --communication_backend="hccl" \
    --enable_schedule_overlap=true \
    --enable_graph=true \
    --acl_graph_decode_batch_size_limit=4 \
    --draft_model=$DRAFT_MODEL_PATH \
    --num_speculative_tokens=3 \
    --ep_size=32 \
    --dp_size=4 \
    --rank_tablefile=/yourPath/ranktable.json \
    --tool_call_parser=auto \
    > $LOG_FILE 2>&1 &
done
```

### Node1 (worker)

```bash
MASTER_NODE_ADDR="11.87.49.110:19990"
LOCAL_HOST="11.87.49.111"
START_PORT=15890
START_DEVICE=0
LOG_DIR="logs"
NNODES=32
LOCAL_NODES=16
export HCCL_IF_BASE_PORT=48439
unset HCCL_OP_EXPANSION_MODE

for (( i=0; i<$LOCAL_NODES; i++ ))do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i));  LOG_FILE="$LOG_DIR/node_$i.log"
  nohup numactl -C $((DEVICE*40))-$((DEVICE*40+39)) $XLLM_PATH \
    --model $MODEL_PATH \
    --host $LOCAL_HOST \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --node_rank=$((i + LOCAL_NODES)) \
    --max_memory_utilization=0.85 \
    --max_tokens_per_batch=8192 \
    --max_seqs_per_batch=128 \
    --block_size=128 \
    --enable_prefix_cache=true \
    --enable_chunked_prefill=true \
    --communication_backend="hccl" \
    --enable_schedule_overlap=true \
    --enable_graph=true \
    --acl_graph_decode_batch_size_limit=4 \
    --draft_model=$DRAFT_MODEL_PATH \
    --num_speculative_tokens=3 \
    --ep_size=32 \
    --dp_size=4 \
    --rank_tablefile=/yourPath/ranktable.json \
    --tool_call_parser=auto \
    > $LOG_FILE 2>&1 &
done
```

### ranktable样例

 [A3 ranktable配置](https://www.hiascend.com/document/detail/zh/canncommercial/900/API/hcclug/hcclug_000066.html)

 [A2 ranktable配置](https://www.hiascend.com/document/detail/zh/canncommercial/900/API/hcclug/hcclug_000067.html)

 （注意A3与A2的ranktable格式差异）

## device NUMA亲和性查看

命令：

```bash
npu-smi info -t topo
```

前述命令中

```bash
numactl -C $((DEVICE*12))-$((DEVICE*12+11))
```

表示该进程绑在对应亲和的核上，可根据机器具体情况修改绑定的核id

## EX3.Glm-5 权重量化 (GLM5.2 量化指导待更新)

### 安装msmodelslim

```bash
pip install transformers==5.2.0

git clone https://gitcode.com/Ascend/msmodelslim.git
cd msmodelslim
bash install.sh
```
### 量化执行
```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu:0 \
  --model_type GLM-5 \
  --quant_type w8a8 \
  --trust_remote_code True
```

## PD分离

### etcd\xllm-service 安装

#### PD分离部署

`xllm`支持PD分离部署，这需要与另一个开源库[xllm service](https://github.com/xLLM-AI/xllm-service)配套使用。

##### xLLM Service依赖

首先，我们下载安装`xllm service`，与安装编译`xllm`类似：

```bash
git clone https://github.com/xLLM-AI/xllm-service.git
cd xllm-service
git submodule init
git submodule update
```

##### etcd安装

`xllm_service`依赖[etcd](https://github.com/etcd-io/etcd)，使用etcd官方提供的[安装脚本](https://github.com/etcd-io/etcd/releases)进行安装，其脚本提供的默认安装路径是`/tmp/etcd-download-test/etcd`，我们可以手动修改其脚本中的安装路径，也可以运行完脚本之后手动迁移：

```bash
mv /tmp/etcd-download-test/etcd /path/to/your/etcd
```

##### xLLM Service编译

先应用patch:

```bash
sh prepare.sh
```

再执行编译:

```bash
mkdir -p build
cd build
cmake ..
make -j 8
cd ..
```

!!! warning "可能的错误"
    这里能会遇到关于`boost-locale`和`boost-interprocess`的安装错误：`vcpkg-src/packages/boost-locale_x64-linux/include: No such     file or directory`,`/vcpkg-src/packages/boost-interprocess_x64-linux/include: No such file or directory`
    我们使用`vcpkg`重新安装这些包:
    ```bash
    /path/to/vcpkg remove boost-locale boost-interprocess
    /path/to/vcpkg install boost-locale:x64-linux
    /path/to/vcpkg install boost-interprocess:x64-linux
    ```

### PD分离运行

启动etcd:

```bash
./etcd-download-test/etcd --listen-peer-urls 'http://localhost:2390'  --listen-client-urls 'http://localhost:2389' --advertise-client-urls  'http://localhost:2391'
```

跨机配置时，etcd参考如下：

```bash
/tmp/etcd-download-test/etcd --listen-peer-urls 'http://0.0.0.0:3390' --listen-client-urls 'http://0.0.0.0:3389' --advertise-client-urls 'http://11.87.191.82:3389'
```

启动xllm service:

```bash
ENABLE_DECODE_RESPONSE_TO_SERVICE=true ./xllm_master_serving --etcd_addr="127.0.0.1:12389" --http_server_port 28888 --rpc_server_port 28889 --tokenizer_path=/export/home/models/GLM-5-W8A8/
```

跨机配置时，启动xllm service:

```bash
ENABLE_DECODE_RESPONSE_TO_SERVICE=true ../xllm-service/build/xllm_service/xllm_master_serving --etcd_addr="11.87.191.82:3389" --http_server_port 38888 --rpc_server_port 38889 --tokenizer_path=/export/home/models/GLM-5-W8A8/
```
- 启动Prefill实例
```bash
  BATCH_SIZE=256
  #推理最大batch数量
  XLLM_PATH="./myxllm/xllm/build/xllm/core/server/xllm"
  #推理入口文件路径（上一步中编译产物）
  MODEL_PATH=/export/home/models/GLM-5-w8a8/
  #模型路径（此处为int量化的Glm-5）
  DRAFT_MODEL_PATH=/export/home/models/GLM-5-MTP/

  MASTER_NODE_ADDR="11.87.49.110:10015"
  LOCAL_HOST="11.87.49.110"
  # Service Port
  START_PORT=18994
  START_DEVICE=0
  LOG_DIR="logs"
  NNODES=16

  for (( i=0; i<$NNODES; i++ ))
  do
    PORT=$((START_PORT + i))
    DEVICE=$((START_DEVICE + i))
    LOG_FILE="$LOG_DIR/node_$i.log"
    nohup numactl -C $((i*40))-$((i*40+39)) $XLLM_PATH \
      --model $MODEL_PATH  --model_id glmmoe \
      --host $LOCAL_HOST \
      --port $PORT \
      --master_node_addr=$MASTER_NODE_ADDR \
      --nnodes=$NNODES \
      --node_rank=$i \
      --max_memory_utilization=0.86 \
      --max_tokens_per_batch=5000 \
      --max_seqs_per_batch=$BATCH_SIZE \
      --communication_backend=hccl \
      --enable_schedule_overlap=true \
      --enable_prefix_cache=false \
      --enable_chunked_prefill=false \
      --enable_graph=true \
      --draft_model $DRAFT_MODEL_PATH \
      --num_speculative_tokens 1 \
      --tool_call_parser=auto \
      --enable_disagg_pd=true \
      --instance_role=PREFILL \
      --etcd_addr=$LOCAL_HOST:3389 \
      --transfer_listen_port=$((36100 + i)) \
      --disagg_pd_port=8877 \
      > $LOG_FILE 2>&1 &
  done

  #--etcd_addr=$LOCAL_HOST:3389  参考etcd中advertise-client-urls的配置
  #--instance_role=DECODE     PD配置，DECODE\PREFILL
  ```

- 启动Decode实例

  ```bash
    BATCH_SIZE=256
  #推理最大batch数量
  XLLM_PATH="./myxllm/xllm/build/xllm/core/server/xllm"
  #推理入口文件路径（上一步中编译产物）
  MODEL_PATH=/export/home/models/GLM-5-w8a8/
  #模型路径（此处为int量化的Glm-5）
  DRAFT_MODEL_PATH=/export/home/models/GLM-5-MTP/

  MASTER_NODE_ADDR="11.87.49.110:10015"
  LOCAL_HOST="11.87.49.110"
  # Service Port
  START_PORT=18994
  START_DEVICE=0
  LOG_DIR="logs"
  NNODES=16

  for (( i=0; i<$NNODES; i++ ))
  do
    PORT=$((START_PORT + i))
    DEVICE=$((START_DEVICE + i))
    LOG_FILE="$LOG_DIR/node_$i.log"
    nohup numactl -C $((i*40))-$((i*40+39)) $XLLM_PATH \
      --model $MODEL_PATH  --model_id glmmoe \
      --host $LOCAL_HOST \
      --port $PORT \
      --master_node_addr=$MASTER_NODE_ADDR \
      --nnodes=$NNODES \
      --node_rank=$i \
      --max_memory_utilization=0.86 \
      --max_tokens_per_batch=5000 \
      --max_seqs_per_batch=$BATCH_SIZE \
      --communication_backend=hccl \
      --enable_schedule_overlap=true \
      --enable_prefix_cache=false \
      --enable_chunked_prefill=false \
      --enable_graph=true \
      --draft_model $DRAFT_MODEL_PATH \
      --num_speculative_tokens 1 \
      --tool_call_parser=auto \
      --enable_disagg_pd=true \
      --instance_role=DECODE \
      --etcd_addr=$LOCAL_HOST:3389 \
      --transfer_listen_port=$((36100 + i)) \
      --disagg_pd_port=8877 \
      > $LOG_FILE 2>&1 &
  done

  #--etcd_addr=$LOCAL_HOST:3389  参考etcd中advertise-client-urls的配置
  #--instance_role=DECODE     PD配置，DECODE\PREFILL
  ```

  需要注意：

- PD分离需要读取`/etc/hccn.conf`文件，确保将物理机上的该文件映射到了容器中

- `etcd_addr`需与`xllm_service`的`etcd_addr`相同
  测试命令和上面类似，注意`curl http://localhost:{PORT}/v1/chat/completions ...`的`PORT`选择为启动xLLM service的`http_server_port`。

- 多机部署P或者Q时(例如部署两个P)，需要增加--rank_tablefile来完成通信。
