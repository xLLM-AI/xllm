# PD分离
## 背景
LLM在线推理服务通常需要满足TTFT和TPOT两项性能指标，而传统的Contiguous Batching调度策略将Prefill和Decode请求混合在一起调度，导致P和D会互相抢占计算资源，影响性能指标无法最大程度的利用计算资源。为解决上述矛盾，将Prefill和Decode两阶段拆分到独立的计算资源并行执行，从而同时降低TTFT和TPOT并提升吞吐量。
## 功能介绍
xLLM PD分离功能主要通过以下三个模块实现：

- **etcd**: 存储实例信息等元数据
- **xLLM Service**: 调度请求和管理所有计算实例
- **xLLM**: 请求计算实例

整体架构图如下：
![xLLM PD分离架构图](../../assets/pd_architecture.jpg)
## 功能使用示例
### 使用准备
#### 安装相关依赖
- **xLLM**: 参见[安装编译](../getting_started/quick_start.md)
- **xLLM Service**: 参见[PD分离部署](../getting_started/disagg_pd.md)

### 启动PD分离服务
1. 启动etcd
```
./etcd
```
2. 启动xLLM Service
```bash
ENABLE_DECODE_RESPONSE_TO_SERVICE=true ./xllm_master_serving --etcd_addr="127.0.0.1:12389" --http_server_port 28888 --rpc_server_port 28889 --tokenizer_path=/path/to/tokenizer_config_dir/
```
3. 启动xLLM

4. 以Qwen2-7B为例
    - 启动Prefill实例
        ```bash
        /path/to/xllm --model=Qwen2-7B-Instruct \
               --port=8010 \
               --devices="npu:0" \
               --master_node_addr="127.0.0.1:18888" \
               --enable_prefix_cache=false \
               --enable_chunked_prefill=false \
               --enable_disagg_pd=true \
               --instance_role=PREFILL \
               --etcd_addr=127.0.0.1:12389 \
               --transfer_listen_port=26000 \
               --disagg_pd_port=7777 \
               --node_rank=0 \
               --nnodes=1
        ```
    - 启动Decode实例
        ```bash
        /path/to/xllm --model=Qwen2-7B-Instruct \
               --port=8020 \
               --devices="npu:1" \
               --master_node_addr="127.0.0.1:18898" \
               --enable_prefix_cache=false \
               --enable_chunked_prefill=false \
               --enable_disagg_pd=true \
               --instance_role=DECODE \
               --etcd_addr=127.0.0.1:12389 \
               --transfer_listen_port=26100 \
               --disagg_pd_port=7787 \
               --node_rank=0 \
               --nnodes=1
        ```
    需要注意：
    
    - PD分离需要读取`/etc/hccn.conf`文件，确保将物理机上的该文件映射到了容器中
  
    - `etcd_addr`需与`xllm_service`的`etcd_addr`相同

!!! tip "Chunked prefill的Prefill实例支持prefix cache"
    使用chunked-prefill PD调度器时，Prefill实例已支持prefix cache。开启后，调度器会先匹配已有prefix cache block，再计算当前chunk budget，避免重复计算已缓存的prompt block。
    ``` shell
    --enable_chunked_prefill=true
    --enable_prefix_cache=true
    ```

!!! tip "Decode实例开启prefix cache（推荐）"
    从xLLM v0.x开始，Decode实例支持开启prefix cache。**当请求带 `best_of > 1` 时（best-of-N采样），Decode实例必须开启prefix cache**，否则请求会被拒绝。开启后扩展出的候选序列会通过prefix cache复用第一条序列的prompt KV，避免重复算力与显存浪费。
    ``` shell
    --enable_prefix_cache=true
    ```

!!! info "PD分离下的best_of_n支持"
    - **必要条件**：Decode实例 `--enable_prefix_cache=true`；任意请求 `best_of > 1` + Decode `enable_prefix_cache=false` 会被拒绝并返回错误。
    - **流程**：Prefill实例只对第一条序列做prefill并把KV发到Decode；Decode实例在收到first token后通过prefix cache将prompt KV共享给扩展出的 `best_of-1` 条候选序列。
    - **MLU平台**：暂不支持 PD + best_of_n（受 `normalize_mlu` 限制）。
    - **`best_of != n`时**：流式输出仍会被强制关闭（与non-PD行为一致）。
