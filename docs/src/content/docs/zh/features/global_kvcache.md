---
title: "全局多级KV Cache"
sidebar:
  order: 51
---

## 背景

长上下文推理在自回归解码过程中需要持续读取历史 KV Cache。随着模型规模和上下文窗口增长，设备显存容量和带宽会成为主要瓶颈。仅使用 Device Cache 时，即使相同前缀已经被其他请求或其他 xLLM 实例计算过，冷请求仍然需要重新执行 Prefill。

xLLM 将 Device Prefix Cache 扩展为三级缓存：

| 层级 | 用途 | 生命周期 |
|---|---|---|
| Device HBM | 当前 Forward 使用的最低延迟 KV | 设备本地 |
| Host Cache | Pinned CPU 内存中的传输缓冲区和可复用 Host Prefix Cache | xLLM 进程 |
| Mooncake Store | 在多个 xLLM 进程以及进程重启之间共享的分布式 KV 对象 | Store 集群 |

请求首先探测 Host Prefix Cache。缺失的完整 Block 可以从 Mooncake Store 读取到预分配的 Host Block，再按层恢复到 HBM，从而跳过已命中前缀的重复计算。已完成的 HBM Block 会异步写回 Host，随后写入 Mooncake Store。

## 架构

部署中可以包含以下组件：

- **etcd**：注册计算实例并同步服务元数据。
- **xLLM Service**：路由请求并管理 Fused 或 PD 分离实例。
- **xLLM**：持有 Device/Host KV Cache 并执行推理。
- **Mooncake Store**：提供分布式、可跨进程复用的 KV 对象存储层。

服务级整体架构如下：

![xLLM 全局多级KV Cache](../../assets/globalkvcache_architecture.png)

## Block 流转

Fused 实例和 PD 分离中的 Prefill 实例使用完整的 Mooncake 准入、Host 恢复和写回流程。Decode 也保持 Store 开启，用于自身的 Host/Mooncake 写回；但 Decode 的请求准入仍然只探测 Device Prefix。

```mermaid
sequenceDiagram
    autonumber

    participant Client as Client / xLLM Service
    participant Scheduler as Scheduler
    participant BlockMgr as HierarchyBlockManagerPool
    participant Engine as Engine / RemoteWorker
    participant Result as PrefetchResult / Async Callback
    participant Worker as TP Workers
    participant Store as Mooncake Store
    participant Host as Host Cache
    participant HBM as Device HBM

    rect rgb(235, 245, 255)
        Note over Client,Store: 阶段一：请求准入与 Mooncake Store 预取

        Client->>Scheduler: add_request(request)
        Scheduler->>BlockMgr: prefetch_from_storage(request)
        BlockMgr->>Host: 探测 Host Prefix Cache
        Host-->>BlockMgr: 返回已有 blocks 与 holes
        BlockMgr->>Host: 为 holes 分配 G2H 目标 blocks
        Host-->>BlockMgr: 返回 Host block IDs

        Note over BlockMgr,Store: 若 Host 已覆盖全部 prefix，则跳过 Store RPC
        BlockMgr->>Engine: prefetch_from_storage(G2H infos)
        Engine->>Result: 创建 worker_count × block_count 结果矩阵

        par 所有 TP Rank 并行
            Engine->>Worker: PrefetchFromStorage(G2H batch)
            Worker->>Store: BatchIsExist(keys)
            Store-->>Worker: key existence bitmap
            opt 仅对存在的 keys
                Worker->>Store: BatchGet(existing keys, Host tensors)
                Store-->>Worker: KV 写入 Host tensors
            end
            Worker-->>Result: rank-local bitmap 与完成状态
        end

        loop Scheduler admission poll
            Scheduler->>BlockMgr: update_prefetch_result(timeout)
            BlockMgr->>Result: completed()?
        end
        BlockMgr->>Result: merged_hits()
        Result-->>BlockMgr: 所有 TP bitmap 逻辑 AND
        Note right of Result: 只有所有 TP Rank 都命中时才能发布该 block

        BlockMgr->>Host: 释放 Store miss 的目标 blocks
        BlockMgr->>Host: cache Store hit blocks
        BlockMgr->>BlockMgr: 计算连续可用 prefix 并 mount Host state
        BlockMgr-->>Scheduler: Prefetch 完成
        Scheduler->>Scheduler: AdmissionReady / enqueue_ready_request
        Note over Scheduler,Result: Worker 不直接回调 Scheduler
    end

    rect rgb(240, 255, 240)
        Note over Scheduler,HBM: 阶段二：Host KV 恢复到 HBM 并执行 Forward

        Scheduler->>BlockMgr: allocate(sequence, num_tokens)
        BlockMgr->>BlockMgr: 合并 Device Prefix 与已 mount 的 Host Prefix
        BlockMgr->>HBM: 分配缺失的 Device blocks
        HBM-->>BlockMgr: 返回 Device block IDs
        BlockMgr->>Host: best-effort 分配后续 D2H 目标 blocks
        Host-->>BlockMgr: 返回预留 Host block IDs
        BlockMgr->>BlockMgr: 发布 Device Prefix 元数据
        Note over BlockMgr,HBM: 元数据发布受 token cursor 限制，但发生在物理 H2D 完成之前
        BlockMgr->>BlockMgr: 构建按层 H2D plan

        Scheduler->>BlockMgr: transfer_blocks(batches)
        BlockMgr->>Engine: enqueue TransferBlocks(H2D, batch_id)
        BlockMgr-->>Scheduler: 调度返回，不等待 H2D copy 完成

        par 所有 TP Rank 并行
            Engine->>Worker: 注册 H2D transfer
            Worker->>Worker: 创建 LayerSynchronizer(batch_id)
            Worker->>Worker: 异步调度 load_from_host
            Worker-->>Engine: registration ACK 与 scheduled block 数
            Engine->>Worker: Forward(batch_id)，排在注册 RPC 之后
            Worker->>Worker: 挂载 LayerSynchronizer(batch_id)

            loop 每个 layer copy range
                Worker->>Host: 读取 Host KV tensors
                Host-->>Worker: Host KV
                Worker->>HBM: 异步 H2D copy 并记录 event
                Worker->>Worker: 当前计算层等待 event
                Worker->>HBM: event 完成后读取 KV Cache
            end

            Worker-->>Engine: Forward output
        end

        Note over Scheduler,Worker: 不存在 H2D-complete 回调 Scheduler
    end

    rect rgb(255, 245, 235)
        Note over Scheduler,Store: 阶段三：HBM Block 写回 Host 与 Mooncake

        Scheduler->>BlockMgr: deallocate(completed sequence)
        BlockMgr->>BlockMgr: 发布已完成的 Device Prefix 元数据
        BlockMgr->>BlockMgr: 收集 HBM → 已预留 Host block pairs
        BlockMgr->>BlockMgr: reset sequence，offload pair 保留 block 引用

        Scheduler->>BlockMgr: transfer_offload_blocks()
        BlockMgr->>Engine: 异步提交 D2H2G plans

        par 所有 TP Rank 并行
            Engine->>Worker: TransferKvBlocks(D2H2G)
            Worker->>Worker: copy stream wait_stream(compute stream)
            Worker->>HBM: 读取 Device KV
            HBM-->>Worker: Device KV
            Worker->>Host: D2H copy 并同步 copy stream
            Worker->>Store: BatchIsExist(keys)

            alt Store key 不存在
                Worker->>Store: BatchPut(keys, Host tensors)
                Store-->>Worker: Put results
            else Store key 已存在
                Worker->>Worker: 跳过覆盖并计为已存在
            end

            Note right of Worker: BatchPut 部分失败只记录日志<br/>不会改变 D2H 成功状态
            Worker-->>Engine: D2H 成功时返回完整 block count
        end

        Engine-->>Result: 所有 TP futures
        Result->>Result: 校验每个 TP 返回 expected block count
        Result-->>BlockMgr: future callback(copy_ok)
        BlockMgr->>HBM: 无论 copy_ok 与否都释放 offload 持有的 Device blocks

        alt 所有 TP Rank 的 D2H/RPC 都成功
            BlockMgr->>Host: 发布 Host Prefix Cache
        else 任一 TP Rank D2H/RPC 失败
            BlockMgr->>Host: 不发布 Host Prefix，并释放预留 Host blocks
        end

        Note over Scheduler,Result: offload completion 由 BlockManager callback 处理，不回调 Scheduler
    end
```

## PD 分离

在 PD 分离场景中，Mooncake Store 准入和 Host→HBM 恢复发生在 **Prefill** 实例。Decode 会在 Prefill 开始前预分配目标 Device Block，并且在请求准入时只探测 Device Prefix Cache；Decode 不会 mount Host alias、从 Mooncake 获取 prefix 或调度 Host→Device 恢复。Decode 仍需开启 Store 并配置 Host Cache 容量，用于自身的写回流程。

```mermaid
sequenceDiagram
    autonumber

    participant Client as Client / xLLM Service
    participant PSched as PREFILL Scheduler
    participant PBlock as PREFILL BlockManager
    participant PWorker as PREFILL TP Workers
    participant Store as Mooncake Store
    participant Host as PREFILL Host Cache
    participant PHBM as PREFILL HBM
    participant DService as DECODE Service / Scheduler
    participant DBlock as DECODE BlockManager
    participant KVTransfer as PD KV Transfer（Mooncake）
    participant DHBM as DECODE HBM

    rect rgb(235, 245, 255)
        Note over Client,PHBM: 阶段一：PREFILL admission 与 Mooncake 恢复

        Client->>PSched: add_request(request, decode_address)
        PSched->>PBlock: prefetch_from_storage(request)
        PBlock->>PWorker: TP 并行 PrefetchFromStorage(G2H)
        PWorker->>Store: BatchIsExist / BatchGet
        Store-->>PWorker: KV 写入预注册 Host tensors
        PWorker->>Host: 命中的 Host blocks 已填充
        PWorker-->>PBlock: rank-local bitmap（经 PrefetchResult）
        PBlock->>PBlock: TP 逻辑 AND 并 mount Host state
        PSched->>PBlock: 轮询 update_prefetch_result
        PBlock-->>PSched: Prefetch 完成
        PSched->>PSched: enqueue_ready_request → PREFILL dispatch queue
    end

    rect rgb(250, 240, 255)
        Note over PSched,DHBM: 阶段二：先在 DECODE 预分配目标 blocks

        PSched->>DService: AddNewRequests(prompt metadata)
        DService->>DBlock: try_allocate(DECODE sequence)
        DBlock->>DHBM: 仅探测 Device Prefix Cache
        DBlock->>DHBM: 为未命中部分分配 Device blocks
        Note over Store,DBlock: DECODE 准入不获取 Host/Mooncake prefix，也不安排 H2D restore<br/>Store 仍为 DECODE 写回保持开启
        DBlock-->>DService: allocation success
        DService->>DService: 收集 D block IDs + remote_shared_num
        DService-->>PSched: allocation response

        PSched->>PSched: 保存 TransferKVInfo
        PSched->>PSched: transfer cursor 跳过 D-side shared prefix
        PSched->>PSched: request 写入 PREFILL request_queue
    end

    rect rgb(240, 255, 240)
        Note over PSched,DHBM: 阶段三：PREFILL Forward 与 P→D KV 传输

        PSched->>PBlock: allocate PREFILL sequence
        PBlock->>Host: 使用已 mount 的 Store/Host prefix
        PBlock->>PHBM: 分配 Device blocks
        PBlock->>PBlock: 构建 Host→HBM restore plan
        PSched->>PBlock: transfer_blocks(batches)
        PBlock->>PWorker: 注册 H2D plan + batch_id
        PSched->>PWorker: PREFILL Forward
        Note over PWorker,PHBM: Forward 按 batch_id 挂载 LayerSynchronizer<br/>并等待所需的 H2D events

        alt PUSH
            PWorker->>KVTransfer: push_kv_blocks_async(P local → D remote)
            Note over PWorker,KVTransfer: 与 PREFILL 按 layer 同步推进<br/>并跳过 D-side shared blocks
            KVTransfer->>PHBM: 读取已计算的 PREFILL KV
            KVTransfer->>DHBM: Push 到预分配的 Decode blocks
            PWorker->>PWorker: Forward 返回前等待 KV push 完成
            PWorker-->>PSched: Forward output / first token

            PSched->>DService: FirstGeneration(token, mode=PUSH)
            DService->>DService: append first token，不执行 PULL
            DService->>DService: enqueue Decode request
            DService-->>PSched: FirstGeneration success
        else PULL
            PWorker-->>PSched: Forward output / first token
            PSched->>DService: FirstGeneration(token + P source metadata, mode=PULL)
            DService->>KVTransfer: pull_kv_blocks(P source → D destination)
            KVTransfer->>PHBM: 读取 PREFILL KV
            KVTransfer->>DHBM: 写入 Decode blocks / recurrent state
            KVTransfer-->>DService: pull success
            DService->>DService: pull 成功后才 enqueue Decode request
            DService-->>PSched: FirstGeneration success
        end

        PSched->>PBlock: FirstGeneration 成功后 cache_prefill_blocks
        PSched->>PBlock: deallocate PREFILL sequence
        Note over PBlock,Store: PREFILL 随后走通用异步 D2H→Host→Mooncake 写回流程
    end

    rect rgb(255, 245, 235)
        Note over DService,DHBM: 阶段四：DECODE 执行

        DService->>DHBM: Decode Forward 使用 Device blocks
        DService-->>Client: Token stream
    end
```

调度路径同时支持 `PUSH` 和 `PULL`。`kv_cache_transfer_type` 默认使用 `Mooncake`；下面的示例为了清晰仍显式设置该参数。标准 PD 部署需要在 Prefill 和 Decode 都开启全局 Mooncake Store，并为两个角色设置不重叠的 `store_local_hostname` 基础端口区间。

## 部署

### 前置条件

- 编译并安装 [xLLM](/zh/getting_started/quick_start/)。
- 使用服务路由或 PD 分离时，安装 [xLLM Service](https://github.com/xLLM-AI/xllm-service)。
- 编译或安装 Mooncake Store 的 `mooncake_master` 和 `mooncake_client`。
- 预留足够的 Host 内存。Mooncake Store 要求 `--enable_prefix_cache=true` 且 `--host_blocks_factor > 1`。

使用 Mooncake etcd 高可用模式时，需要先安装 Go，然后在构建 xLLM 和随仓库提供的 Mooncake 二进制时显式开启 HA 后端：

```bash
MAX_JOBS=32 SKIP_EXPORT=1 \
  python setup.py build --device npu --enable-ha true
cmake --build build/cmake.linux-aarch64-cpython-311 \
  --target mooncake_master mooncake_client -j32
```

可直接复用的 HA master、独立 Store client 和 xLLM 参数脚本位于 `scripts/kvcache_store/`。

### 启动最小 Mooncake Store

下面的 TCP 示例使用 Mooncake P2P handshake，因此不需要额外启动 Transfer Engine metadata service：

```bash
export MC_STORE_CLUSTER_ID=xllm-mooncake

mooncake_master \
  --rpc_address=0.0.0.0 \
  --rpc_port=50051
```

至少启动一个持有存储资源的 Store client：

```bash
mooncake_client \
  --host=0.0.0.0:50053 \
  --port=50052 \
  --global_segment_size=4GB \
  --master_server_address=127.0.0.1:50051 \
  --metadata_server=P2PHANDSHAKE \
  --protocol=tcp
```

### 启动 Mooncake Store 高可用集群

先启动可供所有 Mooncake master 访问的 etcd 集群。然后在每个 master 节点启动一个实例；所有实例使用相同的 etcd endpoints 和 `cluster_id`，但 `rpc_address` 必须是各自可达的地址：

```bash
mooncake_master \
  --enable_ha=true \
  --ha_backend_type=etcd \
  --ha_backend_connstring="10.0.0.1:2379;10.0.0.2:2379;10.0.0.3:2379" \
  --cluster_id=xllm-mooncake \
  --rpc_address=10.0.1.11 \
  --rpc_port=50051
```

Store client 和 xLLM 不再绑定单个 master 地址，而是通过 etcd 自动发现并跟随当前 leader：

```bash
export MC_STORE_CLUSTER_ID=xllm-mooncake
MOONCAKE_HA_ENTRY='etcd://10.0.0.1:2379;10.0.0.2:2379;10.0.0.3:2379'

mooncake_client \
  --host=0.0.0.0:50053 \
  --port=50052 \
  --global_segment_size=4GB \
  --master_server_address="${MOONCAKE_HA_ENTRY}" \
  --metadata_server=P2PHANDSHAKE \
  --protocol=tcp

/path/to/xllm \
  --enable_prefix_cache=true \
  --host_blocks_factor=4 \
  --enable_kvcache_store=true \
  --store_protocol=tcp \
  --store_master_server_address="${MOONCAKE_HA_ENTRY}" \
  --store_metadata_server=P2PHANDSHAKE \
  --store_local_hostname=127.0.0.1:12345
```

`store_master_server_address` 的 `etcd://` 前缀用于选择 HA leader-discovery 后端；后面的 endpoint 列表不带 `http://` 前缀。自定义 `cluster_id` 时，所有 Mooncake master、Store client 和 xLLM 进程都必须使用相同的 `MC_STORE_CLUSTER_ID`。

### 启动 etcd 与 xLLM Service

服务路由和 PD 分离需要该步骤；单独启动 Fused xLLM 时不要求使用 xLLM Service：

```bash
./etcd \
  --listen-peer-urls=http://0.0.0.0:10999 \
  --listen-client-urls=http://0.0.0.0:10998
```

```bash
./xllm_master_serving \
  --etcd_addr=127.0.0.1:10998 \
  --http_server_port=28888 \
  --rpc_server_port=28889 \
  --tokenizer_path=/path/to/tokenizer_config_dir/
```

### Fused xLLM 示例

```bash
/path/to/xllm \
  --model=/path/to/model \
  --model_id=my-model-revision-v1 \
  --enable_prefix_cache=true \
  --host_blocks_factor=4 \
  --enable_kvcache_store=true \
  --store_protocol=tcp \
  --store_master_server_address=127.0.0.1:50051 \
  --store_metadata_server=P2PHANDSHAKE \
  --store_local_hostname=127.0.0.1:12345 \
  --prefetch_batch_size=8 \
  --prefetch_timeout=30000
```

`store_local_hostname` 是 Transfer Engine 基础 endpoint。每个 Worker 使用 `base_port + worker_rank`，因此整个端口区间都必须空闲且网络可达。

使用 RDMA 时，设置 `--store_protocol=rdma`，并通过 `DEVICE_NAMES` 环境变量指定 Mooncake RDMA 设备。如果没有设置 `DEVICE_NAMES`，xLLM 会回退到 TCP。

### PD 分离示例

两个角色都需要正常配置 [PD 分离](/zh/features/disagg_pd/)参数并开启 Store。Prefill 和 Decode 必须使用不同的 `store_local_hostname` 基础端口：

```bash
/path/to/xllm \
  --enable_disagg_pd=true \
  --instance_role=PREFILL \
  --kv_cache_transfer_type=Mooncake \
  --kv_cache_transfer_mode=PUSH \
  --enable_prefix_cache=true \
  --host_blocks_factor=4 \
  --enable_kvcache_store=true \
  --store_protocol=tcp \
  --store_master_server_address=127.0.0.1:50051 \
  --store_metadata_server=P2PHANDSHAKE \
  --store_local_hostname=127.0.0.1:12345
```

Decode 使用不同的本地 endpoint 区间开启 Store：

```bash
/path/to/xllm \
  --enable_disagg_pd=true \
  --instance_role=DECODE \
  --kv_cache_transfer_type=Mooncake \
  --kv_cache_transfer_mode=PUSH \
  --enable_prefix_cache=true \
  --host_blocks_factor=4 \
  --enable_kvcache_store=true \
  --store_protocol=tcp \
  --store_master_server_address=127.0.0.1:50051 \
  --store_metadata_server=P2PHANDSHAKE \
  --store_local_hostname=127.0.0.1:13345
```

所有 `KVCacheStoreConfig` 参数参见 [CLI 参数说明](/zh/cli_reference/)。

## 正确性与运维说明

- 只有当**所有 TP Rank**都报告命中时，从 Store 读取的 Block 才会 mount 到 Host Prefix Cache。
- `prefetch_timeout` 到期后会停止下发新的预取 batch，但请求准入仍会等待所有在途 TP batch 完成；`0` 表示无限等待。
- H2D registration 不等待物理拷贝。Forward 通过 `batch_id` 挂载 `LayerSynchronizer`，并在对应计算层等待；Scheduler 不会收到 H2D-complete 回调。
- 写回时，只有所有 TP Rank 的 D2H/RPC 都成功，Host Prefix 才会发布。Mooncake `BatchPut` 是 best-effort；Store 部分写入失败只记录日志，不会使已经成功的 Host copy 失效。
- 已存在的 Store 对象不会被覆盖。每次权重或配置版本变化时必须使用新的 `model_id`，并按需轮换或清理 Store namespace。
- PD 场景中，Prefill 和 Decode 都需要开启 Store。两个角色会复用 Worker Rank，并且每个 Worker 会绑定 `base_port + worker_rank`，因此必须使用不重叠的 `store_local_hostname` 基础端口区间。
