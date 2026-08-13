# Mooncake Store 高可用启动脚本

本目录提供以下脚本：

- `start_mooncake_ha_service.sh`：启动一个 etcd HA 模式的 `mooncake_master`。每个 master 节点运行一个实例。
- `start_mooncake_client.sh`：启动一个独立、持有实际内存资源的 `mooncake_client`。
- `xllm_mooncake_ha_args.sh`：生成安全的 Bash 参数数组，供 Fused 或 PD 模式的 xLLM 使用。

这些脚本不会启动 etcd。运行前需要准备可被所有 Mooncake master、client 和 xLLM 进程访问的 etcd 集群。

## 1. 编译 HA 产物

Mooncake etcd wrapper 需要 Go 1.25.10 或兼容工具链：

```bash
MAX_JOBS=32 SKIP_EXPORT=1 \
  python setup.py build --device npu --enable-ha true

cmake --build build/cmake.linux-aarch64-cpython-311 \
  --target mooncake_master mooncake_client -j32
```

启用 HA 时，如果没有 Go 1.25.10，`third_party/dependencies.sh` 会参考
Mooncake 的依赖安装方式，依次尝试官方源和国内镜像，并自动安装到
`/usr/local/go`。`pre_build` 会在 `enable-ha` 开启时自动调用该脚本；
Mooncake 升级 Go 版本时，需要同步更新脚本中的 `GOVER`。

脚本会优先使用 `PATH` 中的 Mooncake 二进制，否则自动查找 `build/lib.*/xllm/`。也可以显式设置：

```bash
export MOONCAKE_MASTER_BIN=/path/to/mooncake_master
export MOONCAKE_CLIENT_BIN=/path/to/mooncake_client
```

## 2. 启动 Mooncake HA service

所有 master 必须使用相同的 etcd endpoints 和 cluster ID。`MOONCAKE_RPC_ADDRESS` 必须是 client 和 xLLM 可以访问的地址。

在第一个 master 节点执行：

```bash
export MOONCAKE_ETCD_ENDPOINTS='10.0.0.1:2379;10.0.0.2:2379;10.0.0.3:2379'
export MOONCAKE_CLUSTER_ID=xllm-mooncake
export MOONCAKE_RPC_ADDRESS=10.0.1.11
export MOONCAKE_RPC_PORT=50051
export MOONCAKE_METRICS_PORT=19003

scripts/kvcache_store/start_mooncake_ha_service.sh start
```

在其他 master 节点使用相同的 endpoints 和 cluster ID，只修改本节点的可达地址：

```bash
# master 2
export MOONCAKE_RPC_ADDRESS=10.0.1.12
scripts/kvcache_store/start_mooncake_ha_service.sh start

# master 3
export MOONCAKE_RPC_ADDRESS=10.0.1.13
scripts/kvcache_store/start_mooncake_ha_service.sh start
```

如果在同一台机器启动多个 master，还需要设置不同的 RPC、metrics 端口和实例名：

```bash
MOONCAKE_INSTANCE_NAME=master-50061 \
MOONCAKE_RPC_ADDRESS=127.0.0.1 \
MOONCAKE_RPC_PORT=50061 \
MOONCAKE_METRICS_PORT=19013 \
scripts/kvcache_store/start_mooncake_ha_service.sh start
```

## 3. 启动独立 Mooncake client

至少需要一个持有全局内存 segment 的 client。`MOONCAKE_CLIENT_HOST` 是该 client 的数据传输地址，需要包含端口且能被 xLLM 节点访问：

```bash
export MOONCAKE_ETCD_ENDPOINTS='10.0.0.1:2379;10.0.0.2:2379;10.0.0.3:2379'
export MOONCAKE_CLUSTER_ID=xllm-mooncake
export MOONCAKE_CLIENT_HOST=10.0.2.21:50053
export MOONCAKE_CLIENT_PORT=50052
export MOONCAKE_GLOBAL_SEGMENT_SIZE=32GB
export MOONCAKE_PROTOCOL=tcp

scripts/kvcache_store/start_mooncake_client.sh start
```

可以启动多个独立 client 扩展容量；每个 client 必须使用唯一的 `MOONCAKE_CLIENT_HOST` 和 service port。

## 4. xLLM 必需参数

设置 etcd endpoints、cluster ID 和当前 xLLM 进程的 Store endpoint，然后 source 参数脚本：

```bash
export MOONCAKE_ETCD_ENDPOINTS='10.0.0.1:2379;10.0.0.2:2379;10.0.0.3:2379'
export MOONCAKE_CLUSTER_ID=xllm-mooncake
export XLLM_STORE_LOCAL_HOSTNAME=10.0.3.31:12345

source scripts/kvcache_store/xllm_mooncake_ha_args.sh

/path/to/xllm \
  --model=/path/to/model \
  --model_id=my-model \
  "${XLLM_MOONCAKE_HA_ARGS[@]}"
```

生成的参数如下：

```text
--enable_prefix_cache=true
--host_blocks_factor=4
--enable_kvcache_store=true
--store_protocol=tcp
--store_master_server_address=etcd://10.0.0.1:2379;10.0.0.2:2379;10.0.0.3:2379
--store_metadata_server=P2PHANDSHAKE
--store_local_hostname=10.0.3.31:12345
--prefetch_batch_size=8
--prefetch_timeout=30000
```

可用以下环境变量覆盖默认值：

| 环境变量 | 默认值 | 对应 xLLM 参数 |
| --- | --- | --- |
| `XLLM_HOST_BLOCKS_FACTOR` | `4` | `host_blocks_factor` |
| `XLLM_STORE_PROTOCOL` | `tcp` | `store_protocol` |
| `XLLM_STORE_METADATA_SERVER` | `P2PHANDSHAKE` | `store_metadata_server` |
| `XLLM_PREFETCH_BATCH_SIZE` | `8` | `prefetch_batch_size` |
| `XLLM_PREFETCH_TIMEOUT` | `30000` | `prefetch_timeout` |

直接执行参数脚本可以查看经过 shell 转义的参数数组：

```bash
MOONCAKE_ETCD_ENDPOINTS='10.0.0.1:2379;10.0.0.2:2379;10.0.0.3:2379' \
XLLM_STORE_LOCAL_HOSTNAME=10.0.3.31:12345 \
scripts/kvcache_store/xllm_mooncake_ha_args.sh
```

不要把含分号的 HA 地址拼接到未加引号的命令字符串中。使用该脚本生成的 Bash 数组可以避免 endpoint 被 shell 按分号截断。

## 5. PD 模式额外参数

Prefill 和 Decode 都需要使用上一节的 Store 参数，并额外配置：

```text
--enable_disagg_pd=true
--instance_role=PREFILL              # Decode 进程使用 DECODE
--kv_cache_transfer_type=Mooncake
--kv_cache_transfer_mode=PUSH
--etcd_addr=<xllm-service使用的etcd地址>
--transfer_listen_port=<每个rank唯一端口>
--disagg_pd_port=<每个rank唯一端口>
```

每个 xLLM rank 的 `store_local_hostname` 必须唯一，Prefill 和 Decode 也不能复用 endpoint。例如 TP2：

```text
Prefill rank 0: 10.0.3.31:12345
Prefill rank 1: 10.0.3.31:12346
Decode  rank 0: 10.0.3.31:13345
Decode  rank 1: 10.0.3.31:13346
```

所有 master、独立 client、Prefill 和 Decode 进程必须使用同一个 `MOONCAKE_CLUSTER_ID`/`MC_STORE_CLUSTER_ID`。

## 6. 管理进程

```bash
scripts/kvcache_store/start_mooncake_ha_service.sh status
scripts/kvcache_store/start_mooncake_ha_service.sh stop

scripts/kvcache_store/start_mooncake_client.sh status
scripts/kvcache_store/start_mooncake_client.sh stop
```

默认日志目录为 `logs/mooncake/`。生产环境建议由 systemd、Kubernetes 或其他进程管理器调用脚本的 `run` 子命令，以获得外部拉起和监控能力。
