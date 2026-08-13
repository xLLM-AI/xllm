---
title: "Global Multi-Level KV Cache"
sidebar:
  order: 51
---

## Background

Long-context inference repeatedly reads historical KV cache during autoregressive decoding. As model sizes and context windows grow, device memory capacity and bandwidth become major constraints. A device-only cache also makes a cold request recompute a prefix even when the same prefix was produced by an earlier request or another xLLM instance.

xLLM extends the device prefix cache into a three-level hierarchy:

| Tier | Purpose | Lifetime |
|---|---|---|
| Device HBM | Lowest-latency KV used by the current forward pass | Device-local |
| Host cache | Pinned CPU-memory staging and reusable host prefix cache | xLLM process |
| Mooncake Store | Distributed KV objects shared across xLLM processes and restarts | Store cluster |

A request first checks the Host prefix cache. Missing full blocks can be fetched from Mooncake Store into preallocated Host blocks, restored to HBM layer by layer, and reused without recomputing the matched prefix. Completed HBM blocks are asynchronously copied back to Host memory and then written to Mooncake Store.

## Architecture

The deployment can contain the following components:

- **etcd**: Registers compute instances and synchronizes service metadata.
- **xLLM Service**: Routes requests and manages fused or disaggregated Prefill/Decode instances.
- **xLLM**: Owns the device and Host KV caches and executes inference.
- **Mooncake Store**: Provides the distributed, process-independent KV object tier.

The service-level architecture is shown below:

![xLLM Global Multi-Level KV Cache](../../assets/globalkvcache_architecture.png)

## Block Lifecycle

Fused instances and the Prefill side of disaggregated PD use the complete Mooncake admission, Host restore, and write-back path. Decode keeps Store enabled for its Host/Mooncake write-back path, while its request-admission path remains Device-prefix-only.

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
        Note over Client,Store: Phase 1: request admission and Mooncake prefetch

        Client->>Scheduler: add_request(request)
        Scheduler->>BlockMgr: prefetch_from_storage(request)
        BlockMgr->>Host: Probe Host Prefix Cache
        Host-->>BlockMgr: Existing blocks and holes
        BlockMgr->>Host: Allocate G2H destinations for holes
        Host-->>BlockMgr: Host block IDs

        Note over BlockMgr,Store: If Host already covers the prefix, Store RPCs are skipped
        BlockMgr->>Engine: prefetch_from_storage(G2H infos)
        Engine->>Result: Create worker-by-block result matrix

        par All TP ranks
            Engine->>Worker: PrefetchFromStorage(G2H batch)
            Worker->>Store: BatchIsExist(keys)
            Store-->>Worker: Existence bitmap
            opt Existing keys only
                Worker->>Store: BatchGet(existing keys, Host tensors)
                Store-->>Worker: Fill Host tensors
            end
            Worker-->>Result: Rank-local bitmap and completion
        end

        loop Admission polling
            Scheduler->>BlockMgr: update_prefetch_result(timeout)
            BlockMgr->>Result: completed()?
        end
        BlockMgr->>Result: merged_hits()
        Result-->>BlockMgr: Logical AND across all TP ranks
        Note right of Result: A block is publishable only when every TP rank hits

        BlockMgr->>Host: Release Store-miss destinations
        BlockMgr->>Host: Cache Store-hit blocks
        BlockMgr->>BlockMgr: Compute reachable prefix and mount Host state
        BlockMgr-->>Scheduler: Prefetch complete
        Scheduler->>Scheduler: AdmissionReady / enqueue_ready_request
        Note over Scheduler,Result: Workers do not directly callback the Scheduler
    end

    rect rgb(240, 255, 240)
        Note over Scheduler,HBM: Phase 2: Host-to-HBM restore and forward

        Scheduler->>BlockMgr: allocate(sequence, num_tokens)
        BlockMgr->>BlockMgr: Merge Device and mounted Host prefixes
        BlockMgr->>HBM: Allocate missing Device blocks
        HBM-->>BlockMgr: Device block IDs
        BlockMgr->>Host: Best-effort allocate future D2H destinations
        Host-->>BlockMgr: Reserved Host block IDs
        BlockMgr->>BlockMgr: Publish Device Prefix metadata
        Note over BlockMgr,HBM: Metadata publication is token-cursor bounded but precedes physical H2D completion
        BlockMgr->>BlockMgr: Build layer-wise H2D plan

        Scheduler->>BlockMgr: transfer_blocks(batches)
        BlockMgr->>Engine: Enqueue TransferBlocks(H2D, batch_id)
        BlockMgr-->>Scheduler: Return after dispatch, without waiting for H2D completion

        par All TP ranks
            Engine->>Worker: Register H2D transfer
            Worker->>Worker: Create LayerSynchronizer(batch_id)
            Worker->>Worker: Schedule load_from_host asynchronously
            Worker-->>Engine: Registration ACK with scheduled block count
            Engine->>Worker: Forward(batch_id), ordered after registration
            Worker->>Worker: Attach LayerSynchronizer(batch_id)

            loop Each layer-copy range
                Worker->>Host: Read Host KV tensors
                Host-->>Worker: Host KV
                Worker->>HBM: Async H2D copy and record event
                Worker->>Worker: Current compute layer waits for event
                Worker->>HBM: Read KV after the event completes
            end

            Worker-->>Engine: Forward output
        end

        Note over Scheduler,Worker: There is no H2D-complete callback to the Scheduler
    end

    rect rgb(255, 245, 235)
        Note over Scheduler,Store: Phase 3: HBM-to-Host-to-Mooncake write-back

        Scheduler->>BlockMgr: deallocate(completed sequence)
        BlockMgr->>BlockMgr: Publish completed Device Prefix metadata
        BlockMgr->>BlockMgr: Collect HBM to reserved-Host block pairs
        BlockMgr->>BlockMgr: Reset sequence while offload pairs retain block references

        Scheduler->>BlockMgr: transfer_offload_blocks()
        BlockMgr->>Engine: Submit asynchronous D2H2G plans

        par All TP ranks
            Engine->>Worker: TransferKvBlocks(D2H2G)
            Worker->>Worker: Copy stream waits for compute stream
            Worker->>HBM: Read Device KV
            HBM-->>Worker: Device KV
            Worker->>Host: D2H copy and stream synchronization
            Worker->>Store: BatchIsExist(keys)

            alt Store key is absent
                Worker->>Store: BatchPut(keys, Host tensors)
                Store-->>Worker: Put results
            else Store key already exists
                Worker->>Worker: Skip overwrite and count it as present
            end

            Note right of Worker: Partial BatchPut failure is logged only<br/>and does not change D2H success
            Worker-->>Engine: Full block count when D2H succeeds
        end

        Engine-->>Result: TP futures
        Result->>Result: Validate every TP result against the expected block count
        Result-->>BlockMgr: Future callback(copy_ok)
        BlockMgr->>HBM: Always release offload-held Device blocks

        alt Every TP D2H/RPC succeeds
            BlockMgr->>Host: Publish Host Prefix Cache
        else Any TP D2H/RPC fails
            BlockMgr->>Host: Publish nothing and release reserved Host blocks
        end

        Note over Scheduler,Result: Offload completion is handled by the BlockManager callback, not the Scheduler
    end
```

## Disaggregated PD

In disaggregated PD, Mooncake Store admission and Host-to-HBM restore run on the **Prefill** instance. The Decode instance allocates destination Device blocks before Prefill starts and only probes its Device Prefix Cache during admission; it does not mount Host aliases, fetch a prefix from Mooncake, or schedule Host-to-Device restoration. Decode still enables Store and Host cache capacity for its write-back path.

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
    participant KVTransfer as PD KV Transfer (Mooncake)
    participant DHBM as DECODE HBM

    rect rgb(235, 245, 255)
        Note over Client,PHBM: Phase 1: PREFILL admission and Mooncake restore

        Client->>PSched: add_request(request, decode_address)
        PSched->>PBlock: prefetch_from_storage(request)
        PBlock->>PWorker: TP-parallel PrefetchFromStorage(G2H)
        PWorker->>Store: BatchIsExist / BatchGet
        Store-->>PWorker: Fill registered Host tensors
        PWorker->>Host: Store-hit Host blocks are ready
        PWorker-->>PBlock: Rank-local bitmap through PrefetchResult
        PBlock->>PBlock: TP logical AND and mount Host state
        PSched->>PBlock: Poll update_prefetch_result
        PBlock-->>PSched: Prefetch complete
        PSched->>PSched: enqueue_ready_request to PREFILL dispatch queue
    end

    rect rgb(250, 240, 255)
        Note over PSched,DHBM: Phase 2: allocate Decode destinations first

        PSched->>DService: AddNewRequests(prompt metadata)
        DService->>DBlock: try_allocate(DECODE sequence)
        DBlock->>DHBM: Probe Device Prefix Cache only
        DBlock->>DHBM: Allocate blocks for the missing suffix
        Note over Store,DBlock: DECODE admission does not fetch Host/Mooncake prefix or schedule H2D restore<br/>Store remains enabled for DECODE write-back
        DBlock-->>DService: Allocation success
        DService->>DService: Collect D block IDs and remote_shared_num
        DService-->>PSched: Allocation response

        PSched->>PSched: Save TransferKVInfo
        PSched->>PSched: Advance transfer cursor past D-side shared prefix
        PSched->>PSched: Enqueue request into PREFILL request_queue
    end

    rect rgb(240, 255, 240)
        Note over PSched,DHBM: Phase 3: PREFILL forward and P-to-D KV transfer

        PSched->>PBlock: Allocate PREFILL sequence
        PBlock->>Host: Use mounted Store/Host prefix
        PBlock->>PHBM: Allocate Device blocks
        PBlock->>PBlock: Build Host-to-HBM restore plan
        PSched->>PBlock: transfer_blocks(batches)
        PBlock->>PWorker: Register H2D plan and batch_id
        PSched->>PWorker: PREFILL Forward
        Note over PWorker,PHBM: Forward attaches LayerSynchronizer by batch_id<br/>and waits for the required H2D events

        alt PUSH
            PWorker->>KVTransfer: push_kv_blocks_async(P local to D remote)
            Note over PWorker,KVTransfer: Transfer advances layer by layer with PREFILL<br/>and skips D-side shared blocks
            KVTransfer->>PHBM: Read computed PREFILL KV
            KVTransfer->>DHBM: Push into preallocated Decode blocks
            PWorker->>PWorker: Wait for KV push before returning Forward
            PWorker-->>PSched: Forward output / first token

            PSched->>DService: FirstGeneration(token, mode=PUSH)
            DService->>DService: Append first token without PULL
            DService->>DService: Enqueue Decode request
            DService-->>PSched: FirstGeneration success
        else PULL
            PWorker-->>PSched: Forward output / first token
            PSched->>DService: FirstGeneration(token + P source metadata, mode=PULL)
            DService->>KVTransfer: pull_kv_blocks(P source to D destination)
            KVTransfer->>PHBM: Read PREFILL KV
            KVTransfer->>DHBM: Write Decode blocks and recurrent state
            KVTransfer-->>DService: Pull success
            DService->>DService: Enqueue only after pull succeeds
            DService-->>PSched: FirstGeneration success
        end

        PSched->>PBlock: cache_prefill_blocks after FirstGeneration succeeds
        PSched->>PBlock: Deallocate PREFILL sequence
        Note over PBlock,Store: PREFILL then uses the common asynchronous D2H-to-Host-to-Mooncake write-back path
    end

    rect rgb(255, 245, 235)
        Note over DService,DHBM: Phase 4: DECODE execution

        DService->>DHBM: Decode Forward uses Device blocks
        DService-->>Client: Token stream
    end
```

Both `PUSH` and `PULL` are supported by the scheduler path. `kv_cache_transfer_type` defaults to `Mooncake`; the examples below set it explicitly for clarity. Enable the global Mooncake Store tier on both Prefill and Decode, using disjoint `store_local_hostname` base-port ranges.

## Deployment

### Prerequisites

- Build and install [xLLM](/en/getting_started/quick_start/).
- Install [xLLM Service](https://github.com/xLLM-AI/xllm-service) when service routing or disaggregated PD is required.
- Build or install the Mooncake Store `mooncake_master` and `mooncake_client` binaries.
- Reserve enough Host memory. Mooncake Store requires `--enable_prefix_cache=true` and `--host_blocks_factor > 1`.

For Mooncake's etcd-backed high availability mode, install Go first and explicitly enable the HA backends when building xLLM and the bundled Mooncake binaries:

```bash
MAX_JOBS=32 SKIP_EXPORT=1 \
  python setup.py build --device npu --enable-ha true
cmake --build build/cmake.linux-aarch64-cpython-311 \
  --target mooncake_master mooncake_client -j32
```

Ready-to-use HA master, independent Store client, and xLLM argument scripts are available under `scripts/kvcache_store/`.

### Start a Minimal Mooncake Store

The following TCP example uses Mooncake's P2P handshake, so no separate Transfer Engine metadata service is required:

```bash
export MC_STORE_CLUSTER_ID=xllm-mooncake

mooncake_master \
  --rpc_address=0.0.0.0 \
  --rpc_port=50051
```

Start at least one resource-owning Store client:

```bash
mooncake_client \
  --host=0.0.0.0:50053 \
  --port=50052 \
  --global_segment_size=4GB \
  --master_server_address=127.0.0.1:50051 \
  --metadata_server=P2PHANDSHAKE \
  --protocol=tcp
```

### Start a High-Availability Mooncake Store Cluster

First start an etcd cluster reachable by every Mooncake master. Then start one master instance on each master node. All instances use the same etcd endpoints and `cluster_id`, while `rpc_address` must identify the reachable address of that specific instance:

```bash
mooncake_master \
  --enable_ha=true \
  --ha_backend_type=etcd \
  --ha_backend_connstring="10.0.0.1:2379;10.0.0.2:2379;10.0.0.3:2379" \
  --cluster_id=xllm-mooncake \
  --rpc_address=10.0.1.11 \
  --rpc_port=50051
```

Store clients and xLLM use etcd to discover and follow the current leader instead of binding to one master address:

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

The `etcd://` prefix in `store_master_server_address` selects the HA leader-discovery backend. Do not add `http://` to the endpoint list after that prefix. When using a custom `cluster_id`, every Mooncake master, Store client, and xLLM process must use the same `MC_STORE_CLUSTER_ID`.

### Start etcd and xLLM Service

This step is required for service routing and disaggregated PD, but not for a standalone fused xLLM process:

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

### Fused xLLM Example

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

`store_local_hostname` is a base Transfer Engine endpoint. Each worker uses `base_port + worker_rank`, so the entire port range must be free and reachable.

For RDMA, set `--store_protocol=rdma` and export `DEVICE_NAMES` with the Mooncake RDMA devices. If `DEVICE_NAMES` is absent, xLLM falls back to TCP.

### Disaggregated PD Example

Use the normal [Disaggregated PD](/en/features/disagg_pd/) flags and enable Store on both roles. Use different `store_local_hostname` base ports for Prefill and Decode:

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

Enable Store on Decode with a different local endpoint range:

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

See the [CLI Reference](/en/cli_reference/) for all `KVCacheStoreConfig` parameters.

## Correctness and Operational Notes

- A Store-fetched block is mounted into the Host Prefix Cache only when **every TP rank** reports a hit for that block.
- `prefetch_timeout` stops issuing new prefetch batches after the timeout, but admission still waits for every in-flight TP batch to finish. `0` waits indefinitely.
- H2D registration does not wait for the physical copy. Forward attaches a `LayerSynchronizer` using `batch_id` and waits at the corresponding layers. The Scheduler receives no H2D-complete callback.
- Host Prefix publication after write-back is gated by successful D2H/RPC results from every TP rank. Mooncake `BatchPut` is best-effort; a partial Store write failure is logged but does not invalidate a successful Host copy.
- Existing Store objects are not overwritten. Use a unique `model_id` for every weight/configuration revision and rotate or clean the Store namespace when a checkpoint changes.
- In PD, Prefill and Decode both enable Store. They must use disjoint `store_local_hostname` base-port ranges because both roles reuse worker ranks and each worker binds `base_port + worker_rank`.
