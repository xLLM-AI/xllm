---
title: "MoE Load Balancing (EPLB)"
sidebar:
  order: 71
---
## Overview

Mixture-of-Experts (MoE) models route tokens to different experts. Online request distributions are rarely uniform, so a subset of experts can remain hot and make their Expert Parallel (EP) ranks the bottleneck for an entire batch.

xLLM Expert Parallel Load Balancing (EPLB) creates redundant physical replicas of hot logical experts and adjusts the expert placement one layer at a time while the service is running. EPLB does not change the model's logical expert count or routing result. It maintains a logical-to-physical mapping and migrates the weights required by that mapping.

The current implementation includes the control plane, load aggregation, pluggable policies, an asynchronous prepare protocol, and layer-by-layer weight activation. This document focuses on the DeepSeek V4 NPU `npu_torch/FusedMoE` path. Other models can use the same runtime protocol only after implementing model hooks such as `prepare_expert_weight`, `start_expert_weight_transfer`, and `update_expert_weight`.

## Experts and Slots

EPLB distinguishes between two concepts:

- **Logical expert**: an expert ID produced by the model router. The model architecture determines this count.
- **Physical expert slot**: one resident copy of expert weights on an EP rank. A logical expert may have multiple physical replicas.

With the current one-worker-per-EP-rank deployment, each rank owns:

```text
local_physical_experts = logical_experts / ep_size + redundant_experts_num
```

The global physical slot count is:

```text
global_physical_experts = logical_experts + ep_size * redundant_experts_num
```

The logical expert count must therefore be divisible by `ep_size`. `redundant_experts_num` adds extra slots on every rank and increases both expert-weight memory and EPLB staging memory.

## Current Architecture

| Component | Responsibility |
|---|---|
| `LLMEngine` | Collects load and prepare tokens from every worker and places `EplbInfo` commands in the next forward input. |
| `EplbManager` | Owns the update timer, active placement, per-layer prepare/activate state machine, timeout handling, and commit. |
| `EplbAggregator` | Decodes physical-slot counters into per-round deltas, preserves physical load, and aggregates logical-expert load. |
| `IEplbPolicy` | Produces a candidate placement and per-layer update mask from logical load, physical load, and the active placement. |
| `EplbExecutor` | Runs asynchronous prepare work on each worker and starts migration and activation at forward boundaries. |
| Model EPLB hooks | Maintain logical-to-physical maps, build P2P plans, migrate weights, and activate the new weights. |
| `ProcessGroup` | Batches HCCL point-to-point send/recv calls through a dedicated `ProcessGroupHCCL`; the current all-connected super-node uses HCCS as its data plane. |

The control plane uses `EplbInfo` with the following fields:

- `prepare_layer_id`: the MoE layer whose new placement must be prepared.
- `prepare_token`: a unique token for this prepare attempt. It isolates late results after a timeout.
- `expert_ids`: the global physical-slot table for the pending placement.
- `update_layer_id`: a prepared layer that may be activated at this forward boundary.
- `activation_token`: a unique activation token used to prove that the corresponding forward actually completed.

## Runtime Flow

A complete rebalance uses the following sequence:

1. Each worker records the token count for every local physical expert slot during MoE forward. Counts are prefix encoded along the slot dimension, and `EplbAggregator` restores the per-slot values with a difference operation.
2. `LLMEngine` collects load from every worker. The placement that was active when a sample was submitted is queued with that sample, so an old sample cannot later be interpreted with a newly activated placement.
3. The first valid load sample starts the `eplb_update_interval` timer. When it expires and no other rebalance is in progress, the manager asks the selected policy to recompute a candidate for every layer.
4. Each candidate is evaluated against the current measured physical peak. The manager publishes only layers whose placement changes and whose improvement reaches `eplb_min_peak_load_improvement`, one layer at a time with `prepare_layer_id`, `prepare_token`, and the new `expert_ids`.
5. The worker's `EplbExecutor` background thread computes the logical-to-physical map, changed slots, and the EP-wide P2P migration plan. A worker atomically reports the matching token only after prepare succeeds. A failed prepare is never reported as ready.
6. After every rank reports the same prepare token, the manager publishes `update_layer_id`. On the DeepSeek V4 path, each worker materializes staging weights and starts P2P before forward, then waits for migration and activates the weights after forward.
7. Only after the corresponding forward output returns with `activation_token` does the manager commit policy state, switch the active placement, and reset the physical-load window. Late or duplicate tokens are ignored.

EPLB commands are emitted only at a forward boundary with at least one non-empty DP batch, where every non-empty DP batch is decoding and the input is not graph warmup. Empty DP ranks may participate in this boundary. This prevents a switch while prefill or graph capture may still read the old weights. As a consequence, an already prepared layer remains pending when the workload never reaches an all-decode boundary.

## Load Collection

The manager maintains two load views:

- **Logical expert load**: physical replica load is aggregated through the active slot table into logical expert IDs. Policies use it to identify hot experts and choose replicas.
- **Physical slot load**: the `[layer, rank, slot]` shape is preserved so a policy can measure the actual slowest rank under the current placement instead of assuming equal load across replicas of the same logical expert.

DeepSeek V4 uses the same load semantics across eager execution, ACL graph, and EP2 fused dispatch:

- Graph warmup does not enter the manager's load window.
- ACL graph uses a per-token mask to remove graph padding and synthetic tokens on empty DP ranks.
- With `eplb_use_decode_only_load=true`, eager execution also records only real decode tokens and removes prefill rows from mixed batches.
- Fused dispatch records receiver-observed physical expert counts returned by the operator when possible. When a mixed phase cannot be filtered from aggregate operator counts, the implementation falls back to route-ID counting.

## Rebalance Policies

Select a policy with `eplb_policy_kind`. The value is case-insensitive. An unknown value falls back to `greedy` instead of stopping the rebalance thread.

| Policy | Placement Algorithm | Intended Use |
|---|---|---|
| `balanced` | Select replicas by maximum load reduction, then equal-capacity LPT packing across every rank in the all-connected HCCS super-node. | Default policy and recommended for the current super-node deployment. |
| `greedy` | Historical xLLM greedy replica selection and LPT packing. | Compatibility with historical behavior. |

Historical policy names remain compatibility aliases for `balanced`. New configurations should use `balanced`.

Both policies use the same publication gate:

- The current peak comes from measured physical-slot load. The candidate peak is estimated from the candidate placement.
- A migration is published only when the peak reduction reaches `eplb_min_peak_load_improvement`.
- Every update interval recomputes every layer; no previous-load similarity gate can skip evaluation.
- A logical expert has at most one replica on the same rank.
- A prepare timeout calls the policy's `abort_layer` hook, rolling back the candidate state instead of treating an undeployed placement as active.

## DeepSeek V4 Weight Switching

The DeepSeek V4 `FusedMoE` path maintains stable-address active weights, pending staging weights, and a `log2phy` map. Weight switching behaves as follows:

- Weight storage is expanded at startup to include each rank's redundant expert slots.
- After loading the main model weights, reusable staging buffers are warmed so the first rebalance does not allocate large NPU buffers on demand. The MTP model does not reserve a duplicate staging pool.
- Prepare computes host-side slot maps and P2P plans only. It does not materialize private-format NPU tensors concurrently with a forward that may be reading those tensors.
- Experts already resident on the same rank are copied into the pending buffer. Non-resident experts migrate through the dedicated EPLB `ProcessGroupHCCL`.
- `batch_isend_irecv` ultimately calls `ProcessGroupHCCL::send/recv`. HCCL is the communication library API, while rank-to-rank data in the current all-connected super-node travels over HCCS; EPLB does not model host boundaries.
- Send and receive operations for all tensors share one batched plan. Large payloads are chunked, and the implementation limits outstanding payload volume.
- P2P starts before forward and is awaited after forward. The current forward always uses the old placement, while the next forward observes the new placement.
- When fewer than half of the slots change, activation copies only changed slots. When at least half change, the implementation builds a complete pending tensor and activates it with one full-tensor `copy_` per weight tensor instead of issuing many small copies.
- Active tensor objects are not replaced. Their contents and maps are updated in place to preserve stable addresses required by ACL graph.

## EP2, MC2, and ACL Graph

The EPLB control plane does not require `expert_parallel_degree=2`. That setting selects the DeepSeek V4 EP Level 2 dispatch/combine path. `enable_fused_mc2` then selects the EP2 MoE operator:

| `enable_fused_mc2` | Behavior |
|---:|---|
| `-1` | Automatic selection. It resolves to `1` when `expert_parallel_degree=2`, otherwise to `0`. |
| `0` | Uses `moe_distribute_dispatch_v2 + group_gemm + moe_distribute_combine_v2`. |
| `1` | Uses `DispatchFFNCombine` when quantization, dtype, operator, and graph preparation requirements are satisfied. |
| `2` | Uses `DispatchGmmCombineDecode` for all-decode batches when its requirements are satisfied. |

If the requested MC2 path does not satisfy dtype, quantization, TP, operator availability, or graph preparation requirements, the runtime selects an available regular MoE path. MC2 changes MoE execution and physical-load collection, but it does not change the EPLB policy or layer-by-layer switching protocol.

EPLB works with ACL graph enabled or disabled. In graph mode, expert weights and the `log2phy` map keep stable storage, while the decode mask is aligned per DP rank and padded with zeros so graph padding cannot affect the load signal.

Decode graph buckets created by one `AclGraphExecutor` share a fixed capture stream and one private graph pool. This lets buckets reuse graph scratch addresses instead of retaining an independent set of expandable segments for every warmed bucket. The optimization prevents graph-pool memory from growing linearly with the bucket count, but the graph pool, persistent tensors, and runtime caches still consume device memory. Actual usage depends on the model, warmed buckets, and concurrency settings.

## Configuration

EPLB options can be provided through gflags or the xLLM JSON configuration. The manager snapshots `redundant_experts_num`, `eplb_update_interval`, `eplb_min_peak_load_improvement`, `eplb_policy_kind`, and `eplb_prepare_timeout_seconds` into `EplbOptions` when it is created. `eplb_use_decode_only_load` and `expert_parallel_degree` are model-execution settings read by their runtime paths. The supported operating model is still to finalize configuration before startup and restart the service after changing it.

The defaults below are the xLLM core gflags/JSON defaults. Deployment scripts may override them explicitly, so verify the effective command line and startup logs.

| Option | Default | Constraint and Meaning |
|---|---:|---|
| `enable_eplb` | `false` | Enables dynamic expert load balancing. |
| `redundant_experts_num` | `1` | Extra physical expert slots on every EP rank. Must be non-negative. |
| `eplb_update_interval` | `1000` | Rebalance interval in seconds, starting with the first valid load sample. Must be non-negative. The example below explicitly overrides it to `300`. |
| `eplb_min_peak_load_improvement` | `0.05` | Minimum slowest-rank improvement required by every policy. Range `[0, 1]`. |
| `eplb_policy_kind` | `balanced` | `balanced` (default) or `greedy`; historical names map to `balanced`, while unknown values fall back to `greedy`. |
| `eplb_use_decode_only_load` | `false` | Whether eager execution records only real decode tokens. Graph execution always filters padding. |
| `eplb_prepare_timeout_seconds` | `30` | Timeout while waiting for every rank to prepare the same layer. Must be positive. The layer is skipped and rolled back on timeout. |
| `expert_parallel_degree` | `0` | EP mode setting. Value `2` enables an eligible EP2 dispatch/combine path. |
| `enable_fused_mc2` | `-1` (automatic) | Valid values are `-1/0/1/2`. Automatic mode resolves to `1` with `expert_parallel_degree=2`, otherwise to `0`. |

## Launch Example

The following example represents 16 workers/EP ranks in one all-connected HCCS super-node and uses a 300-second test interval. `ep_size` must equal the actual worker count, and the model's logical expert count must be divisible by 16.

```bash
--enable_eplb=true \
--ep_size=16 \
--expert_parallel_degree=2 \
--redundant_experts_num=1 \
--eplb_update_interval=300 \
--eplb_policy_kind=balanced \
--eplb_min_peak_load_improvement=0.05 \
--eplb_use_decode_only_load=true \
--enable_fused_mc2=1
```

See [EP Parallelism](./moe_params.md) for EP and DP/TP topology combinations.

## Runtime Constraints

- EPLB aggregation currently requires `ep_size == worker_num`, which means one worker per EP rank.
- `ep_size` must be greater than 1, and the logical expert count must be divisible by `ep_size`.
- Redundant slots and staging buffers consume additional device memory. Leave capacity for both EPLB staging and the KV cache before startup.
- Weight activation occurs only at an all-decode forward boundary. It is never forced during graph warmup or a mixed prefill boundary.
- Configuration is snapshotted when the manager is created. Restart the service after changing startup options.

## Shared Memory Compatibility

EPLB control fields and worker output pass through xLLM forward shared memory. The current `SharedMemoryManager` uses layout v2 with a magic value, version, payload size, and generation. A global generation lock serializes open/unlink/create transitions for objects with the same name, preventing workers from mapping different shared-memory generations.

When upgrading from an older xLLM binary, stop every old process that may access the same shared-memory names before starting the new version. Rolling old and new binaries together is not supported. An incompatible size, magic value, or layout version fails immediately; the new process does not resize or clear the old object.

## Observability and Troubleshooting

| Log Prefix | Meaning |
|---|---|
| `EPLB manager start` | Effective policy, MoE layer count, device count, and physical slots per device. |
| `EPLB rebalance` | Policy runtime and number of layers selected for update. |
| `EPLB placement benefit` | Current peak, candidate peak, improvement ratio, and benefit-gate result. |
| `prepare_expert_weight` | Layer, slot-table size, duration, and failure state for worker background prepare. |
| `materialize_expert_weight` | Time spent materializing DeepSeek V4 staging tensors at a forward boundary. |
| `update_expert_weight` | Layer, rank, MC2 mode, changed slots, P2P count, transfer time, activation time, and total time. |
| `EPLB heartbeat` | Liveness and progress for the rebalance, manager, executor, and P2P bucket paths. |
| `EPLB staging reservation warmed` | Staging bytes and tensor count reserved during startup. |

For a stalled switch, first verify that every rank reports the same `prepare_token`, then inspect `tasks_failed_since_last`, `layers_timed_out`, and `update_expert_weight`. Partial P2P participation normally appears as a prepare timeout. An incompatible shared-memory layout fails during startup with a layout error.
