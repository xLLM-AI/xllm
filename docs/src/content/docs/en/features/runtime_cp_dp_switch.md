---
title: "Runtime CP↔DP Mode Switching"
sidebar:
  order: 91
---
## Background

xLLM supports two parallel layouts for a fixed world size:

- **CP_PREFILL** (`cp=N, dp=1`): context parallelism across all workers. Long prompts split their KV compute across workers, minimizing TTFT for large-input requests.
- **DP_DECODE** (`cp=1, dp=N`): data parallelism, `N` independent decoder replicas. Higher concurrency and throughput for small requests, at the cost of no per-request work-splitting.

Traffic patterns rarely stay in one regime forever. Long-context ingest bursts favor CP; concurrent short-request traffic favors DP. Static provisioning per instance forces the operator to overprovision one lane or accept degraded metrics on the other.

**Runtime CP↔DP mode switching** lets an xLLM instance flip between the two layouts on demand, without a service restart, in ~500 ms.

## When to use it

- Traffic mix shifts across the day (long-context batch jobs at night, chat traffic during the day).
- You want to test-drive both layouts on a single instance rather than maintain separate deployments.
- You are building an autoscaling controller that decides layout per instance and needs a machine-actuated switch RPC.

Do **not** use it if:

- You are debugging a numerical accuracy problem — start with a fixed layout to eliminate flip as a variable.
- Your workload is uniformly one shape — the resident cost of holding both ATB graphs (~600 MB / NPU + extra graph workspace) is not free.

## How it works

### Startup: dual-graph mode

When the worker starts with `--enable_runtime_cp_dp_switch=true`, it initializes both parallel layouts up front:

- Two `CollectiveCommunicator`s. One with `cp_size=N,dp_size=1`, one with `cp_size=1,dp_size=N`. Each brings up its own HCCL comm domain on a distinct master port (offset by `--dual_mode_port_stride`, default 256).
- Two `ParallelArgs` snapshots wrapped in a `DualParallelArgs` with an `atomic<Mode>` discriminator. `active()` reads the current snapshot with acquire semantics.
- Four ATB nodes per decoder layer: `cp_prefill`, `cp_decode`, `dp_prefill`, `dp_decode`. Weights are shared; only the graph binding differs.

The initial mode is chosen from `--cp_size` at startup (`cp>1 → CP_PREFILL`, else `DP_DECODE`).

### The switch: an RPC that atomically flips the layout

Callers invoke `xllm.proto.ModeSwitchService.SwitchMode` on the instance's `--disagg_pd_port` (both prefill and decode instances host the service):

```bash
curl -sS -X POST \
    "http://<host>:<disagg_pd_port>/xllm.proto.ModeSwitchService/SwitchMode" \
    -d '{"target_mode":1,"timeout_ms":30000}'
```

`target_mode` is `0` for CP_PREFILL, `1` for DP_DECODE. The response carries `{"ok":true,"current_mode":<0|1>}` on success.

**Serialize P then D.** The decode side rebuilds datadist links using the prefill side's advertised `dp_size` from etcd. If you flip both in parallel and D wins the race, D reads P's stale dp_size and links the wrong topology. Ad-hoc curl scripts should always flip the prefill instance first, wait for its response, then flip the decode instance.

### The orchestration steps

`ModeSwitchService::SwitchMode` on each instance runs, in order:

1. **`scheduler.pause(WAIT)`** — stop admitting new batches; existing in-flight steps finish.
2. **`wait_until_paused(timeout_ms)`** — verify the drain completed; return an error to the caller if not (`{"error":"scheduler drain timeout"}`), leaving mode unchanged.
3. **`begin_switch()`** — acquire a `std::shared_mutex` unique lock. Six read paths in `DisaggPDScheduler` (`dispatch_requests`, `try_allocate`, `decode_schedule`, `decode_recv_first_generation`, `link_instance`, `unlink_instance`) hold this shared; the unique lock waits them out.
4. **`engine.switch_mode(target)`** — fan out `SwitchMode` RPC to every worker. Each worker flips its `DualParallelArgs::active_` atomic and refreshes its cached `parallel_args_` and `ModelContext`. The engine also updates its own `cp_size_/dp_size_/dp_local_size_` under the `paired = max(cp,dp)` invariant so subsequent scheduler polls see the new layout.
5. **`rebuild_after_flip(new_dp_size)`** — reconstruct `BlockManagerPool` for the new dp_size. Reuses the old pool's `Options` so KV-cache-cap-derived config isn't recomputed. Publishes the new pool pointer to `XServiceClient` (release-store on `atomic<const Pool*>`), which the heartbeat / reconcile threads acquire-load once per iteration.
6. **`re_register_dp_size(new)`** — write the new dp_size back to the etcd `InstanceInfo` so peers observe the post-flip topology when they reconcile.
7. **`gate.unlock()` + `scheduler.resume()`** — release the write lock and start admitting new batches. Steady-state cost after this point is one atomic acquire-load per step.
8. **`relink_after_flip`** — decode instances only. Unlink stale datadist connections and relink using the peer's fresh dp_size fetched from etcd. Runs *outside* the switch gate to avoid a deadlock between the datadist worker RPC handler and the caller thread that already holds the gate.

### Steady-state runtime concerns

**Lopsided-batch defer (DP mode).** DP forward requires every dp_rank to enter the collective with a shape-consistent input. If one dp_rank has a real batch and the other is empty, entering forward would either hang the peer (empty rank returns early) or race real KV blocks (empty rank fabricates a fake input). The scheduler defers such steps: if `active_dp_size > 1` and any dp_rank has an empty batch, `step()` returns without dispatching to the engine. A 100 ms backdoor forces the step if the imbalance persists — better to risk a fake-input path than to starve indefinitely. In practice the backdoor never fires (DP dispatch converges in <10 ms).

**Diagnostic logging.** The `enable_flip_verbose_log` gflag (default `false`) gates 12 per-request / per-step FLIPDIAG log sites. Flip lifecycle events (`switch_mode`, `rebuild_after_flip`, `relink_after_flip`, `lopsided_backdoor`, `EARLY_RETURN`) log unconditionally. Toggle at runtime via brpc's built-in endpoint (a `DEFINE_validator` marks the flag reloadable):

```bash
curl "http://<host>:<disagg_pd_port>/flags/enable_flip_verbose_log?setvalue=true"
```

## Failure modes and root causes

Six layers of concurrency issues surfaced during development. Each is documented here so future regressions have a starting point:

| # | Symptom | Root cause | Fix |
|---|---------|------------|-----|
| 1 | rank0 zombies during rebuild | dispatch handlers race `unique_ptr::reset` on `kv_cache_manager_` | `switch_gate_` unique_lock + pause/drain gate |
| 2 | rank0 dies ~5 s after every flip | `XServiceClient` heartbeat holds a raw pointer to the old (destroyed) pool | `std::atomic<const BlockManagerPool*>` + release/acquire; heartbeat pins one snapshot per iteration |
| 3 | flip-back CHECK failure `num_free_blocks_==free_blocks_.size()-1` | `BlockManagerImpl` destructor CHECK too strict; sequences release out of the original invariant window | Downgrade CHECK → WARNING |
| 4 | `PushKvBlocks failed, ret=5010b007` (LLM_NOT_YET_LINK) | Datadist link topology stale after a peer's `dp_size` change | `re_register_dp_size` + `relink_after_flip` (P-side skip; D-side fetches fresh `InstanceInfo`; serialized P → D at the caller) |
| 5 | DP forward hangs on `batch_sizes=[N,0]` | `dp_global_token_nums` says 0 for empty rank while worker's fake-input path made it 1; DP collective sizes disagree | Clamp `dp_global_token_nums[dp_rank] = max(1, ...)` in `LLMEngine::prepare_inputs` when `dp_size>1` |
| 6 | DP forward still occasionally hangs on lopsided batches | ATB attention op hangs when one shard uses real block_tables and another the placeholder path | Scheduler-layer defer: skip lopsided batches, 100 ms backdoor as safety valve |

## Verification

The reference `verify_switch.sh` regression exercises the following pattern:

1. HTTP probe to confirm the service is up.
2. Reset both P and D to CP mode as a preflight.
3. Warmup request.
4. **CP burst** (default 6 concurrent, `max_tokens=6`).
5. Idle wait until both P and D report `Running requests: 0`.
6. **Flip 0 → 1** (serialized P then D). Print each side's latency in ms.
7. Idle wait, **DP burst**.
8. Idle wait, **flip 1 → 0**.
9. Idle wait, **CP-post burst**.

Success criteria: 18/18 responses across the three bursts, both flips complete without `drain timeout`, no `main process disappeared` in either instance's rank0 log.

## Configuration reference

| Flag | Default | Purpose |
|------|---------|---------|
| `--enable_runtime_cp_dp_switch` | `false` | Bring up two `CollectiveCommunicator`s + four ATB nodes at startup. Required for flip to work. |
| `--dual_mode_port_stride` | `256` | Port offset between the CP and DP master ports. 256 avoids TIME_WAIT collisions on rapid restarts. |
| `--enable_flip_verbose_log` | `false` | Emit per-request/per-step FLIPDIAG lines. Runtime-toggleable via brpc `/flags` endpoint. |
| `--disagg_pd_port` | `7777` (non-disagg) / instance-specific (disagg) | Port `ModeSwitchService` listens on. |

## Not yet covered

- Real 61-layer DeepSeek-V3.2-w8a8 model validation on multi-host disagg (single-host validated).
- Cross-machine disagg flip.
- `lm_eval` numerical parity sweep between CP and DP with a shared seed.
- Long-run stability (>1 h with periodic flips), including memory / connection leak checks.
- Automatic mode-switch controller (framework scaffolding exists in this PR; wiring to real traffic signals is a follow-on PR).
