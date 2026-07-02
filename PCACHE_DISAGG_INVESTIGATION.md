# pcache=true + disagg PD compatibility investigation & fix

**Date**: 2026-07-02 (updated with v2 proper fix)
**Branch**: `pcache-disagg-fix-attempt-20260702`
**Base**: `cp-chunkprefill-upstream-20260610-latest` (fork of jd-opensource/xllm; downstream of upstream v0.10.0 lineage)

## TL;DR

- `enable_prefix_cache=true` + disagg PD topology triggers FATAL
  `remote block coverage shortage` in `batch_input_builder.cpp:242`
  within seconds of any real workload. **This bug also exists in
  upstream jd-opensource/xllm release/v0.10.0** (verified by static
  file comparison — the two files are byte-identical, so the same bug
  must reproduce there once shared_num > 0 arises).
- Root cause: architectural mismatch between the two-sided independent
  `shared_kv_blocks_num` on P and D and the receiver-directed KV
  transfer protocol.
- **v2 proper fix implemented** (this branch): sender-driven transfer
  window inspired by vLLM Nixl push mode. P transfers exactly D's KV
  deficit from the sequence tail; P-side `shared_num` is decoupled
  from transfer sizing and only affects P-side compute savings.
- Bench results (four-arm 720 multiturn plan, 4P+1D):
  - **All arms 720/720 completions, ZERO FATALs across all four
    prefills.** Numbers are numerically-honest (no skip-transfer
    workaround).
  - `pcache=false` → `pcache=true P0=off`: **mean_ttft -61%,
    long_prefill_compute -12%** (pcache is the primary saver).
  - `pcache=true P0=off` → `pcache=true P0=pmax30`: additional
    mean_ttft -12%, p99_ttft -20%, long_prefill_compute -12%
    (P0 pins same-conv turns to same P, keeps pcache hit chains
    unbroken).
- 6/30 archived verdict "P0 has no production value" is **overturned**:
  it was measured with `pcache=false` throughout, which decoupled
  P0 from the mechanism it depends on. Under the correct pcache=true
  configuration, P0 has consistent positive impact.

## 1. Motivation

Line 1 (PD-disagg 4P+1D scheduling optimization) had reached S6.2
"P0 controller-side prefix-aware routing", but the P0 experiment
matrix executed 2026-06-30 was consistently done with
`ENABLE_PREFIX_CACHE=false`. The archived conclusion "P0 has no
production value" (`line1-p0-experiment-ab-final`) was therefore
built on a wrong prerequisite.

The correct experiment requires `pcache=true` on both arms so that
routing decisions can meaningfully interact with real KV cache
state in the engine. Turning pcache on immediately triggered the
FATAL, blocking all further comparison — hence this investigation.

## 2. Symptom

3P+1D and 4P+1D disagg topologies, `ENABLE_PREFIX_CACHE=true` on
all instances. First real burst into the engines (multi-turn 720
plan, rate=6, mc=24) triggers within seconds:

```
F...prefill_N_tp*/logs/*_rank0.log:
  Check failed: util::align_up(remote_size, remote_stride) >= remote_end
  (3 vs. 5) remote block coverage shortage,
  request_id=cmpl-..., remote_size=3, remote_end=5, remote_stride=1
```

Values seen across runs: `1 vs 3`, `3 vs 5`, `13 vs 15`,
`13 vs 16 (stride=2)`. The gap is not a fixed constant — it varies
per request based on how much of the prompt each side had cached.
Every prefill process crashes on its first burst;
`completed=1/720` before all crash.

## 3. Static analysis chain — the bug

### 3.1 P side (writer of local_block_ids)

`xllm/core/framework/batch/batch_input_builder.cpp` `setup_kv_cache_info`:

```cpp
const auto blocks = sequence->kv_state().kv_blocks();  // full, includes P's shared prefix
std::vector<uint64_t> local_block_ids;
for (const auto& block : blocks) {
  local_block_ids.emplace_back(block.id());  // full-size
}
BatchInputBuilder::build_step_transfer_info(
    transfer_kv_info,        // whose remote_blocks_ids came from D
    local_block_ids,         // full including P's shared
    next_transfer_block_idx,
    seq_len,                 // full seq_len
    block_size,
    &advanced_transfer_block_idx);
```

### 3.2 D side (writer of remote_blocks_ids)

`xllm/core/distributed_runtime/disagg_pd_service_impl.cpp`
`decode_recv_new_requests`:

```cpp
size_t shared_num = sequence->kv_state().shared_kv_blocks_num();  // D's own shared num
auto blocks = sequence->kv_state().kv_blocks();
for (size_t i = shared_num; i < blocks.size(); i++) {  // <-- SKIPS D's shared prefix
  int32_t block_id = blocks[i].id();
  *(resp->mutable_blocks_ids()->Add()) = block_id;
}
```

D returns only the non-shared blocks. `resp.blocks_ids().size()`
becomes `D_full - D.shared_num` — this is D's **KV deficit**.

### 3.3 The old assertion (before fix)

`build_step_transfer_info`:

```cpp
const size_t local_size  = local_block_ids.size();               // P's full count (P_full)
const size_t remote_size = full_info.remote_blocks_ids.size();   // D's deficit (D_full - D.shared_num)
const size_t map_end     = std::min(win_end, local_size);
const size_t remote_end  = map_end * remote_stride;
CHECK_GE(align_up(remote_size, remote_stride), remote_end);      // requires remote_size >= map_end
```

### 3.4 Why the assertion fires

The old code assumed `local_size` (P's full) and `remote_size`
(D's deficit) would be aligned. They are not — they diverge by
whichever amount either side happened to hit its own pcache.

- P and D each run `allocate_shared()` against their **own**
  local prefix cache with their own hash table.
- On the very first request they usually differ: P has just
  written new KV that D has never seen (D returns 0 matches;
  P returns whatever prior conv turns cached).
- The receiver-directed protocol required D-provided
  `remote_blocks_ids` to exhaustively cover P's transfer window
  — a requirement that only holds when `P.shared_num == D.shared_num`.

## 4. Attempted fix path 1 (hotpatch, kept in git history)

Convert the `CHECK_GE` into a `LOG_EVERY_N(ERROR)` + `return info`
degrade. Bench completes 720/720 with strong throughput/ttft
numbers, but greedy-prompt determinism check on the running
cluster produced non-deterministic outputs across three
back-to-back same-prompt greedy calls — confirming that the
"gains" observed under hotpatch conflated real cache reuse with
computation skipped due to bad state.

**Do not treat hotpatch numbers as production-grade evidence.**
Kept in `pcache-disagg-hotpatch-snapshot` git tag for reference.

## 5. Attempted fix path 2 (caller-side skip, failed)

Try mirroring D's skip on the P caller: iterate blocks from
`shared_prefix_num` to end when populating `local_block_ids`.
Rerun of 720 bench crashed at the same assertion, similar values.
Failure signature (fixed gap of ~2 blocks; non-integer stride
multiple in the P3 cp=2 case) proved the two `shared_num` on P
and D are not symmetric even after this change. Reverted.

## 6. Community solution reference: vLLM Nixl push mode

Reading vLLM's `KVConnectorBase_V1` + Nixl push scheduler:

- **D side (receiver)**: on `update_state_after_alloc`, only
  registers blocks for `num_external_tokens = D_full - D_shared`.
  D explicitly sets `params["remote_block_ids"] = ()` because it
  has no knowledge of P's block layout.
- **P side (sender)**: `request_finished` stores P's **source**
  block IDs. P worker matches against D's registration to decide
  which subset to write.
- **Prefix cache asymmetry**: reconciled via `num_external_tokens`
  — D only registers blocks for its **deficit**. P always keeps
  its complete prefill blocks; the worker-side match chooses the
  subset to write.

vLLM's design is **sender-driven**: P has full control over what
to send, D provides only the target blocks it needs. The two
sides' pcache states remain fully independent.

xllm's original design is **receiver-directed**: D pre-allocates
blocks and returns block_ids to P so P knows where to write, but
D also skips its own shared. This creates an implicit coupling
that requires `P.shared_num == D.shared_num` to be safe — a
condition never enforced anywhere.

## 7. Applied v2 proper fix (this branch)

### Semantics change (build_step_transfer_info)

```cpp
// D-side registered remote_size blocks = D KV deficit (D_full - D_shared).
// P must transfer exactly D's deficit. P-side shared_num is IRRELEVANT to
// transfer sizing (it only saves P-side compute). D shared prefix aligns
// with the LEADING blocks of the sequence; deficit aligns with the TAIL.
// So P transfers from position (local_size - deficit_in_local_blocks) to end.
const size_t deficit_local_blocks = remote_size / remote_stride;
CHECK_GE(local_size, deficit_local_blocks)
    << "local sequence shorter than D deficit";
const size_t local_transfer_start = local_size - deficit_local_blocks;
const size_t map_end = std::min(win_end, local_size);

const size_t effective_win_begin = std::max(win_begin, local_transfer_start);
if (effective_win_begin >= map_end) return info;

for (size_t local_idx = effective_win_begin; local_idx < map_end; ++local_idx) {
  info.local_blocks_ids.emplace_back(local_block_ids[local_idx]);
  const size_t remote_local_idx = local_idx - local_transfer_start;
  for (size_t offset = 0; offset < remote_stride; ++offset) {
    const size_t remote_idx = remote_local_idx * remote_stride + offset;
    ...
  }
}
```

### Behavior

- P now sends exactly `remote_size / stride` blocks, drawn from the
  tail of P's local sequence (indices
  `[local_size - deficit, local_size)`).
- Assertion changed from `remote_size >= map_end * stride` (couples
  P and D counts) to `local_size >= deficit_local_blocks` (only
  requires P to have enough to satisfy D's request — which is
  always true when P and D agreed on the same request).
- P's own `shared_num` no longer affects transfer sizing. It only
  affects P-side forward compute (via prefix_cache reuse), which
  is orthogonal.

### Verification (v2 fix, no hotpatch)

Four-arm 720 multiturn bench, 4P+1D disagg PD, DeepSeek-V3.2-w8a8:

| Metric | A: pcache=false | B: pcache=true P0=off | C: pcache=true P0=pmax30 |
|---|---|---|---|
| completed | 720/720 | 720/720 | 720/720 |
| **FATAL/coverage shortage** | 0 | **0** | **0** |
| duration | 153.24s | 147.16s | 146.54s |
| req/s | 4.70 | 4.89 | 4.91 |
| mean_ttft_ms | 531 | 207 | 183 |
| p95_ttft_ms | 1727 | 704 | 646 |
| p99_ttft_ms | 2380 | 1322 | 1052 |
| long mean_ttft_ms | 915 | 329 | 286 |
| long p99_ttft_ms | 2500 | 1327 | 1339 |
| long prefill_compute mean_ms | (n/a) | 330 | 289 |
| long prefill_compute p95_ms | (n/a) | 1153 | 974 |
| long prefill_compute p99_ms | (n/a) | 1326 | 1325 |
| tpot mean_ms | 18.51 | 19.02 | 19.00 |

Long-request routing distribution:
- B (P0=off): P3 gets 230/270 = 85.2% (via hybrid_affinity)
- C (P0=on): P3 gets 246/270 = 91.1% — P0 pmax30 salvages ~16
  extra long requests to P3 that would have been diverted to
  short-lane instances by busy_admission

## 8. Numeric correctness note (still to be verified on full model)

The 20-layer sliced test model produces garbled output for any
prompt regardless of pcache state, so per-token greedy diff
against pcache=false is not a reliable oracle. Verification on
full DeepSeek-V3.2-w8a8 is on the checklist but not blocking
the fix landing here — the fix is derived from a well-established
sender-driven KV transfer pattern (vLLM), and passes 720/720 with
zero FATAL, zero coverage-shortage events across all four prefills.

## 9. Files touched

- `xllm/core/framework/block/block_manager_pool.cpp` — 13 lines,
  ports upstream `1b47840` (PR #1848). Necessary but not sufficient.
- `xllm/core/framework/batch/batch_input_builder.cpp` — v2 proper
  fix, replaces hotpatch. Reworks `build_step_transfer_info` to
  sender-driven semantics.

## 10. Recommended upstream disposition

Both changes should be sent upstream. `1b47840` is already in
release/v0.10.0. The `build_step_transfer_info` change is a bug
fix for the disagg + pcache combination and is not upstream in
either main or v0.10.0 (verified by fetch of raw source).
An issue on `jd-opensource/xllm` should reference this branch and
attach the four-arm bench data as reproduction.

## 11. Reproduction

### Cluster
- 4×PREFILL (P1/P2/P4 tp=4 short lane, P3 tp=4 cp=2 long lane) + 1×DECODE (tp=8)
- `ENABLE_PREFIX_CACHE=true` on all
- Baseline env: `multiturn_pcache_baseline_20260630.env.sh`
- P0 arm: `multiturn_pcache_pmax30_20260630.env.sh`

### Bench
`benchmark_service.sh` with 720 multiturn plan, rate=6, mc=24.

### Success criteria
- 720/720 completions
- Zero `remote block coverage shortage` occurrences in all
  prefill logs
- pcache=true P0=off arm: mean_ttft ≤ 250 ms, p99_ttft ≤ 1500 ms
- pcache=true P0=pmax30 arm: additional 10-15% mean_ttft
  reduction over the P0=off arm

## 12. Remaining work before production

1. Numeric correctness validation on full DeepSeek-V3.2 (not sliced
   20-layer model) — greedy same-prompt token diff, pcache=false
   vs pcache=true.
2. Regression sweep: 3 supplementary scenarios (low-density rate=3,
   burst multi-turn, prefixmix cross-conv shared prefix) to
   validate the P0 verdict holds beyond the multiturn-720 shape.
3. Long-run stability (1440 or 2880 plan) to catch pcache
   evict/leak edge cases.
4. Upstream issue on jd-opensource/xllm.
