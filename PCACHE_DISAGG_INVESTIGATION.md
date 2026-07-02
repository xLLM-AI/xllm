# pcache=true + disagg PD compatibility investigation

**Date**: 2026-07-02
**Branch**: `pcache-disagg-fix-attempt-20260702`
**Base**: `cp-chunkprefill-upstream-20260610-latest` (fork of jd-opensource/xllm)

## TL;DR

- `enable_prefix_cache=true` + disagg PD topology triggers FATAL
  `remote block coverage shortage` in `batch_input_builder.cpp:242`
  within minutes of any real workload.
- Root cause is **architectural, not local**: prefill and decode instances
  each hit their own pcache with independent `shared_kv_blocks_num`, but
  the KV transfer protocol has no offset negotiation to reconcile the
  two counts.
- Backporting upstream `1b47840` (PR #1848 "fix prefix match_kv recompute")
  is necessary but insufficient — that fix addresses the P-side allocation
  count, not the P/D shared_num mismatch on the transfer path.
- Applied a temporary hotpatch (log-and-degrade at the assertion) to
  unblock benchmarking and gather profiling evidence. Runs to 720/720
  completion with strong observed throughput/ttft, but greedy token
  comparison shows **non-deterministic output** — the "improvement" is
  partly downstream masking of missed KV transfers, not verified numeric
  gain.
- **Proper fix requires engine-team-level protocol changes.** Options
  outlined in Section 5.

## 1. Motivation

We wanted to validate whether P0 controller-side prefix-aware routing
provides real production value. Prior conclusion (dated 2026-06-30,
memory `line1-p0-experiment-ab-final`) was "P0 no production value,
work archived", but that verdict was reached with `ENABLE_PREFIX_CACHE=false`
throughout — a config error that was only recognized after the archive.

The correct experiment requires `pcache=true` on both the P0-off baseline
arm and the P0-on pmax30 arm, so the routing decision can meaningfully
interact with real KV cache state in the engine. That is what this branch
aims to unblock.

## 2. Symptom

3P+1D and 4P+1D disagg topologies, `ENABLE_PREFIX_CACHE=true` on all
instances. First real burst into the engines (multi-turn 720 plan,
rate=6, mc=24) triggers within seconds:

```
F...prefill_N_tp*/logs/*_rank0.log:
  Check failed: util::align_up(remote_size, remote_stride) >= remote_end
  (3 vs. 5) remote block coverage shortage,
  request_id=cmpl-..., remote_size=3, remote_end=5, remote_stride=1
```

Values seen across multiple runs: `1 vs 3`, `3 vs 5`, `13 vs 15`,
`13 vs 16 (stride=2)`. The gap is not a fixed constant — it is
`remote_end - remote_size` where the mismatch varies per request based
on how much of the prompt each side had cached.

Every prefill process crashes on its first burst. Bench progresses to
`completed=1/720` before all four prefills die.

## 3. Static analysis chain

### 3.1 P side (writer of local_block_ids)

`xllm/core/framework/batch/batch_input_builder.cpp` `setup_kv_cache_info`:

```cpp
const auto blocks = sequence->kv_state().kv_blocks();  // full, includes shared prefix
std::vector<uint64_t> local_block_ids;
for (const auto& block : blocks) {
  local_block_ids.emplace_back(block.id());  // full-size
}
// ...
BatchInputBuilder::build_step_transfer_info(
    transfer_kv_info,        // whose remote_blocks_ids came from D
    local_block_ids,         // full including shared
    next_transfer_block_idx,
    seq_len,                 // full seq_len
    block_size,
    &advanced_transfer_block_idx);
```

### 3.2 D side (writer of remote_blocks_ids)

`xllm/core/distributed_runtime/disagg_pd_service_impl.cpp`
`decode_recv_new_requests`:

```cpp
size_t shared_num = sequence->kv_state().shared_kv_blocks_num();
auto blocks = sequence->kv_state().kv_blocks();
for (size_t i = shared_num; i < blocks.size(); i++) {  // <-- SKIPS D's shared prefix
  int32_t block_id = blocks[i].id();
  *(resp->mutable_blocks_ids()->Add()) = block_id;
  block_ids.push_back(block_id);
}
```

D returns only the non-shared blocks. `resp.blocks_ids().size()`
becomes `D_full - D.shared_num`.

### 3.3 P side scheduler (importer of D's response)

`xllm/core/scheduler/disagg_pd_scheduler.cpp`:

```cpp
TransferKVInfo info;
for (auto& bid : resps.resps()[i].blocks_ids()) {
  info.remote_blocks_ids.emplace_back(bid);  // takes D's skipped list verbatim
}
sequence->kv_state().set_transfer_kv_info(std::move(info));
```

### 3.4 The assertion

`build_step_transfer_info`:

```cpp
const size_t local_size  = local_block_ids.size();      // = P_full
const size_t remote_size = full_info.remote_blocks_ids.size();  // = D_full - D.shared_num
const size_t win_end     = ceil_div(seq_len, block_size);
const size_t map_end     = std::min(win_end, local_size);
const size_t remote_stride = kv_split_size_effective();
const size_t remote_end  = map_end * remote_stride;
CHECK_GE(align_up(remote_size, remote_stride), remote_end);
```

### 3.5 Why the assertion fires

`local_size` is P's full count (including P's own shared prefix).
`remote_size` is D's full count minus D's shared prefix.

If `P.shared_num == D.shared_num == k`:
`remote_size = full - k`, `local_size = full`, `map_end = full`,
`remote_end = full`. Assertion FAILS by `k`.

If `P.shared_num != D.shared_num`:
Assertion FAILS by an even more variable amount.

There is no reason for the two shared counts to be equal — each side
runs its own `allocate_shared()` against its own local prefix cache
using its own hash table. On the very first request they usually differ
because P has just written new KV that D has never seen (D returns 0
matches; P returns whatever prior conv turns cached).

## 4. Failed fix attempt: caller-side skip

Attempted:

```cpp
// P side, in setup_kv_cache_info
const size_t shared_prefix_num = sequence->kv_state().shared_kv_blocks_num();
size_t block_idx = 0;
for (const auto& block : blocks) {
  block_ids.push_back(block.id());              // stays full for local forward
  if (block_idx >= shared_prefix_num) {
    local_block_ids.emplace_back(block.id());   // only non-shared
  }
  ++block_idx;
}
```

Rationale: mirror D's skip so `local_size` and `remote_size` become
symmetric (both = full - shared).

**Failed**. Rerun of 720 bench crashed at the same assertion, with
similar values (`1 vs 3`, `3 vs 5`, `12 vs 26 stride=2`). The failure
signature (fixed gap of ~2 blocks, or non-integer stride multiple in
`P3`'s cp=2 case) proves the two `shared_num` on P and D are not
symmetric even after this change.

Mode B verdict: this cannot be fixed by reconciling one caller;
it needs a protocol-level change.

## 5. Proper fix options (open questions for engine team)

### (a) Add `remote_shared_offset` to TransferKVInfo proto

Extend `TransferKVInfo` (both `xllm/proto/worker.proto` and
`xllm/core/common/types.h`) with:

```proto
uint32 remote_shared_offset = N;  // D's shared_kv_blocks_num, informs P
```

D populates it in `decode_recv_new_requests` before responding. P uses
it in `build_step_transfer_info` to compute the correct `remote_end`
and `remote_idx` mapping. Requires proto version bump.

### (b) D sends full blocks_ids, P decides skip

D returns `blocks[0..full]` unconditionally (no `for i = shared_num` skip),
plus a small counter `d_shared_num`. P receives full remote list, applies
its own skip logic based on the smaller of `P.shared_num` and `D.shared_num`
(the actually-shared prefix that both agree on). Wire size increases
slightly for shared blocks that never travel, but semantics get clean.

### (c) Hash-based coordination

At request enqueue, P and D exchange prefix hashes and settle on a common
`shared_num` before either allocates. Cleanest semantically but adds an
extra RPC round-trip on the request-critical-path.

### Recommendation

Prefer (a) or (b). (a) is minimally invasive (proto field + two files
touched); (b) is more permissive (P side has full flexibility). Neither
is a leaf-level patch — both require coordinated updates across engine
scheduler, disagg_pd_service, batch_input_builder, and possibly worker.

## 6. Comparable systems

### vLLM disaggregated prefill

- Uses `KVConnector` abstraction with explicit `bind_kv_caches` +
  `send_kv_caches_and_hidden_states` / `recv_kv_caches_and_hidden_states`
  interfaces.
- Prefix caching (block hash allocator) runs on both P and D
  independently, but the transfer path relies on P telling D exactly
  which block indices it produced this step, not on D pre-allocating
  and returning a target list.
- Effectively equivalent to option (b): the receiver adjusts to
  the sender's payload rather than having a pre-negotiated target.

### SGLang disaggregation

- Uses `MooncakeKVSender/Receiver` for token-level KV transfer.
- Radix-tree-based prefix cache lives on the "prefill node"; the
  "decode node" pulls tokens on-demand. No two-sided shared_num
  mismatch because the decode side does not maintain its own prefix
  hash for cross-machine sharing — it only caches locally.

### Implication for xllm

xllm's disagg design (D pre-allocates blocks and returns block_ids to P
so P knows where to write) is the source of the P/D shared_num
mismatch. Both vLLM and SGLang avoid this by making the transfer path
sender-driven rather than receiver-directed.

If we adopt option (b) (D sends full list, P skips), the semantic
becomes closer to vLLM's model. This is the smaller change and is
recommended.

## 7. Empirical evidence gathered under hotpatch

(caveat: numeric correctness NOT verified — see Section 8)

Three-arm 720 bench, 4P+1D topology, multiturn plan rate=6 mc=24:

| metric | A: pcache=false | B: pcache=true, P0=off | C: pcache=true, P0=pmax30 |
|---|---|---|---|
| completed | 720/720 | 720/720 | 720/720 |
| duration | 153.24s | 145.42s | 144.01s |
| req/s | 4.70 | 4.95 | 5.00 |
| mean_ttft_ms | 531 | 201 | 111 |
| p95_ttft_ms | 1727 | 606 | 231 |
| p99_ttft_ms | 2380 | 1344 | 973 |

Engine-side stage_trace:

| metric | B | C |
|---|---|---|
| long prefill_compute_ms mean | 333 | 109 (-67%) |
| long prefill_compute_ms p95 | 1148 | 157 (-86%) |
| long queue_before_prefill_ms | 16.3 | 15.1 |

Long-request routing distribution:

| instance | B | C |
|---|---|---|
| P3 (long lane) | 245 (91%) | 260 (96%) |

P0 pmax30 amplifies P3's magnetism for prefix-hit long requests, and
P3's engine-side pcache eliminates two-thirds of the actual prefill
compute for those.

## 8. Numeric correctness caveat

Same greedy prompt sent 3 times (temperature=0, seed fixed) to the
running hotpatched cluster produced three different outputs:

- Run 1 (cache miss):  ` WHY WHYjenkszeleshnerzemirkaleyounger-than-than...`
- Run 2 (cache hit):   ` WHY WHYjenkszeleshnerzemetaexpressivityfeetwidebandwidths...`
- Run 3 (cache hit):   ` WHY WHYjenkszeleshnerzemetaexpressivityfeetwidecurrrently...`

First 8 tokens match, then divergence. Runs 2 and 3 agree for another
~10 tokens, then also diverge. This confirms the hotpatch's skip-transfer
behavior corrupts KV state in a non-deterministic way. The "-67%
prefill compute" number likely conflates real cache reuse with
computation skipped due to bad state.

**Do not treat the hotpatch numbers as production-grade evidence.**
They establish direction and magnitude but not the final claim.

## 9. Reproduction

### Build

```bash
# In xllm_chunk_latest source tree
cp path/to/patched/block_manager_pool.cpp   xllm/core/framework/block/
cp path/to/patched/batch_input_builder.cpp  xllm/core/framework/batch/
cd build/cmake.linux-aarch64-cpython-311/
ninja xllm  # ~5-10 min incremental
```

### Cluster

- 4×PREFILL (P1/P2/P4 tp=4 short lane, P3 tp=4 cp=2 long lane) + 1×DECODE (tp=8)
- `ENABLE_PREFIX_CACHE=true` on all
- Baseline env: `multiturn_pcache_baseline_20260630.env.sh`
- P0 arm: `multiturn_pcache_pmax30_20260630.env.sh`

### Bench

`benchmark_service.sh` with 720 multiturn plan, rate=6, mc=24.

## 10. Files touched

- `xllm/core/framework/block/block_manager_pool.cpp` — 13 lines,
  ports upstream `1b47840`
- `xllm/core/framework/batch/batch_input_builder.cpp` — 15 lines,
  hotpatch (revert before merging any real fix)

## 11. Next steps

1. File upstream issue on jd-opensource/xllm covering Sections 3, 4, 5.
2. Discuss options (a) / (b) / (c) with engine team; recommend (b).
3. Once proper fix is available, revert hotpatch commit (`8ab2bcb`),
   apply real fix, rerun three-arm 720, greedy-token-diff verify.
4. Only then are the pmax30 numbers publishable.
