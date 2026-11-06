---
title: "Adaptive Speculation"
sidebar:
  order: 85
---

<!-- Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

Adaptive speculation adjusts draft generation or target verification to improve
predicted output throughput. The two switches describe different execution
controls, rather than naming a speculative algorithm:

| Control | Meaning | Current implementation and rationale |
|:--------|:--------|:-------------------------------------|
| `enable_adaptive_draft_step` | Change how many sequential draft forward steps are executed before verification. | MTP generates candidates sequentially, so a shorter depth avoids draft forward work as well as shortening verification. |
| `enable_adaptive_target_verify` | Change how many already-generated candidates the target verifies. | DSpark uses its confidence head; MTP uses selected-token draft probabilities. Both keep draft generation width fixed and prune target work. |

Both controls default to `false` and cannot currently be enabled together.
An algorithm not listed above is not implicitly supported by the generic name.
Parameter types and defaults are listed in the
[CLI reference](/en/cli_reference/#speculativeconfig).

## Adaptive draft steps (MTP)

`num_speculative_tokens` sets the maximum depth. Before submitting a decode
batch, the engine selects one depth for that whole batch, using prefix
acceptance probabilities from completed verification results and a calibrated
full-cycle cost for each candidate depth. The objective is
`B * (1 + sum(prefix_acceptance[:depth])) / cost_ms`.

Startup measures the current MTP service, including its draft forwards, target
verification, sampling, prefix refresh and result retirement. It shares the SPS
and additive fitting implementation with target verification, but does not reuse
DSpark's measured costs. Predictions use physical draft request capacity and
padded target graph rows. Online feedback comes from ordinary result retirement,
without additional confidence D2H or a TP decision broadcast.

Current support requires an NPU GLM DSA Python model, ordinary prewarmed ACL
graphs, asynchronous scheduling, task pipeline, and DP=CP=1. Unified MTP graphs
and PD disaggregation are not supported. Apply the settings to every rank of an
MTP launch. This example starts rank zero of a 16-rank service; start the other
ranks with their own `node_rank` and listening port:

```bash
export HCCL_OP_EXPANSION_MODE=AIV
xllm --model=/models/target --draft_model=/models/mtp \
  --backend=llm --model_impl=python --npu_kernel_backend=TORCH \
  --nnodes=16 --node_rank=0 --host=127.0.0.1 --port=8000 \
  --master_node_addr=127.0.0.1:27000 \
  --speculative_algorithm=MTP --num_speculative_tokens=5 \
  --enable_adaptive_draft_step=true --enable_graph=true \
  --python_graph_backend=aclgraph --disable_graph_warmup=false \
  --enable_unified_mtp_graph=false --enable_schedule_overlap=true \
  --enable_task_pipeline=true --dp_size=1 --cp_size=1 \
  --max_seqs_per_batch=16 --acl_graph_decode_batch_size_limit=16
```

### Optional draft-step configuration

`adaptive_draft_step_config` is the **path to a JSON configuration file**, not
an enable switch or a precomputed cost table. Omit it to use built-in defaults;
the service still calibrates costs at startup and logs all effective settings.
The file has two independent sections:

| Section | Purpose | Defaults |
|:--------|:--------|:---------|
| `policy.window_size` | Completed-batch window for per-position prefix acceptance rates. | `20` batches |
| `policy.update_interval` | Rescore candidate depths after this many completed batches. | `5` batches |
| `policy.switch_min_gain` | Minimum predicted relative throughput gain for changing depth. | `0.05` (strictly more than 5%) |
| `calibration` | Startup sampling grid, context/KV coverage, warmup and measurement windows, and SPS/additive fitting choice. | Additive model, actual captured request buckets up to 128, prefill up to 2048 and KV coverage up to 4096, subject to model/cache capacity. |

For example, pass `--adaptive_draft_step_config=/configs/draft-steps.json` with:

```json
{
  "version": 1,
  "policy": {
    "window_size": 20,
    "update_interval": 5,
    "switch_min_gain": 0.05
  },
  "calibration": {
    "model": "additive",
    "request_buckets": [1, 2, 4, 8, 16],
    "contexts": [
      {"prefill_tokens": 2048, "min_kv_length": 1, "max_kv_length": 4096}
    ],
    "warmup_steps": 8,
    "window_steps": 32,
    "windows_per_point": 3,
    "token_seed": 1
  }
}
```

Either section may be omitted to keep its defaults. Explicit sampling must fit
the actual prewarmed graphs and KV capacity; it changes startup coverage/time,
not the client's prompt or sampling parameters. Each depth gets its own measured
costs, and loading precomputed `step_cost_models` is not supported by this option.

## Adaptive target verification with MTP logits

For a greedy MTP Draft5/7, `enable_adaptive_target_verify=true` executes all
configured draft steps, then prunes verification using
`exp(selected_logit - logsumexp(draft_logits))` as an acceptance proxy. This
is not a measured target acceptance probability. It reuses the existing
prefix-allocation, packing and commit kernels rather than another MTP-only
pruning kernel.

The Host chooses a padded target graph budget from feedback retired through
round n-2 and full-cycle costs measured by the current MTP service. Current
probabilities stay on Device for prefix allocation; their small asynchronous
feedback copy shares normal result retirement, without a new draft-to-target
Host wait. Packed Target hidden states are gathered by Device offsets before
MTP context publication and prefix refresh.

Use the MTP launch above with these settings instead of adaptive draft steps:

```bash
--num_speculative_tokens=5 --enable_adaptive_draft_step=false \
--enable_adaptive_target_verify=true --enable_probabilistic_draft=false
```

This path requires ordinary Python GLM DSA ACL graphs, asynchronous scheduling,
task pipeline and DP=CP=1. Greedy and random targets (including mixed batches)
reuse the existing ragged verifier; the draft must remain greedy. Sampled
target IDs reach TP consensus before prefix commitment or context publication.
Unified MTP graphs and simultaneous adaptive draft steps are rejected before
execution. Startup costs use a greedy target; random sampling adds sampler work
that is not measured by that calibration. The `adaptive_target_verify_*` cost-table and
startup calibration options below also apply; never reuse a DSpark cost table
for MTP.

## Adaptive target verification (DSpark)

The current DSpark implementation uses lagged confidence feedback and calibrated
full-step costs to select a verification budget, keeping draft generation width
fixed at startup.

The adaptive-pruning flag and JSON key are now `enable_adaptive_target_verify`
instead of `enable_adaptive_spec_prune` or `enable_adaptive_speculative_decode`. Update existing launch
scripts and JSON configurations; the old name is not an alias.

### Supported configuration

The supported adaptive task pipeline uses an NPU GLM DSA Python target, a DSpark
draft with a confidence head, `num_speculative_tokens=5` or `7`, greedy draft sampling,
CP=1, ACL graphs, schedule overlap and task pipeline. Ordinary ACL graphs also
support data parallelism. All DP shards receive the same retired n-2 confidence
metadata and select a common physical target graph, with separate local draft
budgets. This adds no Device readback or collective between Draft and Target.
Graph warmup must be
enabled. Add `--enable_unified_mtp_graph=true` to capture each complete adaptive
round, including the greedy heads, confidence consensus, packing, verification
and accepted-context update. Unified graphs currently require DP=1 and `temperature=0`;
positive or mixed temperatures must use ordinary graphs. Such requests are
rejected before execution when unified graphs are enabled, never silently
decoded greedily. Fixed-width DSpark and other algorithms retain their own
execution contracts.

For DP=2 use `--dp_size=2 --enable_unified_mtp_graph=false`. Calibration request
buckets are per-DP-shard capacities: a bucket of four uses eight simultaneous
synthetic requests. Keep the global sequence/token limits large enough for all
shards. Saved cost tables include DP topology in their compatibility metadata;
do not reuse a DP=1 table for DP=2. Both fixed-width and adaptive DSpark reject
DP>1 with `--enable_unified_mtp_graph=true`; the requested mode is never silently
replaced with split graphs.

### Launch example

Add the following settings to **every rank** of an existing multi-rank launch.
This command illustrates rank zero of a 16-rank service; start the remaining
ranks with their own `node_rank` and listening port. Create the output directory
before launch and choose an output file that does not already exist:

```bash
mkdir -p /results
export HCCL_OP_EXPANSION_MODE=AIV
xllm --model=/models/target --draft_model=/models/dspark \
  --backend=llm --model_impl=python --npu_kernel_backend=TORCH \
  --nnodes=16 --node_rank=0 --host=127.0.0.1 --port=8000 \
  --master_node_addr=127.0.0.1:27000 \
  --speculative_algorithm=DSpark --num_speculative_tokens=7 \
  --enable_adaptive_target_verify=true \
  --enable_graph=true --python_graph_backend=aclgraph \
  --disable_graph_warmup=false --enable_schedule_overlap=true \
  --enable_task_pipeline=true --enable_unified_mtp_graph=false \
  --dp_size=1 --cp_size=1 --max_seqs_per_batch=16 \
  --acl_graph_decode_batch_size_limit=16 \
  --adaptive_target_verify_cost_table_output=/results/dspark-costs.json
```

Send requests with an explicit temperature: `temperature=0` selects greedy
target decoding; a positive value such as `temperature=0.5` selects random
target sampling. Different temperatures can share a batch. Target top-k, top-p
and min-p filtering use the ordinary sampler's semantics. The draft remains
greedy by default; probabilistic drafts are not admitted.

For random targets, the rejection sampler maps request parameters through the
device-side packed-row owners and draws one target token per verification row.
With a deterministic draft this is exact point-mass rejection sampling: matching
draws accept the draft prefix, and the first mismatch supplies the residual
token; an all-matching prefix uses its terminal draw as the bonus. All TP ranks
agree on the draws before acceptance and context-KV publication. Target/draft
forward and adaptive packing retain ACL graph replay; random sampling and
prefix commit execute on the task stream outside the model graph, without
reading the packed boundaries back to Host. Per-request seeds, logprobs,
structured constraints, minimum-length stop masks, token bias and non-neutral penalties remain unsupported
and are rejected before execution rather than silently ignored.

The budget is still selected ahead of the round from eligible lagged confidence;
current target draws never feed back into the same round's clipping decision.
Greedy-draft confidence need not predict stochastic acceptance accurately.
Startup calibration currently uses greedy target requests, so its full-step
costs do not include random-sampling overhead. Random-target correctness does
not imply the same acceptance rate, optimal budgets or performance as greedy
targets; measure them separately.

For Draft5, change only `--num_speculative_tokens=5` and select a new output
file such as `/results/dspark-costs-draft5.json`. The draft generates five
candidates; adaptive verification can select a shorter prefix. The three
device kernels use width-specific compiled specializations, and Slot storage
is sized to the selected width rather than padded to Draft7. The width is
fixed at startup, included in graph binding keys and saved-table compatibility,
and cannot change between replays. Draft5 and Draft7 costs must not be mixed.
Unversioned legacy tables without a `draft_width` field are treated as Draft7;
newly fitted bare tables record their width explicitly.

### Startup calibration

No HTTP calibration client, external fitting command or second service launch
is needed. Startup first warms the graphs, drives the real asynchronous
prefill/decode pipeline, collects in-memory completion windows, fits costs on
CPU and installs the same immutable table on every worker before ready. This
does not add per-round timing Events, profiling D2H copies or file writes to
normal serving. Existing result-readiness Events and asynchronous
result/confidence feedback remain necessary.

#### Built-in defaults

With no calibration file or saved table, the built-in settings are:

| Setting | Effective default |
|:--------|:------------------|
| `model` | `additive` |
| `draft_width` | Selected `num_speculative_tokens`; `5` or `7` |
| `request_buckets` | Actual captured physical draft request buckets |
| `target_graph_rows` | All legal captured target tiers, including full width, for each request bucket |
| `contexts` | One interval: prefill 2048 tokens, KV coverage 1–4096 |
| `warmup_steps` | `8` |
| `window_steps` | `32` complete decode intervals per window |
| `windows_per_point` | `3` |
| `token_seed` | `1` |

The default KV upper bound is capped by the model's context limit. For a shorter
context, prefill is reduced to reserve the worst-case generated tokens and both
in-flight slots. If the default sampling steps still do not fit, startup fails
with a request for a smaller explicit configuration. Sampling every captured
bucket can increase startup time; use a JSON file to restrict the grid.

#### Custom configuration

To override the defaults, add `--adaptive_target_verify_calibration=/configs/dspark.json`.
The width is inferred from the launch configuration; an optional `draft_width`
field in the JSON must match it. For example:

```json
{
  "model": "additive",
  "request_buckets": [1, 2, 4, 8, 16],
  "contexts": [
    {"prefill_tokens": 1868, "min_kv_length": 1800, "max_kv_length": 4096}
  ],
  "warmup_steps": 8,
  "window_steps": 32,
  "windows_per_point": 3,
  "token_seed": 1
}
```

#### Cost models and coverage

`model` accepts `sps`, `additive`, or `graph`. The SPS model is a one-dimensional
steps-per-second table over physical target graph rows `G`; cost is
`1000 / SPS(G)` milliseconds. The additive model fits full-step seconds as
`bias + alpha(Bcap) + theta(G)`, where `Bcap` is the physical draft request
bucket, not logical HTTP concurrency. These are full-step costs: do not add draft
or commit time again. The additive probe grid must be connected through shared
G values. Request buckets and any optional `target_graph_rows` must be captured;
the grid must retain full-width coverage. Context intervals must not overlap and
must fit the model, cache and worst-case sampling progress.

Unified startup captures at most four target tiers per actual request count
`B`: the first prepared buckets covering `B`, `2B`, `4B` and
`B * (draft_width + 1)`. For example, Draft5 at `B=8` uses `G=8,16,32,48`.
Calibration uses only complete `B/G` bindings captured for both pipeline slots;
an explicit unsupported tier fails before sampling. Transient request sets use
their prewarmed full-width binding and do not count as stable samples. Serving
never captures a missing binding. Split and unified costs are distinct, and
saved bundles validate the unified setting before installation.

Both default and file-based startup log `DSpark adaptive effective configuration`
with the source, all effective settings (including omitted defaults) and actual
captured geometry. No serving request causes profiling, fitting or graph
recapture. A KV range or request bucket without a calibrated profile uses the
already-prewarmed full-width admission policy, not an eager retry.

### Graph-cost-aware verification on NPU

Set the startup calibration `model` to `graph` to retain the median full-step
latency for each measured `(physical request bucket Bcap, physical target graph
G, KV interval)` binding. Costs for the same G remain distinct across Bcap;
latency jumps and non-monotonic costs between graph buckets are preserved.
The existing budget planner uses n−2 or older confidence and maximizes expected
output tokens divided by the measured full-step cost among prewarmed graphs.

The graph-cost selection approach references
[DeepInfer MTPVarlen](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/csrc/engine/mtp_varlen.hpp).
xLLM measures stable completion windows in the existing NPU asynchronous
pipeline, including draft, verification, commit and retirement. CUDA timing
Events are not imported, and draft/commit costs are not counted twice. The
resulting table is constructed at startup and installed consistently on all
workers; this mode does not add online timing regression.

For example, create `/configs/npu-graph-cost.json`:

```json
{
  "model": "graph",
  "request_buckets": [1, 2, 4, 8],
  "contexts": [
    {"prefill_tokens": 128, "min_kv_length": 1, "max_kv_length": 1024}
  ],
  "warmup_steps": 4,
  "window_steps": 8,
  "windows_per_point": 3,
  "token_seed": 1
}
```

Use these options in the NPU adaptive-target-verification launch command above:

```bash
--adaptive_target_verify_calibration=/configs/npu-graph-cost.json \
--adaptive_target_verify_cost_table_output=/results/npu-graph-costs.json
```

Request buckets must exist in the service's captured geometry. Omitting
`target_graph_rows` measures every legal prewarmed graph for each Bcap. An
explicit sparse selection must retain the captured full-width graph for every
Bcap. Unmeasured graphs are excluded from cost scoring. If the current logical
batch has no measured full-width cost, admission uses the existing prewarmed
full-width policy. A table without measured physical-request-bucket full-width
coverage is rejected before serving.
`graph_costs` stores `draft_requests`, `target_graph_rows` and `step_time_ms`.
Graph tables require an explicit `max_kv_length` and a versioned bundle with
compatibility metadata; they cannot use the legacy bare-table loading path.
The enclosing bundle retains model, topology, Draft5/7 and KV coverage checks.

Exact graph tables do not require the connected-grid assumption of additive
fitting or pool different Bcap measurements into one G-only cost. Their zero
training error only means cell medians were retained; it does not establish a
serving speedup. Compare TPOT, throughput and output equality against SPS and
additive modes with the same weights, context, concurrency and graph settings.

### Reuse a saved cost table

To reuse a saved bundle on a later compatible launch, replace the calibration
and output flags with `--adaptive_target_verify_cost_table=/results/dspark-costs.json`.
The source, saved calibration settings and effective coverage are logged.
Checkpoint/configuration and parallelism/execution compatibility are checked;
saved-table loading skips sampling/fitting, not graph warmup. Random-token
calibration and a low training error do not prove a speedup: compare real-request
TPOT, acceptance and accepted length against a fixed-width baseline.
