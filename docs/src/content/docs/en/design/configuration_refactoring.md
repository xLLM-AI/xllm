---
title: "Configuration Refactoring"
---
<!--
Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

## Goal and staged delivery

The target startup flow takes a model path, discovers hardware and derives a
suitable configuration. GFlags, CLI, JSON and API inputs remain available as
explicit constraints. Recommendations must account for model capabilities,
topology, memory and workload objectives; measured performance is required before
claiming an improvement.

The first implementation introduces requested inputs for Scheduler and explicit
topology injection for WorkerMemory. Existing numeric defaults and runtime
policies remain in effect. Automatic model/hardware tuning, explicit AUTO syntax
and unified Python input handling are subsequent steps.

## Separate intent, policy and state

```text
GFlags / JSON / API
        |
typed requested inputs + provenance
        |
model / hardware / deployment facts
        |
resolve and validate
        |
BootPlan -> load and observe -> bounded calibration -> ServingPlan
        |
domain configurations -> domain-owned runtime state
```

Requested input records retain omission and source. Effective configurations
contain typed values for consumers. Plans describe startup actions and resource
layout. Queues, observed memory, page ownership and execution progress stay with
their runtime owners. ModelArgs, QuantArgs, TokenizerArgs, per-request parameters
and small component Options retain their separate responsibilities.

BootPlan must be available before worker spawning and communication setup.
ServingPlan can use post-load observations, but calibration needs initial
execution/cache resources. Finalize capacity and layout before serving; do not
silently resize allocated layouts. Initial sleep defers execution calibration
until weights become resident on wakeup.

## First migration contracts

- Capture `FlagInputSnapshot` after GFlags parsing, before any legacy JSON
  writeback, and immediately copy typed Scheduler values. The snapshot records
  supplied names, not values. GFlags can mark a direct `FLAGS_` assignment as
  modified during a later query.
- GFlags retains its parser, help and supported flagfile/environment imports.
  The adapter records them as GFLAGS input, including values explicitly equal
  to their defaults. It does not infer omission by comparing values.
- `SchedulerConfigInput` stores optional requested values and their source.
  JSON fills omitted fields; explicit GFlags/API values win. Explicit false,
  zero and empty strings are values. Missing, null and object entries keep the
  existing JsonReader absent-value behavior; incompatible scalar types still
  fail parsing.
- `resolve_scheduler_config` uses only the typed input and existing defaults.
  It reads and writes no flag values. `SchedulerConfig::initialize(input)`
  applies the result and performs one-way flag writeback for legacy consumers.
  The old `from_flags/from_json/initialize()` entry points remain adapters.
- WorkerMemory validates the topology in `WorkerMemoryOptions` once at RPC
  setup, then passes world size and rank to the server. Each process still owns
  exactly one local device. Existing memory resources and RPC lifetimes retain
  their owners.

Startup dump format remains the existing non-default effective-value JSON.
It is not a provenance report or a serialized execution plan.

## Next migrations and acceptance

1. Extend typed input adapters across domains and Python without turning
   omitted defaults into user constraints. Keep entry-point behavior differences
   explicit and fix them separately.
2. Establish a shared resolution boundary and a deterministic legacy policy
   before adding automatic recommendations.
3. Migrate EPLB, Scheduler/cache and engine/worker/executor consumers by domain;
   remove global reads as each injected policy becomes authoritative.
4. Discover model/hardware capabilities, derive BootPlan, then build ServingPlan
   from memory observations and bounded calibration.
5. Enable default optimization only for validated model/backend combinations.

Each slice must connect input, resolution, injection and real consumers.
Validate actual GFlags parsing, precedence and compatibility, then compare
minimal startup and explicit overrides across native/Python and worker roles.
Evaluate startup cost, memory headroom, throughput and latency against a
replayable manually configured baseline before changing optimization defaults.
