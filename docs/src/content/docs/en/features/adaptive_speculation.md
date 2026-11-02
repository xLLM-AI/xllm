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

The task-pipeline and unified-MTP execution paths used by adaptive speculation
are currently unsupported.

The `enable_task_pipeline` and `enable_unified_mtp_graph` switches remain
available, both defaulting to `false`. Keep them disabled. Setting either
switch to `true` fails at startup because the feature is currently unsupported.

## Retained configuration

The adaptive options remain part of the configuration interface:

| Option | Intended control | Availability |
|:-------|:-----------------|:-------------------------|
| `enable_adaptive_draft_step` | Adjust the number of sequential draft forward steps before target verification. | Default `false`; currently unsupported. |
| `enable_adaptive_target_verify` | Adjust how many already-generated draft proposals the target verifies. | Default `false`; currently unsupported. |
| `adaptive_draft_step_config` | Configure the draft-step policy and startup calibration. | Retained configuration only. |
| `adaptive_target_verify_*` | Configure verification-budget selection, calibration, and cost-table input or output. | Retained configuration only. |

Keep both adaptive feature switches at `false`.
Configuration files and saved cost tables do not enable the missing execution
path. See [CLI reference](/en/cli_reference/#speculativeconfig) for parameter
types and defaults.
