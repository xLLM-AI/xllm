---
title: "自适应投机推理"
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

自适应投机推理依赖的 task pipeline 和 unified MTP 执行路径目前暂不支持。

`enable_task_pipeline` 和 `enable_unified_mtp_graph` 开关继续保留，默认值均为
`false`。请保持关闭。任一开关设为 `true` 时，启动会因功能暂不支持而报错。

## 保留的配置

自适应相关选项仍保留在配置接口中：

| 选项 | 控制含义 | 可用性 |
|:-----|:---------|:-------------|
| `enable_adaptive_draft_step` | 调整一次 target 验证前顺序执行的 draft forward 步数。 | 默认 `false`；目前暂不支持。 |
| `enable_adaptive_target_verify` | 调整已生成的 draft 候选中交给 target 验证的数量。 | 默认 `false`；目前暂不支持。 |
| `adaptive_draft_step_config` | 配置 draft 步数策略与启动标定。 | 仅保留配置接口。 |
| `adaptive_target_verify_*` | 配置验证预算选择、标定与成本表的输入或输出。 | 仅保留配置接口。 |

请将两个自适应功能开关均保持为 `false`。配置文件或已保存的成本表
不能启用缺失的执行路径。参数类型和默认值见 [CLI reference](/zh/cli_reference/#speculativeconfig)。
