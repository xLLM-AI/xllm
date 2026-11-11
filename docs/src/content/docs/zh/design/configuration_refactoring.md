---
title: "配置重构"
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

## 目标与分阶段交付

目标启动流程只要求模型路径，自动发现硬件并推导适合的配置。保留 GFlags、CLI、
JSON 和 API 作为显式约束。推荐必须考虑模型能力、拓扑、内存及工作负载目标；
性能改善需要实测支持。

首批实现 Scheduler 的请求输入模型，以及 WorkerMemory 的显式拓扑注入。
现有数值默认值和运行策略继续生效。模型与硬件自动调优、显式 AUTO 语法及
Python 统一输入将在后续阶段接入。

## 分开表达意图、策略与状态

```text
GFlags / JSON / API
        |
带类型的请求输入与来源
        |
模型 / 硬件 / 部署事实
        |
统一推导与校验
        |
BootPlan -> 加载并观测 -> 有界校准 -> ServingPlan
        |
领域配置 -> 领域对象持有的运行状态
```

请求输入保留未指定语义和来源；有效配置向消费者提供有类型的值；
计划表达启动动作与资源布局。队列、内存观测、页面所有权、执行进度仍由原有
运行时对象负责。ModelArgs、QuantArgs、TokenizerArgs、请求参数及小型组件
Options 保持各自职责。

创建 worker 和通信组前必须形成 BootPlan。ServingPlan 可以使用加载后的观测，
但校准本身需要初始执行与缓存资源。开始服务前应确定最终容量和布局，不隐式修改
已经分配的布局。初始 sleep 模式在唤醒、权重驻留后再执行运行校准。

## 首批迁移契约

- GFlags 解析完成后、任何旧 JSON 写回之前捕获 `FlagInputSnapshot`，
  并立即复制 Scheduler 的有类型输入。快照保存显式名称，不保存数值；
  后续 GFlags 查询可能把直接 `FLAGS_` 赋值识别为修改。
- 保留 GFlags 的解析、帮助、flagfile 和环境导入语义，统一记录为 GFLAGS 来源。
  显式传入等于默认值的参数仍属于用户输入，不通过数值比较推断是否省略。
- `SchedulerConfigInput` 保存可选值及来源。JSON 只填充未指定字段，
  显式 GFlags/API 值优先。false、0、空字符串都是有效输入；缺项、null、
  object 延续 JsonReader 的缺省语义，不兼容的标量类型仍报解析错误。
- `resolve_scheduler_config` 只使用请求输入和现有默认值，不读写 flag 值。
  `SchedulerConfig::initialize(input)` 应用结果，并单向写回 flag 以兼容旧消费者。
  保留旧 `from_flags/from_json/initialize()` 适配入口。
- WorkerMemory 在 RPC 初始化边界一次校验 `WorkerMemoryOptions` 中的拓扑，
  将 world size 和 rank 传给 server。仍然要求每个进程只服务一个本地设备；
  内存资源及 RPC 生命周期继续由原对象持有。

启动 dump 保持原来的“非默认有效值 JSON”格式；它尚不包含来源报告或执行计划。

## 后续迁移与验收

1. 按领域扩展输入适配，并接入 Python；不把省略参数的默认值变成用户约束。
   记录各入口现有行为差异，独立修正。
2. 建立共享推导边界和确定的兼容策略，再加入自动推荐。
3. 依次迁移 EPLB、Scheduler/cache、engine/worker/executor 消费者；
   注入策略成为唯一有效来源后，删除对应全局读取。
4. 发现模型与硬件能力，先形成 BootPlan，再结合内存观测及有界校准构造 ServingPlan。
5. 仅在完成验证的模型和后端组合中默认开启优化。

每个切片都必须接通输入、推导、注入及真实消费。验证真实 GFlags 解析、
覆盖优先级和兼容性，再比较 native/Python 与不同 worker 角色的最小启动及显式覆盖。
切换默认调优策略前，用可重放的人工配置基线比较启动开销、内存余量、吞吐和延迟。
