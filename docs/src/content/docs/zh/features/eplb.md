---
title: "MoE负载均衡（EPLB）"
sidebar:
  order: 71
---
## 概述

Mixture-of-Experts（MoE）模型会根据路由结果把 token 分配给不同专家。在线请求的输入分布通常不是均匀的，因此部分专家可能持续成为热点，导致其所在 EP rank 成为整批请求的瓶颈。

xLLM 的 Expert Parallel Load Balancing（EPLB）通过为热点逻辑专家创建冗余物理副本，并在服务运行期间逐层调整专家分布，降低最慢 EP rank 的负载。EPLB 不改变模型的逻辑专家数量或路由结果，只维护逻辑专家到物理槽位的映射并迁移对应权重。

当前实现包含完整的控制面、负载聚合、可插拔策略、异步 prepare 协议和逐层权重切换。本文重点描述 DeepSeek V4 的 NPU `npu_torch/FusedMoE` 路径；其他模型只有在实现 `prepare_expert_weight`、`start_expert_weight_transfer` 和 `update_expert_weight` 等模型 hook 后，才能使用同一套运行时协议。

## 专家与槽位

EPLB 区分以下两个概念：

- **逻辑专家（logical expert）**：模型路由器输出的专家 ID，数量由模型结构决定。
- **物理专家槽位（physical expert slot）**：某个 EP rank 上实际保存的一份专家权重。一个逻辑专家可以有多个物理副本。

在当前一 worker 对应一 EP rank 的部署中，每个 rank 的物理槽位数为：

```text
local_physical_experts = logical_experts / ep_size + redundant_experts_num
```

全局物理槽位数为：

```text
global_physical_experts = logical_experts + ep_size * redundant_experts_num
```

因此，逻辑专家数必须能被 `ep_size` 整除。`redundant_experts_num` 增加的是每个 rank 的额外专家槽位，也会增加权重显存和 EPLB staging 显存。

## 当前架构

| 组件 | 主要职责 |
|---|---|
| `LLMEngine` | 从所有 worker 收集负载与 prepare token，把 `EplbInfo` 命令写入下一次 forward 输入。 |
| `EplbManager` | 管理更新定时器、活动分布、逐层 prepare/activate 状态机、超时和提交。 |
| `EplbAggregator` | 把各 rank 的物理槽位计数还原为本轮增量，同时保留物理负载并聚合逻辑专家负载。 |
| `IEplbPolicy` | 根据逻辑负载、物理负载和当前分布生成候选布局及逐层更新掩码。 |
| `EplbExecutor` | 在每个 worker 上执行异步 prepare，并在 forward 边界启动迁移和激活。 |
| 模型 EPLB hook | 维护逻辑到物理映射、生成 P2P 计划、迁移权重并切换活动权重。 |
| `ProcessGroup` | 通过专用 `ProcessGroupHCCL` 批量调用 HCCL 点对点 send/recv；当前全互联超节点的数据面使用 HCCS。 |

控制面通过 `EplbInfo` 传递以下字段：

- `prepare_layer_id`：需要准备新布局的 MoE 层。
- `prepare_token`：本次 prepare 的唯一 token，用于隔离超时后的迟到结果。
- `expert_ids`：该层更新后的全局物理槽位表。
- `update_layer_id`：已经准备完成、可以在本次 forward 边界激活的层。
- `activation_token`：激活命令的唯一 token，用于确认对应 forward 已真正完成。

## 运行流程

一次完整的 rebalance 按以下顺序执行：

1. 每个 worker 在 MoE forward 中记录各层本地物理专家槽位的 token 数。计数按槽位前缀和编码，`EplbAggregator` 再通过差分恢复每个槽位的实际负载。
2. `LLMEngine` 收集所有 worker 的负载，并把采样时生效的专家分布与负载一起入队。这样即使随后发生激活，旧样本也不会被错误解释为新布局的负载。
3. 第一份有效负载到达后启动 `eplb_update_interval` 定时器。定时器到期且没有其他 rebalance 正在进行时，manager 要求策略为每个层重新计算候选布局。
4. 每个候选都用当前实测物理负载计算最慢 rank 改善率。只有布局发生变化且改善率达到 `eplb_min_peak_load_improvement` 的层才会发布，并逐层发送 `prepare_layer_id`、`prepare_token` 和新的 `expert_ids`。
5. worker 的 `EplbExecutor` 在后台线程计算逻辑到物理映射、变化槽位和 EP-wide P2P 迁移计划。prepare 成功后，worker 原子地上报对应 token；失败不会伪装成 ready。
6. 所有 rank 都上报同一个 prepare token 后，manager 发布该层的 `update_layer_id`。在 DeepSeek V4 路径中，worker 在 forward 前物化 staging 权重并启动 P2P，在 forward 后等待迁移完成并激活新权重。
7. 对应 forward 的输出携带 `activation_token` 返回 engine 后，manager 才提交策略状态、切换活动分布，并清空物理负载窗口。迟到或重复 token 会被忽略。

EPLB 命令只会在至少一个 DP batch 非空、所有非空 DP batch 都处于 decode、且不是 graph warmup 的 forward 边界下发。空 DP rank 可以参与该边界。这样可以避免 prefill 或 graph capture 仍在读取旧权重时发生切换。副作用是持续没有纯 decode 边界时，已经准备好的层会延迟激活。

## 负载统计

manager 同时维护两种负载：

- **逻辑专家负载**：把所有物理副本的负载按当前槽位表聚合到逻辑专家 ID，用于选择热点专家和生成副本。
- **物理槽位负载**：保留 `[layer, rank, slot]` 维度，用于计算当前布局下真实的最慢 rank，而不是假设同一逻辑专家的副本负载均匀。

DeepSeek V4 对 eager、ACL graph 和 EP2 fused dispatch 路径使用同一套负载语义：

- graph warmup 不进入 manager 的统计窗口。
- ACL graph 使用每 token mask 过滤 graph padding 和空 DP rank 的合成 token。
- `eplb_use_decode_only_load=true` 时，eager 路径也只记录真实 decode token，mixed batch 中的 prefill 行会被过滤。
- fused dispatch 在可用时记录算子返回的 receiver-observed 物理专家计数；mixed phase 无法直接按 token 过滤时，回退到路由 ID 统计。

## Rebalance 策略

通过 `eplb_policy_kind` 选择策略。字符串大小写不敏感，未知值会回退到 `greedy`，不会中断 rebalance 线程。

| 策略 | 布局算法 | 适用场景 |
|---|---|---|
| `balanced` | 先按最大负载下降选择副本，再在全互联 HCCS 超节点的所有 rank 间做等容量 LPT packing。 | 默认策略，推荐用于当前超节点部署。 |
| `greedy` | 历史 xLLM 贪心副本选择和 LPT packing。 | 兼容历史行为。 |

历史策略名称仍作为兼容别名解析为 `balanced`，新配置统一使用 `balanced`。

两种策略使用相同的发布门控：

- 当前峰值使用实测物理槽位负载计算；候选峰值根据候选布局估算。
- 只有预计峰值下降比例达到 `eplb_min_peak_load_improvement` 才发布迁移。
- 每个更新周期都会重新计算所有层，不再根据上一轮逻辑负载相似度提前跳过。
- 同一逻辑专家在同一 rank 上最多放置一个副本。
- prepare 超时会调用策略的 `abort_layer` 回滚该层候选状态，不会把未部署布局当成已生效布局。

## DeepSeek V4 权重切换

DeepSeek V4 的 `FusedMoE` 路径维护稳定地址的活动权重、pending staging 权重和 `log2phy` 映射。权重切换具有以下行为：

- 启动时扩展每层权重，为每个 rank 的冗余专家槽位预留存储。
- 主模型在权重加载后预热可复用 staging buffer，避免第一次 rebalance 临时申请大块 NPU 显存；MTP 模型不重复预留这组 buffer。
- prepare 阶段只生成 host 侧槽位映射和 P2P 计划，避免后台线程在 forward 读取 private-format 权重时并发物化张量。
- 同 rank 已驻留的专家直接复制到 pending buffer；非驻留专家通过专用 EPLB `ProcessGroupHCCL` 迁移。
- `batch_isend_irecv` 最终调用 `ProcessGroupHCCL::send/recv`。HCCL 是通信库接口，当前全互联超节点上的 rank 间数据通过 HCCS 传输；EPLB 不再区分机器边界。
- 所有 tensor 的 send/recv 被合并到一次批量计划，大 payload 会分块并限制同时在途的数据量。
- P2P 在 forward 前启动，forward 完成后等待通信并激活，因此本次 forward 始终使用旧布局，下一次 forward 才看到新布局。
- 变化槽位少于一半时只复制变化槽位；变化槽位达到一半时，先完整构造 pending tensor，再用整 tensor `copy_` 激活，避免对大批槽位逐个发起小 copy。
- 活动 tensor 本身不替换，只原地更新内容和映射，满足 ACL graph 对稳定地址的要求。

## EP2、MC2 与 ACL Graph

EPLB 控制面不要求 `expert_parallel_degree=2`。该参数决定 DeepSeek V4 是否使用 EP Level 2 的 dispatch/combine 路径。`enable_fused_mc2` 进一步选择 EP2 MoE 算子：

| `enable_fused_mc2` | 行为 |
|---:|---|
| `-1` | 自动选择。`expert_parallel_degree=2` 时解析为 `1`，否则解析为 `0`。 |
| `0` | 使用 `moe_distribute_dispatch_v2 + group_gemm + moe_distribute_combine_v2` 路径。 |
| `1` | 满足量化、dtype、算子和 graph 准备条件时使用 `DispatchFFNCombine`。 |
| `2` | 纯 decode 且满足前置条件时使用 `DispatchGmmCombineDecode`。 |

当请求的 MC2 路径不满足 dtype、量化格式、TP、算子可用性或 graph 准备条件时，运行时会选择可用的普通 MoE 路径。MC2 模式只改变 MoE 执行和物理负载的采集方式，不改变 manager 的策略和逐层切换协议。

ACL graph 开启或关闭时都可以使用 EPLB。graph 模式下，专家权重和 `log2phy` 映射保持稳定存储，decode mask 按 DP rank 对齐并补零，避免 graph padding 污染负载。

同一个 `AclGraphExecutor` 创建的 decode graph bucket 共用一个固定 capture stream 和一个 private graph pool，使不同 bucket 可以复用 graph 临时地址，避免每个预热 bucket 都独立保留一组 expandable segment。该优化避免 graph pool 显存随 bucket 数量线性增长，但 graph pool、persistent tensor 和运行时缓存仍会占用显存，实际占用取决于模型、预热 bucket 和并发配置。

## 配置参数

EPLB 参数既可以通过 gflags 传入，也可以写入 xLLM JSON 配置。manager/policy 使用的 `redundant_experts_num`、`eplb_update_interval`、`eplb_min_peak_load_improvement`、`eplb_policy_kind` 和 `eplb_prepare_timeout_seconds` 会在 manager 创建时快照到 `EplbOptions`。`eplb_use_decode_only_load` 和 `expert_parallel_degree` 属于模型执行配置，由对应运行时路径读取；受支持的使用方式仍是在启动前确定配置，修改后重启服务。

下表默认值指 xLLM 核心 gflags/JSON 配置的默认值。部署脚本可以显式覆盖这些值，应以实际启动命令和启动日志为准。

| 参数 | 默认值 | 约束与说明 |
|---|---:|---|
| `enable_eplb` | `false` | 开启动态专家负载均衡。 |
| `redundant_experts_num` | `1` | 每个 EP rank 的额外物理专家槽位数，必须大于等于 0。 |
| `eplb_update_interval` | `1000` | 从第一份有效负载开始计时的 rebalance 间隔，单位为秒，必须大于等于 0。本文示例显式覆盖为 `300`。 |
| `eplb_min_peak_load_improvement` | `0.05` | 所有策略候选布局需要达到的最慢 rank 最小预计改善比例，范围 `[0, 1]`。 |
| `eplb_policy_kind` | `balanced` | `balanced`（默认）或 `greedy`；历史名称兼容映射到 `balanced`，未知值回退到 `greedy`。 |
| `eplb_use_decode_only_load` | `false` | eager 路径是否只统计真实 decode token。graph 路径始终过滤 padding。 |
| `eplb_prepare_timeout_seconds` | `30` | 所有 rank 等待同一层 prepare 完成的超时，必须大于 0。超时后跳过并回滚该层。 |
| `expert_parallel_degree` | `0` | EP 模式参数；设为 `2` 时启用满足条件的 EP2 dispatch/combine 路径。 |
| `enable_fused_mc2` | `-1`（自动） | 有效值为 `-1/0/1/2`；自动模式在 `expert_parallel_degree=2` 时解析为 `1`，否则解析为 `0`。 |

## 启动示例

下面示例表示单一全互联 HCCS 超节点中的 16 个 worker/EP rank，并使用 300 秒测试周期。`ep_size` 必须与实际 worker 数一致，模型逻辑专家数必须能被 16 整除。

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

EP 和 DP/TP 组合方式参见 [EP 并行](./moe_params.md)。

## 运行约束

- 当前 EPLB 聚合要求 `ep_size == worker_num`，即一个 worker 对应一个 EP rank。
- `ep_size` 必须大于 1，模型逻辑专家数必须能被 `ep_size` 整除。
- 开启冗余槽位和 staging buffer 会增加显存占用，应在启动前给 KV cache 和 EPLB staging 留出余量。
- 权重切换只在纯 decode forward 边界执行，不会在 graph warmup 或 mixed prefill 边界强行激活。
- 配置在 manager 创建时快照，修改启动配置后必须重启服务才能生效。

## 共享内存兼容性

EPLB 控制字段和 worker 输出会经过 xLLM forward 共享内存。当前 `SharedMemoryManager` 使用 layout v2，包含 magic、version、payload size 和 generation，并通过全局 generation lock 串行化同名对象的 open/unlink/create，防止不同 worker 映射到两个 SHM generation。

从旧版 xLLM 升级时，必须先停止所有可能访问相同 SHM 名称的旧进程，再启动新版本。新旧二进制滚动混跑不受支持；不兼容的大小、magic 或版本会直接失败，不会调整大小或清空旧对象。

## 可观测性与排障

| 日志前缀 | 含义 |
|---|---|
| `EPLB manager start` | 实际策略、MoE 层数、device 数和每 device 物理槽位数。 |
| `EPLB rebalance` | 本轮策略耗时和需要更新的层数。 |
| `EPLB placement benefit` | 策略的当前峰值、候选峰值、改善比例和收益门控结果。 |
| `prepare_expert_weight` | worker 后台 prepare 的层、槽位表大小、耗时和失败状态。 |
| `materialize_expert_weight` | DeepSeek V4 在 forward 边界物化 staging tensor 的耗时。 |
| `update_expert_weight` | 层、rank、MC2 模式、变化槽位、P2P 数、迁移耗时、激活耗时和总耗时。 |
| `EPLB heartbeat` | rebalance、manager、executor 和 P2P bucket 的存活及进度信息。 |
| `EPLB staging reservation warmed` | 启动时预留的 staging bytes 和 tensor 数。 |

排障时先确认所有 rank 的 `prepare_token` 是否一致，再检查 `tasks_failed_since_last`、`layers_timed_out` 和 `update_expert_weight`。如果只有部分 rank 进入 P2P，通常会表现为 prepare 超时；如果 SHM 不兼容，服务会在启动阶段直接报告 layout 错误。
