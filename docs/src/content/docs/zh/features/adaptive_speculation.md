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

自适应投机推理通过调整 draft 生成或 target 验证的工作量，提高预测的输出吞吐。
两个开关描述的是不同的执行控制方式，而不是某一种投机算法的名称：

| 控制开关 | 本身含义 | 当前接入及适用原因 |
|:---------|:---------|:-------------------|
| `enable_adaptive_draft_step` | 调整一次 target 验证前实际执行的顺序 draft forward 步数。 | MTP 顺序生成候选，减少步数既能省去 draft forward，也能缩短验证长度。 |
| `enable_adaptive_target_verify` | 调整已经生成的候选中，交给 target 验证的数量。 | DSpark 使用 confidence head，MTP 使用选中 token 的 draft 概率；均保持 draft 生成宽度不变，只裁剪验证工作。 |

两个开关默认均为 `false`，当前不能同时开启；通用命名不表示其他算法已经接入。
参数类型和默认值见 [CLI reference](/zh/cli_reference/#speculativeconfig)。

## 自适应 draft 步长（MTP）

`num_speculative_tokens` 是最大步数。每次提交 decode batch 前，engine 根据
已完成验证结果中的逐位置前缀接受概率，以及各候选步数的完整轮成本，为整个
batch 选择统一步数，使 `B * (1 + sum(prefix_acceptance[:步数])) / cost_ms` 最大。

启动时实测当前 MTP 服务的 draft forward、target 验证、采样、prefix refresh
和结果回收。它与 target 验证控制复用 SPS/二维加性拟合代码，不复用 DSpark 的
实测成本；预测使用 draft 的物理请求容量和 target padding 后的图行数。
在线反馈直接来自原有结果回收，不额外增加 confidence D2H 或 TP 决策广播。

当前接入要求 NPU GLM DSA Python 模型、普通预热 ACL graph、异步调度、task
pipeline 和 DP=CP=1，不支持 unified MTP graph 或 PD 分离。在 MTP 启动脚本的
每个 rank 应用以下配置；下面展示 16 rank 服务的 rank 0，其余 rank 使用各自
的 `node_rank` 和监听端口：

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

### 可选的步长配置文件

`adaptive_draft_step_config` 是 **JSON 配置文件的路径**，不是开启功能的开关，
也不是预先测好的成本表。不传时使用内置默认值，服务仍会启动标定，并打印完整
生效配置。配置文件分为两部分：

| 配置项 | 用途 | 默认值 |
|:-------|:-----|:-------|
| `policy.window_size` | 逐位置前缀接受率的已完成 batch 统计窗口。 | `20` 个 batch |
| `policy.update_interval` | 每完成多少个 batch 重新评分各候选步数。 | `5` 个 batch |
| `policy.switch_min_gain` | 切换步数所需的最小预测相对吞吐收益。 | `0.05`（要求严格超过 5%） |
| `calibration` | 启动采样桶、上下文/KV 覆盖、预热与测量窗口，以及 SPS/二维加性模型选择。 | 二维加性模型、实际捕获的请求桶（最多 128）、最多 2048 的 prefill 和最多 4096 的 KV，受模型与缓存容量限制。 |

例如，传入 `--adaptive_draft_step_config=/configs/draft-steps.json`，文件内容为：

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

其中任一部分均可省略并使用默认值。显式采样范围必须落在真实预热图和 KV 容量
内；这些设置影响启动标定的覆盖范围与耗时，不修改客户端请求或采样参数。
各候选步数均独立实测成本；此参数不支持加载预计算的 `step_cost_models`。

## 基于 MTP logits 的自适应 target 验证

greedy MTP Draft5/7 可开启 `enable_adaptive_target_verify=true`。每轮仍执行
全部 draft 步数，再以 `exp(selected_logit - logsumexp(draft_logits))` 作为
接受概率的预测分数。它不等于真实 target 接受概率。前缀分配、打包及提交复用
现有 kernel，不增加另一套 MTP 专用裁剪 kernel。

Host 仅用 n−2 及更早回收的反馈，以及当前 MTP 服务实测的完整轮成本，提前
决定 padding 后的 target 图预算。当前轮概率保留在 Device 上用于前缀分配，
小量异步反馈与正常结果一起回收，不在 draft 与 target 之间新增 Host 等待。
紧凑 target 输出的 hidden state 按 Device offsets 恢复后再更新 MTP 上下文
及 prefix refresh。

在上面的 MTP 启动命令中，用以下选项替换自适应步长设置：

```bash
--num_speculative_tokens=5 --enable_adaptive_draft_step=false \
--enable_adaptive_target_verify=true --enable_probabilistic_draft=false
```

此路径要求普通 Python GLM DSA ACL graph、异步调度、task pipeline 和 DP=CP=1。
greedy 或随机 target（包括混合 batch）复用现有 ragged 验证器，draft 必须保持
greedy。随机 target 的 token 在前缀提交和上下文更新之前先达成 TP 共识。
unified MTP graph 以及同时开启自适应 draft 步长会在执行前拒绝。启动成本标定
使用 greedy target，未计入随机 target 的额外采样工作。下述 `adaptive_target_verify_*`
成本表与启动标定选项也适用，但不能将 DSpark 实测成本表用于 MTP。

## 自适应 target 验证（DSpark）

当前 DSpark 实现使用滞后的 confidence 反馈和已标定的完整步成本选择验证预算，
draft 的生成宽度在启动时固定。

自适应裁剪的命令行选项和 JSON 配置项已从
`enable_adaptive_spec_prune`（更早为 `enable_adaptive_speculative_decode`）
重命名为 `enable_adaptive_target_verify`。
请同步更新现有启动脚本和 JSON 配置，旧名称不作为别名保留。

### 支持的配置

当前自适应 task pipeline 支持 NPU 上的 GLM DSA Python target、带 confidence
head 的 DSpark draft、`num_speculative_tokens=5` 或 `7`、greedy draft、CP=1、
ACL graph、异步调度和 task pipeline。普通 ACL graph 也支持 DP：所有分片接收
一致的 n−2 confidence 元数据，选择共同的物理 target graph 桶，并保留各自的
draft 验证预算，不在 draft 与 target 之间增加 Device 回读或 collective。
必须启用 graph warmup；添加
`--enable_unified_mtp_graph=true` 可捕获完整自适应轮次，包括 greedy heads、
confidence 共识、打包、验证及已接受 context 更新。unified graph 当前仅支持
DP=1 和 `temperature=0`；正温度或混合温度请求必须使用普通 graph，若开启 unified graph
则会在执行前拒绝，不会静默改成 greedy。固定步数 DSpark 和其他算法的执行限制
保持各自原有约定。

DP=2 使用 `--dp_size=2 --enable_unified_mtp_graph=false`。标定请求桶表示每个
DP 分片的容量，例如桶 4 会同时构造 8 条合成请求；全局请求数和 token 上限
需要容纳所有分片。已保存成本表的兼容性元数据包含 DP 拓扑，不能直接复用
DP=1 的表来运行 DP=2。固定步数和自适应 DSpark 都会拒绝 DP>1 与
`--enable_unified_mtp_graph=true` 的组合，不会静默改为 split graph。

### 启动示例

在已有多 rank 启动脚本的**每个 rank**中加入下列配置。下面展示 16 rank 服务的
rank 0；其余 rank 使用各自的 `node_rank` 和监听端口。启动前创建输出目录，
并选择尚不存在的输出文件：

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

客户端显式传入 temperature：`temperature=0` 使用 greedy target；
`temperature=0.5` 等正值使用随机 target 采样。不同 temperature 可以混合成批，
target 的 top-k、top-p 和 min-p 过滤复用普通采样器的语义。
draft 默认为 greedy；暂不支持 probabilistic draft。

随机 target 路径由 rejection sampler 按 Device 上的 packed-row owners 映射
每请求参数，对每个验证行采样。由于 draft 是确定性的，这与点质量提议分布的
拒绝采样等价：连续匹配的结果接受 draft 前缀，首次不匹配的采样结果就是拒绝后
的替代 token；全部匹配时使用末行采样结果作为 bonus。各 TP rank 在接受判断、
context KV 更新前先统一采样结果。target/draft forward 和自适应打包继续走
ACL graph；随机采样和前缀提交在模型图外、task stream 上执行，不将打包边界
回读到 Host。每请求 seed、logprobs、结构化约束、最短输出长度 stop mask、token bias 及非中性 penalty
仍不支持，会在执行前拒绝，不会静默忽略。

预算仍提前使用符合轮次约束的历史 confidence 决定，本轮 target 采样结果不参与
本轮裁剪。greedy draft 的 confidence 不一定能准确预测随机采样接受率。
当前启动标定使用 greedy target 请求，完整步成本尚不包含随机采样开销；支持
随机采样的正确性不代表接受率、预算最优性和性能与 greedy 相同，需要单独测量。

使用 Draft5 时，将步数改为 `--num_speculative_tokens=5`，并选择新的输出文件，
例如 `/results/dspark-costs-draft5.json`。draft 本身生成五个候选，自适应验证可
选择更短前缀。三个 device kernel 按宽度编译特化，Slot 按所选宽度分配，不会
填充到 Draft7。宽度在启动时固定，纳入 graph binding 和已保存成本表的兼容性
校验，不能在 replay 之间改变；Draft5 和 Draft7 的成本不能混用。
无版本旧表若未记录 `draft_width`，按 Draft7 处理；新拟合的裸表会显式记录宽度。

### 启动标定

标定不需要 HTTP 客户端、手动拟合命令或第二次启动：服务先预热图，再驱动真实
异步 prefill/decode 流水线，在内存中收集完成时间窗口，随后在 CPU 拟合成本，
并在就绪前向所有 worker 安装同一份不可变成本表。正常服务不会增加逐轮计时
Event、耗时采样 D2H 或文件写入；原有结果就绪 Event、结果回读和 confidence
异步反馈仍然保留。

#### 内置默认值

未指定标定文件和已保存成本表时，内置默认值为：

| 配置项 | 实际默认值 |
|:-------|:-----------|
| `model` | `additive` |
| `draft_width` | 启动参数 `num_speculative_tokens`，取值为 `5` 或 `7` |
| `request_buckets` | 实际已捕获的物理 draft request 桶 |
| `target_graph_rows` | 每个 request 桶对应的所有合法已捕获 target 桶，包括完整宽度 |
| `contexts` | 一个区间：prefill 2048 token，KV 覆盖 1–4096 |
| `warmup_steps` | `8` |
| `window_steps` | 每个窗口 `32` 个完整 decode 间隔 |
| `windows_per_point` | `3` |
| `token_seed` | `1` |

默认 KV 上限不会超过模型上下文限制；较短上下文会缩短 prefill，为最坏情况下的
生成进度和两个在途 Slot 预留空间。如果默认采样步数仍然无法容纳，启动会明确
报错，要求提供更小的显式配置。对所有已捕获桶采样可能增加启动时间，可用 JSON
限制标定网格。

#### 自定义配置

指定 `--adaptive_target_verify_calibration=/configs/dspark.json` 可覆盖默认配置。
宽度由启动参数推导；若 JSON 显式指定 `draft_width`，其值必须与启动参数一致。例如：

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

#### 成本模型与覆盖范围

`model` 支持 `sps`、`additive` 和 `graph`。一维 SPS 表按物理 target graph 行数 `G`
描述每秒完整步数，成本为 `1000 / SPS(G)` 毫秒；二维加性模型拟合
`bias + alpha(Bcap) + theta(G)` 秒，其中 `Bcap` 是物理 draft request 桶，
不是 HTTP 逻辑并发数。这些都是完整一步的成本，不能再次加上 draft/commit 耗时。
加性模型的二维网格必须通过共同的 G 值连通；request 桶和可选的 `target_graph_rows` 必须
已经捕获，并保留完整宽度覆盖。上下文区间不能重叠，且必须满足模型、KV 容量和
最坏采样进度的限制。

Unified 启动为每个实际请求数 `B` 捕获最多四个 target 桶：分别取覆盖 `B`、
`2B`、`4B` 和 `B * (draft_width + 1)` 的首个 prepared 桶。例如 Draft5 的
`B=8` 对应 `G=8,16,32,48`。标定仅使用两个流水 Slot 都已捕获的完整 `B/G`
绑定；显式指定未支持的桶会在采样前失败。过渡请求集合使用其已预热的完整宽度
绑定推进，不计为稳定样本。服务阶段不会补捕获缺失绑定。Split 和 unified 的
成本分别标定，加载已保存的 bundle 时会校验 unified 设置。

不管使用默认值还是传入文件，启动日志都会输出
`DSpark adaptive effective configuration`，包含来源、补齐默认项后的生效配置
及实际捕获的图几何。服务过程中不做标定、拟合或 graph recapture。未标定的
KV 区间或 request 桶使用已经预热的完整宽度准入策略，不做 eager 重试。

### Graph 成本感知验证（NPU）

将启动标定配置中的 `model` 设为 `graph`，按实际预热的
`(物理请求桶 Bcap, 物理验证图 G, KV 区间)` 保留完整步耗时的中位数。
相同 G 在不同 Bcap 下可以有不同成本；相邻 G 的延迟跳变和非单调关系也会保留。
预算规划器继续用 n−2 及更早的 confidence 计算预期输出量，并按
`预期输出 token 数 / 实测完整步毫秒数` 比较已预热的候选图。

这种按图成本选择预算的方式参考
[DeepInfer 的 MTPVarlen](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/csrc/engine/mtp_varlen.hpp)。
xLLM 使用现有 NPU 异步流水线的稳定完成窗口，成本包含 draft、验证、提交及回收；
不把 CUDA 计时事件迁入 NPU，也不重复计入 draft/commit 耗时。此模式的成本表仍然
在启动阶段生成并一致地安装到所有 worker，在线成本回归不属于此模式。

例如创建 `/configs/npu-graph-cost.json`：

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

在上面的 NPU 自适应 target 验证启动命令中使用：

```bash
--adaptive_target_verify_calibration=/configs/npu-graph-cost.json \
--adaptive_target_verify_cost_table_output=/results/npu-graph-costs.json
```

请求桶必须存在于该服务的预热结果中；省略 `target_graph_rows` 会测量对应 Bcap
下所有合法预热图。显式选择较少的图时，每个 Bcap 的完整宽度图必须保留。
没有实测成本的图不参与成本评分；缺少当前逻辑批次的完整宽度成本时，采用已有的
预热完整宽度准入策略。无法覆盖物理请求桶完整宽度的表会在服务就绪前被拒绝。
`graph_costs` 保存 `draft_requests`、`target_graph_rows` 和 `step_time_ms` 三个字段，
Graph 表必须声明 `max_kv_length`，并通过带版本及兼容性信息的 bundle 加载；
不能使用旧版裸表入口。外围 bundle 继续校验模型、并行拓扑、Draft5/7 和 KV 区间。

精确表不需要加性拟合的连通性假设，也不将不同 Bcap 的窗口合并成一个 G 成本。
其训练误差为零仅表示保存了各单元的中位数，不代表真实请求已经获得性能提升；
应在相同权重、上下文、并发及 Graph 设置下比较 SPS/加性模式与 `graph` 模式的
TPOT、吞吐和输出一致性。

### 复用已保存成本表

后续兼容的启动可移除标定和输出参数，改为
`--adaptive_target_verify_cost_table=/results/dspark-costs.json`。日志会打印成本表来源、
保存的标定配置和生效覆盖范围，并校验权重/量化配置及并行、执行设置的兼容性。
加载表只跳过采样与拟合，不跳过 graph warmup。随机 token 标定和较小训练误差
本身不能证明提速，仍应使用真实请求对比固定步数基线的 TPOT、接受率和接受长度。
