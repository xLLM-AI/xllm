# Adaptive Speculative Decoding 设计稿

## 背景

当前 xLLM 的 MTP 投机解码使用静态 `num_speculative_tokens`。每个 decode step 固定执行 K 次 draft model，然后将 `K + 1` 个 token 送入 target model 验证。静态 K 实现简单，但不同 batch size、不同请求局部和不同 draft 置信度下，最优验证宽度可能不同：

- draft 接受率高时，较大的验证宽度可以摊薄 target validate 成本。
- draft 接受率低时，较大的验证宽度会浪费 target validate 宽度。
- batch size 和 prefix length 变化时，validate 的边际耗时也会变化。

本设计的第一个版本只实现 validate 前自适应剪枝：draft model 仍按配置的 `num_speculative_tokens` 完整推理，adaptive controller 在 target model validate 前选择每个请求要送入验证的 draft 前缀长度。

此方案参考 DSPark 自适应投机推理算法（区别主要在于计算接受概率时 DSPark 使用模型本身的置信度头，本实现先使用 draft model 输出的 softmax logits），但还未专门针对 graph mode 和 ZOS 做适配，后续还可以进一步支持对 draft 的 early stop。

## 目标

v1 目标：

1. 只支持 MTP。
2. 默认关闭 adaptive，静态 MTP 行为保持兼容。
3. `num_speculative_tokens` 继续表示最大推测长度 `max_sl`。
4. adaptive 开启后，draft 仍完整执行 `max_sl` 步，只在 validate 前按 draft selected token 概率剪枝。
5. 支持 per-request 不同的验证长度，validate 阶段通过 ATB kernel 的 variable q_len 能力只计算有效 tokens，真正减少 attention/FFN 计算量。
6. 支持 TP/CP 多 rank 场景下同步剪枝结果，避免不同 rank validate width 不一致。
7. graph mode、DP/EP 场景下暂时关闭 adaptive，退回静态 MTP。

非目标：

1. v1 不直接中断 draft loop，不做 draft early stop。
2. v1 不支持 Eagle3 和 Suffix。
3. v1 不接入 TPOT SLO，只预留后续设计空间。
4. v1 不改变 scheduler token/KV 预算，仍按 `max_sl` 保守调度。

## 术语

- `max_sl`：配置的最大 speculative length，即 `num_speculative_tokens`。
- `prefix_len_i`：第 i 个请求本轮送入 target validate 的 draft token 前缀长度，范围为 `[0, max_sl]`。
- `effective_sl`：本轮 batch 的最大 validate draft 宽度，等于 `max(prefix_len_i)`，运行时最小取 1。用于截断 draft_outputs 和作为 RejectionSampler 的 padded width。
- `per_seq_val_tokens`：每个请求实际参与 validate 的 token 数，等于 `prefix_len_i + 1`。ATB kernel 按此值只计算有效 tokens。
- `num_val_tokens`：target validate 的 padded 输入宽度，等于 `effective_sl + 1`，用于 RejectionSampler 的张量对齐。
- `path_prob_i(k)`：第 i 个请求前 k 个 draft token 全部被接受的估计概率。

## 参数

配置保持：

```text
num_speculative_tokens = max_sl
```

新增实验参数：

```text
enable_adaptive_speculative_decode = false
adaptive_speculative_min_gain = 0.0
```

含义：

- `enable_adaptive_speculative_decode`：总开关。默认 false。
- `adaptive_speculative_min_gain`：把一个候选 token 加入验证集所需的最小相对吞吐收益。默认 0，表示只要估计吞吐严格提升即可加入。

不新增单独的 adaptive 最大 SL 参数，直接复用 `num_speculative_tokens` 作为上限。也不提供 target/draft step time 手写参数。

## 核心策略

MTP 每步 draft output 的 `sample_output.probs` 会被压缩成 selected-only 概率。对每个请求按位置计算路径概率：

```text
path_prob_i(k) = product_{j=1..k} p_i(j)
```

其中 `p_i(j)` 是 draft 第 j 步采样 token 经过 softmax 后的概率。由于概率都在 `[0, 1]`，同一个请求内 `path_prob_i(k)` 随 k 单调不增，所以如果后置 token 被选中，前置 token 必须一起被选中。

controller 把所有请求、所有位置的候选 token 展平成候选列表：

```text
candidate = (seq_id, prefix_len, path_prob)
```

然后按 `path_prob` 从大到小排序，从空验证集开始贪心尝试加入候选。每次尝试只会把某个请求的 `prefix_len` 增大到候选位置；如果加入后估计吞吐提升超过 `adaptive_speculative_min_gain`，则接受该候选，否则回滚并跳过该候选继续尝试后面的候选。

这里选择"跳过而不是停止"的原因是：某个候选可能因为边际 validate time 成本过高而不划算，但后面另一个请求的低位置候选成本更低，仍然可能提升整体吞吐。

估计吞吐公式：

```text
score(prefixes)
  = expected_emitted(prefixes)
    / estimated_time(prefixes)

expected_emitted(prefixes)
  = batch_size + sum_i sum_{k=1..prefix_len_i} path_prob_i(k)

estimated_time(prefixes)
  = full_draft_time(max_sl)
    + estimated_validate_time(prefixes)
```

其中 `estimated_validate_time` 按 per-seq 累加（而非按 batch max 估计）：

```text
estimated_validate_time(prefixes)
  = intercept
    + sum_i (batch_ms + query_token_ms * q_i + query_prefix_ms * q_i * kv_i)
```

其中 `q_i = prefix_len_i + 1`，`kv_i` 是第 i 个请求的 kv cache 长度。

说明：

- `batch_size` 表示每个请求至少会由 target 产生一个 token。
- `full_draft_time(max_sl)` 使用本轮完整 draft loop 的实测耗时，因此不需要估计 draft step time。
- `estimated_validate_time` 来自 `ProfileManager` 拟合出的 validate predictor 的系数，按 per-seq 展开计算。
- 如果 predictor 不可用（`SpeculativeProfileRegistry` 为空），整个 adaptive 路径不会被启用（入口门控）。

## 时间估计

v1 复用现有 ProfileManager 的离线 profile/线性拟合方式。开启 step-time profile 且 MTP `num_speculative_tokens > 1` 时，ProfileManager 额外采样 validate-like workload，并拟合：

```text
T_validate(B, q_len, prefix)
  = c0
    + c1 * B
    + c2 * B * q_len
    + c3 * B * q_len * prefix
```

其中：

- `B` 是 decode batch size。
- `q_len = sl + 1` 是 target validate query 宽度。
- `prefix` 是 batch 的平均 prefix length。
- 不引入 `q_len^2` 项，因为 speculative validate 的 q_len 通常较小，二次项容易被噪声放大。

ProfileManager 拟合完成后：

1. 在 master 进程本地写入 `SpeculativeProfileRegistry`。
2. 通过 `SetSpeculativeValidateTimePredictor` RPC 广播到 target engine 的 worker rank。
3. 每个 worker 在 runtime 进程本地查询 registry，预测不同 `sl` 对应的 validate time。
4. registry 中有 predictor 是 adaptive 路径启用的前置条件（门控），确保 profile 阶段不会触发 adaptive 的 HCCL broadcast。

profile 采样时增加了 KV cache 容量检查，跳过超出可用 blocks 90% 的 batch_size × prefix_len 组合，防止 profile 阶段 OOM。

## Per-Seq Variable-Length Validate

v1 支持 batch 内每个请求不同的 validate token 数（`per_seq_val_tokens`）：

1. `prepare_validate_inputs` 按 per-seq q_len 构造 ATB kernel 输入（`q_seq_lens`/`q_cu_seq_lens`），kernel 只计算有效 tokens。
2. `fill_validate_input_from_draft_outputs` 按 draft step 分组，用 `index_select` + `index_copy_` 向量化填充 draft tokens。
3. Target model forward 输出 flat `[total_tokens, vocab]` logits。
4. **Fast-path**：如果所有 seq 都保持满 SL（`total_tokens == batch * max_val_tokens`），直接 `view` 零拷贝，无额外开销。
5. **Slow-path**：per-seq 不均匀时，用 `index_copy_` 将 flat logits scatter 到 `[batch * max_val_tokens, vocab]` 的 padded layout，padding 位填 `-1e9`。
6. RejectionSampler 使用 padded layout 做 rejection sampling，padding 位自然被 reject。
7. `apply_pruned_prefix_lengths` 在 rejection sampling 之后按 per-seq prefix 裁剪输出。

## MTP Worker 接入

`MTPWorkerImpl::step_decode()` 的自适应路径：

1. 按静态 MTP 原逻辑完整执行 `max_sl` 步 draft。
2. 记录本轮完整 draft loop 的实测耗时 `draft_latency_ms`。
3. 从 `draft_outputs` 提取 `[batch, max_sl]` 的 selected token 概率矩阵。
4. 计算每个 seq 的 kv_len 作为 `per_seq_kv_lens`。
5. 控制 rank 调用 `AdaptiveSpeculativeController::select_pruned_prefix_lengths()` 得到每个请求的 `prefix_len_i`。
6. 将 `prefix_lengths` 同步给其它 rank。
7. 如果同步失败，则退回所有请求 `prefix_len_i = max_sl`。
8. 计算 `effective_sl = max(1, max(prefix_lengths))` 和 `per_seq_val_tokens`。
9. 截断 `draft_outputs` 到 `effective_sl`。
10. 用 `per_seq_val_tokens` 构造 variable-length validate input。
11. Target validate 后，根据 total_tokens 是否等于 padded_total 选择 fast-path 或 slow-path padding。
12. Rejection sampling + per-request prefix 裁剪。
13. Cache 写入、metrics 统计都使用剪枝后的结果。

静态路径保持原逻辑，不调用 controller，也不消费 profile predictor。

## 剪枝后的 Validate 输出

因为 batch 内每个请求的 `prefix_len_i` 可以不同，而 RejectionSampler 的张量宽度取 batch 内最大值，所以 rejection sampler 之后需要逐行修正输出：

1. 对 `prefix_len_i < effective_sl` 的请求，保留 `[0, prefix_len_i]` 的输出。
2. `prefix_len_i` 位置作为 target 修正 token/bonus token 保留，保证每个请求至少推进一个 token。
3. `prefix_len_i + 1` 之后的位置置为 `-1`。
4. 对应 embedding 位置清零，避免后续 cache 读取未被选中的 validate 结果。

这样 `EmbeddingCache::write_target_context()` 可以继续按 accepted prefix 解析 `next_tokens`，`all_draft_accepted` 也会基于本轮 `effective_sl` 正确判断。

## Metrics

现有指标继续复用：

- `speculative_num_draft_tokens_total`
- `speculative_num_accepted_tokens_total`

adaptive prune 下，draft token 计数不再使用 `batch_size * effective_sl`，而是使用：

```text
selected_draft_tokens = sum_i prefix_len_i
```

accepted token 计数只统计每个请求 `[0, prefix_len_i)` 范围内真正被接受的 draft token，避免把剪掉的位置当成 reject。

## 多 Rank 同步

TP/CP 多 rank 场景下，target validate width 必须一致，否则 collective shape 和调用顺序可能分叉。v1 使用控制 rank 决策：

1. 所有 rank 都完整执行相同的 draft loop。
2. 控制 rank 计算 `prefix_lengths`。
3. 控制 rank 将 `prefix_lengths` 广播给其它 rank。
4. 所有 rank 使用同一个 `effective_sl` 和同一组 per-request prefix 做 validate。

同步路径：

- Torch backend：优先使用 `parallel_args_.tp_group_`，否则使用 `process_group_`，通过小 `int32` tensor broadcast 同步。
- NPU ATB backend：Torch `ProcessGroup` 可能不可用，因此复用 ATB mapping 中的 `ATTN_TP` parallel info 初始化 HCCL control comm，并通过 `HcclAllReduce` 同步 prefix list。

如果没有可用同步组，adaptive prune 自动退回静态 `max_sl`，避免 rank 本地剪枝导致 hang。

## Graph Mode

graph mode 暂不支持 adaptive prune。主要原因是当前 graph 路径把 speculative validate width 固定为 `num_decoding_tokens = num_speculative_tokens + 1`：

1. graph capture/replay 需要固定 shape，动态 `effective_sl + 1` 会改变 validate 输入 token 数。
2. executor 中多处使用固定 `num_decoding_tokens` 反推 batch size。
3. persistent graph buffer、seq lens padding、metadata copy 都假设 decode width 静态。
4. 多 rank 下所有 rank 必须选择同一个 bucket，否则 collective shape 会不一致。

当前实现中，`enable_graph=true` 时 `AdaptiveSpeculativeController` 自动关闭 adaptive，继续使用静态 MTP。

后续可以用 graph bucket 支持，例如预捕获 `SL in {1, 2, 4, 8}`，运行时按剪枝结果选择 bucket，并在 bucket 内 mask 未选中的位置。

## DP/EP 暂缓

DP/EP 场景下，不同 rank 可能持有不同请求子集，本地最优 `prefix_lengths` 不同；同时 `dp_global_token_nums`、expert dispatch、padding 和 graph bucket 都依赖全局 token shape。v1 为了控制变量，DP/EP 大于 1 时关闭 adaptive，退回静态 MTP。

后续支持可以考虑：

1. scheduler/master 侧根据历史接受率预先选择全局 SL。
2. DP group 内 all-gather 本地建议，再取 min/median/max。
3. 将 DP/EP token shape 和 adaptive bucket 统一管理。

## 风险和边界

1. v1 不减少 draft 计算，只减少 target validate 宽度；如果瓶颈主要在 draft model，收益有限。
2. profile predictor 未就绪时（服务启动中），adaptive 路径不会启用，直到 profile 完成并设置 registry。
3. 如果 draft selected probs 未定义，adaptive controller 无法工作，应退回静态路径。
4. NPU ATB 多卡必须能初始化控制通信组；否则自动退回静态 MTP。
5. 剪枝后每行 prefix 不同，cache 和 metrics 必须只消费被选中的前缀。
6. Score 公式中 `intercept` 占总 estimated_time 比例较大时（如 ~70%），per-token 边际裁剪激励偏弱，SL 较小时裁剪效果有限。

## 验证计划

1. 默认关闭 adaptive，跑现有 MTP 用例，确认行为不变。
2. 开启 adaptive prune，构造高概率 draft，确认 prefix 接近 `max_sl`。
3. 开启 adaptive prune，构造低概率 draft，确认 prefix 可以降到 0 或较小值。
4. 检查 batch 内不同 prefix 时，target context cache 不越界，`all_draft_accepted` 不误判。
5. 检查 accepted/draft counters 中 draft token 分母等于 `sum(prefix_lengths)`。
6. TP/ATB 多卡确认所有 rank 使用同一个 `effective_sl`，不再出现 validate width 分叉导致 hang。
7. compare 脚本对比 static MTP 和 adaptive-prune MTP 的吞吐、TPOT、MTP acceptance rate。

## 当前实现范围

已落地：

1. `SpeculativeConfig` / `Options` / spawn worker 参数传播 adaptive prune 配置。
2. 新增 `AdaptiveSpeculativeController`，实现基于 draft path probability 的 validate 前剪枝，score 公式按 per-seq 累加估计 validate time。
3. 新增 `SpeculativeProfileRegistry`，保存 ProfileManager 拟合的 validate time predictor，同时作为 adaptive 路径的启用门控。
4. 新增 `SetSpeculativeValidateTimePredictor` worker RPC，把 predictor 广播到 worker runtime 进程。
5. `MTPWorkerImpl::step_decode()` 完整 draft 后进行 adaptive prune，支持 per-seq variable-length validate。
6. Per-seq validate 通过 ATB kernel 的 `q_seq_lens` 减少实际计算量；RejectionSampler 使用 fast-path/slow-path padding 对齐。
7. validate 输出按 per-request prefix 裁剪。
8. metrics 按选中的 draft token 统计 acceptance rate。
9. graph mode、非 MTP、DP/EP 大于 1 时自动关闭 adaptive。
10. profile 阶段增加 KV cache 容量检查，防止超出 blocks 导致 hang。

未落地，留作独立后续 feat：

1. Draft early stop，即直接中断 draft loop。
2. TPOT SLO 控制。
3. Graph bucket 支持。
4. ZOS 适配。
5. 历史 accept length EMA / batch-size 分段 SL range。
6. adaptive 专用 metrics。
