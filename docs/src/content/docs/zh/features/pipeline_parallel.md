---
title: GLM 流水线并行与 DCPP
---

<!-- Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. -->

Python/NPU GLM 路径（`model_type=glm_moe_dsa`）支持 eager PP × TP，
适用于 GLM-5.3 W8A8。`pp_size` 默认 `1`；16 个 worker 设置 PP2 后，
每级使用 TP8。

## 模型与缓存分层

stage `s` 持有 `[floor(n_layers*s/pp_size), floor(n_layers*(s+1)/pp_size))`。
首级加载 embedding，末级加载最终 norm 和 LM head，各级只加载本级权重。
checkpoint 保留全局层号，KV/index cache 使用本地层号；公共 block 数按
占用最大的 stage 计算，并考虑各级实际拥有的完整 indexer 层数。

相同 TP lane 通过 HCCL PP 子组传输 hidden 和 residual。下一级从共享
indexer 层开始时，还会传输 int32 top-k。GLM-5.3 有 78 层，PP2 的第二级
从共享层 39 开始，因此必须保留这份跨级状态。末级 TP rank 0 负责采样。
GLM 的 PP 适配继续复用主线 DeepSeek/GLM 的公共 decoder 执行循环。

## 连续 chunk 调度

调度器为每条请求预留至多 `pp_size` 个连续 prefill chunk 的 KV 空间，
总量受原有 batch token budget 限制，并在 batch 上记录有序 chunk 计划。
动态模式按每个 chunk 各自的历史长度预测大小；固定模式复用同一条流水线。

engine 在请求之间交错执行 chunk，单条长请求的连续 chunk 也可同时在途。
输入准备阶段分别记录位置、前缀长度和采样目标。每一级保持相同执行顺序，
等待本级全部 TP worker 完成后再复用缓冲区，不同级可以同时执行不同 chunk。
所有 stage 成功后才应用输出；只有 prompt 的最后一个 chunk 采样首个输出 token。

纯 decode batch 在 sequence 边界拆成至多 `pp_size` 个子批次；单请求 decode
的后续 token 依赖前一个采样结果，无法以此方式重叠。

## GLM-5.3 W8A8 启动参数

在现有 16-worker 启动命令的每个 rank 上加入：

```bash
--model=/mnt/models/GLM-5.3-w8a8 \
--model_impl=python \
--pp_size=2 --dp_size=1 --ep_size=1 --cp_size=1 \
--enable_graph=false --enable_schedule_overlap=false \
--enable_task_pipeline=false --enable_shm=false \
--enable_in_batch_prefix_cache=false \
--enable_chunked_prefill=true --enable_dynamic_chunking=true \
--max_tokens_per_batch=16384 --max_tokens_per_chunk_for_prefill=4096 \
--dynamic_chunk_min_tokens=256 --dynamic_chunk_profile_samples=16
```

`python_model_path` 应指向当前 checkout。`pp_size` 必须整除 worker 数量且
不超过模型层数。batch 预算应容纳多个基础 chunk，才能让单条长请求填充流水线。
关闭 `enable_dynamic_chunking` 使用固定 chunk PP，不进行启动 profiling。

当前要求在线 NPU Python GLM、eager、DP=EP=CP=1，不使用 layerwise/KV split。
启动时拒绝 graph、worker task pipeline、投机解码、EPLB、PD、主机缓存卸载、
外部 KV 存储、XTensor、in-batch prefix sharing、multi-stream-parallel 和
beam-search kernel；beam search 请求也会在调度前拒绝。

启动 profiling 测量单个 chunk 穿过整个流水线的耗时，目前不分别拟合各 stage
成本，也不进行在线校准；模型按层数分配。预测算法见[动态分块预填充](../dynamic_chunked_prefill)。

## 验证范围

回归覆盖权重/层归属、residual/top-k 等价性、四进程通信、stage 内顺序和跨级重叠、
公共 KV 容量、chunk 预算及最终 chunk 采样。真实 HCCL、全模型精度和性能需要在
目标 NPU 上验证；不能仅凭这些回归测试认定存在推理加速。
