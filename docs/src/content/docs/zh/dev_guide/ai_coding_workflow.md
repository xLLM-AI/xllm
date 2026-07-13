---
title: "AI Coding 工作流"
sidebar:
  order: 4
---

本文总结一套面向 xLLM 开发者的实践工作流，用于 NPU 推理优化、回归定位，以及为
PR 准备可复现的验证证据。

完整 workflow 知识库维护在
[:simple-github: xllm-workflow](https://github.com/xLLM-AI/xllm-workflow)。
当你需要可直接加载的 agent skills、Prompt 模板、artifact schemas 或模型优化历史时，
可以使用该仓库。

![xLLM AI Coding 工作流](../../assets/xllm-ai-coding-workflow-zh.png)

## 什么时候使用

当改动需要普通代码 review 之外的证据时，建议使用这套工作流：

- 优化 TTFT、TPOT、TPS、内存使用或 serving 并发能力；
- 公平比较 xLLM 与 vLLM-Ascend、SGLang NPU 或其他 serving 框架；
- 定位乱码输出、数据集掉分、GPU/NPU 不一致、OOM、图 replay 失败、HCCL 问题或运行时 crash；
- 在 NPU 相关 PR 合入前做验证；
- 判断是否需要算子迁移或 kernel-level 实验。

不要把一次 smoke run 当作正式结论。正式的性能和精度结论应包含精确命令、环境、
workload、原始 artifacts 和归一化 summary。

## 证据闭环

对性能优化和 correctness-sensitive 改动，推荐遵循下面的闭环：

```text
target -> baseline -> profiling -> patch -> accuracy -> performance -> record
```

核心原则是把 benchmark、profiling 和 accuracy 证据分开。Profiling 用于解释瓶颈，
不能替代 warmed-up before/after 性能对比。

完整优化任务可以按下面阶段推进：

| 阶段 | 目的 | 产物 |
|---|---|---|
| 目标与环境 | 定义目标、模型、框架 commit、硬件、CANN/runtime 版本、workload 和 SLA。 | Run manifest |
| 历史知识 | 查询历史模型 PR、失败尝试和已知风险路径。 | History notes |
| 公平基线 | 改代码或参数前先跑 warmed-up baseline。 | 原始 metrics 和 summary |
| 证据采集 | 根据症状采集 profiling、capacity、pipeline、compute 或 accuracy 证据。 | 诊断报告 |
| Patch | 每轮尽量只做一个有意义、可 review 的改动。 | Code diff |
| 验证 | 根据改动重新运行 accuracy、performance、build 和 UT 检查。 | 验证表 |
| 沉淀 | 保存命令、指标、失败尝试、风险说明和后续工作。 | 可复用经验 |

## 开发者检查清单

在提交或更新 NPU 优化 PR 前，PR 描述应能回答这些问题：

- 使用了什么模型、tokenizer、dtype、设备型号、设备数量和框架 commit？
- 精确的启动命令和 benchmark 命令是什么？
- baseline 是否经过 warmup，并且运行在干净设备上？
- profiling 结果是否只作为诊断证据使用？
- 跑到了哪个精度验证等级？如果有失败样例，保存在哪里？
- patch 改了什么？还剩哪些风险？
- 其他开发者可以用哪些 artifacts 复现结论？

## 推荐产物

正式结果建议在同一个 run root 下保存：

- 包含环境和命令细节的 `manifest.md` 或 `manifest.yaml`；
- 原始 evalscope 或 benchmark 输出；
- 归一化的 `metrics.json` 或 `summary.md`；
- 使用 profiling 诊断时的 profiling report 和 timeline notes；
- 精度任务中的 `failed_cases.jsonl` 或等价坏例记录；
- 说明改了什么、为什么安全、如何验证的 PR notes。

workflow 仓库提供了这些共享 schema：

- `references/run-manifest-template.md`
- `references/perf-artifact-schema.md`
- `references/profiling-artifact-schema.md`
- `references/accuracy-artifact-schema.md`

## 相关 Skills

workflow 仓库包含面向任务的 skills，可由 Codex、Claude Code、opencode 或其他本地
agent runtime 加载：

| 任务 | Skill |
|---|---|
| 端到端优化 | `xllm-npu-sota-loop` |
| 服务启动和 evalscope 采集 | `xllm-npu-eval-runner` |
| 公平框架对比 | `xllm-npu-benchmark` |
| msprof / MindStudio 分析 | `xllm-npu-profiler` |
| Decode bubble 和 rank skew 分析 | `xllm-npu-pipeline-analysis` |
| HBM、KV cache 和 OOM 分析 | `xllm-npu-capacity-planner` |
| FLOPs、MFU 和理论下界估算 | `xllm-npu-compute-simulation` |
| 精度回归定位 | `xllm-npu-accuracy-debug` |
| Crash 或运行时事故定位 | `xllm-npu-incident-triage` |
| NPU 代码 review | `xllm-npu-code-review` |
| 算子迁移 | `xllm-npu-op-migration` |

这些 skills 是工程纪律的辅助工具。最终结论仍应基于可复现的 xLLM artifacts 和可
review 的代码改动。
