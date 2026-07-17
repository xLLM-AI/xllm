---
title: "运行时 CP↔DP 模式切换"
sidebar:
  order: 91
---
## 背景

在固定 world size 下，xLLM 支持两种并行布局：

- **CP_PREFILL** (`cp=N, dp=1`)：全体 worker 上跑 context parallelism。长 prompt 的 KV 计算被拆到多个 worker 上，能显著降低大输入请求的 TTFT。
- **DP_DECODE** (`cp=1, dp=N`)：data parallelism，`N` 个独立 decoder 副本。小请求下并发和吞吐更高，代价是不再对单请求做工作切分。

真实流量的形态很少长期停留在同一种。长上下文的批量灌入场景更适合 CP；并发的短请求场景更适合 DP。按实例静态配置会迫使运维要么给一条通道超配，要么在另一条通道上接受指标退化。

**运行时 CP↔DP 模式切换** 允许一个 xLLM 实例在不重启服务的前提下，在两种布局之间按需切换，切换耗时约 500 ms。

## 什么时候用

- 一天内流量形态会切换（例如夜间跑长上下文批量任务，白天跑对话流量）。
- 想在同一个实例上验证两种布局，而不是维护两套部署。
- 你在开发一个自动扩缩控制器，需要按实例决策布局，并调用一个可以由机器发起的 switch RPC。

**不建议**在以下场景使用：

- 你在排查数值精度问题——先固定为一种布局，把 flip 当作变量排除掉。
- 你的 workload 长期只有一种形态——同时驻留两张 ATB 图（约 600 MB / NPU + 额外 graph workspace）并非免费。

## 工作原理

### 启动：双图模式

worker 以 `--enable_runtime_cp_dp_switch=true` 启动时，会预先初始化两种并行布局：

- 两个 `CollectiveCommunicator`。一个 `cp_size=N,dp_size=1`，一个 `cp_size=1,dp_size=N`。每个都在独立的 master port 上拉起自己的 HCCL 通信域（端口偏移由 `--dual_mode_port_stride` 控制，默认 256）。
- 两份 `ParallelArgs` 快照被 `DualParallelArgs` 包住，配一个 `atomic<Mode>` 判别器。`active()` 以 acquire 语义读取当前快照。
- 每个 decoder layer 有 4 个 ATB 节点：`cp_prefill`、`cp_decode`、`dp_prefill`、`dp_decode`。权重共享，只是 graph 绑定不同。

初始模式由启动时的 `--cp_size` 决定（`cp>1 → CP_PREFILL`，否则 `DP_DECODE`）。

### 切换：通过 RPC 原子地翻转布局

调用方在实例的 `--disagg_pd_port` 上调用 `xllm.proto.ModeSwitchService.SwitchMode`（prefill 和 decode 实例都会承载这个服务）：

```bash
curl -sS -X POST \
    "http://<host>:<disagg_pd_port>/xllm.proto.ModeSwitchService/SwitchMode" \
    -d '{"target_mode":1,"timeout_ms":30000}'
```

`target_mode` 为 `0` 表示 CP_PREFILL，`1` 表示 DP_DECODE。成功时响应体是 `{"ok":true,"current_mode":<0|1>}`。

**必须先 P 再 D。** decode 侧会根据 etcd 里 prefill 侧广播的 `dp_size` 重建 datadist 链路。如果两侧并行 flip 且 D 抢先，D 读到的是 P 的旧 `dp_size`，会连到错误的拓扑。任何脚本都应确保先切完 prefill、等到响应，再切 decode。

### 编排流程

`ModeSwitchService::SwitchMode` 在每个实例上按以下顺序执行：

1. **`scheduler.pause(WAIT)`** —— 停止接收新 batch；已经在飞的 step 跑完。
2. **`wait_until_paused(timeout_ms)`** —— 确认已经 drain 干净；若未 drain 干净则返回 `{"error":"scheduler drain timeout"}` 给调用方，模式保持不变。
3. **`begin_switch()`** —— 拿 `std::shared_mutex` 的 unique lock。`DisaggPDScheduler` 中的 6 条读路径（`dispatch_requests`、`try_allocate`、`decode_schedule`、`decode_recv_first_generation`、`link_instance`、`unlink_instance`）持有 shared lock；unique lock 等它们全部释放。
4. **`engine.switch_mode(target)`** —— 向所有 worker 广播 `SwitchMode` RPC。每个 worker 翻转自己的 `DualParallelArgs::active_` atomic，并刷新缓存的 `parallel_args_` 和 `ModelContext`。engine 也在 `paired = max(cp,dp)` 不变量下更新自己的 `cp_size_/dp_size_/dp_local_size_`，使后续 scheduler 轮询看到新布局。
5. **`rebuild_after_flip(new_dp_size)`** —— 为新的 dp_size 重建 `BlockManagerPool`。复用旧 pool 的 `Options`，避免重新根据 KV cache cap 推导配置。把新 pool 指针以 release-store 发布给 `XServiceClient`（`atomic<const Pool*>`），心跳/reconcile 线程每轮以 acquire-load 拿一次快照。
6. **`re_register_dp_size(new)`** —— 把新 dp_size 写回 etcd 的 `InstanceInfo`，让对端 reconcile 时观察到 flip 后的拓扑。
7. **`gate.unlock()` + `scheduler.resume()`** —— 释放写锁并重新接收 batch。稳态代价：每个 step 一次原子 acquire-load。
8. **`relink_after_flip`** —— 仅 decode 侧。unlink 旧的 datadist 链路，用刚从 etcd 拉到的 peer 侧 dp_size 重新 link。**这一步在 switch gate 之外执行**，否则会在 datadist worker RPC handler 和已经持有 gate 的调用线程之间产生死锁。

### 稳态运行时需要注意的问题

**Lopsided batch 延迟（DP 模式）。** DP forward 要求每个 dp_rank 以形状一致的输入进入 collective。如果一个 dp_rank 有真实 batch 而另一个是空的，直接进 forward 要么让空 rank 的对端 hang（空 rank 提前返回），要么让空 rank 伪造 fake input 从而竞争真的 KV block。scheduler 会延迟这类 step：若 `active_dp_size > 1` 且任一 dp_rank 是空 batch，`step()` 直接返回，不下发到 engine。100 ms 的 backdoor 会在长时间不平衡时强制放行一次—— 宁可走 fake-input 路径也不能无限饿死。实测 backdoor 几乎不触发（DP dispatch 一般 10 ms 内自动收敛）。

**诊断日志。** `enable_flip_verbose_log` gflag（默认 `false`）门控 12 处 per-request / per-step 的 FLIPDIAG 日志。flip 生命周期事件（`switch_mode`、`rebuild_after_flip`、`relink_after_flip`、`lopsided_backdoor`、`EARLY_RETURN`）无条件打印。可通过 brpc 内建端点在运行时切换（该 flag 用 `DEFINE_validator` 标记为可 reload）：

```bash
curl "http://<host>:<disagg_pd_port>/flags/enable_flip_verbose_log?setvalue=true"
```

## 失效模式与根因

开发过程中暴露了 6 层并发问题，记录在此以便后续回归有起点：

| # | 现象 | 根因 | 修复 |
|---|------|------|------|
| 1 | rebuild 期间 rank0 变僵尸 | dispatch handler 与 `kv_cache_manager_` 上的 `unique_ptr::reset` 竞态 | `switch_gate_` 的 unique_lock + pause/drain gate |
| 2 | 每次 flip 后约 5s rank0 死掉 | `XServiceClient` 心跳持有旧 pool 的裸指针（对象已销毁） | `std::atomic<const BlockManagerPool*>` + release/acquire；心跳每轮 pin 一份快照 |
| 3 | flip 回来时 CHECK 失败 `num_free_blocks_==free_blocks_.size()-1` | `BlockManagerImpl` 析构 CHECK 太严；序列在原不变量窗口之外释放 | 把 CHECK 降级为 WARNING |
| 4 | `PushKvBlocks failed, ret=5010b007`（LLM_NOT_YET_LINK） | peer 的 `dp_size` 改了以后 datadist 链路拓扑变陈旧 | `re_register_dp_size` + `relink_after_flip`（P 侧跳过；D 侧拉最新 `InstanceInfo`；调用方保证 P→D 串行） |
| 5 | DP forward 在 `batch_sizes=[N,0]` 时 hang | `dp_global_token_nums` 上空 rank 显示 0，而 worker 的 fake-input 路径把它写成 1；DP collective 尺寸不一致 | 在 `LLMEngine::prepare_inputs` 里，当 `dp_size>1` 时做 `dp_global_token_nums[dp_rank] = max(1, ...)` clamp |
| 6 | DP forward 在 lopsided batch 上偶尔仍 hang | ATB attention op 在一个 shard 用真实 block_tables、另一个 shard 走 placeholder 路径时 hang | scheduler 层延迟：跳过 lopsided batch，100 ms backdoor 作为安全阀 |

## 验证

参考的 `verify_switch.sh` 回归脚本运行如下流程：

1. HTTP 探测确认服务已启动。
2. 把 P 和 D 都重置到 CP 模式，作为 preflight。
3. Warmup 请求。
4. **CP burst**（默认 6 并发，`max_tokens=6`）。
5. 空闲等待到 P 和 D 都报 `Running requests: 0`。
6. **Flip 0 → 1**（串行 P 后 D），打印两侧延迟（ms）。
7. 空闲等待，跑 **DP burst**。
8. 空闲等待，**flip 1 → 0**。
9. 空闲等待，跑 **CP-post burst**。

成功标准：3 轮 burst 累计 18/18 响应；两次 flip 都不出现 `drain timeout`；两个实例的 rank0 日志中都不出现 `main process disappeared`。

## 配置参考

| Flag | 默认值 | 用途 |
|------|--------|------|
| `--enable_runtime_cp_dp_switch` | `false` | 启动时拉起两个 `CollectiveCommunicator` + 4 个 ATB 节点。flip 依赖此项。 |
| `--dual_mode_port_stride` | `256` | CP 和 DP master port 之间的偏移。256 是为了避免快速重启时 TIME_WAIT 端口碰撞。 |
| `--enable_flip_verbose_log` | `false` | 打印 per-request / per-step 的 FLIPDIAG 行。可通过 brpc `/flags` 端点在运行时切换。 |
| `--disagg_pd_port` | 非 disagg `7777` / disagg 实例侧配置 | `ModeSwitchService` 的监听端口。 |

## 暂未覆盖

- 61 层真实 DeepSeek-V3.2-w8a8 模型在跨机 disagg 下的验证（单机 disagg 已验证）。
- 跨机 disagg 的 flip。
- CP 与 DP 在相同 seed 下的 `lm_eval` 数值对齐扫描。
- 长时间稳定性（>1 h 周期性 flip），包含内存 / 连接泄漏检查。
- 自动 mode-switch 控制器（本 PR 引入了 scaffolding；接入真实流量信号在后续 PR）。
