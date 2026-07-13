---
title: "在线性能采集 (Online Profiling)"
sidebar:
  order: 3
---

## 背景

在线服务部署中，Timeline 性能采集对于定位性能瓶颈至关重要：它能展示时间究竟消耗在主机侧调度、设备侧算子还是通信上。vLLM 通过两个 HTTP 接口 `POST /start_profile` 与 `POST /stop_profile` 实现该能力，在每个 worker 上开启/关闭采集，从而无需重启即可在运行中的服务上采集 timeline。

xLLM 提供等价的能力，并通过 `--profile_backend` 选择两种后端：

- **`torch`（默认）** —— 通过 libtorch 的 Kineto profiler（即 `torch.profiler.profile` 的 C++ 等价实现）在进程内采集 CPU 与 CUDA 活动，并在 `/stop_profile` 时把 Chrome trace 直接写到磁盘。无需任何外部 profiler：正常启动服务、调用这两个接口即可。该方式与 vLLM 默认的 `TorchProfilerWrapper` 对齐。
- **`cuda`** —— 仅开关 CUDA profiler 的 capture range（`cudaProfilerStart()` / `cudaProfilerStop()`）。它本身不记录任何内容，必须与 NVIDIA Nsight Systems（`nsys`）配合使用：服务在 `nsys profile` 下启动，并将 capture range 绑定到 CUDA Profiler API，这两个接口负责开启/关闭 `nsys` 实际记录的窗口。

两种后端目前均仅支持 CUDA，其他芯片后续支持。

## 原理介绍

整体调用链复用了已有的 `sleep`/`wakeup` 广播路径：

```
HTTP POST /start_profile
   -> APIService::StartProfileHttp        (xllm/api_service)
   -> Master::start_profile               (xllm/core/distributed_runtime)
   -> Engine::start_profile               (广播到所有 worker)
        -> WorkerClient::start_profile_async   (进程内本地 worker)
        -> RemoteWorker::start_profile_async   (通过 brpc 的远程 worker)
             -> WorkerService::StartProfile     (worker 服务端处理)
        -> WorkerImpl::start_profile
             -> TorchProfiler::start            (Kineto，默认)
                或 CudaProfiler::start          (cudaProfilerStart，--profile_backend=cuda)
```

Engine 会将请求并发广播到每个 worker，并等待全部确认。由于 xLLM 在单进程内为每个设备运行一个 worker 线程，而 Kineto/CUPTI 是进程级的，因此每个 profiler 都被封装为进程级、幂等的单例：首次 `start` 会为整个进程开启采集窗口，重复或并发的 `start`/`stop` 调用会被合并处理。`/stop_profile` 会关闭采集窗口。

对于 `torch` 后端，profiler 在 worker 的计算线程（即执行 forward 的线程）上开启与关闭，从而能采集到主机侧的 CPU 算子。在 `/stop_profile` 时，由 libtorch 自行写出 Chrome trace，文件落到 `--profile_dir`（未设置时为当前工作目录）。对于 `cuda` 后端，trace 的输出位置由 `nsys` 控制（其 `-o` 参数），而非由 xLLM 管理。

## 使用方式

该功能默认关闭，需显式开启。启动服务时开启采集：

```shell
--enable_online_profile=true
```

`enable_online_profile`、`profile_backend` 与 `profile_dir` 也可以通过 JSON 配置文件设置。仅当 `--enable_online_profile=true` 时这两个接口才会生效；否则会返回错误并提示如何开启该功能。

### 默认后端（`torch`，无需 nsys）

正常启动服务，并指定 trace 的输出目录即可：

```shell
<你的 xllm serve 启动命令> \
    --enable_online_profile=true \
    --profile_dir=/path/to/traces   # 可选；默认为当前目录
```

随后，针对运行中的服务：

```shell
# 开始采集
curl -X POST http://127.0.0.1:9977/start_profile

# ... 发送需要采集的推理请求 ...

# 停止采集（此时写出 trace）
curl -X POST http://127.0.0.1:9977/stop_profile
```

（请将 host/port 替换为你的服务地址。）在 `/stop_profile` 时，每个 worker 会在 `--profile_dir` 下写出名为 `xllm_rank<rank>_<pid>_<timestamp>.pt.trace.json` 的 Chrome trace，其绝对路径会打印在服务日志中。

### `cuda` 后端（仅 capture range，需要 nsys）

若改用 capture range 方式，设置 `--profile_backend=cuda`，并在 `nsys` 下启动服务、将 capture range 绑定到 CUDA Profiler API：

```shell
# --capture-range=cudaProfilerApi 让 nsys 恰好在 cudaProfilerStart/Stop 被调用时
# 开始/停止采集；--capture-range-end=repeat 允许在一次会话中进行多次 start/stop 循环。
nsys profile \
    --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    --capture-range-end=repeat \
    -o xllm_profile \
    <你的 xllm serve 启动命令> --enable_online_profile=true --profile_backend=cuda
```

用同样的 `/start_profile` 与 `/stop_profile` 接口控制采集窗口。当服务进程退出时，`nsys` 会将报告写入 `xllm_profile.nsys-rep`。

## 查看 trace

对于 `torch` 后端，在 [Perfetto](https://ui.perfetto.dev)、`chrome://tracing` 或 TensorBoard 中打开生成的 `.pt.trace.json`。

对于 `cuda` 后端，在 Nsight Systems GUI 中打开生成的 `.nsys-rep`，或在命令行汇总：

```shell
nsys stats xllm_profile.nsys-rep
```

## 注意事项

- 当前仅支持 **CUDA**。其他芯片上调用这两个接口会返回错误。
- 仅当 `--enable_online_profile=true` 时这两个接口才会生效；否则会返回错误并提示如何开启该功能。
- 使用 `--profile_backend=cuda` 时，`cudaProfilerStart`/`cudaProfilerStop` 只有在服务运行于 `nsys`（或 `ncu`）这类配置了 `--capture-range=cudaProfilerApi` 的 profiler 之下时才会生效。未挂载此类 profiler 时调用接口无害，但不会产生 trace。对于多进程 / 多 GPU 场景，`nsys` 推荐使用 `--trace-fork-before-exec=true` 启动，以便子 worker 进程也被采集。
- 采集会带来运行时开销，仅建议在诊断时开启，不要在稳态生产服务中长期开启，且采集窗口应尽量短。
