---
title: "Online Profiling"
sidebar:
  order: 3
---

## Background

Timeline profiling is essential for diagnosing performance bottlenecks in an online serving deployment: where time is spent across host scheduling, device kernels, and communication. vLLM exposes this through two HTTP endpoints, `POST /start_profile` and `POST /stop_profile`, which toggle profiling on every worker so traces can be collected on a live server without restarting it.

xLLM provides the equivalent capability with platform-specific backends, selected by `--profile_backend`:

- **`torch` (default on non-NPU builds)** — records CPU and CUDA activities in-process via libtorch's Kineto profiler (the C++ equivalent of `torch.profiler.profile`) and writes a Chrome trace to disk on `/stop_profile`. No external profiler is required: just launch the server normally and drive the two endpoints. This mirrors vLLM's default `TorchProfilerWrapper`.
- **`cuda`** — only toggles the CUDA profiler capture range (`cudaProfilerStart()` / `cudaProfilerStop()`). It records nothing on its own and must be paired with NVIDIA Nsight Systems (`nsys`): the server is launched under `nsys profile` with a capture range tied to the CUDA Profiler API, and the two endpoints open and close the window that `nsys` records.

**`ascend` (default on NPU)** uses the CANN profiling API to collect NPU tasks, AscendCL calls, and HCCL communication. It supports both native C++ and Python models, including ACLGraph replay. Raw data is flushed on stop and exported with `msprof`. This backend does not collect PyTorch CPU operator stacks.

## Introduction

The control flow mirrors the existing `sleep`/`wakeup` broadcast path:

```
HTTP POST /start_profile
   -> APIService::StartProfileHttp        (xllm/api_service)
   -> Master::start_profile               (xllm/core/distributed_runtime)
   -> Engine::start_profile               (broadcast to all workers)
        -> WorkerClient::start_profile_async   (local worker, in-process)
        -> RemoteWorker::start_profile_async   (remote worker, over brpc)
             -> WorkerService::StartProfile     (worker server handler)
        -> WorkerImpl::start_profile
             -> TorchProfiler::start            (Kineto, default)
                or CudaProfiler::start          (cudaProfilerStart, --profile_backend=cuda)
                or NpuProfiler::start           (CANN, default on NPU)
```

The engine fans the request out to every local and remote worker concurrently and waits for all of them to acknowledge. Profiler state is managed by a singleton in each worker process. The NPU backend shares CANN initialization within a process and tracks capture separately for each device. `/stop_profile` stops the devices and flushes each process's output.

For the `torch` backend, the profiler is enabled and disabled on the worker's compute thread (the one that runs the forward pass), so host-side CPU operators are captured. On `/stop_profile`, libtorch writes the Chrome trace itself; the file goes to `--profile_dir` (the current working directory when unset). For the `cuda` backend, the trace output location is controlled by `nsys` (its `-o` flag), not by xLLM.

## Usage

Profiling is opt-in. Start the server with profiling enabled:

```shell
--enable_online_profile=true
```

`enable_online_profile`, `profile_backend`, and `profile_dir` can also be set via the JSON config file. The endpoints only act when `--enable_online_profile=true`; otherwise they respond with an error explaining how to enable the feature.

### Ascend NPU backend (`ascend`)

Add these options to the existing NPU server command on every worker node:

```shell
--enable_online_profile=true --profile_backend=ascend --profile_dir=/tmp/xllm-profile
```

Warm up the model, call `POST /start_profile`, send the inference requests to capture, then call `POST /stop_profile`. The same endpoints broadcast to local and remote workers. The engine serializes start/stop requests and stops all workers if a start broadcast fails.

Each process creates a unique `xllm_npu_<pid>_<suffix>` session directory. Devices share process initialization but retain separate capture configurations; the last device to stop finalizes and flushes the session. Repeated starts on an active device and stops on an inactive device are idempotent. A new start after a completed stop creates a new directory. Each transition drains the worker's torch_npu dispatch queue and device work on its compute thread.

After a successful stop, use the absolute session path printed in the server log:

```shell
msprof --export=on --output=/tmp/xllm-profile/xllm_npu_<pid>_<suffix>
```

Inspect the exported operator statistics and `msprof_*.json` timelines with MindStudio Insight or Perfetto. Keep the raw `PROF_*` directories. For distributed serving, collect the output from each worker process/node and verify all expected devices are present. No external collector needs to be attached while serving.

### Kineto backend (`torch`, non-NPU)

Just start the server normally and choose where traces are written:

```shell
<your xllm serve command> \
    --enable_online_profile=true \
    --profile_dir=/path/to/traces   # optional; defaults to the current directory
```

Then, against the running server:

```shell
# Start collecting
curl -X POST http://127.0.0.1:9977/start_profile

# ... send the inference requests you want to profile ...

# Stop collecting (writes the trace at this point)
curl -X POST http://127.0.0.1:9977/stop_profile
```

(Replace the host/port with your server's address.) On `/stop_profile`, each worker writes a Chrome trace named `xllm_rank<rank>_<pid>_<timestamp>.pt.trace.json` into `--profile_dir`. The absolute path is printed in the server log.

### `cuda` backend (capture-range only, requires nsys)

To use the capture-range path instead, set `--profile_backend=cuda` and launch the server under `nsys` with a capture range tied to the CUDA Profiler API:

```shell
# --capture-range=cudaProfilerApi makes nsys start/stop capturing exactly when
# cudaProfilerStart/Stop are called, and --capture-range-end=repeat allows
# multiple start/stop cycles in one session.
nsys profile \
    --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    --capture-range-end=repeat \
    -o xllm_profile \
    <your xllm serve command> --enable_online_profile=true --profile_backend=cuda
```

Drive the window with the same `/start_profile` and `/stop_profile` endpoints. When the server process exits, `nsys` writes the report to `xllm_profile.nsys-rep`.

## Viewing traces

For the `torch` backend, open the generated `.pt.trace.json` in [Perfetto](https://ui.perfetto.dev), `chrome://tracing`, or TensorBoard.

For the `cuda` backend, open the generated `.nsys-rep` in the Nsight Systems GUI, or summarize it on the command line:

```shell
nsys stats xllm_profile.nsys-rep
```

## Notice

- Ascend NPU uses `--profile_backend=ascend`; unsupported backend/platform combinations return an error. Do not combine this backend with another CANN profiler (including `PROFILING_MODE=dynamic` or profiling configured through `aclInit`).
- The two endpoints are only active when `--enable_online_profile=true`; otherwise they respond with an error explaining how to enable the feature.
- With `--profile_backend=cuda`, `cudaProfilerStart`/`cudaProfilerStop` only have an effect when the server is running under a profiler such as `nsys` (or `ncu`) configured with `--capture-range=cudaProfilerApi`. Calling the endpoints without such a profiler attached is harmless but produces no trace. For multi-process / multi-GPU runs, `nsys` recommends launching with `--trace-fork-before-exec=true` so child worker processes are traced.
- Profiling adds runtime overhead. Enable it only for diagnosis, not in steady-state production serving, and keep the capture window short.
