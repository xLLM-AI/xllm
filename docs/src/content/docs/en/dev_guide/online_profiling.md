---
title: "Online Profiling"
sidebar:
  order: 3
---

## Background

Timeline profiling is essential for diagnosing performance bottlenecks in an online serving deployment: where time is spent across host scheduling, device kernels, and communication. vLLM exposes this through two HTTP endpoints, `POST /start_profile` and `POST /stop_profile`, which toggle profiling on every worker so traces can be collected on a live server without restarting it.

xLLM provides the equivalent capability with two backends, selected by `--profile_backend`:

- **`torch` (default)** — records CPU and CUDA activities in-process via libtorch's Kineto profiler (the C++ equivalent of `torch.profiler.profile`) and writes a Chrome trace to disk on `/stop_profile`. No external profiler is required: just launch the server normally and drive the two endpoints. This mirrors vLLM's default `TorchProfilerWrapper`.
- **`cuda`** — only toggles the CUDA profiler capture range (`cudaProfilerStart()` / `cudaProfilerStop()`). It records nothing on its own and must be paired with NVIDIA Nsight Systems (`nsys`): the server is launched under `nsys profile` with a capture range tied to the CUDA Profiler API, and the two endpoints open and close the window that `nsys` records.

Both backends are CUDA only for now. Support for other backends will be added later.

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
```

The engine fans the request out to every worker concurrently and waits for all of them to acknowledge. Because xLLM runs one worker thread per device inside a single process and Kineto/CUPTI is process-global, each profiler is wrapped in a process-wide, idempotent singleton: the first `start` opens the collection window for the whole process, and repeated/overlapping `start`/`stop` calls are coalesced. `/stop_profile` closes the window.

For the `torch` backend, the profiler is enabled and disabled on the worker's compute thread (the one that runs the forward pass), so host-side CPU operators are captured. On `/stop_profile`, libtorch writes the Chrome trace itself; the file goes to `--profile_dir` (the current working directory when unset). For the `cuda` backend, the trace output location is controlled by `nsys` (its `-o` flag), not by xLLM.

## Usage

Profiling is opt-in. Start the server with profiling enabled:

```shell
--enable_online_profile=true
```

`enable_online_profile`, `profile_backend`, and `profile_dir` can also be set via the JSON config file. The endpoints only act when `--enable_online_profile=true`; otherwise they respond with an error explaining how to enable the feature.

### Default backend (`torch`, no nsys required)

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

- Profiling is currently supported on **CUDA only**. On other backends the endpoints return an error.
- The two endpoints are only active when `--enable_online_profile=true`; otherwise they respond with an error explaining how to enable the feature.
- With `--profile_backend=cuda`, `cudaProfilerStart`/`cudaProfilerStop` only have an effect when the server is running under a profiler such as `nsys` (or `ncu`) configured with `--capture-range=cudaProfilerApi`. Calling the endpoints without such a profiler attached is harmless but produces no trace. For multi-process / multi-GPU runs, `nsys` recommends launching with `--trace-fork-before-exec=true` so child worker processes are traced.
- Profiling adds runtime overhead. Enable it only for diagnosis, not in steady-state production serving, and keep the capture window short.
