---
title: "C++ Serving Framework + Python Model Execution Architecture Decision"
sidebar:
  order: 3
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

This document discusses why xLLM adds Python model execution to its existing C++ serving
framework, and why the Python side contains both a model and a `ModelExecutor`. It focuses only
on the architectural choice and does not describe implementation classes, interface details, or
migration steps.

## 1. Situation

### 1.1 xLLM Already Has a Complete C++ Serving Framework

xLLM's existing C++ framework already covers the complete path from request ingestion to result
return:

```text
Input Processing
    -> Continuous Batching Scheduler
    -> Worker Runtime
    -> Sampling / Speculative Decoding
    -> Output Processing
```

- **Input Processing** handles tokenization, chat templates, and multimodal input processing.
- **Continuous Batching Scheduler** handles chunked prefill, prefill/decode scheduling, request
  priority, and schedule overlap.
- **Worker Runtime** receives batches produced by the scheduler, constructs model inputs, and
  manages device resources and the execution flow.
- **Sampling / Speculative Decoding** handles next-token selection as well as draft, verification,
  and token acceptance for speculative decoding methods such as MTP, Eagle, and Suffix.
- **Output Processing** handles asynchronous responses, detokenization, and streaming output.

`KV Cache Manager`, `Distributed Runtime`, and `Observability` support the entire path. The first
two manage KV blocks, prefix cache, multi-level KV cache, master/worker execution, multiple ranks,
multiple nodes, prefill-decode disaggregation, and KV transfer. The last provides profiling,
metrics, and device monitoring.

This C++ framework is the foundation of this architecture decision. Python model execution is
integrated at the model-computation position without changing the serving and runtime
responsibilities described above.

### 1.2 Main Problems with the Current C++ LibTorch Models

The existing C++ LibTorch models already provide stable and optimizable model execution. Python
is not introduced because C++ models cannot meet performance requirements, but because new model
integration and long-term maintenance are increasingly constrained by the following issues.

#### Community Models Must Be Reimplemented

After a new model is released, the implementations that can be referenced directly usually come
from the model vendor, Hugging Face, vLLM, or SGLang. They are primarily Python `nn.Module`
implementations, Python weight loaders, and Python layer APIs. Integrating such a model as a C++
LibTorch model requires re-expressing the model structure, TP partitioning, weight mapping, and
attention calls, followed by a separate numerical-alignment effort.

This is not a simple language translation. When the community implementation later adds a new
attention variant, quantization format, or checkpoint field, the C++ model must also be updated
manually. The community Python implementation and the xLLM C++ implementation therefore become
two copies of model code that must remain synchronized over time.

#### Python-First Tools Require Additional Integration Work

xLLM already supports integrating Triton and TileLang kernels through AOT compilation. The
problem is that kernels often need to support different model sizes, TP sizes, batch and sequence
shapes, data types, quantization formats, and hardware models. AOT compilation requires these
combinations to be prepared in advance and makes it difficult to autotune for runtime shapes on
the actual deployment hardware.

Triton, TileLang, quantization loaders, and many new kernels usually provide Python APIs first. A
C++ model can continue to invoke registered native operators, but adding or tuning a Python-first
kernel requires additional handling for AOT builds, wrappers, caches, type conversions, and
lifecycles.

Python is also better suited to using hooks, per-layer tensor dumps, and profilers for numerical
alignment with community reference implementations. Changes to C++ models usually require
recompilation and relinking, which lengthens the debugging cycle during model migration.

`torch.compile`, Dynamo, FX, and Inductor are also considerations, but they have a lower priority
than community model migration, JIT kernels, quantization, and debugging capabilities. They can
operate directly on Python models, but not on model graphs defined with C++ `torch::nn::Module`.

## 2. Task

The problem to solve is: **reduce the structural rewriting required when migrating community
Python models to xLLM, while preserving the existing C++ serving framework and maintaining the
performance and multi-hardware extensibility required for production inference.**

This task has the following constraints:

1. C++ continues to own Input/Output Processing, Continuous Batching Scheduler, KV Cache Manager,
   Distributed Runtime, Worker Runtime, Sampling, and Speculative Decoding. The overall control
   flow of speculative decoding methods such as MTP remains in C++; Python only performs the
   associated model tensor computation.
2. The Python model should remain close to the responsibilities of a community `nn.Module`,
   managing only model structure, weights, and model computation. It must not duplicate the
   scheduler, KV cache manager, or request state in Python.
3. Persistent inputs, padding, attention plans, and graph state must not be scattered across model
   layers. They require an independent owner for model execution.
4. Batch data, page tables, sequence lengths, slot mappings, and other data already constructed by
   C++ must not be recomputed in Python. The cross-language boundary must remain narrow and tensor
   transfer must remain zero-copy.
5. The target architecture must support different attention backends, collective backends, and
   hardware graph runtimes. Common model code must not depend on FlashInfer, CUDA Graphs, or other
   device-specific implementations.
6. Per-step Python eager overhead must be controlled. Reaching target production performance
   requires graph execution coverage for decode, prefill, and mixed batches.
7. Each rank keeps an independent process and Python interpreter. C++ serving threads do not enter
   Python; only the worker's model-execution path crosses the interpreter boundary.

## 3. Solution

xLLM adopts **C++ Serving Framework + Python Model Execution**: the C++ framework retains its
existing responsibilities, while Python owns the model and the state and execution modes directly
related to model execution.

```text
┌─────────────────────────────────────────────────────────────┐
│ C++ Serving Framework                                       │
│ Input/Output Processing · Continuous Batching Scheduler      │
│ KV Cache Manager · Distributed Runtime · Worker Runtime      │
│ Sampling · Speculative Decoding · Observability              │
└───────────────────────────┬─────────────────────────────────┘
                            │ PyTorch tensor + metadata view
                            │ once per step
┌───────────────────────────┴─────────────────────────────────┐
│ Python Model Execution                                      │
│ Model: model structure · weights · forward · logits          │
│ ModelExecutor/Runner: persistent inputs · padding · graph    │
│ AttentionBackend: plan · execute · backend state            │
│ Distributed: TP group · collective                          │
└───────────────────────────┬─────────────────────────────────┘
                            │ torch.ops / backend API
┌───────────────────────────┴─────────────────────────────────┐
│ Device Kernels and Runtime                                  │
│ xllm_ops · Triton/TileLang · attention kernels · graph API   │
└─────────────────────────────────────────────────────────────┘
```

The responsibilities are divided as follows:

- **C++ framework** owns requests, scheduling, batches, KV cache, distributed workers,
  speculative decoding, sampling, and output processing, and invokes Python model execution once
  per step.
- **Python Model** owns model structure, weight loading, forward computation, and logits. It does
  not hold runner, padding, attention-plan, or graph state.
- **Python ModelExecutor/Runner** owns persistent inputs, padding, execution modes, and the graph
  lifecycle, and invokes an already constructed model.
- **AttentionBackend** owns attention wrappers, workspaces, per-step planning, and
  backend-specific state. For a backend that requires planning, each attention group plans only
  once per step, and all layers reuse that plan.
- **Python Distributed** is responsible only for the TP group and collectives inside the model.
  Worker and rank management, multi-node coordination, prefill-decode disaggregation, and KV
  transfer remain the responsibility of the C++ `Distributed Runtime`.
- **Device ops** select hardware implementations through PyTorch dispatch or a specific backend.
  Model layers do not select CUDA, NPU, or other device implementations.

Common attention metadata remains an output of the C++ InputBuilder. C++ constructs it, and the
Python attention backend only consumes a metadata view and converts it into runtime state for the
current backend. Attention metadata is called out separately to clarify this cross-language
ownership, not to treat it as an independent C++ framework subsystem.

Graphs are managed by concrete Python runners and attention backends. The common model and common
runner interfaces do not prescribe CUDA Graphs, ACL Graphs, or another type of device graph.
Different hardware platforms can provide their own concrete runners without sharing a single
graph inheritance hierarchy.

## 4. Rationale

### 4.1 Comparison of Alternatives

This document compares only the four alternatives directly relevant to the task:

| Alternative | What It Solves | Main Limitation |
|---|---|---|
| Continue using only C++ LibTorch models | Preserves the existing execution path and runtime dependencies | Community Python models still need to be reimplemented, so model migration and long-term synchronization costs remain |
| Let the Python model use a runner directly inside `forward()` | Adds only one Python model entry point and is simple to integrate | The model owns weights, persistent inputs, padding, attention plans, and graph state at the same time, making it difficult to preserve the responsibilities and structure of a community model |
| Python model + C++ model executor | Reuses community models and part of the C++ executor | Persistent inputs, padding, attention plans, and graph state span C++ and Python; adding a Python attention backend or hardware graph still requires modifying the C++ executor |
| Python model + Python `ModelExecutor` | Reuses both the Python model ecosystem and Python execution state; runners, attention backends, and graphs can evolve together in Python | Requires embedded Python, a cross-language metadata contract, and Python graph capabilities |

**Continuing to use only C++ LibTorch models** preserves the validated native path and remains
appropriate for existing models and deployments that cannot accept a Python runtime. However, it
does not solve the primary problem of this task: community models still need to be reimplemented,
and later model changes still require maintaining two copies of the code. The native path
therefore continues to exist, but cannot by itself provide the required model extensibility.

**Letting the Python model use a runner directly** is the shortest integration path, but it puts
two responsibilities with different rates of change into one object. Model structure, weights,
and raw forward computation should remain close to the community implementation; persistent
buffers, padding, attention plans, and graphs change with the backend, batch shape, and hardware.
Coupling them means that migrating a community model still requires understanding xLLM's complete
execution state, and the model can no longer be tested and reused independently of the runner.

**Python model + C++ model executor** preserves model/executor separation and also allows C++ to
prepare persistent inputs and graphs. The problem is that attention plans, workspaces, padding
metadata, and graph capture must remain consistent with the Python attention backend. Either
backend-private state must be exposed to C++, or C++ and Python must each maintain their own state
conversion. Adding another Python backend or hardware graph still requires changes across both
languages.

**Python model + Python `ModelExecutor`** keeps the model close to a community `nn.Module` and
allows execution state to evolve on the same side as Python runners, attention backends, and graph
implementations. C++ provides already prepared batches, tensors, and metadata, while continuing to
own the surrounding serving runtime. The cost is maintaining embedded Python and a cross-language
contract, but both costs have clear boundaries and can be validated independently.

The fourth alternative is therefore selected. The key reason is not that "Python is faster than
C++", but that it simultaneously satisfies three requirements: community model migration,
model/executor responsibility separation, and the ability for backends and graphs to evolve in
Python.

Putting both the model and `ModelExecutor` in Python does not mean that the model manages graphs
itself. They remain separate: the model owns only weights and computation, while the
`ModelExecutor` and runners own execution state. This follows the same basic responsibility split
between Model and ModelRunner/ModelExecutor used in vLLM and SGLang.

### 4.2 Community Model Migration Is the Primary Rationale

Community model implementations include more than a model class. They also include layer
composition, weight names, quantization handling, TP partitioning, and model-specific input rules.
Python model execution can reuse most of this structure and development style, concentrating the
migration effort on xLLM's layer, attention, and weight-loader contracts.

The new-model support documentation of both vLLM and SGLang uses Python model classes and
ModelRunner as extension points. This alone does not prove that xLLM must choose Python, but it
does show that community model implementations, debugging methods, and contributor experience
have formed around Python. Using a similar model and layer vocabulary in xLLM reduces unnecessary
structural rewriting during migration.

### 4.3 JIT Kernels, Quantization, and Debugging Are Secondary Rationales

Python model execution can directly invoke Triton/TileLang JIT kernels, Python-first quantization
loaders, and community kernel packages. Critical kernels can still be provided by C++, CUDA,
AscendC, or other native implementations; selecting a Python model does not mean rewriting kernels
in Python.

Python can also directly use hooks, tensor dumps, and profilers for per-layer comparison with
reference implementations. For new model migration, these tools primarily shorten the cycle for
correctness validation and performance analysis.

`torch.compile` is an additional benefit, but is less important than model migration, JIT,
quantization, and debugging. It is suitable for models or submodules that have already been
validated as stable, such as parts of a VLM encoder. Whether to use it must be determined per model
by evaluating compilation time, correctness, and performance; it is not a mandatory path for all
Python models.

The long-term value of the compile path also includes compiler extensibility. Dynamo backends and
public FX/ATen passes can introduce other graph optimizations or fused-operator replacements while
continuing to reuse the same Python model. The design depends only on these public extension
points, rather than building its long-term interface on the more rapidly changing internal IR of
Inductor.

AOTInductor can compile a Python model into an artifact loaded by C++, but it is not the primary
path of this design. It requires stricter export constraints, fixes kernel configurations at build
time, and cannot JIT or autotune again for new shapes in the deployment environment. It is better
suited to scenarios with a narrow shape range or delivery environments where a Python runtime is
not allowed.

### 4.4 Performance Depends on Graph Execution and a Strict Cross-Language Boundary

Python eager execution adds dispatch and object-construction overhead, especially for decode,
where device computation per step is short. Decode has a relatively stable execution path and
shape, making it suitable for full-graph execution, where capture/replay reduces this host
overhead.

Prefill, chunked prefill, and mixed batches are more dynamic. Their attention planning, sequence
combinations, and execution shapes are difficult to include uniformly in a full graph. The target
architecture uses piecewise graphs: stable parts of the execution path are captured as graph
pieces, while parts that depend on dynamic metadata remain outside the graph. Covering different
execution scenarios with full graphs and piecewise graphs respectively is necessary for Python
model execution to achieve production performance.

Regardless of the graph mechanism, the cross-language boundary must satisfy the following
requirements:

- only one C++-to-Python execute call is made per step;
- tensors and metadata tensors are passed zero-copy;
- stable objects such as KV cache tensors are bound once instead of rebuilding Python wrappers on
  every step;
- data already constructed by C++ is not recomputed in Python;
- attention planning remains outside the captured model and is reused by all layers in the same
  group.

The process model is also a performance condition. When each rank uses an independent process and
interpreter, C++ request-processing, scheduler, and KV cache threads do not contend for the Python
GIL. If multiple ranks are instead executed by multiple threads in one process, the GIL and
interpreter lifecycle must be reevaluated; the performance conclusions of this design cannot be
applied directly.

### 4.5 Multi-Hardware Compatibility Depends on Backend Boundaries

A Python model does not guarantee multi-hardware compatibility by itself. Compatibility comes
from the boundaries among the model, attention backend, graph runner, collectives, and device
operators:

1. the model does not import FlashInfer, CUDA Graphs, or another device-specific runtime;
2. an attention layer only invokes the current backend and does not interpret plans or state;
3. a concrete graph runner manages its own device graph, and the common interface exposes no CUDA
   types;
4. stateless operators select device implementations through PyTorch dispatch;
5. TP communication uses the collective backend corresponding to the tensor device.

vllm-ascend reuses Python models and adapts to Ascend through NPU runners, attention backends,
device operators, and `torch.npu.NPUGraph`. It is cited here only to validate that these boundaries
can support non-CUDA hardware, not to imply that xLLM should copy its runner inheritance hierarchy
or patching approach.

### 4.6 Accepted Costs

After selecting Python model execution, the following long-term costs must be accepted:

- deployment includes a versioned combination of embedded Python, wheels, PyTorch, and the device
  runtime;
- metadata views become a cross-language contract whose ownership, lifetime, and field scope must
  be explicit;
- native C++ models and Python models coexist for some period of time;
- scenarios without graph coverage still incur Python eager overhead;
- the boundaries among models, runners, and backends require continuous review so that execution
  state does not migrate back into model layers.

If major workloads cannot obtain graph coverage over the long term, cross-language overhead
cannot meet performance targets, or community model migration does not become more efficient in
practice, this architecture decision must be reevaluated.
