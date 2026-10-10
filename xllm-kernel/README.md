<!-- Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. -->

# xLLM Kernel

Initial NPU RMSNorm integration for Python-defined models, including
GLM-5.3 (`glm_moe_dsa`). The native implementation, numerical behavior,
Torch schema, fake registration, and C++ callers retain their existing owners.

Install from a source checkout before starting Python model workers:

```bash
python -m pip install --no-deps -e ./xllm-kernel
python setup.py build
```

The main xLLM build also includes `xllm_kernel` in its wheel. The standalone
package can be imported without Torch or an NPU SDK; execution requires a
matching xLLM host, PyTorch, torch_npu, and CANN. Importing the package does
not load the native extension or initialize a device.

The existing host startup calls `xllm_kernel.initialize(device="npu")` after
native operators are loaded. It freezes a preparation-only registry and binds
`npu.xllm_native.rms_norm` before model import/capture. Repeating the same
initialization is safe; changing device or policy requires a worker restart.
A missing native operator, unsupported device, or unknown explicit
implementation fails during preparation. There is no execution-error fallback.

```python
from xllm_kernel import prepare_rms_norm
from xllm_kernel.ops import rms_norm

# Inside an initialized xLLM host:
plan = prepare_rms_norm()
output = rms_norm(input, weight, eps)
# A model may retain plan.function directly for its graph's entire lifetime.
```

`xllm.python.kernels.rms_norm` and the NPU normalization shim delegate to
this API. This covers the shared RMSNorm layers and MLA preprocessing used
by GLM-5.3. Residual/quantized/Gemma RMSNorm and other operator families
remain on their existing paths.

This first adapter uses the complete native tensor domain; it does not create
shape-specific plans or perform per-token selection. Native Torch dispatch
validates arguments. Output is independent of input/weight, and neither input
is modified. Registry diagnostics expose the selected contract, implementation,
execution modes, generation, and exclusion reasons. Runtime logs selection once.

Package tests live in `xllm-kernel/test/`: common import, registry, packaging,
numerical-report and CLI tests are at its root; NPU integration tests are in
`test/npu/`. Run NPU tests only on idle devices. A package import or a passing
fake test does not prove device or whole-model correctness.

## Optional xlite RMSNorm

The package owns `xllm_kernel::xlite_rms_norm`, its PrivateUse1 bridge, and its
fake implementation. It calls the GVirt RMSNorm host launchers on the current
PyTorch NPU stream through the torch_npu submission queue. It does not create
an XRuntime/private stream or synchronize in the operator. Existing C++ model
interfaces and `xllm_ops` schemas are unchanged.

Build against the installed GVirt revision
`4a6dc3102d5778928e4d21116bc4d4694d51e584`, matching Torch/torch_npu and CANN:

```bash
cd xllm-kernel
XLLM_KERNEL_BUILD_XLITE=1 MAX_JOBS=4 python setup.py build_ext --inplace
cd ..
python setup.py build
export XLLM_KERNEL_RMS_NORM_IMPL=npu.xlite.rms_norm
```

For a standalone compiled wheel, use
`XLLM_KERNEL_BUILD_XLITE=1 python -m pip wheel --no-build-isolation --no-deps ./xllm-kernel`.
The wheel contains the bridge and build manifest; GVirt launcher libraries,
Torch, torch_npu and CANN are external deployment dependencies. Startup checks
the recorded versions and launcher-library hashes before loading the bridge.
Ordinary package import and pure-Python installation never load those libraries.

The explicit solution accepts FP16/BF16 inputs and weights on the same NPU,
ND layout, width divisible by 64 and at most 8192, at most `2**32 - 1` input
elements, and finite nonnegative FP32 epsilon. Strided ND views are copied on
the current stream; empty rows are supported. Unsupported arguments fail
before launching the kernel. Fake tensors validate logical shape/dtype/device;
physical NPU format is checked by the real bridge. Model warmup must cover the
intended shapes before readiness. Default selection stays native; there is no
automatic per-token switch to xlite or fallback from it.

## Independent validation tools

The package follows the same ownership split as TokenSpeed:

```text
xllm-kernel/
├── python/xllm_kernel/
│   ├── numerics/       # cases, inputs, reference, validation and precision CLI
│   └── benchmark/      # timing, performance reports and benchmark CLI
└── test/
    ├── test_*.py       # common package and tool tests
    └── npu/            # NPU fixtures and integration tests
```

`benchmark` reuses `numerics` validation before timing. `numerics` does not
import the benchmark timer. Test fixtures belong in `test/`, outside the
installed Python package; there is no package-level `testing` layer.

The numerical and benchmark entry points use the same versioned RMSNorm cases,
CPU-generated inputs, FP32 CPU reference and tolerance rules. They write JSON
and Markdown with coverage, failure reasons, alias/mutation checks and error
summaries. The native solution requires an installed xLLM host extension; xlite
can run with the standalone compiled package, without starting an xLLM server.

```bash
python -m xllm_kernel.numerics --device npu --mode eager \
  --implementation npu.xlite.rms_norm --revision "$(git rev-parse HEAD)" \
  --output numerics.json
python -m xllm_kernel.benchmark --device npu --mode aclgraph \
  --implementation npu.xlite.rms_norm --revision "$(git rev-parse HEAD)" \
  --iterations 50 --output benchmark.json
XLLM_KERNEL_RMS_NORM_IMPL=npu.xlite.rms_norm \
  python -m pytest -q xllm-kernel/test/npu/test_kernel_xlite_rms_norm_npu.py
```

After installing the package, run common tests without an NPU or host extension:

```bash
python -m pytest -q xllm-kernel/test --ignore=xllm-kernel/test/npu
```

NPU integration tests also require the matching xLLM host. Their local
`conftest.py` loads its extension and initializes Python kernels before
collection. Run native and xlite tests in separate processes with the
corresponding `XLLM_KERNEL_RMS_NORM_IMPL`; selection stays fixed for a process.
The NPU host build registers `python_kernel_tests` with CTest and `all_tests`:
`python setup.py test --test-name python_kernel_tests` builds dependencies and
runs each file with its implementation explicitly selected. These tests share
the existing Python device resource lock.

Require idle hardware and monitor ownership externally throughout measurement.
The device test suite requires two visible NPUs for device-guard validation.
Benchmarks validate first and exclude preparation, warmup and capture from
steady-state samples. Eager reports host submission, device-event interval
(including submission gaps) and synchronized wall time. ACLGraph captures 64
independent operations and reports time per operation, median/p90, dispersion
and raw samples. The cache policy is repeated hot inputs, without an explicit
flush. Use complete native A1 / xlite B / native A2 runs with identical case,
mode and timer settings; compare against both baselines and report drift.
Zero applicable cases or failed validation cannot produce a passing report.

CUDA adapters, shape-specialized selection, profiling/replay, compiler
monitoring, tuning and fused/quantized operators remain later phases.
