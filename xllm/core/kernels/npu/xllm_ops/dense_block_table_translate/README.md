<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. See LICENSE. -->

# Dense block table translation

This component implements task 4's standalone Python `prepare` / `run` interface
on Ascend NPU. It translates logical block addresses into layer-specific physical
page metadata. It does not move K/V data.

`X` is contiguous INT32 `[R,C]`; `M` is contiguous INT32 `[B]` or `[L,B]` on the
same NPU. Every negative entry in `X` produces `-1`. Other entries select the
corresponding element from `M`. Outputs have shape `[R,C]` or `[L,R,C]` and own
their storage. Empty rows/columns and nonzero storage offsets are supported.

```python
from solution import prepare, run

prepared = prepare([x, mapping], {})
output = prepared.run()
output_with_validation = run([x, mapping], {})
```

The prepared object retains the input tensors and compiled executable code.
Each invocation reads current input values and uses the calling stream. Shapes,
dtypes, devices, and valid value ranges must remain unchanged after preparation.
See [DESIGN.md](DESIGN.md) for the algorithms, memory layout, and lifecycle.

## Environment and build

Use Linux, Ascend 910C / Atlas A3, CANN 9.1.0, and compatible PyTorch, torch_npu,
and triton-ascend packages. The verified environment uses Python 3.11.4,
PyTorch 2.7.1+cpu, torch_npu 2.7.1.post4, triton-ascend 3.2.1, and NumPy 1.26.4.
A C++17 compiler and Ninja are required for the host extension.

From this directory:

```bash
source /home/developer/Ascend/ascend-toolkit/set_env.sh
export PATH="$HOME/.local/bin:$PATH"
export ASCEND_RT_VISIBLE_DEVICES=0
export MAX_JOBS=2
export TORCH_EXTENSIONS_DIR="$(dirname "$PWD")/dense-block-table-extensions"
python3 -m pip install --user ninja 'numpy>=1.22'
python3 native_loader.py
```

Adjust the CANN installation path and visible device to the allocated environment.
The C++ extension uses the installed PyTorch ABI. Triton compilation and initial
warmup occur during preparation and are excluded from steady-state measurements.
This is a standalone component with its own host extension; it does not add a
new ACLNN installation package or change the existing `xllm_ops` CMake target.

## Validation

```bash
python3 -m unittest discover -s tests -v
python3 collect_environment.py --output ../dense-environment.json
python3 verify.py --output-dir ../dense-results > ../dense-results.log 2>&1
```

Output paths must be new and outside this source directory. The driver invokes
the unmodified official functional verifier, then checks the full convenience
entry point, independent repeated outputs, CPU-oracle agreement on both streams,
and paired performance. Both main performance inputs must pass all four criteria.
Failures, interruptions, and missing required checks return a nonzero exit code.

All four timed inputs use 50 warmups and six alternating paired rounds of 200
complete calls, including final device synchronization. The two main inputs must
meet 75/60 microsecond mean latency limits and show positive mean, median, and
bootstrap lower-bound savings over the supplied baseline. Bootstrap uses 10000
shared resamples, seed 20261031, and linear percentiles.

`official/` contains the byte-preserved task fixtures and reference code. The
local Ruff configuration excludes only those immutable upstream fixtures.
New source files follow the repository formatter and lint configuration.

## Submission materials

`task_submission/` contains the seven reports/logs required by the task and the
raw `evidence/` records supporting their results. `task_submission.zip` contains
the same material set. The implementation, verifier, and reports describe the
standalone task interface and its measured validation scope.
