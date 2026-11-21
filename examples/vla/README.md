<!-- Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0 -->

# GR00T N1.7 offline inference on DCU

This example runs a GR00T N1.7 checkpoint using xLLM's native Qwen3-VL feature
extractor and a PyTorch/Diffusers action head. It supports one DCU, batch one,
and DROID observations. It returns a 40-step action chunk containing end-effector
pose, gripper position, and joint position.

Build and install xLLM with the DCU backend using your platform's normal build
environment. This change adds a DCU-only native binding to the existing build.
Install the optional Python dependencies in the same environment:

```bash
python -m pip install -r examples/vla/requirements.txt
```

Download GR00T-N1.7-3B and the matching Cosmos-Reason2-2B backbone configuration.
Supply a compatible local Qwen3-VL processor directory containing tokenizer and
image-processor files. All model paths below are placeholders.

```bash
python examples/vla/gr00t_droid.py \
  --model /models/GR00T-N1.7-3B \
  --backbone-config /models/Cosmos-Reason2-2B/config.json \
  --processor /models/Qwen3-VL-2B-Instruct \
  --observation /data/droid-observation.npz \
  --instruction "Put the blue block in the green bowl" \
  --device cuda:0 --dtype bfloat16 --attention-backend flash --action-execution eager \
  --seed 42 --output /data/actions.npz
```

HIP PyTorch uses the `cuda:0` device name for DCU. The input NPZ contains
`video.<camera>` arrays (RGB uint8, `[1,T,H,W,3]`) and `state.<group>` arrays
(float32, `[1,1,D]`). Camera names and history length come from the checkpoint's
`processor_config.json`; the tested DROID configuration uses two cameras and
two historical frames. State groups are `eef_9d` (9), `gripper_position` (1),
and `joint_position` (7). The caller supplies the requested history.

The output archive contains `eef_9d` `[1,40,9]`, `gripper_position`
`[1,40,1]`, and `joint_position` `[1,40,7]`, all float32 in physical
action space. The model generates 40 steps per call; the robot client chooses
how many to execute before observing again.

The default uses native BF16 Linear and Flash attention in the text backbone,
vision backbone and action head. Install the FlashAttention runtime required by
your DCU Torch build. Some vendor builds require `FA_SO_PATH` to identify its
FlashAttention shared library; use the path from your runtime installation.
Flash is required when selected: unsupported operations propagate their errors.

Use `--action-execution graph` to capture and replay the action head. The backbone
remains eager. A bounded cache holds two specializations, keyed by sequence
length and visible text/image key counts. Every request updates features, robot
state, embodiment, mask indices and noise. Calls on a policy are serialized.
Graph warmup and capture add first-call cost; report them separately from steady
inference. `--attention-backend math --action-execution eager` selects a reference
path. The validated Flash/Graph policy requires BF16, one DCU and batch one.
Backbone backend environment variables apply per process; use a separate process
when comparing attention backends.

## Python API

```python
from xllm.vla.gr00t import Gr00tDroidPolicy

policy = Gr00tDroidPolicy(
    "/models/GR00T-N1.7-3B",
    "/models/Cosmos-Reason2-2B/config.json",
    "/models/Qwen3-VL-2B-Instruct",
    attention_backend="flash",
    action_execution="graph",
)
actions = policy.get_action(observation, seed=42)
```

`Gr00tModel.predict_processed` also accepts prepared checkpoint tensors and an
explicit `initial_noise` tensor. Comparing implementations should reuse actual
noise tensors, because equal integer seeds need not generate equal noise across
different devices or random-number schedules.

## Tests

After building the checkout's native extension, run:

```bash
python -m pytest -q tests/python/test_gr00t_action.py \
  tests/python/test_gr00t_processing.py tests/python/test_gr00t_backbone.py \
  tests/python/test_gr00t_execution.py
```

The small action-head tests cover seeded inference, immutable explicit noise,
configuration validation and embodiment bounds. If a separately installed
NVIDIA Isaac-GR00T inference package is available, they additionally compare
the independent NVIDIA action head with identical weights and noise.

The native backbone test compares nonzero image inputs and pre-final-norm token
features with Transformers, in FP32 and BF16, on DCU. Checkpoint-dependent
processor tests run when `GR00T_CHECKPOINT` and `GR00T_PROCESSOR_PATH` are set.
Flash tests compare masked action outputs with math and verify fresh state, masks,
embodiment and noise across graph replays. Tests requiring unavailable devices or external checkpoints report skips.

This is an offline policy API. Integration with the main xLLM serving scheduler,
continuous batching, other raw-observation embodiments, and closed-loop robot
success evaluation remain separate work.
