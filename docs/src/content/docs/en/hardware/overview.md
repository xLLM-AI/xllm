---
title: "Hardware Platforms"
description: "Platform-specific guides for running xLLM on supported hardware."
sidebar:
  order: 1
---

xLLM supports multiple accelerator backends for large-scale model inference. This section collects the hardware-specific entry points for environment setup, runtime device selection, launch scripts, and model support.

## Platform Guides

- [NVIDIA GPU](/en/hardware/nvidia_gpu/) - CUDA backend setup and launch entry points.
- [Ascend NPU](/en/hardware/ascend_npu/) - Ascend NPU setup, runtime environment, and HCCL launch notes.
- [Cambricon MLU](/en/hardware/cambricon_mlu/) - MLU backend setup and launch entry points.
- [Hygon DCU](/en/hardware/dcu/) - Hygon DCU backend setup and launch entry points.
- [MetaX MACA](/en/hardware/metax_maca/) - MetaX MACA backend setup and launch entry points.
- [Mthreads MUSA](/en/hardware/musa/) - Mthreads MUSA GPU image, build, and launch entry points (device selection via `MUSA_VISIBLE_DEVICES`).

## Common Workflow

1. Prepare the platform-specific container image from the explicit commands in each platform guide.
2. Build xLLM inside the container, or use a release image that already includes `xllm`.
3. Start the service with the matching device backend in [Launch xllm](/en/getting_started/launch_xllm/).
4. Check model and modality coverage in the [Model Support List](/en/supported_models/).
