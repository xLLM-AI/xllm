---
title: "Quick Start"
sidebar:
  order: 1
---
## Environment Setup

Choose your **Hardware**, host **CPU** architecture, and **Release** or **Dev** image to get the Docker commands for your environment.

<div data-quick-start>

Enable JavaScript to use the installation selector, or follow the hardware-specific instructions below.

</div>

- **Release** images include a pre-built `xllm` binary, ready to run without compiling.
- **Dev** images include the development environment; build xLLM from source inside the container.
- **CPU** selects the host architecture (`x86_64` or `ARM64`), not a CPU-only inference backend. Run `uname -m` on your Linux host if you are unsure.

Install Docker and the appropriate accelerator drivers on the host first. NVIDIA GPU containers also require the NVIDIA Container Toolkit. The commands above mount your home directory and grant access to accelerator devices; use them only with trusted images on a suitable development host.

The commands create named containers that remain after you exit. To re-enter a stopped container, run `docker start -ai <container-name>`, replacing `<container-name>` with the value of `--name` (for example, `xllm-npu`). To try a different image or hardware setup, choose an unused `--name` in the new `docker run` command. Restarting an existing container keeps its original image and configuration.

Browse the [Ascend and NVIDIA image tags](https://quay.io/repository/jd_xllm/xllm-ai?tab=tags), or see the full setup instructions for your hardware:

- [Ascend NPU (A2 / A3)](/en/hardware/ascend_npu/)
- [NVIDIA GPU](/en/hardware/nvidia_gpu/)
- [Cambricon MLU](/en/hardware/cambricon_mlu/)
- [Hygon DCU](/en/hardware/dcu/)
- [MetaX MACA](/en/hardware/metax_maca/)
- [Mthreads MUSA](/en/hardware/musa/)

Image availability and release versions vary by platform. Combinations without a listed image show a notice instead of a Docker command. Cambricon MLU requires a vendor-provided image.

## Build xllm

Skip this step when using a **Release** image: it includes a pre-built binary that you can invoke with `xllm`. For a **Dev** image, run the following commands **inside the container**.

Download xllm and dependencies:
```bash
git clone https://github.com/xLLM-AI/xllm.git
cd xllm

# Install pre-commit for the first time
pip install pre-commit
pre-commit install

git submodule update --init --recursive
```

In a new image, the first compilation of xllm takes a long time because all dependencies in vcpkg need to be compiled, but subsequent compilations will be much faster.
```bash
# Compile cpp binary
python setup.py build

# Compile python wheel
python setup.py bdist_wheel

# Compile python wheel and install
python setup.py install
```

## Launch xllm
Please refer to [How to Launch xllm](/en/getting_started/launch_xllm/).
