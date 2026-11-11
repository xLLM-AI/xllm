---
title: "Ascend NPU"
description: "Run xLLM on Ascend NPUs with the NPU backend."
sidebar:
  order: 2
---

Set up an xLLM environment for Ascend NPU and start a service with the unified `xllm serve` command from [Launch xllm](/en/getting_started/launch_xllm/).

## Prerequisites

- Install the Ascend driver and Docker on the Linux host. Confirm that `npu-smi info` can see the NPUs, and choose an image matching the A2/A3 hardware and host CPU architecture.
- Check the [Model Support List](/en/supported_models/) for your backend and prepare the model weights in a directory accessible inside the container.
- The commands below mount your home directory at the same path. For weights stored elsewhere, add a mount such as `-v /path/to/models:/models:ro` and use the container path in `MODEL_PATH`.

## Start the container

Run the following commands on the host. Choose one development image for your hardware:

| Hardware | Host CPU | Development image |
| --- | --- | --- |
| Ascend A2 | x86_64 | `quay.io/jd_xllm/xllm-ai:xllm-dev-a2-x86-cann9-20260911` |
| Ascend A2 | ARM64 | `quay.io/jd_xllm/xllm-ai:xllm-dev-a2-arm-cann9-20260911` |
| Ascend A3 | ARM64 | `quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260911` |

The example below selects A3 ARM64; change `XLLM_IMAGE` for another platform.

```bash
XLLM_IMAGE="quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260911"
docker pull "$XLLM_IMAGE"
```

```bash
docker run -it \
  --name xllm-npu \
  --ipc=host \
  -u 0 \
  --privileged \
  --network=host \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /usr/local/sbin/:/usr/local/sbin/ \
  -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
  -v /var/log/npu/slog/:/var/log/npu/slog \
  -v /var/log/npu/profiling/:/var/log/npu/profiling \
  -v /var/log/npu/dump/:/var/log/npu/dump \
  -v "$HOME:$HOME" \
  -w "$HOME" \
  "$XLLM_IMAGE" \
  /bin/bash
```

For additional devices, add the corresponding `--device=/dev/davinciN` entries before creating the container.

The container remains after exit. Re-enter it with `docker start -ai xllm-npu` when stopped, or `docker exec -it xllm-npu bash` while it is running.

## Install xLLM

Run inside the container. If your image already includes the xLLM wheel and supports `xllm serve`, skip the source installation. Otherwise, clone the repository (or enter your existing checkout), initialize its submodules, and build and install the wheel for this backend:

```bash
git clone https://github.com/xLLM-AI/xllm.git
cd xllm

# Install pre-commit on first use
pip install pre-commit
pre-commit install

git submodule update --init --recursive
python setup.py install --device npu
```

You can also install directly with `python setup.py install`; the installer automatically detects the backend from the current PyTorch environment when `--device` is omitted.

An NPU wheel includes the complete xllm_ops vendor from the build environment, including shared libraries, operator metadata, tiling libraries, and kernels. Packaging follows the build's vendor selection: `${ASCEND_OPP_PATH}/vendors/custom_xllm_math` when its `op_api/include/aclnnop` directory exists, otherwise the legacy `${ASCEND_OPP_PATH}/vendors/xllm`. Both layouts are packaged at the fixed `xllm/opp/vendors/custom_xllm_math` path used by the launcher and native libraries. Packaging requires complete installed operator resources matching the Git revision of `third_party/xllm_ops`; missing resources or a revision mismatch fail the build. Other vendors are not bundled. Prepare any additional operator packages required by your model according to its deployment instructions.

To install an existing wheel on the target machine without rebuilding, replace the path below with the actual wheel file:

```bash
python -m pip install /path/to/xllm_npu*.whl
```

The target environment still needs CANN, drivers, hardware, and Python/PyTorch versions compatible with the wheel, but does not need a separate installation of the bundled `custom_xllm_math` vendor. The `xllm` launcher automatically configures its shared-library and custom OPP search paths while preserving the base CANN OPP configuration.

`python setup.py build` only builds the artifacts; the unified command requires the installed wheel and its packaged server binary. Check the installed command before continuing:

```bash
command -v xllm
xllm serve --help
```

## Launch the service

Run inside the container after replacing `MODEL_PATH` with your weights directory. This example starts one device. Adjust the backend settings using the target model recipe and [CLI Reference](/en/cli_reference/).

```bash
#!/bin/bash
set -e

export ASCEND_RT_VISIBLE_DEVICES=0

MODEL_PATH="/path/to/model/Qwen3-8B"
PORT=8010
MASTER_NODE_ADDR="127.0.0.1:9748"
NNODES=1

xllm serve \
  --model="$MODEL_PATH" \
  --model_id=Qwen3-8B \
  --port="$PORT" \
  --nnodes="$NNODES" \
  --master_node_addr="$MASTER_NODE_ADDR" \
  --block_size=128 \
  --communication_backend=hccl \
  --max_memory_utilization=0.86 \
  --enable_prefix_cache=false \
  --enable_chunked_prefill=true \
  --enable_schedule_overlap=true \
  --enable_shm=true \
  --enable_graph=true
```

For two devices on one machine, replace the device selection and `NNODES` lines in the script with:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
NNODES=2
```

Run `xllm serve` once. The launcher assigns ranks and per-rank ports, writes `log/node_0.log`, `log/node_1.log`, and so on, and stays in the foreground. Do not pass `--node_rank` or wrap the command in a per-device loop.

## Verify the service

In another shell, wait for model loading to finish and query the rank-0 service (port `8010` in this example):

```bash
curl http://127.0.0.1:8010/health
curl http://127.0.0.1:8010/v1/models
```

The model list should include `Qwen3-8B`, as set by `--model_id`. Use that ID in requests to the [OpenAI-Compatible APIs](/en/basic_usage/openai_api/). If you change `PORT`, update the request URLs as well.

## Platform notes

- The example uses HCCL, `--block_size=128`, shared memory, chunked prefill, schedule overlap, and graph execution. Check the model recipe before changing these settings.
- For multiple independent services, use separate master addresses, service port ranges, log directories, and HCCL base ports.
- Multi-machine NPU deployments may also require `--rank_tablefile`. See [Multi-Machine Deployment](/en/basic_usage/multi_machine/) for network and rank-table preparation, and use the unified launcher described above.
- Kernel development is covered in [TileLang Kernel Development on Ascend NPU](/en/dev_guide/tilelang_ascend_kernel_dev/).
