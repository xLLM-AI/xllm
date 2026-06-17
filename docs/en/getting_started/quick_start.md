# Quick Start

## Environment Setup

All images are stored [here](https://quay.io/repository/jd_xllm/xllm-ai?tab=tags). The docker startup command below uses the dev image as an example.

### NPU

Below are our pre-built dev image.
```bash
# A2 x86
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-x86-cann9-20260605
# A2 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-arm-cann9-20260605
# A3 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260605
```

Container startup command:
```bash
docker run -it \
--ipc=host \
-u 0 \
--name xllm-npu \
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
-v $HOME:$HOME \
-w $HOME \
<docker_image_name> \
/bin/bash
```

### NVIDIA GPU

We provide a [Dockerfile](../../../docker/Dockerfile.cuda) for NVIDIA GPU usage, which can be used to build custom image. Of course, you can also use dev image we built based on the default Dockerfile:
```bash
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-cuda-x86
```

Container startup command:
```bash
sudo docker run -it \
--privileged \
--shm-size '128gb' \
--ipc=host \
--net=host \
--pid=host \
--name=xllm-cuda \
-v $HOME:$HOME \
-w $HOME \
<docker_image_name> \
/bin/bash
```

### MLU

We cannot provide MLU image. If you already have the dev image, you can start the container with the following command:
```bash
sudo docker run -it \
--privileged \
--shm-size '128gb' \
--ipc=host \
--net=host \
--pid=host \
--name xllm-mlu \
-v $HOME:$HOME \
-w $HOME \
<docker_image_name> \
/bin/bash
```

### DCU

Below are our pre-built dev image.
```bash
docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:xllm-dev-dcu-x86-20260602
```

Container startup command:
```bash
docker run -it \
--ipc=host \
-u 0 \
--name xllm-dcu \
--privileged \
--network=host \
--shm-size 256g \
--device=/dev/kfd \
--device=/dev/dri \
--device=/dev/mkfd \
--security-opt seccomp=unconfined \
--group-add video \
-v /opt/hyhal:/opt/hyhal \
-v $HOME:$HOME \
-w $HOME \
<docker_image_name> \
/bin/bash
```

## Build xllm

If you download a release image, i.e., an image with a version number in the tag, you can skip this step because the release image comes with a pre-compiled xllm binary, located at `/usr/local/bin/xllm`.

Download xllm and dependencies:
```bash
git clone https://github.com/jd-opensource/xllm
cd xllm

# Install pre-commit for the first time
pip install pre-commit
pre-commit install

git submodule update --init --recursive
```

The compiled binary file is located at `/path/to/xllm/build/xllm/core/server/xllm`. In a new image, the first compilation of xllm takes a long time because all dependencies in vcpkg need to be compiled, but subsequent compilations will be much faster.
```bash
python setup.py build
```

## Launch xllm
Please refer to [How to Launch xllm](launch_xllm.md).

