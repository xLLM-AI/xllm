# 快速开始

## 环境设置

所有的镜像都存放在[这里](https://quay.io/repository/jd_xllm/xllm-ai?tab=tags)，下面的docker启动命令以开发镜像为例。

### NPU

下面是我们构建好的开发镜像。
```bash
# A2 x86
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-x86-cann9-20260605
# A2 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-arm-cann9-20260605
# A3 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260605
```

容器启动命令如下：
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

我们提供了NVIDIA GPU使用的[Dockerfile](../../../docker/Dockerfile.cuda)，可以构建自定义镜像，当然也可以使用我们根据默认Dockerfile构建的开发镜像：
```bash
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-cuda-x86
```

容器启动命令如下：
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

我们无法提供MLU镜像，如果您已经拥有了相应的开发镜像，那么可以根据下面的命令启动容器：
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

下面是我们构建好的开发镜像。
```bash
docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:xllm-dev-dcu-x86-20260602
```

容器启动命令如下：
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

## 编译xllm

如果下载的是release镜像，即tag中带有版本号的镜像，可以跳过此步，因为release镜像自带编译好的xllm二进制文件，路径为`/usr/local/bin/xllm`。

下载xllm及依赖
```bash
git clone https://github.com/jd-opensource/xllm
cd xllm

# 第一次需要进行pre-commit安装
pip install pre-commit
pre-commit install

git submodule update --init --recursive
```

编译生成的二进制文件位于`/path/to/xllm/build/xllm/core/server/xllm`，在新镜像中，第一次编译xllm耗时较长，因为需要编译vcpkg中的所有依赖，但是后续编译会很快。
```bash
python setup.py build
```

## 启动xllm
请参考 [xllm启动方式](launch_xllm.md)。