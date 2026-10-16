---
title: "快速开始"
sidebar:
  order: 1
---
## 环境设置

选择**硬件平台**、主机 **CPU 架构**以及 **Release** 或 **Dev** 镜像，获取适用于当前环境的 Docker 命令。

<div data-quick-start>

请启用 JavaScript 以使用安装选择器，或按照下方对应硬件的配置指南操作。

</div>

- **Release** 镜像包含编译好的 `xllm` 二进制文件，无需编译即可运行。
- **Dev** 镜像提供开发环境，需要在容器内从源码编译 xLLM。
- **CPU 架构**指主机架构（`x86_64` 或 `ARM64`），并非纯 CPU 推理后端。不确定时，可在 Linux 主机上运行 `uname -m` 查看。

请先在主机上安装 Docker 和对应的加速卡驱动。NVIDIA GPU 容器还需要 NVIDIA Container Toolkit。上述命令会挂载主目录并授予容器访问加速卡设备的权限，请仅在合适的开发主机上使用可信镜像。

上述命令创建的命名容器在退出后仍会保留。重新进入已停止的容器时，运行 `docker start -ai <container-name>`，将 `<container-name>` 替换为 `--name` 的值（例如 `xllm-npu`）。若要尝试其他镜像或硬件配置，请为新的 `docker run` 命令选择一个未使用的 `--name`。重启已有容器会保留其原始镜像和配置。

可浏览 [昇腾和 NVIDIA 镜像标签](https://quay.io/repository/jd_xllm/xllm-ai?tab=tags)，或查看对应硬件的完整配置指南：

- [昇腾 NPU（A2 / A3）](/zh/hardware/ascend_npu/)
- [NVIDIA GPU](/zh/hardware/nvidia_gpu/)
- [寒武纪 MLU](/zh/hardware/cambricon_mlu/)
- [海光 DCU](/zh/hardware/dcu/)
- [沐曦 MACA](/zh/hardware/metax_maca/)
- [摩尔线程 MUSA](/zh/hardware/musa/)

不同平台的镜像可用性和发布版本可能不同。未列出镜像的组合会显示说明，而非 Docker 命令。寒武纪 MLU 需要使用厂商提供的镜像。

## 编译xllm

使用 **Release** 镜像时可跳过此步：镜像内已包含编译好的二进制文件，可以直接调用 `xllm`。使用 **Dev** 镜像时，请在**容器内**执行以下命令。

下载 xllm 及依赖：
```bash
git clone https://github.com/xLLM-AI/xllm.git
cd xllm

# 第一次需要进行 pre-commit 安装
pip install pre-commit
pre-commit install

git submodule update --init --recursive
```

在新镜像中，第一次编译 xllm 耗时较长，因为需要编译 vcpkg 中的所有依赖，但是后续编译会很快。
```bash
# 只编译 cpp 二进制文件
python setup.py build

# 编译 python wheel
python setup.py bdist_wheel

# 编译 python wheel 并安装
python setup.py install
```

## 启动xllm
请参考 [xllm 启动方式](/zh/getting_started/launch_xllm/)。
