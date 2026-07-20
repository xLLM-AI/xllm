---
title: "硬件平台"
description: "xLLM 支持硬件平台的环境、启动和模型支持入口。"
sidebar:
  order: 1
---

xLLM 支持多种加速器后端，用于大模型推理部署。本章节汇总不同硬件平台的环境准备、运行时设备选择、服务启动和模型支持入口。

## 平台指南

- [NVIDIA GPU](/zh/hardware/nvidia_gpu/) - CUDA 后端环境和启动入口。
- [昇腾 NPU](/zh/hardware/ascend_npu/) - 昇腾 NPU 环境、运行时变量和 HCCL 启动注意事项。
- [寒武纪 MLU](/zh/hardware/cambricon_mlu/) - MLU 后端环境和启动入口。
- [海光 DCU](/zh/hardware/dcu/) - 海光 DCU 后端环境和启动入口。
- [沐曦 MACA](/zh/hardware/metax_maca/) - 沐曦 MACA 后端环境和启动入口。
- [摩尔线程 MUSA](/zh/hardware/musa/) - 摩尔线程 MUSA GPU 镜像启动入口。

## 通用流程

1. 根据各平台指南中的显式命令准备对应平台的容器镜像。
2. 在容器内编译 xLLM，或直接使用已经包含 `xllm` 的 release 镜像。
3. 按 [启动 xllm](/zh/getting_started/launch_xllm/) 中对应平台的设备后端启动服务。
4. 在 [模型支持列表](/zh/supported_models/) 中确认模型和模态覆盖情况。
