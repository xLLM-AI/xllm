<!-- Copyright 2022 JD Co.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this project except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

[English](../../README.md) | [中文](./README_zh.md)

<div align="center">
<img src="../assets/logo_with_llm.png" alt="xLLM" style="width:50%; height:auto;">
    
[![Document](https://img.shields.io/badge/Document-black?logo=html5&labelColor=grey&color=red)](https://docs.xllm-ai.com/) [![Docker](https://img.shields.io/badge/Docker-black?logo=docker&labelColor=grey&color=%231E90FF)](https://quay.io/repository/jd_xllm/xllm-ai?tab=tags) [![License](https://img.shields.io/badge/license-Apache%202.0-brightgreen?labelColor=grey)](https://opensource.org/licenses/Apache-2.0) [![report](https://img.shields.io/badge/Technical%20Report-red?logo=arxiv&logoColor=%23B31B1B&labelColor=%23F0EBEB&color=%23D42626)](https://arxiv.org/abs/2510.14686) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/jd-opensource/xllm)
    
</div>

---------------------


### 📢 新闻
<!-- only keep the latest 3 news, others should be folded -->
- 2026-07-06: 🎉 xLLM 正式捐赠给开放原子开源基金会！
- 2026-06-13: 🎉 我们 day-0 支持了[MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) 模型的推理服务，部署请参考[部署文档](https://github.com/jd-opensource/xllm/blob/preview/minimax-m3/testspace/run_minimax_m3.sh)。
- 2026-04-24: 🎉 我们 day-0 支持了[DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) 模型的推理服务，部署请参考[部署文档](https://github.com/jd-opensource/xllm/blob/preview/deepseek-v4-mlu/testspace/run_deepseek_v4.sh)。


<details>
<summary>更多新闻</summary>

- 2026-02-12: 🎉 我们 day-0 支持了最新的[GLM-5](https://github.com/zai-org/GLM-5) 模型的高效推理服务，部署请参考[部署文档](https://github.com/zai-org/GLM-5/blob/main/example/ascend.md)。
- 2025-12-21: 🎉 我们在第一时间内支持了[GLM-4.7](https://github.com/zai-org)模型的高效推理。
- 2025-12-08: 🎉 我们在第一时间内支持了[GLM-4.6V](https://github.com/zai-org/GLM-V)模型的高效推理。
- 2025-12-05: 🎉 我们支持了[GLM-4.5/GLM-4.6](https://github.com/zai-org/GLM-4.5/blob/main/README_zh.md)系列模型.
- 2025-12-05: 🎉 我们支持了[VLM-R1](https://github.com/om-ai-lab/VLM-R1) 模型.
- 2025-12-05: 🎉 我们基于[Mooncake](https://github.com/kvcache-ai/Mooncake)构建了混合 KV 缓存管理机制，支持具备智能卸载与预取能力的全局 KV 缓存管理。
- 2025-10-16: 🎉 我们最近在 arXiv 上发布了我们的 [xLLM 技术报告](https://arxiv.org/abs/2510.14686)，提供了全面的技术蓝图和实施见解。

</details>

## 简介

**xLLM** 是一个高效的开源大模型推理框架，专为**国产芯片**优化设计，提供企业级的服务部署，使得性能更高、成本更低。

<div align="center">
<img src="../assets/xllm_arch.png" alt="xllm_arch" style="width:90%; height:auto;">
</div>

## 亮点

* **顶尖性能**：通过众多先进特性，提供高吞吐、低延迟的推理服务。
* **主流硬件支持**：专为国产AI加速卡打造并深度优化。
* **服务-引擎分离架构**：服务层负责调度与可用性，引擎层专注于计算。
* **企业级部署**：已在京东零售核心业务中大规模落地验证。

## 硬件支持

| 硬件类型           | 简称 | 型号   | 备注                |
| ------------------ | ---- | ------ | ------------------- |
| Ascend NPU         | NPU  | A2, A3 | HDK Driver 25.2.0 + |
| Cambricon MLU      | MLU  | MLU590 |                     |
| Moore Threads GPU  | MUSA | S5000  |                     |
| Hygon DCU          | DCU  | BW1000 |                     |
| MetaX MACA         | MACA | MXC500 |                     |
| Iluvatar CoreX GPU | ILU  | BI150  |                     |

## 入门指南

* [快速开始](https://docs.xllm-ai.com/zh/getting_started/quick_start/)
* [启动xLLM](https://docs.xllm-ai.com/zh/getting_started/launch_xllm/)
* [在线服务](https://docs.xllm-ai.com/zh/getting_started/online_service/)
* [离线推理](https://docs.xllm-ai.com/zh/getting_started/offline_service/)
* [模型支持](https://docs.xllm-ai.com/zh/supported_models/)


## 社区支持

<div align="center">
  <img src="../assets/wechat_qrcode.png" alt="qrcode3" width="50%" />
</div>


## 致谢
本项目的实现得益于以下开源项目: 

- [ScaleLLM](https://github.com/vectorch-ai/ScaleLLM) - 采用了ScaleLLM中构图方式和借鉴Runtime执行。
- [Mooncake](https://github.com/kvcache-ai/Mooncake) - 依赖构建了多级KV Cache管理机制。
- [brpc](https://github.com/apache/brpc) - 依赖brpc构建了高性能http service。
- [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp) - 依赖tokenizers-cpp构建了c++ tokenizer。
- [safetensors](https://github.com/huggingface/safetensors) - 依赖其c binding safetensors能力。
- [Partial JSON Parser](https://github.com/promplate/partial-json-parser) - xLLM的C++版本JSON解析器，参考Python与Go实现的设计思路。
- [concurrentqueue](https://github.com/cameron314/concurrentqueue) - 高性能无锁Queue.

感谢以下合作的高校实验室：

- [THU-MIG](https://ise.thss.tsinghua.edu.cn/mig/projects.html)（清华大学软件学院、北京信息科学与技术国家研究中心）
- USTC-Cloudlab（中国科学技术大学云计算实验室）
- [Beihang-HiPO](https://github.com/buaa-hipo)（北京航空航天大学HiPO研究组）
- PKU-DS-LAB（北京大学数据结构实验室）
- PKU-NetSys-LAB（北京大学网络系统实验室）
- [TJU-TANKLab](https://flashserve.org/) (天津大学TANK实验室)

感谢以下为xLLM作出贡献的[开发者](https://github.com/jd-opensource/xllm/graphs/contributors)

<a href="https://github.com/jd-opensource/xLLM/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/xllm" />
</a>


## 引用

如果你觉得这个仓库对你有帮助，欢迎引用我们：
```
@article{liu2025xllm,
  title={xLLM Technical Report},
  author={Liu, Tongxuan and Peng, Tao and Yang, Peijun and Zhao, Xiaoyang and Lu, Xiusheng and Huang, Weizhe and Liu, Zirui and Chen, Xiaoyu and Liang, Zhiwei and Xiong, Jun and others},
  journal={arXiv preprint arXiv:2510.14686},
  year={2025}
}
```
