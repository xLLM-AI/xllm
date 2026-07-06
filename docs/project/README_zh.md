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
<p align="center">
| <a href="https://docs.xllm-ai.com/"><b>Documentation</b></a> |  <a href="https://arxiv.org/abs/2510.14686"><b>Technical Report</b></a> |
</p>

### 📢 新闻
- 2026-06-13: 🎉 我们 day-0 支持了[MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) 模型的推理服务，部署请参考[部署文档](https://github.com/jd-opensource/xllm/blob/preview/minimax-m3/testspace/run_minimax_m3.sh)。
- 2026-04-24: 🎉 我们 day-0 支持了[DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) 模型的推理服务，部署请参考[部署文档](https://github.com/jd-opensource/xllm/blob/preview/deepseek-v4-mlu/testspace/run_deepseek_v4.sh)。
- 2026-02-12: 🎉 我们 day-0 支持了最新的[GLM-5](https://github.com/zai-org/GLM-5) 模型的高效推理服务，部署请参考[部署文档](https://github.com/zai-org/GLM-5/blob/main/example/ascend.md)。
- 2025-12-21: 🎉 我们在第一时间内支持了[GLM-4.7](https://github.com/zai-org)模型的高效推理。
- 2025-12-08: 🎉 我们在第一时间内支持了[GLM-4.6V](https://github.com/zai-org/GLM-V)模型的高效推理。
- 2025-12-05: 🎉 我们支持了[GLM-4.5/GLM-4.6](https://github.com/zai-org/GLM-4.5/blob/main/README_zh.md)系列模型.
- 2025-12-05: 🎉 我们支持了[VLM-R1](https://github.com/om-ai-lab/VLM-R1) 模型.
- 2025-12-05: 🎉 我们基于[Mooncake](https://github.com/kvcache-ai/Mooncake)构建了混合 KV 缓存管理机制，支持具备智能卸载与预取能力的全局 KV 缓存管理。
- 2025-10-16: 🎉 我们最近在 arXiv 上发布了我们的 [xLLM 技术报告](https://arxiv.org/abs/2510.14686)，提供了全面的技术蓝图和实施见解。

## 简介

**xLLM** 是一个高效的开源大模型推理框架，专为**国产芯片**优化设计，提供企业级的服务部署，使得性能更高、成本更低。

<div align="center">
<img src="../assets/xllm_arch.png" alt="xllm_arch" style="width:90%; height:auto;">
</div>

## 为什么选择 xLLM

* **顶尖性能**：通过全图化多流水线执行编排、图融合优化、投机推理、MoE专家动态负载均衡以及全局多级KV缓存管理，实现高吞吐、低延迟的推理服务。
* **主流硬件支持**：专为国产AI加速卡深度优化设计，广泛支持 NPU、MLU、ILU、MUSA、DCU、MACA 等多种硬件。
* **服务-引擎分离架构**：采用服务-引擎分离的架构设计，服务层提供在离线请求弹性调度、动态PD分离、EPD混合执行及高可用容错能力，引擎层专注于高效计算。
* **企业级部署**：已全面落地京东零售核心业务——涵盖智能客服、风控、供应链优化、广告推荐等场景——实现高性能、低成本的生产级部署。

---
## 硬件支持

| 硬件类型 | 型号   | 备注            |
| -------- | ------ | --------------- |
| NPU      | A2, A3 | HDK Driver 25.2.0 + |
| MLU      |        |                 |
| ILU      | BI150  |                 |
| MUSA     | S5000  |                 |
| DCU      | BW1000 |                 |
| MACA     | MXC500 |                 |

此外，请在[模型支持列表](../zh/supported_models.md)查看不同硬件上的模型支持情况。

---

## 快速开始

请参考[快速开始文档](../zh/getting_started/quick_start.md)。

---

## 社区支持

<div align="center">
  <img src="../assets/wechat_qrcode.png" alt="qrcode3" width="50%" />
</div>

---

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
