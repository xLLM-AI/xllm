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

**xLLM** 是一个高效的开源大模型推理框架，专为**国产芯片**优化设计，提供企业级的服务部署，使得性能更高、成本更低。该框架采用**服务-引擎分离的推理架构**，通过服务层的在离线请求弹性调度、动态PD分离、EPD混合机制及高可用容错设计，结合引擎层的多流并行计算、图融合优化、投机推理、动态负载均衡及全局KV缓存管理，实现推理效率突破性提升。xLLM整体架构和功能如下图所示：

<div align="center">
<img src="../assets/xllm_arch.png" alt="xllm_arch" style="width:90%; height:auto;">
</div>

**xLLM** 已支持主流大模型（如 *DeepSeek-V3.1*，*Qwen2/3*等）在国产芯片上的高效部署，助力企业实现高性能、低成本的 AI 大模型应用落地。xLLM已全面落地京东零售核心业务，涵盖智能客服、风控、供应链优化、广告推荐等多种场景。


## 核心特性
xLLM 提供了强大的智能计算能力，通过硬件系统的算力优化与算法驱动的决策控制，联合加速推理过程，实现高吞吐、低延迟的分布式推理服务。

**全图化/多层流水线执行编排**
- 框架调度层的异步解耦调度，减少计算空泡；
- 模型图层的计算和通信异步并行，重叠计算与通信；
- 算子内核层的异构计算单元深度流水，重叠计算与访存。

**动态shape的图执行优化**
- 基于参数化与多图缓存方法的动态尺寸适配，提升静态图灵活性；
- 受管控的显存池，保证地址安全可复用；
- 集成适配性能关键的自定义算子（如 *PageAttention*, *AllReduce*）。

**高效显存优化**
- 离散物理内存与连续虚拟内存的映射管理；
- 按需分配内存空间，减少内存碎片与浪费；
- 智能调度内存空间，增加内存页复用，减小分配延迟；
- 国产芯片相应算子适配。

**全局多级KV Cache管理**
- 多级缓存的kv智能卸载与预取；
- 以kv cache为中心的分布式存储架构；
- 多节点间kv的智能传输路由。

**算法优化**
- 投机推理优化，多核并行提升效率；
- MoE专家的动态负载均衡，实现专家分布的高效调整。

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

## 成为贡献者
请参阅 [CONTRIBUTING_zh.md](../../.github/CONTRIBUTING_zh.md) 获取贡献指南。

---

## 社区支持
如果你在xLLM的开发或使用过程中遇到任何问题，欢迎在项目的Issue区域提交可复现的步骤或日志片段。
如果您有企业内部Slack，请直接联系xLLM Core团队。另外，我们建立了官方微信群，可以访问以下二维码加入。欢迎沟通和联系我们:

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

---

## 许可证

[Apache License](../../LICENSE)

#### xLLM 由 JD.com 提供 
#### 感谢您对xLLM的关心与贡献!

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
