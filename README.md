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

[English](./README.md) | [中文](./docs/project/README_zh.md)

<div align="center">
<img src="docs/assets/logo_with_llm.png" alt="xLLM" style="width:50%; height:auto;">
    
[![Document](https://img.shields.io/badge/Document-black?logo=html5&labelColor=grey&color=red)](https://docs.xllm-ai.com/) [![Docker](https://img.shields.io/badge/Docker-black?logo=docker&labelColor=grey&color=%231E90FF)](https://quay.io/repository/jd_xllm/xllm-ai?tab=tags) [![License](https://img.shields.io/badge/license-Apache%202.0-brightgreen?labelColor=grey)](https://opensource.org/licenses/Apache-2.0) [![report](https://img.shields.io/badge/Technical%20Report-red?logo=arxiv&logoColor=%23B31B1B&labelColor=%23F0EBEB&color=%23D42626)](https://arxiv.org/abs/2510.14686) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/jd-opensource/xllm)
    
</div>

---------------------


### 📢 News
<!-- only keep the latest 3 news, others should be folded -->
- 2026-07-06: 🎉 xLLM is officially donated to the OpenAtom Foundation!
- 2026-06-13: 🎉 We day-0 support the [MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) model, please refer to the [Deployment Document](https://github.com/jd-opensource/xllm/blob/preview/minimax-m3/testspace/run_minimax_m3.sh) for deployment.
- 2026-04-24: 🎉 We day-0 support the [DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) model, please refer to the [Deployment Document](https://github.com/jd-opensource/xllm/blob/preview/deepseek-v4-mlu/testspace/run_deepseek_v4.sh) for deployment.

<details>
<summary>More News</summary>

- 2026-02-12: 🎉 We day-0 support high-performance inference for the [GLM-5](https://github.com/zai-org/GLM-5) model, please refer to the [Deployment Document](https://github.com/zai-org/GLM-5/blob/main/example/ascend.md) for deployment.
- 2025-12-21: 🎉 We day-0 support high-performance inference for the [GLM-4.7](https://github.com/zai-org) model.
- 2025-12-08: 🎉 We day-0 support high-performance inference for the [GLM-4.6V](https://github.com/zai-org/GLM-V) model.
- 2025-12-05: 🎉 We now support high-performance inference for the [GLM-4.5/GLM-4.6](https://github.com/zai-org/GLM-4.5/blob/main/README_zh.md) series models.
- 2025-12-05: 🎉 We now support high-performance inference for the [VLM-R1](https://github.com/om-ai-lab/VLM-R1) model.
- 2025-12-05: 🎉 We build hybrid KV cache management based on [Mooncake](https://github.com/kvcache-ai/Mooncake), supporting global KV cache management with intelligent offloading and prefetching.
- 2025-10-16: 🎉 We recently have released our [xLLM Technical Report](https://arxiv.org/abs/2510.14686) on arXiv, providing comprehensive technical blueprints and implementation insights.

</details>

## Overview

**xLLM** is an **efficient LLM inference framework**, specifically optimized for **Chinese AI accelerators**, enabling enterprise-grade deployment with enhanced efficiency and reduced cost.
<div align="center">
<img src="docs/assets/xllm_arch.png" alt="xllm_arch" style="width:90%; height:auto;">
</div>

## Highlights

* **Top-tier Performance**: Delivers high-throughput, low-latency inference through many advanced features.
* **Mainstream Hardware Support**: Purpose-built and deeply optimized for Chinese AI accelerators.
* **Service-Engine Decoupled Architecture**: Service layer handles scheduling and availability; engine layer handles computation.
* **Enterprise-grade Deployment**: Battle-tested at scale across JD.com's core retail business.

## Hardware Support

| Hardware           | Abbreviation | Example | Remark              |
| ------------------ | ------------ | ------- | ------------------- |
| Ascend NPU         | NPU          | A2, A3  | HDK Driver 25.2.0 + |
| Cambricon MLU      | MLU          | MLU590  |                     |
| Moore Threads GPU  | MUSA         | S5000   |                     |
| Hygon DCU          | DCU          | BW1000  |                     |
| MetaX MACA         | MACA         | MXC500  |                     |
| Iluvatar CoreX GPU | ILU          | BI150   |                     |


## Getting Started

* [Quick Start](https://docs.xllm-ai.com/en/getting_started/quick_start/)
* [Launch xLLM](https://docs.xllm-ai.com/en/getting_started/launch_xllm/)
* [Online Service](https://docs.xllm-ai.com/en/getting_started/online_service/)
* [Offline Inference](https://docs.xllm-ai.com/en/getting_started/offline_service/)
* [Supported Models](https://docs.xllm-ai.com/en/supported_models/)

## Community & Support

<div align="center">
  <img src="docs/assets/wechat_qrcode.png" alt="qrcode3" width="50%" />
</div>

## Acknowledgment

This project was made possible thanks to the following open-source projects:  
- [ScaleLLM](https://github.com/vectorch-ai/ScaleLLM) - xLLM draws inspiration from ScaleLLM's graph construction method and references its runtime execution. 
- [Mooncake](https://github.com/kvcache-ai/Mooncake) - Build xLLM hybrid KV cache management based on Mooncake.
- [brpc](https://github.com/apache/brpc) - Build high-performance http service based on brpc.
- [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp) - Build C++ tokenizer based on tokenizers-cpp.
- [safetensors](https://github.com/huggingface/safetensors) - xLLM relies on the C binding safetensors capability.
- [Partial JSON Parser](https://github.com/promplate/partial-json-parser) - Implement xLLM's C++ JSON parser with insights from Python and Go implementations.
- [concurrentqueue](https://github.com/cameron314/concurrentqueue) - A fast multi-producer, multi-consumer lock-free concurrent queue for C++11.


Thanks to the following collaborating university laboratories:

- [THU-MIG](https://ise.thss.tsinghua.edu.cn/mig/projects.html) (School of Software, BNRist, Tsinghua University)
- USTC-Cloudlab (Cloud Computing Lab, University of Science and Technology of China)
- [Beihang-HiPO](https://github.com/buaa-hipo) (Beihang HiPO research group)
- PKU-DS-LAB (Data Structure Laboratory, Peking University)
- PKU-NetSys-LAB (NetSys Lab, Peking University)
- [TJU-TANKLab](https://flashserve.org/) (TANK Lab, Tianjin University)

Thanks to all the following [developers](https://github.com/jd-opensource/xllm/graphs/contributors) who have contributed to xLLM.

<a href="https://github.com/jd-opensource/xllm/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/xllm" />
</a>


## Citation

If you think this repository is helpful to you, welcome to cite us:
```
@article{liu2025xllm,
  title={xLLM Technical Report},
  author={Liu, Tongxuan and Peng, Tao and Yang, Peijun and Zhao, Xiaoyang and Lu, Xiusheng and Huang, Weizhe and Liu, Zirui and Chen, Xiaoyu and Liang, Zhiwei and Xiong, Jun and others},
  journal={arXiv preprint arXiv:2510.14686},
  year={2025}
}
```
