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

<p align="center">
| <a href="https://docs.xllm-ai.com/"><b>Documentation</b></a> | <a href="https://arxiv.org/abs/2510.14686"><b>Technical Report</b></a> |
</p>


### 📢 News
- 2026-06-13: 🎉 We day-0 support the [MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) model, please refer to the [Deployment Document](https://github.com/jd-opensource/xllm/blob/preview/minimax-m3/testspace/run_minimax_m3.sh) for deployment.
- 2026-04-24: 🎉 We day-0 support the [DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) model, please refer to the [Deployment Document](https://github.com/jd-opensource/xllm/blob/preview/deepseek-v4-mlu/testspace/run_deepseek_v4.sh) for deployment.
- 2026-02-12: 🎉 We day-0 support high-performance inference for the [GLM-5](https://github.com/zai-org/GLM-5) model, please refer to the [Deployment Document](https://github.com/zai-org/GLM-5/blob/main/example/ascend.md) for deployment.
- 2025-12-21: 🎉 We day-0 support high-performance inference for the [GLM-4.7](https://github.com/zai-org) model.
- 2025-12-08: 🎉 We day-0 support high-performance inference for the [GLM-4.6V](https://github.com/zai-org/GLM-V) model.
- 2025-12-05: 🎉 We now support high-performance inference for the [GLM-4.5/GLM-4.6](https://github.com/zai-org/GLM-4.5/blob/main/README_zh.md) series models.
- 2025-12-05: 🎉 We now support high-performance inference for the [VLM-R1](https://github.com/om-ai-lab/VLM-R1) model.
- 2025-12-05: 🎉 We build hybrid KV cache management based on [Mooncake](https://github.com/kvcache-ai/Mooncake), supporting global KV cache management with intelligent offloading and prefetching.
- 2025-10-16: 🎉 We recently have released our [xLLM Technical Report](https://arxiv.org/abs/2510.14686) on arXiv, providing comprehensive technical blueprints and implementation insights.

## Project Overview

**xLLM** is an **efficient LLM inference framework**, specifically optimized for **Chinese AI accelerators**, enabling enterprise-grade deployment with enhanced efficiency and reduced cost. The framework adopts a **service-engine decoupled** inference architecture, achieving breakthrough efficiency through several  technologies: at the service layer, including elastic scheduling of online/offline requests, dynamic PD disaggregation, a hybrid EPD mechanism for multimodal and high-availability fault tolerance; and at the engine layer, combined with technologies such as multi-stream parallel computing, graph fusion optimization, speculative inference, dynamic load balancing and global KV cache management. The overall architecture is shown below:

<div align="center">
<img src="docs/assets/xllm_arch.png" alt="xllm_arch" style="width:90%; height:auto;">
</div>

**xLLM** already supports efficient deployment of mainstream large models (such as *DeepSeek-V3.1*, *Qwen2/3*, etc.) on Chinese AI accelerators, empowering enterprises to implement high-performance, low-cost AI large model applications. xLLM has been fully deployed in JD.com’s real core retail businesses, covering a variety of scenarios including intelligent customer service, risk control, supply chain optimization, ad recommendation, and more.


## Core Features

**xLLM** delivers robust intelligent computing capabilities. By leveraging hardware system optimization and algorithm-driven decision control, it jointly accelerates the inference process, enabling high-throughput, low-latency distributed inference services.

**Full Graph Pipeline Execution Orchestration**
- Asynchronous decoupled scheduling at the requests scheduling layer, to reduce computational bubbles.
- Asynchronous parallelism of computation and communication at the model graph layer, overlapping computation and communication.
- Pipelining of heterogeneous computing units at the operator kernel layer, overlapping computation and memory access.

**Graph Optimization for Dynamic Shapes**
- Dynamic shape adaptation based on parameterization and multi-graph caching methods to enhance the flexibility of static graph.
- Controlled tensor memory pool to ensure address security and reusability.
- Integration and adaptation of performance-critical custom operators (e.g., *PageAttention*, *AllReduce*).

**Efficient Memory Optimization**
- Mapping management between discrete physical memory and continuous virtual memory.
- On-demand memory allocation to reduce memory fragmentation.
- Intelligent scheduling of memory pages to increase memory reusability.
- Adaptation of corresponding operators for domestic accelerators.

**Global KV Cache Management**
- Intelligent offloading and prefetching of KV in hierarchical caches.
- KV cache-centric distributed storage architecture.
- Intelligent KV routing among computing nodes.

**Algorithm-driven Acceleration**
- Speculative decoding optimization to improve efficiency through multi-core parallelism.
- Dynamic load balancing of MoE experts to achieve efficient adjustment of expert distribution.

---
## Hardware Support

| Hardware | Example | Remark          |
| -------- | ------- | --------------- |
| NPU      | A2, A3  | HDK Driver 25.2.0 + |
| MLU      |         |                 |
| ILU      | BI150   |                 |
| MUSA     | S5000   |                 |
| DCU      | BW1000  |                 |
| MACA     | MXC500  |                 |

Besides, please check the supported models on different hardwares at [Supported Models List](docs/en/supported_models.md).

---

## Quick Start

Please refer to [Quick Start](docs/en/getting_started/quick_start.md) for more details.

--- 

## Contributing
Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for contribution guidelines.

---

## Community & Support
If you encounter any issues along the way, you are welcomed to submit reproducible steps and log snippets in the project's Issues area, or contact the xLLM Core team directly via your internal Slack. In addition, we have established official WeChat groups. You can access the following QR code to join. Welcome to contact us!

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

---

## License
[Apache License](LICENSE)

#### xLLM is provided by JD.com 
#### Thanks for your Contributions!

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
