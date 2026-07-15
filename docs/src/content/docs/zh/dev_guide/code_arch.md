---
title: "代码结构"
sidebar:
  order: 1
---
```
├── xllm/
|   : 主代码目录
│   ├── api_service/               # api服务化实现
│   ├── core/  
│   │   : xllm核心功能代码目录
│   │   ├── common/                
│   │   ├── distributed_runtime/   # 分布式PD服务实现
│   │   ├── framework/             # 引擎执行模块实现
│   │   ├── kernels/               # 国产芯片kernels适配实现
│   │   ├── layers/                # 模型层实现
│   │   ├── platform/              # 多平台兼容层
│   │   ├── runtime/               # worker/executor角色实现
│   │   ├── scheduler/             # 批调度与PD调度实现
│   │   └── util/
│   ├── function_call              # function call实现
│   ├── models/                    # 模型实现
│   ├── processors/                # 多模态模型预处理实现
│   ├── proto/                     # 通信协议
│   ├── pybind/                    # python接口
|   └── server/                    # xLLM服务实例
├── examples/                      # 服务调用示例
├── tools/                         # NPU Timeline生成工具
└── xllm.cpp                       # xLLM启动入口
```
