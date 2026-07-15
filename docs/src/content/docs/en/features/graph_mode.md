---
title: "Graph Mode"
sidebar:
  order: 81
---
## Overview

xLLM supports Graph Mode: computation graphs are pre-captured and replayed in subsequent runs to reduce CPU overhead and improve inference performance. Graph Mode has corresponding implementations on different hardware platforms.

## Feature Description

To optimize Host-side scheduling, graph mode submits a large task from the CPU once and then executes small kernels in a streaming manner on the device, significantly reducing startup time and device bubbles.

In the xLLM engine, Graph Mode provides the following:

### Dynamic Shape Parameterization
  - Key dynamic dimensions other than num_tokens are treated as whole-graph input parameters, including batch_size, kv_seq_lens, q_seq_lens, block_table_size, and the like, for flexibility. During memory allocation and kernel configuration. At graph launch, the actual values of these parameters are passed so kernels use the correct strides to access data.

### Piecewise Graph
  - When some operators do not support graph capture and thus break the full graph, each segment (piece) after the break is captured as a separate graph. This maximizes graph-mode benefits even when the full graph cannot be captured, and is commonly used for prefill and chunked prefill.

### Multi-Shape Reusable Memory Pool
  - To avoid waste from separate memory buffers (input, output, and intermediate tensors) per shape, we use an expandable memory pool. Multiple shapes share the pool base address, with different shapes using different offsets from that base.

## Usage

These capabilities are implemented inside the xLLM engine and are generally controlled through gflags.

The minimal configuration only needs `enable_graph` to turn on Graph Mode for the decode phase:

```shell
--enable_graph=true
```

Common companion flags include:

- `enable_graph`: enables the base Graph Mode capability for the decode phase
- `enable_prefill_piecewise_graph`: enables Piecewise Graph for the prefill phase
- `enable_graph_mode_decode_no_padding`: builds decode graphs with the actual `num_tokens` instead of the padded shape
- `max_tokens_for_graph_mode`: limits the maximum number of tokens covered by Graph Mode; `0` means no limit

If you want to enable both decode Graph and prefill Piecewise Graph, use:

```shell
--enable_graph=true \
--enable_prefill_piecewise_graph=true \
--max_tokens_for_graph_mode=2048
```

If you need decode graph capture without padding, add:

```shell
--enable_graph=true \
--enable_graph_mode_decode_no_padding=true
```

For a more complete description of the flags, see [CLI Reference](/en/cli_reference/).

## Performance Impact

- With Graph Mode enabled, decode-phase throughput **improves by about 8%–10%** on models such as Qwen3-0.6B and Qwen3-1.7B.

## Model Support

The following table lists each model’s support on ACLGraph, CudaGraph, and MLUGraph.

| Model | ACLGraph | CudaGraph | MLUGraph |
|------|----------|-----------|----------|
| Qwen3/Qwen3-MoE | ✅ | ✅ | ✅ |
| DeepseekV3.2 | ✅ | | ✅ |
| GLM4.5/4.6/4.7 | ✅ | | |
| Qwen2.5-VL | | | ✅ |
| Qwen3-VL/Qwen3-VL-MoE | ✅ | | |
| GLM4V | ✅ | | |
| GLM4V-MoE | ✅ | | |

:::caution[Adding Graph Mode support for new models]
Ensure that the kernels used in the computation implement dynamic dimension parameterization; otherwise the graph may break and kernels may need to be re-implemented.

:::
## Related Documentation

- For more detailed Graph Mode design and implementation notes, including ACL Graph / CUDA Graph fundamentals, dynamic dimension parameterization, Piecewise Graph, and multi-shape memory reuse, see: [Graph Mode Design Document](/en/design/graph_mode_design/)

