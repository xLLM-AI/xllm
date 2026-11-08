---
title: "Virtual Memory for KV Cache and Model Weights"
sidebar:
  order: 52
---

xLLM can use virtual memory management (VMM) independently for the KV Cache and model weights. Both options default to `false` and are supported on NPU.

| Flag | Purpose |
|:-----|:--------|
| `enable_vmm_kvcache` | Use VMM for the KV Cache. When disabled, xLLM uses the physical KV Cache implementation. |
| `enable_vmm_weights` | Use VMM for model weights and implicitly enable the manual loader. |

Enable either option on its own, or enable both to share the VMM pool between the KV Cache and model weights. KV-only VMM keeps the configured weight loader and does not imply `enable_manual_loader`. Weight-only VMM keeps the physical KV Cache implementation. Weight VMM currently supports the LLM backend. Model sleep, wakeup and fork require `enable_vmm_weights=true`. Weight-only VMM requires `vmm_elastic=true`; a fixed weight pool would consume memory needed by the physical KV Cache. See the [CLI Reference](/en/cli_reference/) for all flags and defaults.

## Prefix caching with virtual KV Cache

Set `enable_prefix_cache=true` together with `enable_vmm_kvcache=true` to reuse complete KV blocks across requests. Cached blocks retain their virtual addresses and physical pages after a request completes. When a new page allocation reaches the live memory budget, the manager evicts unused prefix blocks in LRU order and retries allocation. Blocks still used by a request or KV transfer remain pinned. An empty page is released after its last block is evicted or returned.

Prefix matching uses the same token and multimodal hashes as the physical KV Cache. Chunked prefill publishes only complete blocks; hybrid attention also requires a recoverable linear-state checkpoint. This option does not enable KV Cache event publishing for VMM.

## Host KV offload with virtual KV Cache

Enable prefix caching and set `host_blocks_factor` greater than `1` to retain completed KV blocks in host memory. After a device prefix is evicted under memory pressure, a matching request can restore it from the host cache into newly mapped VMM blocks. Device and host blocks remain retained until their transfers finish, including when a request is cancelled. Only completed blocks are offloaded; active requests are not swapped out.

Append these flags to an existing NPU launch command:

```bash
--enable_vmm_kvcache=true \
--enable_prefix_cache=true \
--host_blocks_factor=2
```

Host capacity is preallocated from the configured device KV tensor capacity multiplied by `host_blocks_factor`, including in elastic mode. Size `max_cache_size` and the factor for available host RAM. This LLM path supports flat KV and MLA+INDEX tensor layouts with `kv_cache_dtype=auto`; VLM, grouped caches, and hybrid linear-attention conv/SSM caches remain unsupported for VMM HostKV offload.

## MTP with virtual KV Cache

Co-located MTP target and draft models can use `enable_vmm_kvcache=true`. The scheduler maps and releases both caches together, including models whose target and draft KV tensors have different widths. For hybrid attention models, only full-attention KV tensors use VMM; convolution and recurrent states remain resident and are reserved separately from the physical KV page pool.

Append these flags to an existing MTP launch command:

```bash
--draft_model=/models/mtp \
--speculative_algorithm=MTP \
--num_speculative_tokens=3 \
--enable_vmm_kvcache=true \
--enable_vmm_weights=false
```

Hybrid linear-state caches and external draft models do not support weight VMM. Embedded Eagle3 drafts and speculative decoding with PD disaggregation are unsupported with virtual KV Cache. Quantized KV Cache remains unsupported with KV VMM.

## Elastic KV Cache

With `enable_vmm_kvcache=true` and `vmm_elastic=true`, physical backing grows as KV pages are mapped. Idle backing is released after the last block on a page is returned. The VMM pool reserves a stable virtual address arena and activates logical slots as demand grows.

Elastic VMM ignores `max_memory_utilization` during startup sizing, virtual arena reservation, and physical growth. An existing setting such as `max_memory_utilization=0.75` has no effect in this mode. Physical KV Cache and fixed VMM continue to use the ratio. `max_cache_size` remains an optional byte quota for the shared KV/weight pool. Actual growth and request admission use current free memory and preserve headroom for observed execution peaks plus `vmm_memory_reserve_bytes`. Resident graph and operator caches already reduce free memory; execution memory observations continue to update at runtime. On backends with native allocator statistics, memory released outside the pool can provide additional KV capacity without restarting; backends without reliable statistics retain conservative execution watermarks. Use the explicit reserve for direct SDK buffers and execution shapes that have not yet been observed.

Admission converts request token demand into KV blocks using the model's actual cache layout and the minimum available capacity across its workers. MTP embedding slots and hybrid recurrent-state slots must also be available. Running concurrency therefore adapts to request lengths and live memory within the configured scheduler limits. Fresh chunked-prefill requests are checked against their full prompt footprint; decode obtains additional blocks as tokens are generated. This does not reserve every request's maximum possible output in advance.

Append these flags to an existing launch command to leave the byte quota unset and keep an additional 1 GiB free for execution:

```bash
--enable_vmm_kvcache=true \
--vmm_elastic=true \
--max_cache_size=0 \
--vmm_memory_reserve_bytes=1073741824
```

Tune the additional reserve using device measurements. A value of `vmm_memory_reserve_bytes=0` still preserves runtime execution headroom. Elastic VMM follows the configured graph or eager warmup behavior without additional VMM-specific warmup. PD KV transfer and remote weight transfer using Mooncake require a fully resident pool; use `vmm_elastic=false` and a positive `max_memory_utilization`. See [Elastic virtual KV](/en/dev_guide/code_arch/#elastic-virtual-kv) for architecture details.
