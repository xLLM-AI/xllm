include(cc_library)
include(cc_test)

# Multi-tenant LoRA support.
#
#   M1 lora_config      - CLI flags + LoRAConfig struct
#   M3 lora_request     - LoRARequest immutable descriptor
#   M3 lora_registry    - name<->int_id map + refcount drain semantics
#   M2 adapter_loader   - PEFT (adapter_config.json + safetensors) reader
#   MVP lora_runtime    - process singleton wiring loader + registry +
#                         "currently active" whole-block A/B tensors so
#                         forward paths can pick them up in one lookup.
#                         Replaced by proper LoraCacheManager (M4) once
#                         per-adapter and per-layer routing lands.
cc_library(
  NAME
    lora
  HDRS
    lora_config.h
    lora_request.h
    lora_registry.h
    adapter_loader.h
    lora_runtime.h
    lora_context.h
    lora_metrics.h
  SRCS
    lora_config.cpp
    lora_registry.cpp
    adapter_loader.cpp
    lora_runtime.cpp
    lora_context.cpp
    lora_metrics.cpp
  DEPS
    :state_dict
    glog::glog
    gflags::gflags
    torch
)
