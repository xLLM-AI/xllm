/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Thread-local context propagating per-forward LoRA routing info from the
// model forward loop down into individual Linear wrappers. The Linear
// wrappers themselves have a fixed forward(input) signature that can't be
// extended without breaking base-class ABI, so we thread the info through
// TLS instead.
//
// Contract:
//   * LlmModelImplBase::forward pushes a Frame at the top of each forward,
//     with adapter_ids (per-seq) and q_seq_lens_vec (per-seq).
//   * Inside the layer for-loop, it updates Frame::layer_index each
//     iteration.
//   * LoRA*ParallelLinearImpl::forward reads the current Frame and, if
//     any adapter_id is non-zero, pulls the corresponding per-layer
//     per-proj (A,B,scaling) from LoRARuntime and applies the delta.
//
// Frames stack (a batch can call into a submodule that itself launches
// more forwards, though in practice we're always one deep). LoRAScope
// RAII pushes/pops.

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <vector>

namespace xllm {

struct LoRAContextFrame {
  // Per-sequence adapter routing. Empty when the batch is pure-base or
  // when LoRA is disabled globally.
  const std::vector<uint64_t>* adapter_ids = nullptr;

  // Per-sequence lengths, index-aligned with adapter_ids. Used to slice
  // the [num_tokens, hidden] hidden state into per-seq chunks.
  const std::vector<int32_t>* q_seq_lens_vec = nullptr;

  // Phase A W2 v2: per-token adapter id, device-side int64 [total_tokens].
  // Prepared by ModelInputParams::to(device) at batch-build time by
  // expanding adapter_ids according to q_seq_lens_vec. Pointer is valid
  // for the duration of the forward pass. nullptr or undefined() when the
  // batch is pure-base (all adapter_ids == 0) or LoRA is disabled -
  // consumers treat that as the fast path and skip delta entirely.
  //
  // Consumers (fused_moe.cpp, LoRA wrappers) may safely combine this with
  // per_layer combine_idx / cu_seq_lens to derive per-expanded-row adapter
  // labels via index_select - all device-side, no CPU->NPU copy in the
  // forward loop (CANN 8.5 + torch_npu 2.7.1 forbid that; see 07-06 notes).
  const torch::Tensor* adapter_ids_per_token = nullptr;

  // Which decoder layer the forward is currently in (0-indexed). Updated
  // by the model's layer loop each iteration. LoRA wrapper uses this to
  // pull the right per-layer weight from LoRARuntime.
  int32_t layer_index = -1;

  // Which architectural family this model is. For Qwen3 (and Qwen2/
  // oxygen) the PEFT canonical form is "layers.{L}.self_attn.q_proj",
  // "layers.{L}.mlp.gate_proj", etc. Other archs may vary; the wrapper
  // uses this + a proj-name hint to compose the LoRARuntime lookup key.
  std::string arch = "qwen3";
};

// Push a new frame. Returns a token used to pop it.
void push_lora_context(LoRAContextFrame frame);

// Pop the top frame. Must match the last push (LIFO).
void pop_lora_context();

// Read the top frame. Returns nullptr if the stack is empty.
const LoRAContextFrame* current_lora_context();

// RAII helper for push/pop.
class LoRAScope {
 public:
  explicit LoRAScope(LoRAContextFrame frame) {
    push_lora_context(std::move(frame));
  }
  ~LoRAScope() { pop_lora_context(); }
  LoRAScope(const LoRAScope&) = delete;
  LoRAScope& operator=(const LoRAScope&) = delete;
};

// Update the layer_index on the top frame in-place. Cheaper than a full
// push/pop and matches how the layer loop iterates. No-op if the stack
// is empty.
void set_lora_context_layer(int32_t layer_index);

}  // namespace xllm
