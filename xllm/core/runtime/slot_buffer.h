/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <span>

#include "core/framework/model/model_input_params.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/platform/stream.h"

namespace xllm {

struct ModelInputCapacity {
  uint32_t max_tokens = 0;
  uint32_t max_sequences = 0;
  uint32_t max_blocks_per_sequence = 0;
};

// Borrows CPU data only until Prepare Ack. Positions have one axis and query
// cumulative lengths contain row ends without a leading zero.
struct ModelInputHostView {
  std::span<const int32_t> token_ids;
  std::span<const int32_t> positions;
  std::span<const int32_t> new_cache_slots;
  std::span<const int32_t> q_seq_lens;
  std::span<const int32_t> kv_seq_lens;
  std::span<const int32_t> q_cu_seq_lens;
  std::span<const int32_t> block_tables;
  uint32_t block_table_width = 0;
};

struct ModelInputBatch {
  BatchForwardType forward_type;
  uint32_t num_actual_sequences = 0;
  uint64_t batch_id = 0;
  bool is_graph_warmup = false;
};

struct SlotBufferCapacity {
  ModelInputCapacity model;
  uint32_t max_unique_tokens = 0;
  uint32_t vocab_size = 0;
  uint32_t max_top_logprobs = 0;
  torch::ScalarType parameter_dtype = torch::kFloat32;
  bool enable_mla = false;
};

struct TokenResultTensors {
  torch::Tensor tokens;        // [S], int64
  torch::Tensor logprobs;      // [S], float32, optional
  torch::Tensor top_tokens;    // [S,K], int64, optional
  torch::Tensor top_logprobs;  // [S,K], float32, optional
};

// Owns one Slot's model/attention inputs, sampling parameters, previous-token
// patch indices and single-token result buffers. Reuse only after retirement.
// Prepare runs on the state thread; patch_previous_tokens runs on the ordered
// task stream after the pipeline's input_ready event. The shared previous-token
// buffer stays outside this object, so retiring its producer Slot cannot
// invalidate the next reader.
class SlotBuffer final {
 public:
  static Status create(const SlotBufferCapacity& capacity,
                       const torch::Device& device,
                       std::unique_ptr<SlotBuffer>& output);
  ~SlotBuffer();
  SlotBuffer(const SlotBuffer&) = delete;
  SlotBuffer& operator=(const SlotBuffer&) = delete;

  // Validate the complete input before any staging write. The owner may add
  // model/KV checks, then prepare exactly this input on the validated stream.
  Status validate(const ModelInputHostView& model,
                  const ModelInputBatch& batch,
                  const SamplingParameters& sampling,
                  uint32_t previous_rows,
                  const Stream& stream) const;
  void prepare(const ModelInputHostView& model,
               const ModelInputBatch& batch,
               const SamplingParameters& sampling,
               const Stream& stream);
  bool has_previous_tokens() const { return gather_count_ != 0; }
  void patch_previous_tokens(const torch::Tensor& previous_tokens);

  const torch::Tensor& tokens() const { return tokens_; }
  const torch::Tensor& positions() const { return positions_; }
  ModelInputParams& model_params() { return model_params_; }
  const ModelInputParams& model_params() const { return model_params_; }
  const SamplingParameters& sampling_params() const { return sampling_params_; }
  torch::Tensor copy_cpu_do_sample() const;
  // Prepared from the sampling parameters together with the input views.
  const TokenResultTensors& device_result() const { return result_device_; }
  // The producer event follows all device writes. An empty result still
  // records a fence, so Consume retires producer work before Slot reuse.
  Status copy_result_to_host(const Stream& stream,
                             const StreamEventPtr& producer_ready);
  // Waits for D2H and returns independent, unpinned CPU values.
  TokenResultTensors take_result();
  void discard_result();
  uint64_t pinned_bytes() const;
  uint64_t device_bytes() const;

 private:
  struct Region {
    uint64_t offset = 0;
    uint64_t bytes = 0;
  };
  struct Layout {
    ModelInputCapacity capacity;
    Region token_ids;
    Region positions;
    Region new_cache_slots;
    Region q_seq_lens;
    Region kv_seq_lens;
    Region q_cu_seq_lens;
    Region block_tables;
    uint64_t block_table_row_stride_bytes = 0;
    uint64_t total_bytes = 0;
  };
  struct ModelTensors {
    torch::Tensor token_ids;
    torch::Tensor positions;
    torch::Tensor new_cache_slots;
    torch::Tensor q_seq_lens;
    torch::Tensor kv_seq_lens;
    torch::Tensor q_cu_seq_lens;
    torch::Tensor block_tables;
  };

  SlotBuffer(SlotBufferCapacity capacity,
             torch::Device device,
             Layout layout,
             uint64_t auxiliary_bytes);
  static bool append_region(uint64_t elements,
                            Region& region,
                            uint64_t& total_bytes);
  static Status make_layout(const ModelInputCapacity& capacity, Layout& layout);
  static torch::Tensor field_view(const torch::Tensor& buffer,
                                  const Region& region);
  static ModelTensors bind_views(const torch::Tensor& buffer,
                                 const Layout& layout);
  Status validate_model(const ModelInputHostView& model) const;
  Status validate_sampling(const SamplingParameters& sampling,
                           uint32_t tokens) const;
  Status validate_previous_tokens(const ModelInputHostView& model,
                                  uint32_t previous_rows) const;
  void prepare_model(const ModelInputHostView& model,
                     const ModelInputBatch& batch);
  void prepare_sampling(const SamplingParameters& sampling);
  void prepare_result();
  void prepare_previous_tokens(const ModelInputHostView& model);

  SlotBufferCapacity capacity_;
  torch::Device device_;
  Layout layout_;
  torch::Tensor host_buffer_;
  torch::Tensor device_buffer_;
  ModelTensors model_host_;
  ModelTensors model_device_;
  ModelInputParams model_params_;
  torch::Tensor tokens_;
  torch::Tensor positions_;
  // Sampling and result storage have equal Host/device capacities.
  uint64_t auxiliary_bytes_ = 0;
  SamplingParameters sampling_host_;
  SamplingParameters sampling_device_;
  SamplingParameters sampling_params_;
  torch::Tensor cpu_do_sample_;
  torch::Tensor host_indices_;
  torch::Tensor device_indices_;
  torch::Tensor gathered_int64_;
  torch::Tensor gathered_int32_;
  torch::Tensor gather_rows_;
  torch::Tensor gather_offsets_;
  torch::Tensor gather_output_int64_;
  torch::Tensor gather_output_int32_;
  uint32_t gather_count_ = 0;
  TokenResultTensors result_host_storage_;
  TokenResultTensors result_device_storage_;
  TokenResultTensors result_host_;
  TokenResultTensors result_device_;
  std::unique_ptr<StreamEvent> result_ready_;
  bool copy_submitted_ = false;
};

}  // namespace xllm
