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

#include <memory>
#include <span>
#include <vector>

#include "core/framework/model/model_input_params.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/platform/stream.h"

namespace xllm {

struct ForwardInput;
struct MtpContextView;

struct ModelInputCapacity {
  uint32_t max_tokens = 0;
  uint32_t max_sequences = 0;
  uint32_t max_blocks_per_sequence = 0;
};

// Borrowed CPU model data used by ordinary Slots and speculative invocations.
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

enum class MtpInvocationKind : uint8_t {
  PREFILL,
  DRAFT,
  VALIDATE,
  BLOCK_DRAFT
};

// A model invocation or completed-row extraction for block-draft prefill.
struct MtpInputSpec {
  ModelInputCapacity model;
  uint32_t hidden_size = 0;
  uint32_t block_size = 0;
  uint32_t num_speculative_tokens = 0;
  MtpInvocationKind kind = MtpInvocationKind::PREFILL;
  uint32_t draft_step = 0;
  bool enable_mla = true;
  int32_t mask_token_id = 0;
  bool context_only = false;
};

struct SlotBufferCapacity {
  ModelInputCapacity model;
  uint32_t max_unique_tokens = 0;
  uint32_t vocab_size = 0;
  uint32_t max_top_logprobs = 0;
  torch::ScalarType parameter_dtype = torch::kFloat32;
  bool enable_mla = false;
  // Zero row capacities use model.max_sequences. Zero result width disables
  // result storage for sampling-only invocations.
  uint32_t max_selected_rows = 0;
  uint32_t max_sample_rows = 0;
  uint32_t max_result_width = 1;
};

struct TokenResultTensors {
  torch::Tensor tokens;        // [S] or [S,W], int64
  torch::Tensor lengths;       // [S], int32, speculative accepted lengths
  torch::Tensor logprobs;      // Same token shape, float32, optional
  torch::Tensor top_tokens;    // Token shape plus [K], int64, optional
  torch::Tensor top_logprobs;  // Token shape plus [K], float32, optional
};

// Owns one Slot's fixed model/attention inputs, sampling parameters, patch
// indices and result buffers. Sampling/result storage is optional for model
// invocations within a speculative round. Reuse only after retirement.
// Prepare runs on the state thread; patch_previous_tokens runs on the ordered
// task stream after the pipeline's input_ready event. The shared previous-token
// buffer stays outside this object, so retiring its producer Slot cannot
// invalidate the next reader.
class SlotBuffer final {
 public:
  static Status create(const SlotBufferCapacity& capacity,
                       const torch::Device& device,
                       std::unique_ptr<SlotBuffer>& output);
  static Status create(const ModelInputCapacity& capacity,
                       bool enable_mla,
                       const torch::Device& device,
                       std::unique_ptr<SlotBuffer>& output);
  Status validate(const ModelInputHostView& input,
                  const ModelInputBatch& batch) const;
  Status prepare(const ModelInputHostView& input,
                 const ModelInputBatch& batch,
                 const Stream& stream);
  Status prepare_empty_shard(bool decode,
                             uint64_t batch_id,
                             const Stream& stream);
  Status prepare_decode_padded(const ModelInputHostView& input,
                               const ModelInputBatch& batch,
                               uint32_t padded_batch_size,
                               const Stream& stream);

  ~SlotBuffer();
  SlotBuffer(const SlotBuffer&) = delete;
  SlotBuffer& operator=(const SlotBuffer&) = delete;

  // The owner supplies unpacked CPU input with checked transport types/layouts.
  // Validate before staging writes; the owner may then add model/KV checks.
  Status validate(const ForwardInput& input,
                  uint32_t previous_rows,
                  const Stream& stream) const;
  // An empty shard with an inherited nonempty forward type gets one reserved
  // padding row for peer collectives, without an actual sequence or sample.
  // All caller storage is released after Prepare; only Slot-owned copies
  // remain.
  // Nonzero padded_batch_size pads ordinary decode to a common captured bucket.
  // Zero keeps the eager input shape.
  // Sampling and previous-token mappings retain their logical row counts.
  void prepare(const ForwardInput& input,
               const Stream& stream,
               uint32_t padded_batch_size = 0);
  bool has_previous_tokens() const { return gather_count_ != 0; }
  void patch_previous_tokens(const torch::Tensor& previous_tokens);

  const torch::Tensor& tokens() const { return tokens_; }
  const torch::Tensor& positions() const { return positions_; }
  ModelInputParams& model_params() { return model_params_; }
  const ModelInputParams& model_params() const { return model_params_; }
  // Configure a model-only buffer before admitting any tasks.
  Status configure_sampling(const SlotBufferCapacity& capacity);
  Status validate_sampling(const SamplingParameters& input,
                           uint32_t model_tokens) const;
  Status prepare_sampling(const SamplingParameters& input,
                          uint32_t model_tokens,
                          const Stream& stream);
  Status bind_result(uint32_t sequences,
                     uint32_t tokens_per_sequence,
                     uint32_t top_logprobs,
                     bool logprobs);
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
  friend struct TaskPipelineTestPeer;
  friend class TaskExecutionPipeline;
  static Status create_mtp_input(const MtpInputSpec& spec,
                                 torch::ScalarType hidden_dtype,
                                 const torch::Device& device,
                                 std::unique_ptr<SlotBuffer>& output);
  static Status plan_mtp_decode(SlotBuffer& input,
                                const ModelInputHostView& base,
                                const ModelInputBatch& batch);
  static Status prepare_mtp_decode(SlotBuffer& input,
                                   const ModelInputHostView& base,
                                   const ModelInputBatch& batch,
                                   const Stream& stream);
  static Status prepare_planned_mtp_decode(SlotBuffer& input,
                                           const ModelInputHostView& base,
                                           const ModelInputBatch& batch,
                                           const Stream& stream,
                                           uint32_t physical_rows = 0);
  static void patch_mtp_decode(SlotBuffer& input,
                               const MtpContextView& binding);

  static Status plan_mtp_prefill(SlotBuffer& input,
                                 const ModelInputHostView& base,
                                 const ModelInputBatch& batch,
                                 std::span<const int32_t> extra_token_ids,
                                 const SamplingParameters& sampling);
  static Status prepare_mtp_prefill(SlotBuffer& input,
                                    const ModelInputHostView& base,
                                    const ModelInputBatch& batch,
                                    std::span<const int32_t> extra_token_ids,
                                    const SamplingParameters& sampling,
                                    const Stream& stream);
  static Status prepare_planned_mtp_prefill(SlotBuffer& input,
                                            const ModelInputHostView& base,
                                            const ModelInputBatch& batch,
                                            const Stream& stream);
  static void patch_mtp_prefill(SlotBuffer& input,
                                const torch::Tensor& target_hidden,
                                const torch::Tensor& sampled_tokens,
                                MtpContextView& binding);
  static void initialize_mtp_context(SlotBuffer& input,
                                     const torch::Tensor& target_hidden,
                                     const torch::Tensor& sampled_tokens,
                                     MtpContextView& binding);

  struct InputScratch {
    MtpInputSpec spec_;
    uint32_t rows_per_sequence_ = 0;
    int32_t first_offset_ = 0;
    uint32_t sequences_ = 0;
    // CPU scratch is reserved at creation and never borrowed after Prepare.
    std::vector<int32_t> tokens_;
    std::vector<int32_t> positions_;
    std::vector<int32_t> slots_;
    std::vector<int32_t> q_lengths_;
    std::vector<int32_t> kv_lengths_;
    std::vector<int32_t> query_ends_;
    std::vector<int32_t> block_tables_;
    ModelInputHostView planned_;

    torch::Tensor offsets_;
    torch::Tensor cache_positions_storage_;
    torch::Tensor block_indices_storage_;
    torch::Tensor block_ids_storage_;
    torch::Tensor cache_offsets_storage_;
    torch::Tensor repair_scratch_storage_;
    torch::Tensor positions_view_;
    torch::Tensor kv_lengths_view_;
    torch::Tensor tokens_view_;
    torch::Tensor cache_positions_;
    torch::Tensor cache_first_column_;
    torch::Tensor model_first_column_;
    torch::Tensor block_indices_;
    torch::Tensor block_ids_;
    torch::Tensor cache_offsets_;
    torch::Tensor cache_offsets_flat_;
    torch::Tensor future_positions_;
    std::vector<int32_t> token_to_sequence_;
    std::vector<uint8_t> published_mask_;
    std::vector<uint32_t> planned_rows_;
    std::vector<uint32_t> published_rows_;
    std::vector<int64_t> sampled_token_rows_;
    torch::Tensor host_indices_;
    torch::Tensor device_indices_;
    torch::Tensor indices_;
    torch::Tensor host_lengths_;
    torch::Tensor device_lengths_;
    torch::Tensor bootstrap_positions_;
    torch::Tensor bootstrap_kv_lengths_;
    torch::Tensor bootstrap_hidden_storage_;
    torch::Tensor bootstrap_hidden_;
    torch::Tensor token_storage_;
    torch::Tensor sampled_token_values_;
    uint64_t pinned_bytes_ = 0;
    uint64_t device_bytes_ = 0;
  };

  std::unique_ptr<InputScratch> input_scratch_;
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
  static ModelInputHostView model_input_view(const ForwardInput& input);
  static Status validate_batch(const ModelInputHostView& model,
                               const BatchInputMeta& batch);
  Status validate_model(const ModelInputHostView& model) const;
  Status validate_previous_tokens(const ModelInputHostView& model,
                                  uint32_t previous_rows) const;
  void prepare_model(const ModelInputHostView& model,
                     const BatchInputMeta& batch,
                     uint32_t padded_batch_size = 0);
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
  uint32_t result_sequences_ = 0;
  uint32_t result_width_ = 0;
  uint32_t result_top_logprobs_ = 0;
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
