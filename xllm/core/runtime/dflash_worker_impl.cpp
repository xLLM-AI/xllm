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

#include "runtime/dflash_worker_impl.h"

#include <glog/logging.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "common/metrics.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/speculative_config.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/process_group.h"
#include "framework/sampling/rejection_sampler.h"
#if defined(USE_MLU)
#include "framework/kv_cache_transfer/mooncake_kv_cache_transfer.h"
#endif
#if defined(USE_NPU)
#include "framework/kv_cache_transfer/kv_transfer_completion.h"
#include "framework/kv_cache_transfer/spec_kv_cache_transfer.h"
#endif
#include "runtime/spec_input_builder.h"
#include "util/json_reader.h"
#include "util/timer.h"

namespace xllm {
namespace {

// Under schedule-overlap the scheduler feeds placeholder (-1) tokens, so
// per-rank sampling RNG can diverge across the tensor-parallel group.
// Broadcasting the sampled draft/accepted tokens to the group's rank 0 keeps
// every rank's cached draft probs and accepted prefixes identical. No-op for a
// single rank (world_size <= 1).
ProcessGroup* spec_broadcast_group(const ParallelArgs& parallel_args) {
  return parallel_args.tp_group_ != nullptr ? parallel_args.tp_group_
                                            : parallel_args.process_group_;
}

void broadcast_spec_tokens(torch::Tensor& tokens,
                           ProcessGroup* pg,
                           int32_t root_rank = 0) {
  if (pg == nullptr || pg->world_size() <= 1 || !tokens.defined()) {
    return;
  }
  tokens = tokens.contiguous();
  pg->broadcast(tokens, root_rank);
}

runtime::Options target_options(const runtime::Options& options) {
  runtime::Options opts = options;
  opts.enable_schedule_overlap(false)
      .is_draft_engine(false)
      .enable_graph_aux_hidden_states(true);
  return opts;
}

runtime::Options draft_options(const runtime::Options& options) {
  runtime::Options opts = options;
  opts.enable_schedule_overlap(false)
      .is_draft_engine(true)
      .num_decoding_tokens(1)
      .num_speculative_tokens(0)
      .enable_graph_aux_hidden_states(false);
  return opts;
}

// Pack a host int32 vector into a pinned CPU tensor and stage an async H2D
// copy onto the caller's active stream. Consolidates the three-line idiom
// `TensorOptions(int, device) + specBuilder::make_cpu_int_tensor(vec) +
// safe_to(...)`.
torch::Tensor cpu_int_vec_to_device(const std::vector<int32_t>& values,
                                    const Device& device) {
  return safe_to(
      specBuilder::make_cpu_int_tensor(values),
      torch::TensorOptions().dtype(torch::kInt).device(device.unwrap()),
      /*non_blocking=*/true);
}

void repeat_sampling_tensor(torch::Tensor& tensor, int32_t repeats) {
  if (tensor.defined()) {
    tensor = tensor.repeat_interleave(/*repeats=*/repeats, /*dim=*/0);
  }
}

void repeat_sampling_params(SamplingParameters& sampling_params,
                            int32_t repeats) {
  repeat_sampling_tensor(sampling_params.frequency_penalties, repeats);
  repeat_sampling_tensor(sampling_params.presence_penalties, repeats);
  repeat_sampling_tensor(sampling_params.repetition_penalties, repeats);
  repeat_sampling_tensor(sampling_params.temperatures, repeats);
  repeat_sampling_tensor(sampling_params.top_p, repeats);
  repeat_sampling_tensor(sampling_params.top_k, repeats);
  repeat_sampling_tensor(sampling_params.unique_token_ids, repeats);
  repeat_sampling_tensor(sampling_params.unique_token_counts, repeats);
  repeat_sampling_tensor(sampling_params.unique_token_ids_lens, repeats);
  repeat_sampling_tensor(sampling_params.do_sample, repeats);
}

void clear_selected_embeddings(ForwardOutput& output) {
  output.sample_output.selected_embeddings = torch::Tensor();
}

void clear_all_output_embeddings(ForwardOutput& output) {
  output.sample_output.embeddings = torch::Tensor();
  clear_selected_embeddings(output);
}

void record_metadata_ready_event(Stream& stream, ForwardInput& input) {
  StreamEventPtr event = stream.record_event();
  if (event == nullptr) {
    stream.synchronize();
  }
  input.metadata_ready_event = event;
}

void wait_metadata_ready_event(const ForwardInput& input, Stream& stream) {
  CHECK(stream.wait_event(input.metadata_ready_event))
      << "failed to wait DFlash metadata ready event";
}

void scale_dp_global_token_nums(ModelInputParams& input_params,
                                int32_t multiplier) {
  for (int32_t& token_num : input_params.parallel.dp_global_token_nums) {
    token_num *= multiplier;
  }
}

std::optional<ForwardOutput> run_llm_no_sync_impl(
    LLMWorkerImpl& worker,
    const ForwardInput& input,
    Stream& prepare_stream,
    Stream& compute_stream,
    ForwardInput* processed_output = nullptr) {
  ForwardInput processed_input;
  worker.prepare_work_before_execute_on_stream(
      input, processed_input, prepare_stream);
  std::optional<ForwardOutput> output =
      worker.execute_no_sync_on_stream(processed_input, compute_stream);
  if (processed_output != nullptr) {
    *processed_output = std::move(processed_input);
  }
  return output;
}

void build_query_rows(const ForwardInput& input,
                      int32_t mask_token_id,
                      int32_t num_speculative_tokens,
                      int32_t block_size,
                      specBuilder::DecodeBuildBuffers& buf,
                      std::vector<int32_t>& selected_idxes,
                      std::vector<int32_t>& q_cu_seq_lens) {
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  const int32_t query_width = num_speculative_tokens + 1;
  specBuilder::DecodeRowContext row_ctx =
      specBuilder::make_decode_row_context(input);
  Slice<int32_t> token_ids = {
      input.token_ids_host.data_ptr<int32_t>(),
      static_cast<size_t>(input.token_ids_host.numel())};
  CHECK_GE(static_cast<int32_t>(token_ids.size()), num_sequences)
      << "DFlash input token_ids size is smaller than num_sequences.";

  buf.out_token_ids.reserve(num_sequences * query_width);
  buf.out_positions.reserve(num_sequences * query_width);
  buf.out_new_cache_slots.reserve(num_sequences * query_width);
  buf.out_kv_seq_lens.reserve(num_sequences);
  buf.out_q_seq_lens.reserve(num_sequences);

  selected_idxes.reserve(num_sequences * num_speculative_tokens);
  q_cu_seq_lens.reserve(num_sequences + 1);
  q_cu_seq_lens.emplace_back(0);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    for (int32_t query_idx = 0; query_idx < query_width; ++query_idx) {
      specBuilder::RowSpec row;
      row.seq_id = seq_id;
      row.token_id = query_idx == 0 ? token_ids[seq_id] : mask_token_id;
      row.position_offset = query_idx;
      row.append_kv_len = false;
      row.append_q_len_one = false;
      row.append_block_table = false;
      specBuilder::append_decode_row(row_ctx, row, block_size, buf);
      if (query_idx > 0) {
        selected_idxes.emplace_back(seq_id * query_width + query_idx);
      }
    }

    specBuilder::append_seq_len_by_layout(buf.out_q_seq_lens, query_width);
    q_cu_seq_lens.emplace_back(q_cu_seq_lens.back() + query_width);
    const int32_t kv_len =
        specBuilder::calc_kv_len(input.input_params.attention.host.kv_seq_lens,
                                 seq_id,
                                 /*offset=*/0) +
        num_speculative_tokens;
    specBuilder::update_kv_seq_lens_and_max(
        buf.out_kv_seq_lens, kv_len, buf.meta.kv_max_seq_len);
  }
}

std::vector<int64_t> build_accepted_context_rows(
    const ForwardInput& input,
    const torch::Tensor& accepted_tokens_cpu,
    int32_t block_size,
    specBuilder::DecodeBuildBuffers& buf) {
  const int32_t batch_size = static_cast<int32_t>(accepted_tokens_cpu.size(0));
  const int32_t token_width = static_cast<int32_t>(accepted_tokens_cpu.size(1));
  CHECK_EQ(input.input_params.meta.num_sequences, batch_size)
      << "DFlash accepted token batch mismatch.";

  specBuilder::DecodeRowContext row_ctx =
      specBuilder::make_decode_row_context(input);
  std::vector<int64_t> accepted_idxes;
  accepted_idxes.reserve(static_cast<size_t>(accepted_tokens_cpu.numel()));
  buf.out_positions.reserve(buf.out_positions.size() +
                            static_cast<size_t>(accepted_tokens_cpu.numel()));
  buf.out_new_cache_slots.reserve(
      buf.out_new_cache_slots.size() +
      static_cast<size_t>(accepted_tokens_cpu.numel()));

  const int64_t* accepted_tokens_data =
      accepted_tokens_cpu.const_data_ptr<int64_t>();
  for (int32_t seq_id = 0; seq_id < batch_size; ++seq_id) {
    const int64_t row_offset = static_cast<int64_t>(seq_id) * token_width;
    for (int32_t token_idx = 0; token_idx < token_width; ++token_idx) {
      if (accepted_tokens_data[row_offset + token_idx] < 0) {
        break;
      }

      specBuilder::RowSpec row;
      row.seq_id = seq_id;
      row.position_offset = token_idx;
      row.append_token = false;
      row.append_kv_len = false;
      specBuilder::append_decode_row(row_ctx, row, block_size, buf);
      accepted_idxes.emplace_back(row_offset + token_idx);
    }
  }

  CHECK(!accepted_idxes.empty())
      << "DFlash accepted context must not be empty.";
  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_positions.size())
      << "DFlash accepted context slots/positions mismatch.";
  return accepted_idxes;
}

}  // namespace

DFlashWorkerImpl::DFlashWorkerImpl(const ParallelArgs& parallel_args,
                                   const torch::Device& device,
                                   const runtime::Options& options)
    : SpeculativeWorkerImpl(parallel_args,
                            device,
                            options,
                            target_options(options)) {
  // DFlash feeds the target's captured intermediate-layer aux hidden states
  // into the draft's context K/V. Under context parallelism the worker only
  // exposes the lm_head-gathered final hidden (see llm_worker_impl.cpp), not
  // the aux hidden, so the draft would silently receive the wrong tensor.
  // Reject cp_size > 1 until aux-hidden plumbing under CP is implemented.
  CHECK_LE(parallel_args.cp_size(), 1)
      << "DFlash speculative decoding does not support context parallelism "
         "(cp_size > 1).";
  draft_impl_ = std::make_unique<LLMWorkerImpl>(
      parallel_args, device, draft_options(options));
}

bool DFlashWorkerImpl::init_model(const std::string& model_weights_path,
                                  int32_t random_seed,
                                  MasterStatus master_status) {
  // DFlash draft attends each block non-causally, which the shared QWen3 model
  // only wires up on the chunked-prefill mask path. Without it the draft falls
  // back to a causal mask and proposal quality silently degrades, so require
  // the flag rather than accept a misconfigured run.
  CHECK(::xllm::SchedulerConfig::get_instance().enable_chunked_prefill())
      << "DFlash requires --enable_chunked_prefill=true.";
  bool result = true;
  const bool loading_target =
      impl_->get_status() == WorkerImpl::Status::UNINITIALIZED;
  if (loading_target) {
    result = SpeculativeWorkerImpl::init_model(
        model_weights_path, random_seed, master_status);
  } else {
    CHECK_EQ(draft_impl_->get_status(), WorkerImpl::Status::UNINITIALIZED);
    // Draft config's use_sliding_window / sliding_window is intentionally
    // ignored: xLLM NPU FIA hard-codes pre_tokens=INT_MAX, next_tokens=0
    // and never plumbs sliding_window into the aclnn call. Attended kv_len
    // on the draft path is bounded by the target sequence length, not by
    // block_size, so enforcing a block_size-vs-window relationship here
    // would compare the wrong quantities.
    result = draft_impl_->WorkerImpl::init_model(
        model_weights_path, random_seed, master_status);
  }

  if (impl_->get_status() == WorkerImpl::Status::LOADED) {
    context_ = impl_->context_;
  }

  if (draft_impl_->get_status() == WorkerImpl::Status::LOADED) {
    // Draft shares the target's lm_head and word embedding to save memory and a
    // redundant matmul.
    auto share_torch_head_and_embedding = [this]() {
      auto head = impl_->get_lm_head();
      draft_impl_->set_lm_head(head);
      auto word_embedding = impl_->get_word_embedding();
      draft_impl_->set_word_embedding(word_embedding);
    };
#if defined(USE_NPU)
    // The DFlash draft body is registered ATB-only, so a TORCH-backend run
    // aborts in create_llm_model before reaching here; the draft always uses
    // the target's NPU (ATB) head and embedding.
    auto head = impl_->get_npu_lm_head();
    draft_impl_->set_npu_lm_head(head);
    auto word_embedding = impl_->get_npu_word_embedding();
    draft_impl_->set_npu_word_embedding(word_embedding);
#else
    share_torch_head_and_embedding();
#endif

    JsonReader reader;
    const std::string config_path = model_weights_path + "/config.json";
    CHECK(reader.parse(config_path))
        << "Failed to parse DFlash draft config: " << config_path;
    std::optional<int32_t> mask_token_id =
        reader.value<int32_t>("dflash_config.mask_token_id");
    CHECK(mask_token_id.has_value())
        << "DFlash draft config requires dflash_config.mask_token_id.";
    mask_token_id_ = mask_token_id.value();

    const ModelArgs& draft_args = draft_impl_->context_.get_model_args();
    const int64_t draft_vocab_size = draft_args.vocab_size();
    CHECK_GT(draft_vocab_size, 0) << "DFlash draft vocab_size must be set.";
    CHECK_GE(mask_token_id_, 0) << "DFlash mask_token_id (" << mask_token_id_
                                << ") must be a valid embedding index (>= 0).";
    CHECK_LT(mask_token_id_, draft_vocab_size)
        << "DFlash mask_token_id (" << mask_token_id_
        << ") must be < draft vocab_size (" << draft_vocab_size << ").";
    const int64_t num_target_layers =
        static_cast<int64_t>(draft_args.layers_to_capture().size());
    CHECK_GT(num_target_layers, 0)
        << "DFlash requires dflash_config.target_layer_ids.";
    expected_context_hidden_size_ =
        static_cast<int64_t>(draft_args.hidden_size()) * num_target_layers;
  }
  return result;
}

std::tuple<int64_t, int64_t> DFlashWorkerImpl::estimate_kv_cache_capacity() {
  const std::tuple<int64_t, int64_t> target_memory =
      impl_->estimate_kv_cache_capacity();
  const std::tuple<int64_t, int64_t> draft_memory =
      draft_impl_->estimate_kv_cache_capacity();
  const int64_t cache_size_in_bytes =
      std::min(std::get<0>(target_memory), std::get<0>(draft_memory));
  const int64_t total_memory =
      std::min(std::get<1>(target_memory), std::get<1>(draft_memory));
  return {cache_size_in_bytes, total_memory};
}

bool DFlashWorkerImpl::allocate_kv_cache(const KVCacheShape& kv_cache_shape) {
  const int64_t num_blocks = kv_cache_shape.key_cache_shape()[0];
  embedding_cache_ = std::make_shared<EmbeddingCache>(num_blocks);

  bool target_allocated = true;
  const WorkerImpl::Status target_status = impl_->get_status();
  if (target_status == WorkerImpl::Status::LOADED) {
    target_allocated = impl_->allocate_kv_cache(kv_cache_shape);
  } else {
    CHECK_EQ(target_status, WorkerImpl::Status::READY);
  }

  bool draft_allocated = true;
  const WorkerImpl::Status draft_status = draft_impl_->get_status();
  if (draft_status == WorkerImpl::Status::LOADED) {
    draft_allocated = draft_impl_->allocate_kv_cache(kv_cache_shape);
  } else {
    CHECK_EQ(draft_status, WorkerImpl::Status::READY);
  }

  return target_allocated && draft_allocated;
}

#if defined(USE_NPU) || defined(USE_MLU)
bool DFlashWorkerImpl::allocate_kv_cache_with_transfer(
    const KVCacheShape& kv_cache_shape) {
  const int64_t num_blocks = kv_cache_shape.key_cache_shape()[0];

  if (kv_cache_transfer_ == nullptr) {
#if defined(USE_NPU)
    kv_cache_transfer_ = std::make_shared<SpecKVCacheTransfer>(
        options_.transfer_listen_port(),
        options_.instance_role(),
        context_.get_model_args().index_n_heads() > 0);
#elif defined(USE_MLU)
    CHECK_EQ(::xllm::DisaggPDConfig::get_instance().kv_cache_transfer_type(),
             "Mooncake")
        << "MLU DFlash push only supports Mooncake KV transfer.";
    kv_cache_transfer_ = std::make_shared<MooncakeKVCacheTransferDefault>(
        device_.index(),
        options_.transfer_listen_port(),
        device_,
        context_.get_model_args().model_type());
#endif

    const int32_t device_id = device_.index();
    kv_cache_transfer_->initialize(device_id);
  }

  bool target_allocated = true;
  const WorkerImpl::Status target_status = impl_->get_status();
  if (target_status == WorkerImpl::Status::LOADED) {
    target_allocated = impl_->allocate_kv_cache_with_transfer(
        kv_cache_transfer_, kv_cache_shape);
  } else {
    CHECK_EQ(target_status, WorkerImpl::Status::READY);
  }

  bool draft_allocated = true;
  const WorkerImpl::Status draft_status = draft_impl_->get_status();
  if (draft_status == WorkerImpl::Status::LOADED) {
    draft_allocated = draft_impl_->allocate_kv_cache_with_transfer(
        kv_cache_transfer_, kv_cache_shape);
  } else {
    CHECK_EQ(draft_status, WorkerImpl::Status::READY);
  }

  embedding_cache_ = std::make_shared<EmbeddingCache>(num_blocks);
  return target_allocated && draft_allocated;
}
#endif

ForwardInput DFlashWorkerImpl::update_input_by_last_step_output(
    ForwardInput& inputs) {
  return inputs;
}

std::optional<ForwardOutput> DFlashWorkerImpl::step_empty(
    const ForwardInput& input) {
  if (!input.input_params.meta.batch_forward_type.is_decode()) {
    std::optional<ForwardOutput> output =
        run_llm_no_sync_impl(*impl_, input, *prepare_stream_, *compute_stream_);
    // Warmup only: prime the draft; its output is unused. Keep it alive until
    // the sync below so the no-sync draft input is not freed while its kernel
    // is still in flight.
    std::optional<ForwardOutput> draft_output = run_llm_no_sync_impl(
        *draft_impl_, input, *prepare_stream_, *compute_stream_);
    // Both forwards launched no-sync, so their staged inputs and the returned
    // target output are still in flight. Sync before returning so a DP idle
    // rank or graph warmup cannot reuse the input buffers, and non-overlap
    // callers do not observe an unfinished target output.
    compute_stream_->synchronize();
    if (output.has_value()) {
      clear_all_output_embeddings(output.value());
    }
    return output;
  }

  const int32_t query_width = options_.num_speculative_tokens() + 1;
  ForwardInput query_input = input;
  query_input.input_params.meta.batch_forward_type =
      BatchForwardType::CHUNKED_PREFILL;
  query_input.input_params.meta.q_max_seq_len = query_width;
  scale_dp_global_token_nums(query_input.input_params, query_width);
  // Warmup only: prime the draft; its output is unused. Keep it alive until the
  // sync below so the no-sync draft input is not freed while the target forward
  // launched next can reuse the buffer.
  std::optional<ForwardOutput> draft_output = run_llm_no_sync_impl(
      *draft_impl_, query_input, *prepare_stream_, *compute_stream_);

  ForwardInput validate_input = input;
  scale_dp_global_token_nums(validate_input.input_params, query_width);
  ForwardOutput output =
      run_llm_no_sync_impl(
          *impl_, validate_input, *prepare_stream_, *compute_stream_)
          .value();
  // See above: sync the no-sync draft and target forwards before returning.
  compute_stream_->synchronize();
  clear_all_output_embeddings(output);
  return output;
}

std::optional<ForwardOutput> DFlashWorkerImpl::step_prefill(
    const ForwardInput& input) {
  Timer timer;
  ForwardInput processed_target_input;
  ForwardOutput output = run_llm_no_sync_impl(*impl_,
                                              input,
                                              *prepare_stream_,
                                              *compute_stream_,
                                              &processed_target_input)
                             .value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());

  const torch::Tensor& embeddings = output.sample_output.embeddings;
  if (embeddings.defined()) {
    CHECK(processed_target_input.positions_host.defined())
        << "DFlash prefill requires processed positions_host.";
    Slice<int32_t> positions = {
        processed_target_input.positions_host.data_ptr<int32_t>(),
        static_cast<size_t>(processed_target_input.positions_host.numel())};
    CHECK_EQ(positions.size(), static_cast<size_t>(embeddings.size(0)))
        << "DFlash prefill hidden/position count mismatch.";
    const std::vector<int32_t>& processed_new_cache_slots =
        processed_target_input.input_params.attention.host.new_cache_slots;
    CHECK_EQ(processed_new_cache_slots.size(), positions.size())
        << "DFlash prefill hidden/cache slot count mismatch.";

    timer.reset();
    write_context_kv(
        processed_target_input,
        embeddings,
        processed_target_input.positions,
        processed_target_input.input_params.attention.device.new_cache_slots);
    COUNTER_ADD(speculative_execution_latency_seconds_draft,
                timer.elapsed_seconds());
  }

  if (input.sampling_params.selected_token_idxes.defined()) {
    embedding_cache_->write_prefill_target_context(
        input.input_params.embedding.embedding_ids,
        input.input_params.embedding.request_ids,
        output.sample_output.next_tokens,
        embeddings,
        input.sampling_params.selected_token_idxes);
    // PD handoff: the decode instance requires get_mtp_bootstrap_embedding()
    // defined before it accepts the request (disagg_pd_scheduler). Compress the
    // full prefill hidden to one row per sequence, as
    // write_prefill_target_context stores it.
    torch::Tensor bootstrap_embeddings = embeddings;
    if (bootstrap_embeddings.size(0) !=
        static_cast<int64_t>(
            input.input_params.embedding.embedding_ids.size())) {
      torch::Tensor bootstrap_idxes =
          input.sampling_params.selected_token_idxes.to(
              torch::dtype(torch::kLong).device(bootstrap_embeddings.device()));
      bootstrap_embeddings =
          bootstrap_embeddings.index_select(/*dim=*/0, bootstrap_idxes);
    }
    output.sample_output.embeddings = bootstrap_embeddings.detach();
    clear_selected_embeddings(output);
  } else {
    clear_all_output_embeddings(output);
  }

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  return output;
}

std::optional<ForwardOutput> DFlashWorkerImpl::step_decode(
    const ForwardInput& raw_input) {
  ForwardInput input = raw_input;
  ForwardInput validate_input;

  CHECK(embedding_cache_ != nullptr)
      << "DFlash embedding cache is not allocated";

  const auto& embedding = input.input_params.embedding;
  if (embedding.mtp_bootstrap_embeddings.defined()) {
    CHECK(input.token_ids_host.defined())
        << "DFlash bootstrap requires host token ids";
    CHECK(input.token_ids_host.device().is_cpu())
        << "DFlash bootstrap host token ids must be on CPU";
    CHECK_EQ(input.token_ids_host.scalar_type(), torch::kInt)
        << "DFlash bootstrap host token ids must be int32";

    torch::Tensor bootstrap_embeddings =
        safe_to(embedding.mtp_bootstrap_embeddings,
                torch::dtype(dtype_).device(device_));
    CHECK_EQ(bootstrap_embeddings.size(0),
             static_cast<int64_t>(embedding.mtp_bootstrap_row_idxes.size()))
        << "DFlash bootstrap row count mismatch";

    Slice<int32_t> token_ids = {
        input.token_ids_host.data_ptr<int32_t>(),
        static_cast<size_t>(input.token_ids_host.numel())};
    for (int32_t i = 0;
         i < static_cast<int32_t>(embedding.mtp_bootstrap_row_idxes.size());
         ++i) {
      const int32_t row_idx = embedding.mtp_bootstrap_row_idxes[i];
      CHECK_GE(row_idx, 0) << "DFlash bootstrap row index should be valid";
      CHECK_LT(row_idx, static_cast<int32_t>(embedding.embedding_ids.size()))
          << "DFlash bootstrap row index exceeds embedding ids";
      CHECK_LT(row_idx, static_cast<int32_t>(embedding.request_ids.size()))
          << "DFlash bootstrap row index exceeds request ids";
      CHECK_LT(static_cast<int64_t>(row_idx), input.token_ids_host.numel())
          << "DFlash bootstrap row index exceeds token ids";
      embedding_cache_->write_mtp_bootstrap_context(
          embedding.embedding_ids[row_idx],
          embedding.request_ids[row_idx],
          token_ids[row_idx],
          bootstrap_embeddings[i]);
    }
  }

  std::vector<EmbeddingCache::DecodeState> last_states =
      embedding_cache_->read_decode_states(
          input.input_params.embedding.embedding_ids,
          input.input_params.embedding.request_ids);
  CHECK_EQ(last_states.size(),
           input.input_params.embedding.embedding_ids.size())
      << "DFlash decode target state count mismatch";

  update_decode_step_input(input, last_states);
  DraftBlock draft_block = run_decode_draft(input, validate_input);
  return run_validate(input, draft_block, validate_input);
}

DFlashWorkerImpl::DraftBlock DFlashWorkerImpl::run_decode_draft(
    const ForwardInput& input,
    ForwardInput& validate_input) {
  Timer timer;

  ForwardInput query_input;
  prepare_query_inputs(input, query_input);

  ForwardOutput draft_output =
      run_llm_no_sync_impl(
          *draft_impl_, query_input, *prepare_stream_, *compute_stream_)
          .value();
  // Overlap validate input preparation with the async draft forward: the draft
  // launch above returns immediately, so building validate_input here (it only
  // reads the original input; draft tokens are injected later in
  // fill_validate_input_from_draft_outputs) runs on the host while the draft
  // computes on device, instead of delaying the draft launch.
  prepare_validate_inputs(input, validate_input);
  // Unify the draft next_tokens across the tensor-parallel group before
  // process_draft_sample_output() compresses the probs into the cache, so every
  // rank caches the same selected draft prob under schedule-overlap. No-op for
  // a single rank.
  maybe_broadcast_spec_tokens(draft_output.sample_output.next_tokens);
  process_draft_sample_output(draft_output.sample_output);
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());

  // Draft emits the whole block in one forward; reshape the flat outputs into
  // [batch, num_speculative_tokens] instead of splitting into per-step outputs.
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  const int32_t num_draft_tokens =
      static_cast<int32_t>(draft_output.sample_output.next_tokens.numel());
  CHECK_EQ(num_draft_tokens % num_speculative_tokens, 0)
      << "DFlash draft token count mismatch.";
  const int32_t batch_size = num_draft_tokens / num_speculative_tokens;
  CHECK_EQ(draft_output.sample_output.probs.numel(), num_draft_tokens)
      << "DFlash draft output requires selected draft probs.";

  DraftBlock draft_block;
  draft_block.token_ids = draft_output.sample_output.next_tokens.view(
      {batch_size, num_speculative_tokens});
  draft_block.probs = draft_output.sample_output.probs.view(
      {batch_size, num_speculative_tokens});
  // Keep the draft's no-sync input alive past run_validate's compute-stream
  // sync (see DraftBlock::draft_retained_input).
  draft_block.draft_retained_input = std::move(draft_output.retained_input);
  return draft_block;
}

void DFlashWorkerImpl::fill_validate_input_from_draft_outputs(
    const DraftBlock& draft_block,
    ForwardInput& validate_input,
    Stream& compute_stream) {
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  CHECK(draft_block.token_ids.defined())
      << "DFlash draft token_ids must be defined for validate token fill";
  CHECK_EQ(draft_block.token_ids.dim(), 2)
      << "DFlash draft token_ids must be [batch, num_speculative_tokens]";
  CHECK_EQ(draft_block.token_ids.size(1), num_speculative_tokens)
      << "DFlash draft token_ids width mismatch";
  CHECK(validate_input.token_ids.defined())
      << "DFlash validate token_ids must be prepared before draft token fill";
  CHECK_EQ(validate_input.token_ids.dim(), 1)
      << "DFlash validate token_ids must be flat";
  CHECK_EQ(validate_input.token_ids.numel() % num_val_tokens, 0)
      << "DFlash validate token_ids size must be divisible by validation width";

  const int64_t total_num_val_tokens = validate_input.token_ids.numel();
  const int64_t num_sequences = total_num_val_tokens / num_val_tokens;
  CHECK_EQ(draft_block.token_ids.size(0), num_sequences)
      << "DFlash draft batch must match validate sequence count";
  const torch::TensorOptions token_options = validate_input.token_ids.options();
  c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
  wait_metadata_ready_event(validate_input, compute_stream);
  torch::Tensor validate_token_rows =
      validate_input.token_ids.view({num_sequences, num_val_tokens});

  validate_input.device_tensors_ready = false;
  // Column 0 keeps the real input token; the draft block fills columns
  // [1, num_val_tokens) in one copy rather than per-step.
  torch::Tensor draft_tokens =
      safe_to(draft_block.token_ids, token_options, /*non_blocking=*/true);
  using ISlice = torch::indexing::Slice;
  validate_token_rows.index({ISlice(), ISlice(1, num_val_tokens)})
      .copy_(draft_tokens, /*non_blocking=*/true);
  validate_input.device_tensors_ready = true;
  // Publish this compute-stream write so the target's prepare stage (which
  // consumes validate_input.token_ids under ACL-graph double buffering) waits
  // for the copy to complete before staging into the graph's persistent
  // buffer. Without this, prepare could read stale/placeholder token ids.
  record_metadata_ready_event(compute_stream, validate_input);
}

std::optional<ForwardOutput> DFlashWorkerImpl::run_validate(
    const ForwardInput& input,
    const DraftBlock& draft_block,
    ForwardInput& validate_input) {
  Timer timer;
  fill_validate_input_from_draft_outputs(
      draft_block, validate_input, *compute_stream_);
  ForwardOutput target_output =
      run_llm_no_sync_impl(
          *impl_, validate_input, *prepare_stream_, *compute_stream_)
          .value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());

  timer.reset();
  SampleOutput val_output =
      validate(input.sampling_params, draft_block, target_output);
  COUNTER_ADD(speculative_execution_latency_seconds_validation,
              timer.elapsed_seconds());

  // target forward and validate()'s reads share compute_stream_, so they
  // serialize without a cross-stream wait; the sync below only makes the
  // accepted tokens host-visible for the D2H copy and context-cache write.
  maybe_broadcast_spec_tokens(val_output.next_tokens);
  compute_stream_->synchronize();
  val_output.next_tokens = val_output.next_tokens.to(torch::kCPU);
  write_target_context_to_cache(input, val_output);

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  val_output.embeddings = torch::Tensor();
  target_output.sample_output = val_output;
  return target_output;
}

SampleOutput DFlashWorkerImpl::validate(
    const SamplingParameters& sampling_params,
    const DraftBlock& draft_block,
    const ForwardOutput& target_output) {
  // Draft already emits the whole block [batch, num_speculative_tokens]; feed
  // it straight to the verifier without the per-step select/view/cat round
  // trip. The shared rejection sampler uses MTP's dense contract, so
  // reconstruct dense draft probs unless the selected-only optimization is on.
  const int32_t vocab_size =
      static_cast<int32_t>(target_output.logits.size(/*dim=*/-1));
  const bool enable_opt_validate_probs =
      ::xllm::SpeculativeConfig::get_instance().enable_opt_validate_probs();
  auto [draft_token_ids, draft_probs] =
      specBuilder::draftProbs::build_validate_tensors_from_block(
          draft_block.token_ids,
          draft_block.probs,
          vocab_size,
          enable_opt_validate_probs);
  return validate(sampling_params, draft_token_ids, draft_probs, target_output);
}

SampleOutput DFlashWorkerImpl::validate(
    const SamplingParameters& sampling_params,
    const torch::Tensor& draft_token_ids,
    const torch::Tensor& draft_probs,
    const ForwardOutput& target_output) {
  const int32_t num_val_tokens = options_.num_speculative_tokens() + 1;
  // Derive batch_size from the target logits rows rather than next_tokens so
  // the reshape stays valid regardless of how the target was sampled.
  const int32_t num_logits_rows =
      static_cast<int32_t>(target_output.logits.size(/*dim=*/0));
  CHECK_EQ(num_logits_rows % num_val_tokens, 0)
      << "DFlash validate target logits rows must be divisible by validation "
         "width";
  const int32_t batch_size = num_logits_rows / num_val_tokens;
  const int32_t vocab_size =
      static_cast<int32_t>(target_output.logits.size(/*dim=*/-1));

  using torch::indexing::None;
  using ISlice = torch::indexing::Slice;
  torch::Tensor bonus_token_ids =
      target_output.sample_output.next_tokens
          .index({"...", ISlice(num_val_tokens - 1, None, num_val_tokens)})
          .view({-1, 1});

  torch::Tensor target_logits =
      target_output.logits.view({batch_size, num_val_tokens, vocab_size});

  auto rejection_sampler =
      std::make_unique<RejectionSampler>(sampling_params.do_sample,
                                         sampling_params.all_random_sample,
                                         sampling_params.all_greedy_sample,
                                         target_output.logprobs,
                                         target_output.max_top_logprobs,
                                         enable_fused_kernel_);

  SampleOutput sample_output =
      rejection_sampler->forward(draft_token_ids.to(bonus_token_ids),
                                 draft_probs.to(target_logits.device()),
                                 target_logits,
                                 bonus_token_ids,
                                 /*mask_out_rejected_tokens=*/true);

  const torch::Tensor& embeddings = target_output.sample_output.embeddings;
  sample_output.embeddings =
      embeddings.view({batch_size, num_val_tokens, embeddings.size(-1)});
  return sample_output;
}

void DFlashWorkerImpl::process_draft_sample_output(
    SampleOutput& sample_output) {
  if (sample_output.probs.defined()) {
    CHECK(sample_output.next_tokens.defined())
        << "DFlash draft sample_output.next_tokens must be defined when probs "
           "exist";
    CHECK_EQ(sample_output.next_tokens.dim(), 1)
        << "DFlash draft cache expects next_tokens [batch], got "
        << sample_output.next_tokens.sizes();
    CHECK(sample_output.probs.dim() == 1 || sample_output.probs.dim() == 2)
        << "DFlash draft cache expects probs [batch] or [batch,vocab], got "
        << sample_output.probs.sizes();
    CHECK_EQ(sample_output.probs.size(0), sample_output.next_tokens.size(0))
        << "DFlash draft cache probs/token batch mismatch";
    sample_output.probs = specBuilder::draftProbs::compress_for_cache(
        sample_output.probs, sample_output.next_tokens);
  }
}

void DFlashWorkerImpl::maybe_broadcast_spec_tokens(torch::Tensor& tokens) {
  if (get_optimization_config().enable_spec_token_broadcast &&
      enable_schedule_overlap()) {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    broadcast_spec_tokens(tokens, spec_broadcast_group(parallel_args_));
  }
}

void DFlashWorkerImpl::update_decode_step_input(
    ForwardInput& input,
    const std::vector<EmbeddingCache::DecodeState>& last_states) const {
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  CHECK_EQ(last_states.size(), static_cast<size_t>(num_sequences))
      << "DFlash decode context state count mismatch";
  const bool enable_cache_correction = enable_schedule_overlap();

  std::vector<int32_t> token_ids_vec;
  std::vector<int32_t> positions_vec;
  std::vector<int32_t> kv_seq_lens_vec;
  token_ids_vec.reserve(num_sequences);
  positions_vec.reserve(num_sequences);
#if defined(USE_NPU)
  kv_seq_lens_vec.reserve(num_sequences);
#else
  kv_seq_lens_vec.reserve(num_sequences + 1);
#endif

  const torch::Tensor& token_ids_cpu = input.token_ids_host;
  const torch::Tensor& positions_cpu = input.positions_host;
  Slice<int32_t> input_token_ids = {token_ids_cpu.data_ptr<int32_t>(),
                                    static_cast<size_t>(token_ids_cpu.numel())};
  Slice<int32_t> input_positions = {positions_cpu.data_ptr<int32_t>(),
                                    static_cast<size_t>(positions_cpu.numel())};

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    CHECK_LT(static_cast<size_t>(seq_id), input_token_ids.size())
        << "DFlash decode context token seq_id out of range, seq_id=" << seq_id;
    CHECK_LT(static_cast<size_t>(seq_id), input_positions.size())
        << "DFlash decode context position seq_id out of range, seq_id="
        << seq_id;
    const EmbeddingCache::DecodeState& state = last_states[seq_id];
    const int32_t input_token_id = input_token_ids[seq_id];
    const bool input_is_fake_token = input_token_id < 0;
    // Rewrite fake input tokens to the last committed real token so KV cache
    // scatter has a valid id; only apply the recorded position offset when the
    // cached state is still valid.
    const bool rewrite_fake_token =
        enable_cache_correction && input_is_fake_token;
    const bool use_cache_correction = rewrite_fake_token && state.valid;
    const int32_t position_offset =
        use_cache_correction ? state.position_offset : 0;
    const int32_t current_position = input_positions[seq_id] + position_offset;
    const int32_t current_kv_len = specBuilder::calc_kv_len(
        input.input_params.attention.host.kv_seq_lens, seq_id, position_offset);
    const int32_t expected_kv_len = current_position + 1;

    CHECK_EQ(expected_kv_len, current_kv_len)
        << "DFlash decode context position/kv_len mismatch, seq_id=" << seq_id
        << ", current_position=" << current_position
        << ", current_kv_len=" << current_kv_len;

    token_ids_vec.emplace_back(rewrite_fake_token ? state.token_id
                                                  : input_token_id);
    positions_vec.emplace_back(current_position);
    specBuilder::append_seq_len_by_layout(kv_seq_lens_vec, current_kv_len);
  }

  input.token_ids_host = specBuilder::make_cpu_int_tensor(token_ids_vec);
  input.positions_host = specBuilder::make_cpu_int_tensor(positions_vec);
  input.input_params.attention.host.kv_seq_lens = std::move(kv_seq_lens_vec);
  input.device_tensors_ready = false;
}

void DFlashWorkerImpl::prepare_validate_inputs(const ForwardInput& input,
                                               ForwardInput& validate_input) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  ForwardInput prepared_input = input;
  prepared_input.metadata_ready_event.reset();
  SpeculativeWorkerImpl::prepare_validate_inputs(prepared_input,
                                                 validate_input);
  validate_input.input_params.embedding.input_embedding = torch::Tensor();
  record_metadata_ready_event(*prepare_stream_, validate_input);
}

void DFlashWorkerImpl::prepare_query_inputs(const ForwardInput& input,
                                            ForwardInput& query_input) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  query_input = input;
  query_input.device_tensors_ready = false;
  ModelInputParams& input_params = query_input.input_params;
  input_params.embedding.input_embedding = torch::Tensor();

  specBuilder::DecodeBuildBuffers buf;
  std::vector<int32_t> selected_idxes;
  std::vector<int32_t> q_cu_seq_lens;
  build_query_rows(input,
                   mask_token_id_,
                   options_.num_speculative_tokens(),
                   options_.block_size(),
                   buf,
                   selected_idxes,
                   q_cu_seq_lens);
  const int32_t query_width = options_.num_speculative_tokens() + 1;
  // DFlash emits query_width rows per seq unconditionally, so DP shape
  // symmetry holds by construction. Catch scheduler regressions that break
  // the invariant (see MTP dp_enabled idle-rank branch).
  const int32_t num_sequences_query = input.input_params.meta.num_sequences;
  CHECK_EQ(static_cast<int32_t>(buf.out_positions.size()),
           num_sequences_query * query_width)
      << "DFlash per-seq row count must be uniform (query_width=" << query_width
      << ", num_sequences=" << num_sequences_query << ")";

  specBuilder::set_token_position_tensors(query_input,
                                          buf.out_token_ids,
                                          buf.out_positions,
                                          input.token_ids.options(),
                                          input.positions.options());
  input_params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  specBuilder::update_input_params(input_params,
                                   buf,
                                   query_width,
                                   std::move(buf.out_q_seq_lens),
                                   std::move(q_cu_seq_lens),
                                   buf.meta.kv_max_seq_len,
                                   std::move(buf.out_kv_seq_lens),
                                   /*update_block_tables=*/false);
  scale_dp_global_token_nums(input_params, query_width);
  input_params.attention.rebuild_device_buffer(device_);

  torch::TensorOptions idx_options =
      torch::TensorOptions().dtype(torch::kInt).device(device_);
  // Pinned-host + async H2D on prepare_stream_ (the file's idiom), so the copy
  // overlaps instead of a blocking non-pinned transfer every decode step.
  query_input.sampling_params.selected_token_idxes =
      safe_to(specBuilder::make_cpu_int_tensor(selected_idxes),
              idx_options,
              /*non_blocking=*/true);
  query_input.sampling_params.sample_idxes =
      torch::arange(static_cast<int64_t>(selected_idxes.size()), idx_options);
  // Force the draft sampler to emit selected-token probabilities even on the
  // greedy path (temperature=0); the rejection sampler needs them to verify
  // the block. Without this the greedy sampler skips probs entirely.
  query_input.sampling_params.return_probs = true;
  repeat_sampling_params(query_input.sampling_params,
                         options_.num_speculative_tokens());
  query_input.device_tensors_ready = true;
}

void DFlashWorkerImpl::write_context_kv(
    const ForwardInput& input,
    const torch::Tensor& context_hidden,
    const torch::Tensor& positions_device,
    const torch::Tensor& new_cache_slots_device) {
  CHECK(context_hidden.defined()) << "DFlash context hidden is undefined.";
  CHECK_EQ(context_hidden.dim(), 2) << "DFlash context hidden must be 2D.";
  CHECK_EQ(context_hidden.size(1), expected_context_hidden_size_)
      << "DFlash context hidden size must be hidden_size * "
      << "target_layer_ids.size().";

  CHECK(context_hidden.device() == device_.unwrap())
      << "DFlash context hidden must already be on the compute device.";

  // Both the target forward that produced context_hidden and this pass run
  // on compute_stream_, so no explicit event dance is needed — the stream
  // orders them. Model methods below use torch ops on the same stream.
  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();

#if defined(USE_NPU)
  // PD PUSH: the draft context-KV scattered below is not covered by the target
  // push, so wire the draft push here. Overwrite layer_synchronizer with a
  // draft-sized one (one event per draft layer); the target's was already
  // waited on before this runs, so the overwrite is safe. No-op when
  // transfer_kv_infos is empty. NPU-only: the scatter's record_event loop is
  // NPU-gated, so a non-NPU push would stall on events that never record.
  KVTransferCompletion kv_transfers;
  if (options_.kv_cache_transfer_mode() == "PUSH" &&
      !input.transfer_kv_infos.empty()) {
    std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer =
        std::make_shared<NPULayerSynchronizerImpl>(
            draft_impl_->context_.get_model_args().n_layers());
    const_cast<ModelInputParams*>(&(input.input_params))
        ->parallel.layer_synchronizer = layer_synchronizer;
    kv_transfers.add(kv_cache_transfer_->push_kv_blocks_async(
        input.transfer_kv_infos,
        draft_impl_->context_.get_parallel_args(),
        layer_synchronizer,
        /*is_spec_draft=*/true));
  }
#endif

  ModelOutput scatter_output =
      draft_impl_->write_context_kv(context_hidden,
                                    positions_device,
                                    new_cache_slots_device,
                                    input.input_params);
  // Model returns an empty ModelOutput (hidden_states undefined) when the
  // per-layer NPULayerSynchronizer::record_event fails mid-scatter under
  // PD-PUSH transfer. Fail fast instead of silently continuing: subsequent
  // layers won't have been scattered and the PD transfer side would block
  // forever waiting on the missing per-layer event.
  CHECK(scatter_output.hidden_states.defined())
      << "DFlash context-KV scatter failed (layer_synchronizer record_event "
         "returned false); PD-PUSH transfer would deadlock.";

  // The context-KV scatter is the last device op of the step and is launched
  // no-sync. Under schedule-overlap the scheduler dispatches the next step's
  // host prep as soon as this step's host call returns, while this scatter may
  // still be in flight. Sync so the next step's draft query reads a fully
  // written context-KV cache instead of a partially scattered one.
  compute_stream_->synchronize();

#if defined(USE_NPU)
  // Wait for the draft KV push (if any) so the source draft cache is not
  // overwritten by the next step while the transfer is still reading it
  // (mirrors step_internal's wait_kv_push()). No-op when no push was issued.
  CHECK(kv_transfers.wait()) << "DFlash draft context-KV push failed";
#endif
}

void DFlashWorkerImpl::write_target_context_to_cache(
    const ForwardInput& input,
    const SampleOutput& validate_output) {
  const torch::Tensor& accepted_embeddings = validate_output.embeddings;
  CHECK(accepted_embeddings.defined())
      << "DFlash validate target embeddings are undefined.";
  CHECK_EQ(accepted_embeddings.dim(), 3)
      << "DFlash validate target embeddings must be [batch,width,hidden].";

  torch::Tensor accepted_tokens = validate_output.next_tokens;
  CHECK(accepted_tokens.defined()) << "DFlash accepted tokens are undefined.";
  if (accepted_tokens.scalar_type() != torch::kInt64) {
    accepted_tokens = accepted_tokens.to(torch::kInt64);
  }
  DCHECK(accepted_tokens.is_contiguous())
      << "DFlash accepted tokens must be contiguous (guaranteed by upstream "
         ".to(kCPU) and .to(kInt64) branches); check for stride changes if "
         "this fires.";

  CHECK_EQ(accepted_tokens.dim(), 2)
      << "DFlash accepted tokens must be [batch,width].";
  const int64_t batch_size = accepted_tokens.size(0);
  const int64_t token_width = accepted_tokens.size(1);
  CHECK_EQ(accepted_embeddings.size(0), batch_size)
      << "DFlash accepted token/embedding batch mismatch.";
  CHECK_EQ(accepted_embeddings.size(1), token_width)
      << "DFlash accepted token/embedding width mismatch.";

  specBuilder::DecodeBuildBuffers buf;
  std::vector<int64_t> accepted_idxes = build_accepted_context_rows(
      input, accepted_tokens, options_.block_size(), buf);
  torch::TensorOptions host_index_options = torch::TensorOptions()
                                                .dtype(torch::kLong)
                                                .device(torch::kCPU)
                                                .pinned_memory(true);
  torch::TensorOptions device_index_options =
      torch::TensorOptions()
          .dtype(torch::kLong)
          .device(accepted_embeddings.device());
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  torch::Tensor accepted_index =
      safe_to(torch::tensor(accepted_idxes, host_index_options),
              device_index_options,
              /*non_blocking=*/true);
  torch::Tensor flat_embeddings = accepted_embeddings.reshape(
      {batch_size * token_width, accepted_embeddings.size(/*dim=*/2)});
  torch::Tensor context_hidden =
      flat_embeddings.index_select(/*dim=*/0, accepted_index);
  torch::Tensor positions_device =
      cpu_int_vec_to_device(buf.out_positions, device_);
  torch::Tensor new_cache_slots_device =
      cpu_int_vec_to_device(buf.out_new_cache_slots, device_);
  // Publish the prepare_stream_ work (index_select producing context_hidden +
  // pinned H2D copies for positions/slots) so compute_stream_ waits for it
  // before the model reads these tensors. Without this, torch does not
  // enforce cross-stream ordering: the write_context_kv scatter could launch
  // before index_select finishes, producing corrupt KV cache. The prefill
  // caller runs its producer on compute_stream_, so no event is needed there.
  StreamEventPtr context_hidden_ready_event = prepare_stream_->record_event();
  if (context_hidden_ready_event != nullptr) {
    CHECK(compute_stream_->wait_event(context_hidden_ready_event))
        << "failed to wait DFlash context hidden ready event";
  } else {
    prepare_stream_->synchronize();
  }
  write_context_kv(
      input, context_hidden, positions_device, new_cache_slots_device);
  CHECK(!input.input_params.embedding.embedding_ids.empty())
      << "DFlash target context cache write requires embedding ids";
  embedding_cache_->write_target_context(
      input.input_params.embedding.embedding_ids,
      input.input_params.embedding.request_ids,
      validate_output.next_tokens,
      validate_output.embeddings,
      options_.num_speculative_tokens());
}

}  // namespace xllm
