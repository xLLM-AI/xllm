/* Copyright 2025-2026 The xLLM Authors.

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
#include "core/distributed_runtime/worker_service.h"

#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include <brpc/stream.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "common/types.h"
#include "core/distributed_runtime/worker_model_codec.h"
#include "core/framework/speculative/metrics.h"
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "core/kv_cache/transfer/kv_transport_codec.h"
#include "core/kv_cache/transfer/prefetch_result.h"
#include "core/util/tensor_helper.h"
#include "runtime/dit_forward_params.h"
#include "runtime/forward_params.h"
#include "runtime/rec_forward_params.h"
#include "runtime/vlm_forward_params.h"
#include "util/timer.h"

namespace xllm {
namespace {

std::vector<SpeculativeTokenStats> record_speculative_metrics(
    const ForwardOutput& output,
    const torch::Tensor& cpu_next_tokens,
    bool speculative_decode) {
  if (output.is_graph_warmup || !speculative_decode) {
    return {};
  }
  if (output.spec_verify_layouts.empty()) {
    CHECK(!cpu_next_tokens.defined() || cpu_next_tokens.dim() != 2)
        << "speculative verify output is missing row metadata";
    return {};
  }
  // Reuse the completed output copy instead of issuing another D2H for metrics.
  SpecMetrics metrics =
      make_spec_metrics(cpu_next_tokens, output.spec_verify_layouts);
  record_spec_metrics(metrics);
  return std::move(metrics.sequence_stats);
}

template <typename Input>
int32_t get_decode_sequence_count(const Input& input) {
  const int64_t num_seqs = input.sampling_params.num_sample_seqs();
  if (num_seqs > 0) {
    return static_cast<int32_t>(num_seqs);
  }
  if (input.input_host_sample_count != kUnknownPackedSampleCount) {
    return input.input_host_sample_count;
  }
  if (!input.runtime.input_host_buffer_has_layout) {
    return 0;
  }
  Input unpacked_input;
  const bool unpacked = detail::unpack_from_input_host_buffer(
      input,
      torch::Device(torch::kCPU),
      torch::kFloat32,
      unpacked_input,
      /*materialize_device_buffer=*/false);
  return unpacked ? static_cast<int32_t>(
                        unpacked_input.sampling_params.num_sample_seqs())
                  : 0;
}

template <typename Input>
void stabilize_host_views(Input& input) {
  input.token_ids_host = materialize_cpu_tensor(input.token_ids_host);
  input.positions_host = materialize_cpu_tensor(input.positions_host);
  input.input_params.attention.host.block_tables =
      materialize_cpu_tensor(input.input_params.attention.host.block_tables);
}

bool has_cpu_llm_output(const ForwardOutput& output) {
  const auto on_cpu = [](const torch::Tensor& tensor) {
    return !tensor.defined() || tensor.device().is_cpu();
  };
  const auto& sample = output.sample_output;
  const auto& beam = output.beam_search_output;
  const bool cpu_tensors =
      on_cpu(output.next_tokens_host.defined() ? output.next_tokens_host
                                               : sample.next_tokens) &&
      on_cpu(output.expert_load_data) && on_cpu(sample.embeddings) &&
      on_cpu(sample.logprobs) && on_cpu(sample.top_tokens) &&
      on_cpu(sample.top_logprobs) && on_cpu(beam.src_seq_idxes) &&
      on_cpu(beam.out_tokens) && on_cpu(beam.out_logprobs);
  return cpu_tensors &&
         std::all_of(sample.mm_embeddings.begin(),
                     sample.mm_embeddings.end(),
                     [&](const auto& sequence) {
                       return std::all_of(
                           sequence.begin(), sequence.end(), on_cpu);
                     });
}

class WorkerPrefetchSession final
    : public brpc::StreamInputHandler,
      public std::enable_shared_from_this<WorkerPrefetchSession> {
 public:
  WorkerPrefetchSession(Worker* worker,
                        ThreadPool* threadpool,
                        StoragePrefetchRequest request)
      : worker_(worker), threadpool_(threadpool), request_(std::move(request)) {
    CHECK(worker_ != nullptr);
    CHECK(threadpool_ != nullptr);
    CHECK(request_.valid());
  }

  void retain() {
    std::lock_guard<std::mutex> lock(mutex_);
    keepalive_ = shared_from_this();
  }

  void release() {
    std::lock_guard<std::mutex> lock(mutex_);
    keepalive_.reset();
  }

  void start(brpc::StreamId stream_id) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (state_ != State::CREATED) {
        return;
      }
      stream_id_ = stream_id;
      state_ = State::RUNNING_BATCH;
    }
    schedule_batch();
  }

  int on_received_messages(brpc::StreamId id,
                           butil::IOBuf* const messages[],
                           size_t size) override {
    if (size != 1 || messages[0]->length() != 1) {
      fail_and_close(id);
      return -1;
    }

    uint8_t control_byte = 0;
    messages[0]->copy_to(&control_byte, sizeof(control_byte));
    const PrefetchControl control = static_cast<PrefetchControl>(control_byte);
    bool run_next = false;
    bool close = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (state_ == State::RUNNING_BATCH && control == PrefetchControl::STOP) {
        stop_after_batch_ = true;
      } else if (state_ != State::WAITING_DECISION) {
        state_ = State::FAILED;
        close = true;
      } else if (control == PrefetchControl::STOP) {
        state_ = State::COMPLETED;
        close = true;
      } else if (control == PrefetchControl::CONTINUE &&
                 last_prefix_hit_units_ ==
                     request_.batch_unit_count(batch_index_) &&
                 batch_index_ + 1 < request_.batch_count()) {
        ++batch_index_;
        state_ = State::RUNNING_BATCH;
        run_next = true;
      } else {
        state_ = State::FAILED;
        close = true;
      }
    }

    if (run_next) {
      schedule_batch();
    } else if (close) {
      brpc::StreamClose(id);
    }
    // A close triggered by anything other than a STOP is a protocol error;
    // a STOP-driven close is the normal completion path.
    const bool protocol_error = close && control != PrefetchControl::STOP;
    return protocol_error ? -1 : 0;
  }

  void on_idle_timeout(brpc::StreamId id) override { fail_and_close(id); }

  void on_failed(brpc::StreamId /*id*/,
                 int /*error_code*/,
                 const std::string& /*error_text*/) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != State::COMPLETED) {
      state_ = State::FAILED;
    }
  }

  void on_closed(brpc::StreamId /*id*/) override {
    std::shared_ptr<WorkerPrefetchSession> self = shared_from_this();
    std::lock_guard<std::mutex> lock(mutex_);
    state_ = State::CLOSED;
    keepalive_.reset();
  }

 private:
  enum class State : uint8_t {
    CREATED = 0,
    RUNNING_BATCH = 1,
    WAITING_DECISION = 2,
    COMPLETED = 3,
    FAILED = 4,
    CLOSED = 5,
  };

  void schedule_batch() {
    std::shared_ptr<WorkerPrefetchSession> self = shared_from_this();
    threadpool_->schedule([self = std::move(self)]() { self->run_batch(); });
  }

  void run_batch() {
    size_t batch_index = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (state_ != State::RUNNING_BATCH) {
        return;
      }
      batch_index = batch_index_;
    }

    const auto [transfer_begin, transfer_end] =
        request_.batch_transfer_range(batch_index);
    Slice<BlockTransferInfo> all_transfers(request_.transfer_infos);
    Slice<BlockTransferInfo> batch =
        all_transfers.slice(transfer_begin, transfer_end);
    std::vector<uint8_t> logical_hits = worker_->prefetch_kv_blocks(batch);
    const std::optional<uint8_t> prefix_hit_units =
        request_.count_prefix_hit_units(batch_index, logical_hits);
    if (!prefix_hit_units.has_value()) {
      fail_and_close(stream_id_);
      return;
    }

    bool close_after_result = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (state_ != State::RUNNING_BATCH || batch_index != batch_index_) {
        return;
      }
      last_prefix_hit_units_ = *prefix_hit_units;
      close_after_result = stop_after_batch_;
      state_ = close_after_result ? State::COMPLETED : State::WAITING_DECISION;
    }

    butil::IOBuf result;
    result.append(&*prefix_hit_units, sizeof(*prefix_hit_units));
    if (brpc::StreamWrite(stream_id_, result) != 0) {
      fail_and_close(stream_id_);
    } else if (close_after_result) {
      brpc::StreamClose(stream_id_);
    }
  }

  void fail_and_close(brpc::StreamId id) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (state_ == State::CLOSED) {
        return;
      }
      state_ = State::FAILED;
    }
    brpc::StreamClose(id);
  }

  Worker* worker_ = nullptr;
  ThreadPool* threadpool_ = nullptr;
  StoragePrefetchRequest request_;
  std::mutex mutex_;
  brpc::StreamId stream_id_ = brpc::INVALID_STREAM_ID;
  size_t batch_index_ = 0;
  size_t last_prefix_hit_units_ = 0;
  bool stop_after_batch_ = false;
  State state_ = State::CREATED;
  std::shared_ptr<WorkerPrefetchSession> keepalive_;
};

}  // namespace

WorkerService::WorkerService(runtime::Options options,
                             const torch::Device& device)
    : options_(options), initialized_(false), device_(device) {
  device_.set_device();
  device_.init_device_context();
  stream_ = device_.get_stream_from_pool();
  threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/4,
      /*init_func=*/[this]() mutable { device_.set_device(); },
      /*cpu_binding=*/false,
      /*pool_name=*/"WorkerService.request");
}

WorkerService::WorkerService(runtime::Options options,
                             const torch::Device& device,
                             std::unique_ptr<Worker> worker)
    : options_(options),
      initialized_(true),
      device_(device),
      worker_(std::move(worker)) {
  device_.set_device();
  device_.init_device_context();
  stream_ = device_.get_stream_from_pool();
  threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/4,
      /*init_func=*/[this]() mutable { device_.set_device(); },
      /*cpu_binding=*/false,
      /*pool_name=*/"WorkerService.request");
}

WorkerService::~WorkerService() {
  // The polling thread may be idle in input_read. Cancel that wait before
  // joining, while the worker and its execution resources are still alive.
  if (input_shm_manager_) {
    input_shm_manager_->stop_input_read();
  }
  if (polling_thread_ && polling_thread_->joinable()) {
    polling_thread_->join();
  }
}

void WorkerService::set_worker(std::unique_ptr<Worker> worker) {
  worker_ = std::move(worker);
  initialized_ = true;
}

void WorkerService::create_polling_shm_thread(
    std::unique_ptr<ForwardSharedMemoryManager> input_shm_manager,
    std::unique_ptr<ForwardSharedMemoryManager> output_shm_manager) {
  CHECK(!polling_thread_) << "SHM polling thread has already been created.";
  input_shm_manager_ = std::move(input_shm_manager);
  if (options_.backend() == "llm") {
    polling_thread_ = std::make_unique<std::thread>(
        &WorkerService::poll_llm_shm, this, std::move(output_shm_manager));
  } else if (options_.backend() == "vlm") {
    polling_thread_ = std::make_unique<std::thread>(
        &WorkerService::poll_vlm_shm, this, std::move(output_shm_manager));
  } else if (options_.backend() == "rec") {
    polling_thread_ = std::make_unique<std::thread>(
        &WorkerService::poll_rec_shm, this, std::move(output_shm_manager));
  } else {
    CHECK_EQ(options_.backend(), "dit") << "Unsupported worker backend";
    polling_thread_ = std::make_unique<std::thread>(
        &WorkerService::poll_dit_shm, this, std::move(output_shm_manager));
  }
}

void WorkerService::Hello(::google::protobuf::RpcController* controller,
                          const proto::Status* request,
                          proto::Status* response,
                          ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  if (!initialized_) {
    ctrl->SetFailed("Server is not initialized");
  } else {
    response->set_ok(true);
  }
  return;
}

void WorkerService::InitModel(::google::protobuf::RpcController* controller,
                              const proto::InitModelRequest* request,
                              proto::Status* response,
                              ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    auto model_weights_path = request->model_weights_path();
    auto random_seed = request->random_seed();
    auto init_future =
        worker_->init_model_async(model_weights_path,
                                  random_seed,
                                  MasterStatus(request->master_status()));
    bool status = std::move(init_future).get();
    if (!status) {
      response->set_ok(false);
      return;
    }

    response->set_ok(true);
  });
  return;
}

void WorkerService::collect_llm_output(const ForwardOutput& output,
                                       ForwardOutput& cpu_output,
                                       bool cpu_ready_result) {
  cpu_output.prepared_token = output.prepared_token;
  cpu_output.json_object_errors = output.json_object_errors;
  const auto& sample = output.sample_output;
  const auto& beam = output.beam_search_output;
  auto copy_output_to_host = [&]() {
    // Every D2H, including expert data, runs under the selected output stream.
    cpu_output.expert_load_data =
        safe_to(output.expert_load_data, torch::kCPU, /*non_blocking=*/true);
    cpu_output.sample_output.embeddings =
        safe_to(sample.embeddings,
                torch::dtype(torch::kFloat32).device(torch::kCPU),
                /*non_blocking=*/true);
    auto& mm_embeddings = cpu_output.sample_output.mm_embeddings;
    mm_embeddings.clear();
    mm_embeddings.reserve(sample.mm_embeddings.size());
    for (const auto& sequence : sample.mm_embeddings) {
      std::vector<torch::Tensor> cpu_sequence;
      cpu_sequence.reserve(sequence.size());
      for (const torch::Tensor& embedding : sequence) {
        cpu_sequence.emplace_back(
            safe_to(embedding, torch::kCPU, /*non_blocking=*/true));
      }
      mm_embeddings.emplace_back(std::move(cpu_sequence));
    }
    cpu_output.sample_output.next_tokens =
        safe_to(output.next_tokens_host.defined() ? output.next_tokens_host
                                                  : sample.next_tokens,
                torch::kCPU,
                /*non_blocking=*/true);
    if (cpu_output.sample_output.next_tokens.defined()) {
      cpu_output.sample_output.logprobs =
          safe_to(sample.logprobs, torch::kCPU, /*non_blocking=*/true);
      // Beam results already carry the final tokens and logprobs.
      if (!beam.src_seq_idxes.defined()) {
        cpu_output.sample_output.top_tokens =
            safe_to(sample.top_tokens, torch::kCPU, /*non_blocking=*/true);
        cpu_output.sample_output.top_logprobs =
            safe_to(sample.top_logprobs, torch::kCPU, /*non_blocking=*/true);
      }
    }
    // Keep optional beam results independent of sample next_tokens.
    cpu_output.beam_search_output.src_seq_idxes =
        safe_to(beam.src_seq_idxes, torch::kCPU, /*non_blocking=*/true);
    if (beam.src_seq_idxes.defined()) {
      cpu_output.beam_search_output.out_tokens =
          safe_to(beam.out_tokens, torch::kCPU, /*non_blocking=*/true);
      cpu_output.beam_search_output.out_logprobs =
          safe_to(beam.out_logprobs,
                  torch::dtype(torch::kFloat32).device(torch::kCPU),
                  /*non_blocking=*/true);
    }
  };
  const bool use_default_stream = !options_.enable_schedule_overlap();
  if (cpu_ready_result) {
    copy_output_to_host();
  } else if (use_default_stream) {
    copy_output_to_host();
    device_.synchronize_default_stream();
  } else {
    c10::StreamGuard stream_guard = stream_->set_stream_guard();
    if (output.ready_event != nullptr) {
      CHECK(stream_->wait_event(output.ready_event))
          << "wait LLM output ready event failed";
    }
    copy_output_to_host();
    stream_->synchronize();
  }
#if defined(USE_NPU)
  if (!use_default_stream) {
    DeviceMonitor::get_instance().update_active_activation_memory(
        device_.index());
  }
#endif
}

void WorkerService::step_llm_model(
    LlmForwardInput& input,
    ForwardOutput& cpu_output,
    std::vector<SpeculativeTokenStats>& speculative_token_stats) {
  speculative_token_stats.clear();
  const bool task_pipeline = options_.enable_task_pipeline();
  const bool worker_prepare =
      task_pipeline && worker_->task_pipeline_uses_worker_prepare();
  const bool worker_rebuilds_host_views =
      input.runtime.input_host_buffer_has_layout &&
      input.runtime.input_host_buffer.defined() &&
      input.runtime.input_host_buffer.device().is_cpu() &&
      input.runtime.input_host_buffer.is_contiguous() &&
      input.runtime.input_host_buffer.numel() > 0 &&
      !input.runtime.device_tensors_ready;
  if (options_.enable_schedule_overlap() && !worker_rebuilds_host_views &&
      (!task_pipeline || worker_prepare)) {
    // Packed preparation rebuilds views from its retained owner. Device-ready
    // inputs reuse their existing views, which can still borrow shared memory.
    stabilize_host_views(input);
  }
  auto future = worker_->step_async(input);
  if (!options_.enable_schedule_overlap()) {
    auto output = std::move(future).get();
    if (!output.has_value()) {
      return;
    }
    const bool cpu_ready_result =
        task_pipeline && (!worker_prepare || has_cpu_llm_output(*output));
    collect_llm_output(*output, cpu_output, cpu_ready_result);
    speculative_token_stats =
        record_speculative_metrics(*output,
                                   cpu_output.sample_output.next_tokens,
                                   options_.enable_speculative_decode());
    return;
  }
  if (task_pipeline) {
    // Do not publish placeholders until the Worker accepts and owns the input.
    auto ack = std::move(future).get();
    CHECK(!ack.has_value()) << "LLM task Step must return PrepareAck";
  }
  const auto int_options = torch::TensorOptions().device(torch::kCPU);
  if (worker_->is_driver()) {
    const int32_t sequence_count = get_decode_sequence_count(input);
    cpu_output.sample_output.next_tokens = torch::arange(
        -1, -(sequence_count + 1), -1, int_options.dtype(torch::kInt32));
    if (!task_pipeline) {
      std::move(future).deferValue([](auto&&) {});
    }
  }
  cpu_output.expert_load_data =
      torch::zeros({1, 1}, int_options.dtype(torch::kInt64));
}

void WorkerService::poll_llm_shm(
    std::unique_ptr<ForwardSharedMemoryManager> output_shm_manager) {
  device_.set_device();
  // Graph updates and task preparation must stay on the ordered worker stream.
  const InputDeviceMaterializationPolicy policy =
      options_.enable_task_pipeline() ||
              (options_.enable_schedule_overlap() && options_.enable_graph())
          ? InputDeviceMaterializationPolicy::DEFER_TO_WORKER_PREPARE
          : InputDeviceMaterializationPolicy::MATERIALIZE_ON_READ;
  while (true) {
    LlmForwardInput input;
    if (!input_shm_manager_->input_read(input, device_, policy)) {
      break;
    }
    Timer timer;
    ForwardOutput cpu_output;
    std::vector<SpeculativeTokenStats> speculative_token_stats;
    step_llm_model(input, cpu_output, speculative_token_stats);
    const bool write_ok = output_shm_manager->llm_output_write(
        cpu_output, speculative_token_stats);
    CHECK(write_ok) << "llm worker output shared memory write failed";
    COUNTER_ADD(worker_service_latency_seconds, timer.elapsed_seconds());
  }
}

void WorkerService::ExecuteLLMModel(
    ::google::protobuf::RpcController* controller,
    const proto::LLMForwardRequest* request,
    proto::LLMForwardResponse* response,
    ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    if (!initialized_) {
      controller->SetFailed("Worker is not initialized");
      return;
    }
    if (options_.backend() != "llm") {
      controller->SetFailed("LLM RPC does not match the configured backend");
      return;
    }
    Timer timer;
    if (!request->has_packed_input()) {
      controller->SetFailed("LLM requires a packed input payload");
      return;
    }
    LlmForwardInput input;
    // The native reader checks schema, layout, and domain before preparation.
    if (!input.from_packed_proto(request->packed_input())) {
      controller->SetFailed("Invalid LLM input domain, schema, or layout");
      return;
    }
    ForwardOutput cpu_output;
    std::vector<SpeculativeTokenStats> speculative_token_stats;
    step_llm_model(input, cpu_output, speculative_token_stats);
    if (!encode_llm_output(cpu_output, speculative_token_stats, response)) {
      controller->SetFailed("Failed to serialize LLM model output");
      return;
    }
    COUNTER_ADD(worker_service_latency_seconds, timer.elapsed_seconds());
  });
}

void WorkerService::GetLastLLMStepResult(
    ::google::protobuf::RpcController* controller,
    const proto::Empty* /*request*/,
    proto::LLMForwardResponse* response,
    ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    if (!initialized_) {
      controller->SetFailed("Worker is not initialized");
      return;
    }
    if (options_.backend() != "llm" || !options_.enable_schedule_overlap()) {
      controller->SetFailed(
          "LLM last-step results require the matching overlap backend");
      return;
    }
    auto output = std::move(worker_->get_last_step_result_async()).get();
    ForwardOutput cpu_output;
    std::vector<SpeculativeTokenStats> speculative_token_stats;
    if (output.has_value()) {
      const bool task_pipeline = options_.enable_task_pipeline();
      const bool worker_prepare =
          task_pipeline && worker_->task_pipeline_uses_worker_prepare();
      const bool cpu_ready_result =
          task_pipeline && (!worker_prepare || has_cpu_llm_output(*output));
      collect_llm_output(*output, cpu_output, cpu_ready_result);
      speculative_token_stats =
          record_speculative_metrics(*output,
                                     cpu_output.sample_output.next_tokens,
                                     options_.enable_speculative_decode());
    }
    // Embedding-only and beam-only results are valid completed outputs.
    if (!encode_llm_output(cpu_output, speculative_token_stats, response)) {
      controller->SetFailed("Failed to serialize LLM last-step output");
    }
  });
}

void WorkerService::collect_vlm_output(const ForwardOutput& output,
                                       ForwardOutput& cpu_output) {
  cpu_output.prepared_token = output.prepared_token;
  cpu_output.json_object_errors = output.json_object_errors;
  const auto& sample = output.sample_output;
  const auto& beam = output.beam_search_output;
  auto copy_output_to_host = [&]() {
    // Every D2H, including expert data, runs under the selected output stream.
    cpu_output.expert_load_data =
        safe_to(output.expert_load_data, torch::kCPU, /*non_blocking=*/true);
    cpu_output.sample_output.embeddings =
        safe_to(sample.embeddings,
                torch::dtype(torch::kFloat32).device(torch::kCPU),
                /*non_blocking=*/true);
    auto& mm_embeddings = cpu_output.sample_output.mm_embeddings;
    mm_embeddings.clear();
    mm_embeddings.reserve(sample.mm_embeddings.size());
    for (const auto& sequence : sample.mm_embeddings) {
      std::vector<torch::Tensor> cpu_sequence;
      cpu_sequence.reserve(sequence.size());
      for (const torch::Tensor& embedding : sequence) {
        cpu_sequence.emplace_back(
            safe_to(embedding, torch::kCPU, /*non_blocking=*/true));
      }
      mm_embeddings.emplace_back(std::move(cpu_sequence));
    }
    cpu_output.sample_output.next_tokens =
        safe_to(output.next_tokens_host.defined() ? output.next_tokens_host
                                                  : sample.next_tokens,
                torch::kCPU,
                /*non_blocking=*/true);
    if (cpu_output.sample_output.next_tokens.defined()) {
      cpu_output.sample_output.logprobs =
          safe_to(sample.logprobs, torch::kCPU, /*non_blocking=*/true);
      // Beam results already carry the final tokens and logprobs.
      if (!beam.src_seq_idxes.defined()) {
        cpu_output.sample_output.top_tokens =
            safe_to(sample.top_tokens, torch::kCPU, /*non_blocking=*/true);
        cpu_output.sample_output.top_logprobs =
            safe_to(sample.top_logprobs, torch::kCPU, /*non_blocking=*/true);
      }
    }
    // Keep optional beam results independent of sample next_tokens.
    cpu_output.beam_search_output.src_seq_idxes =
        safe_to(beam.src_seq_idxes, torch::kCPU, /*non_blocking=*/true);
    if (beam.src_seq_idxes.defined()) {
      cpu_output.beam_search_output.out_tokens =
          safe_to(beam.out_tokens, torch::kCPU, /*non_blocking=*/true);
      cpu_output.beam_search_output.out_logprobs =
          safe_to(beam.out_logprobs,
                  torch::dtype(torch::kFloat32).device(torch::kCPU),
                  /*non_blocking=*/true);
    }
  };
  c10::StreamGuard stream_guard = stream_->set_stream_guard();
  if (output.ready_event != nullptr) {
    CHECK(stream_->wait_event(output.ready_event))
        << "wait VLM output ready event failed";
  }
  copy_output_to_host();
  stream_->synchronize();
#if defined(USE_NPU)
  if (options_.enable_schedule_overlap()) {
    DeviceMonitor::get_instance().update_active_activation_memory(
        device_.index());
  }
#endif
}

void WorkerService::step_vlm_model(
    VlmForwardInput& input,
    ForwardOutput& cpu_output,
    std::vector<SpeculativeTokenStats>& speculative_token_stats) {
  speculative_token_stats.clear();
  CHECK(!options_.enable_task_pipeline())
      << "VLM does not support task pipeline";
  const bool worker_rebuilds_host_views =
      input.runtime.input_host_buffer_has_layout &&
      input.runtime.input_host_buffer.defined() &&
      input.runtime.input_host_buffer.device().is_cpu() &&
      input.runtime.input_host_buffer.is_contiguous() &&
      input.runtime.input_host_buffer.numel() > 0 &&
      !input.runtime.device_tensors_ready;
  if (options_.enable_schedule_overlap() && !worker_rebuilds_host_views) {
    // Device-ready inputs can still borrow host views from shared memory.
    stabilize_host_views(input);
  }
  auto future = worker_->step_async(input);
  if (!options_.enable_schedule_overlap()) {
    auto output = std::move(future).get();
    if (output.has_value()) {
      collect_vlm_output(*output, cpu_output);
      speculative_token_stats =
          record_speculative_metrics(*output,
                                     cpu_output.sample_output.next_tokens,
                                     options_.enable_speculative_decode());
    }
    return;
  }
  const auto int_options = torch::TensorOptions().device(torch::kCPU);
  if (worker_->is_driver()) {
    const int32_t sequence_count = get_decode_sequence_count(input);
    cpu_output.sample_output.next_tokens = torch::arange(
        -1, -(sequence_count + 1), -1, int_options.dtype(torch::kInt32));
    std::move(future).deferValue([](auto&&) {});
  }
  cpu_output.expert_load_data =
      torch::zeros({1, 1}, int_options.dtype(torch::kInt64));
}

void WorkerService::poll_vlm_shm(
    std::unique_ptr<ForwardSharedMemoryManager> output_shm_manager) {
  device_.set_device();
  const InputDeviceMaterializationPolicy policy =
      options_.enable_schedule_overlap() && options_.enable_graph()
          ? InputDeviceMaterializationPolicy::DEFER_TO_WORKER_PREPARE
          : InputDeviceMaterializationPolicy::MATERIALIZE_ON_READ;
  while (true) {
    VlmForwardInput input;
    if (!input_shm_manager_->input_read(input, device_, policy)) {
      break;
    }
    Timer timer;
    ForwardOutput cpu_output;
    std::vector<SpeculativeTokenStats> speculative_token_stats;
    step_vlm_model(input, cpu_output, speculative_token_stats);
    const bool write_ok = output_shm_manager->vlm_output_write(
        cpu_output, speculative_token_stats);
    CHECK(write_ok) << "vlm worker output shared memory write failed";
    COUNTER_ADD(worker_service_latency_seconds, timer.elapsed_seconds());
  }
}

void WorkerService::ExecuteVLMModel(
    ::google::protobuf::RpcController* controller,
    const proto::VLMForwardRequest* request,
    proto::VLMForwardResponse* response,
    ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    if (!initialized_) {
      controller->SetFailed("Worker is not initialized");
      return;
    }
    if (options_.backend() != "vlm") {
      controller->SetFailed("VLM RPC does not match the configured backend");
      return;
    }
    Timer timer;
    if (!request->has_packed_input()) {
      controller->SetFailed("VLM requires a packed input payload");
      return;
    }
    VlmForwardInput input;
    // The native reader checks schema, layout, and domain before preparation.
    if (!input.from_packed_proto(request->packed_input())) {
      controller->SetFailed("Invalid VLM input domain, schema, or layout");
      return;
    }
    ForwardOutput cpu_output;
    std::vector<SpeculativeTokenStats> speculative_token_stats;
    step_vlm_model(input, cpu_output, speculative_token_stats);
    if (!encode_vlm_output(cpu_output, speculative_token_stats, response)) {
      controller->SetFailed("Failed to serialize VLM model output");
      return;
    }
    COUNTER_ADD(worker_service_latency_seconds, timer.elapsed_seconds());
  });
}

void WorkerService::GetLastVLMStepResult(
    ::google::protobuf::RpcController* controller,
    const proto::Empty* /*request*/,
    proto::VLMForwardResponse* response,
    ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    if (!initialized_) {
      controller->SetFailed("Worker is not initialized");
      return;
    }
    if (options_.backend() != "vlm" || !options_.enable_schedule_overlap()) {
      controller->SetFailed(
          "VLM last-step results require the matching overlap backend");
      return;
    }
    auto output = std::move(worker_->get_last_step_result_async()).get();
    ForwardOutput cpu_output;
    std::vector<SpeculativeTokenStats> speculative_token_stats;
    if (output.has_value()) {
      collect_vlm_output(*output, cpu_output);
      speculative_token_stats =
          record_speculative_metrics(*output,
                                     cpu_output.sample_output.next_tokens,
                                     options_.enable_speculative_decode());
    }
    // Embedding-only and beam-only results are valid completed outputs.
    if (!encode_vlm_output(cpu_output, speculative_token_stats, response)) {
      controller->SetFailed("Failed to serialize VLM last-step output");
    }
  });
}

void WorkerService::step_rec_model(
    RecForwardInput& input,
    ForwardOutput& cpu_output,
    std::vector<SpeculativeTokenStats>& speculative_token_stats) {
  speculative_token_stats.clear();
  CHECK(!options_.enable_task_pipeline())
      << "REC does not support task pipeline";
  CHECK(!options_.enable_schedule_overlap())
      << "REC does not support schedule overlap";
  // RecWorkerImpl captures input by reference. Wait before releasing this
  // owner.
  auto result = std::move(worker_->step_async(input)).get();
  if (!result.has_value()) {
    return;
  }
  const ForwardOutput& output = *result;
  cpu_output.prepared_token = output.prepared_token;
  cpu_output.json_object_errors = output.json_object_errors;
  const auto& sample = output.sample_output;
  const auto& beam = output.beam_search_output;
  auto copy_output_to_host = [&]() {
    // Every D2H, including expert data, runs under the selected output stream.
    cpu_output.expert_load_data =
        safe_to(output.expert_load_data, torch::kCPU, /*non_blocking=*/true);
    cpu_output.sample_output.embeddings =
        safe_to(sample.embeddings,
                torch::dtype(torch::kFloat32).device(torch::kCPU),
                /*non_blocking=*/true);
    auto& mm_embeddings = cpu_output.sample_output.mm_embeddings;
    mm_embeddings.clear();
    mm_embeddings.reserve(sample.mm_embeddings.size());
    for (const auto& sequence : sample.mm_embeddings) {
      std::vector<torch::Tensor> cpu_sequence;
      cpu_sequence.reserve(sequence.size());
      for (const torch::Tensor& embedding : sequence) {
        cpu_sequence.emplace_back(
            safe_to(embedding, torch::kCPU, /*non_blocking=*/true));
      }
      mm_embeddings.emplace_back(std::move(cpu_sequence));
    }
    cpu_output.sample_output.next_tokens =
        safe_to(output.next_tokens_host.defined() ? output.next_tokens_host
                                                  : sample.next_tokens,
                torch::kCPU,
                /*non_blocking=*/true);
    if (cpu_output.sample_output.next_tokens.defined()) {
      cpu_output.sample_output.logprobs =
          safe_to(sample.logprobs, torch::kCPU, /*non_blocking=*/true);
      // Beam results already carry the final tokens and logprobs.
      if (!beam.src_seq_idxes.defined()) {
        cpu_output.sample_output.top_tokens =
            safe_to(sample.top_tokens, torch::kCPU, /*non_blocking=*/true);
        cpu_output.sample_output.top_logprobs =
            safe_to(sample.top_logprobs, torch::kCPU, /*non_blocking=*/true);
      }
    }
    // REC multi-round results can contain beams without sample next_tokens.
    cpu_output.beam_search_output.src_seq_idxes =
        safe_to(beam.src_seq_idxes, torch::kCPU, /*non_blocking=*/true);
    if (beam.src_seq_idxes.defined()) {
      cpu_output.beam_search_output.out_tokens =
          safe_to(beam.out_tokens, torch::kCPU, /*non_blocking=*/true);
      cpu_output.beam_search_output.out_logprobs =
          safe_to(beam.out_logprobs,
                  torch::dtype(torch::kFloat32).device(torch::kCPU),
                  /*non_blocking=*/true);
    }
  };
  c10::StreamGuard stream_guard = stream_->set_stream_guard();
  if (output.ready_event != nullptr) {
    CHECK(stream_->wait_event(output.ready_event))
        << "wait REC output ready event failed";
  }
  copy_output_to_host();
  stream_->synchronize();
  speculative_token_stats =
      record_speculative_metrics(output,
                                 cpu_output.sample_output.next_tokens,
                                 options_.enable_speculative_decode());
}

void WorkerService::poll_rec_shm(
    std::unique_ptr<ForwardSharedMemoryManager> output_shm_manager) {
  device_.set_device();

  while (true) {
    RecForwardInput input;
    if (!input_shm_manager_->input_read(input, device_)) {
      break;
    }
    Timer timer;
    ForwardOutput cpu_output;
    std::vector<SpeculativeTokenStats> speculative_token_stats;
    step_rec_model(input, cpu_output, speculative_token_stats);
    const bool write_ok = output_shm_manager->rec_output_write(
        cpu_output, speculative_token_stats);
    CHECK(write_ok) << "rec worker output shared memory write failed";
    COUNTER_ADD(worker_service_latency_seconds, timer.elapsed_seconds());
  }
}

void WorkerService::ExecuteRECModel(
    ::google::protobuf::RpcController* controller,
    const proto::RECForwardRequest* request,
    proto::RECForwardResponse* response,
    ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    if (!initialized_) {
      controller->SetFailed("Worker is not initialized");
      return;
    }
    if (options_.backend() != "rec") {
      controller->SetFailed("REC RPC does not match the configured backend");
      return;
    }
    Timer timer;
    if (!request->has_packed_input()) {
      controller->SetFailed("REC requires a packed input payload");
      return;
    }
    RecForwardInput input;
    // The native reader checks schema, layout, and domain before preparation.
    if (!input.from_packed_proto(request->packed_input())) {
      controller->SetFailed("Invalid REC input domain, schema, or layout");
      return;
    }
    ForwardOutput cpu_output;
    std::vector<SpeculativeTokenStats> speculative_token_stats;
    step_rec_model(input, cpu_output, speculative_token_stats);
    if (!encode_rec_output(cpu_output, speculative_token_stats, response)) {
      controller->SetFailed("Failed to serialize REC model output");
      return;
    }
    COUNTER_ADD(worker_service_latency_seconds, timer.elapsed_seconds());
  });
}

void WorkerService::step_dit_model(const DiTForwardInput& input,
                                   ForwardOutput& cpu_output) {
  CHECK(!options_.enable_task_pipeline())
      << "DiT does not support task pipeline";
  auto result = std::move(worker_->step_async(input)).get();
  if (!result.has_value()) {
    return;
  }
  const auto& output = result->dit_forward_output;
  c10::StreamGuard stream_guard = stream_->set_stream_guard();
  if (result->ready_event != nullptr) {
    CHECK(stream_->wait_event(result->ready_event))
        << "wait DiT output ready event failed";
  }
  auto& tensors = cpu_output.dit_forward_output.tensors;
  tensors.reserve(output.tensors.size());
  for (const torch::Tensor& tensor : output.tensors) {
    tensors.emplace_back(safe_to(tensor, torch::kCPU, /*non_blocking=*/true));
  }
  auto& audio_tensors = cpu_output.dit_forward_output.audio_tensors;
  audio_tensors.reserve(output.audio_tensors.size());
  for (const torch::Tensor& tensor : output.audio_tensors) {
    audio_tensors.emplace_back(
        safe_to(tensor, torch::kCPU, /*non_blocking=*/true));
  }
  cpu_output.dit_forward_output.text_output = output.text_output;
  stream_->synchronize();
}

void WorkerService::poll_dit_shm(
    std::unique_ptr<ForwardSharedMemoryManager> output_shm_manager) {
  device_.set_device();

  while (true) {
    DiTForwardInput input;
    if (!input_shm_manager_->input_read(input)) {
      break;
    }
    Timer timer;
    ForwardOutput cpu_output;
    step_dit_model(input, cpu_output);
    const bool write_ok =
        output_shm_manager->dit_output_write(cpu_output.dit_forward_output);
    CHECK(write_ok) << "dit worker output shared memory write failed";
    COUNTER_ADD(worker_service_latency_seconds, timer.elapsed_seconds());
  }
}

void WorkerService::ExecuteDITModel(
    ::google::protobuf::RpcController* controller,
    const proto::DITForwardRequest* request,
    proto::DITForwardResponse* response,
    ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    if (!initialized_) {
      controller->SetFailed("Worker is not initialized");
      return;
    }
    if (options_.backend() != "dit") {
      controller->SetFailed("DIT RPC does not match the configured backend");
      return;
    }
    Timer timer;
    if (!request->has_packed_input()) {
      controller->SetFailed("DIT requires a packed input payload");
      return;
    }
    DiTForwardInput input;
    // The native reader checks schema, layout, and domain before preparation.
    if (!input.from_packed_proto(request->packed_input())) {
      controller->SetFailed("Invalid DIT input domain, schema, or layout");
      return;
    }
    ForwardOutput cpu_output;
    step_dit_model(input, cpu_output);
    if (!encode_dit_output(cpu_output, response)) {
      controller->SetFailed("Failed to serialize DIT model output");
      return;
    }
    COUNTER_ADD(worker_service_latency_seconds, timer.elapsed_seconds());
  });
}

void WorkerService::ProcessGroupTest(
    ::google::protobuf::RpcController* controller,
    const proto::Empty* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    auto future = worker_->process_group_test_async();
    std::move(future).get();
    response->set_ok(true);
  });
  return;
}

void WorkerService::ProfileDeviceMemory(
    ::google::protobuf::RpcController* controller,
    const proto::Empty* request,
    proto::DeviceMemory* response,
    ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    auto future = worker_->estimate_kv_cache_capacity_async();
    std::tuple<int64_t, int64_t> result = std::move(future).get();
    response->set_available_memory(std::get<0>(result));
    response->set_total_memory(std::get<1>(result));
  });
  return;
}

void WorkerService::AllocateKVCache(
    ::google::protobuf::RpcController* controller,
    const proto::AllocateKVCacheRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    const KVCacheShape kv_cache_shape =
        KVCacheShape::from_proto(request->kv_cache_shape());
    auto future = worker_->allocate_kv_cache_async(kv_cache_shape);
    bool status = std::move(future).get();
    response->set_ok(status);
  });
  return;
}

void WorkerService::SetSpeculativeValidateTimePredictor(
    ::google::protobuf::RpcController* controller,
    const proto::SpeculativeValidateTimePredictor* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    SpeculativeProfileRegistry::ValidateTimePredictor predictor;
    predictor.intercept_ms = request->intercept_ms();
    predictor.query_token_ms = request->query_token_ms();
    predictor.query_prefix_ms = request->query_prefix_ms();
    response->set_ok(
        worker_->set_speculative_validate_time_predictor(predictor));
  });
  return;
}

void WorkerService::AllocateKVCacheWithTransfer(
    ::google::protobuf::RpcController* controller,
    const proto::AllocateKVCacheRequest* req,
    proto::Status* resp,
    ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    const KVCacheShape kv_cache_shape =
        KVCacheShape::from_proto(req->kv_cache_shape());
    auto future =
        worker_->allocate_kv_cache_with_transfer_async(kv_cache_shape);
    bool status = std::move(future).get();
    resp->set_ok(status);
  });
  return;
}

void WorkerService::GetCacheInfo(::google::protobuf::RpcController* controller,
                                 const proto::Empty* req,
                                 proto::CacheInfo* resp,
                                 ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    uint64_t cluster_id;
    std::string addr;
    uint16_t listen_port;
    worker_->get_cache_info(cluster_id, addr, listen_port);
    resp->set_cluster_id(cluster_id);
    resp->set_addr(addr);
    resp->set_listen_port(listen_port);
  });
  return;
}

void WorkerService::PullKVCache(::google::protobuf::RpcController* controller,
                                const proto::PullKVCacheRequest* req,
                                proto::Status* resp,
                                ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    std::vector<KVTransferMapping> mappings;
    mappings.reserve(req->mappings_size());
    for (const proto::KVTransferMapping& proto_mapping : req->mappings()) {
      KVTransferMapping mapping;
      mapping.group_id = proto_mapping.group_id();
      mapping.local_ids.assign(proto_mapping.local_ids().begin(),
                               proto_mapping.local_ids().end());
      mapping.remote_ids.assign(proto_mapping.remote_ids().begin(),
                                proto_mapping.remote_ids().end());
      mappings.emplace_back(std::move(mapping));
    }
    auto future =
        worker_->pull_kv_blocks_async(req->cluster_id(), req->addr(), mappings);
    bool status = std::move(future).get();
    resp->set_ok(status);
  });
  return;
}

void WorkerService::TransferBlocks(
    ::google::protobuf::RpcController* controller,
    const proto::BlockTransferInfos* req,
    proto::TransferStatus* resp,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  std::vector<BlockTransferInfo> block_transfer_info;
  uint64_t batch_id = 0;
  if (!proto_to_block_transfer_info(*req, block_transfer_info, &batch_id)) {
    resp->set_success_cnt(0);
    controller->SetFailed("Invalid block transfer request");
    LOG(ERROR) << "Invalid block transfer request.";
    return;
  }

  resp->set_success_cnt(
      worker_->transfer_kv_blocks(batch_id, block_transfer_info));
  return;
}

void WorkerService::PrefetchFromStorage(
    google::protobuf::RpcController* controller,
    const proto::PrefetchRequest* req,
    proto::Status* resp,
    google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  brpc::Controller* cntl = static_cast<brpc::Controller*>(controller);

  StoragePrefetchRequest request;
  if (!request.from_proto(*req)) {
    resp->set_ok(false);
    LOG(ERROR) << "Invalid Mooncake prefetch request.";
    return;
  }

  auto session = std::make_shared<WorkerPrefetchSession>(
      worker_.get(), &copy_threadpool_, std::move(request));

  brpc::StreamId stream_id;
  brpc::StreamOptions stream_options;
  stream_options.idle_timeout_ms = -1;
  stream_options.handler = session.get();
  session->retain();
  if (brpc::StreamAccept(&stream_id, *cntl, &stream_options) != 0) {
    session->release();
    resp->set_ok(false);
    LOG(ERROR) << "Failed to accept stream!";
    return;
  }

  resp->set_ok(true);
  if (google::protobuf::Closure* response_done = done_guard.release()) {
    response_done->Run();
  }
  session->start(stream_id);
}

void WorkerService::LinkCluster(::google::protobuf::RpcController* controller,
                                const proto::ClusterInfo* req,
                                proto::Status* resp,
                                ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    std::vector<uint64_t> cluster_ids(req->cluster_ids().begin(),
                                      req->cluster_ids().end());
    std::vector<std::string> addrs(req->addrs().begin(), req->addrs().end());
    std::vector<uint16_t> ports(req->ports().begin(), req->ports().end());

    bool status = worker_->link_cluster(cluster_ids, addrs, ports);
    resp->set_ok(status);
  });
  return;
}

void WorkerService::UnlinkCluster(::google::protobuf::RpcController* controller,
                                  const proto::ClusterInfo* req,
                                  proto::Status* resp,
                                  ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    std::vector<uint64_t> cluster_ids(req->cluster_ids().begin(),
                                      req->cluster_ids().end());
    std::vector<std::string> addrs(req->addrs().begin(), req->addrs().end());
    std::vector<uint16_t> ports(req->ports().begin(), req->ports().end());

    bool status = worker_->unlink_cluster(cluster_ids, addrs, ports);
    resp->set_ok(status);
  });
  return;
}

void WorkerService::LinkP2P(::google::protobuf::RpcController* controller,
                            const proto::P2PLinkWorkerRequest* req,
                            proto::Status* resp,
                            ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    bool status = worker_->link_p2p(req->remote_addr());
    resp->set_ok(status);
  });
  return;
}

void WorkerService::UnlinkP2P(::google::protobuf::RpcController* controller,
                              const proto::P2PLinkWorkerRequest* req,
                              proto::Status* resp,
                              ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    bool status = worker_->unlink_p2p(req->remote_addr());
    resp->set_ok(status);
  });
  return;
}

void WorkerService::Sleep(::google::protobuf::RpcController* controller,
                          const proto::SleepRequest* req,
                          proto::Status* resp,
                          ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    bool status = worker_->sleep(MasterStatus(req->master_status()));
    resp->set_ok(status);
  });

  return;
}

void WorkerService::Wakeup(::google::protobuf::RpcController* controller,
                           const proto::WakeupRequest* req,
                           proto::Status* resp,
                           ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    WakeupOptions options;
    options.master_status = MasterStatus(req->master_status());
    options.remote_addrs.assign(req->remote_addrs().begin(),
                                req->remote_addrs().end());
    // Unmarshal weight segments
    for (const auto& seg_list : req->src_weight_segments()) {
      std::vector<WeightSegment> segments;
      segments.reserve(seg_list.segments_size());
      for (const auto& proto_seg : seg_list.segments()) {
        segments.emplace_back(proto_seg.offset(), proto_seg.size());
      }
      options.src_weight_segments.push_back(std::move(segments));
    }
    bool status = worker_->wakeup(options);
    resp->set_ok(status);
  });

  return;
}

void WorkerService::StartProfile(::google::protobuf::RpcController* controller,
                                 const proto::Empty* req,
                                 proto::Status* resp,
                                 ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    bool status = worker_->start_profile();
    resp->set_ok(status);
  });

  return;
}

void WorkerService::StopProfile(::google::protobuf::RpcController* controller,
                                const proto::Empty* req,
                                proto::Status* resp,
                                ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    bool status = worker_->stop_profile();
    resp->set_ok(status);
  });

  return;
}

void WorkerService::GetActiveActivationMemory(
    ::google::protobuf::RpcController* controller,
    const proto::Empty* req,
    proto::ActivationMemory* resp,
    ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    auto future = worker_->get_active_activation_memory_async();
    int64_t active_activation_memory = std::move(future).get();
    resp->set_active_activation_memory(active_activation_memory);
  });
  return;
}
}  // namespace xllm
