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
#include "worker_service.h"

#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include <brpc/stream.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "common/device_monitor.h"
#include "common/global_flags.h"
#include "common/metrics.h"
#include "common/types.h"
#include "core/distributed_runtime/comm_channel.h"
#include "core/framework/config/eplb_config.h"
#include "core/framework/speculative/metrics.h"
#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/model/model_input_params.h"
#include "framework/request/sequence.h"
#include "framework/sampling/sampling_params.h"
#include "runtime/forward_params.h"
#include "runtime/params_utils.h"
#include "runtime/rec_forward_params.h"
#include "runtime/vlm_forward_params.h"
#include "util/timer.h"

namespace xllm {
namespace {

template <typename Input>
int32_t get_num_decode_seqs_for_schedule_overlap(const Input& input) {
  if (input.sampling_params.sample_idxes.defined()) {
    return static_cast<int32_t>(input.sampling_params.sample_idxes.size(0));
  }

  if (input.input_host_sample_count != kUnknownPackedSampleCount) {
    return input.input_host_sample_count;
  }

  if (!input.runtime.input_host_buffer_has_layout) {
    return 0;
  }

  Input unpacked_input;
  const bool unpacked =
      detail::unpack_from_input_host_buffer(input,
                                            torch::Device(torch::kCPU),
                                            torch::kFloat32,
                                            unpacked_input,
                                            false);
  if (!unpacked || !unpacked_input.sampling_params.sample_idxes.defined()) {
    return 0;
  }
  return static_cast<int32_t>(
      unpacked_input.sampling_params.sample_idxes.size(0));
}

torch::Tensor clone_cpu_tensor_view(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return tensor;
  }
  CHECK(tensor.device().is_cpu()) << "expected a CPU tensor view";
  return tensor.contiguous().clone();
}

template <typename Input>
void stabilize_schedule_overlap_host_views(Input& input) {
  input.token_ids_host = clone_cpu_tensor_view(input.token_ids_host);
  input.positions_host = clone_cpu_tensor_view(input.positions_host);
  input.input_params.attention.host.block_tables =
      clone_cpu_tensor_view(input.input_params.attention.host.block_tables);
}

bool has_cpu_serialization_inputs(const ForwardOutput& output) {
  const auto on_cpu = [](const torch::Tensor& tensor) {
    return !tensor.defined() || tensor.device().is_cpu();
  };
  const auto& sample = output.sample_output;
  const auto& beam = output.beam_search_output;
  return on_cpu(output.next_tokens_host.defined() ? output.next_tokens_host
                                                  : sample.next_tokens) &&
         on_cpu(output.expert_load_data) && on_cpu(sample.embeddings) &&
         on_cpu(sample.logprobs) && on_cpu(sample.top_tokens) &&
         on_cpu(sample.top_logprobs) && on_cpu(beam.src_seq_idxes) &&
         on_cpu(beam.out_tokens) && on_cpu(beam.out_logprobs) &&
         std::all_of(output.dit_forward_output.tensors.begin(),
                     output.dit_forward_output.tensors.end(),
                     on_cpu);
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

std::vector<SpeculativeTokenStats>
WorkerService::record_speculative_metrics_from_output(
    const ForwardOutput& forward_output) {
  if (forward_output.is_graph_warmup || !options_.enable_speculative_decode()) {
    return {};
  }
  const torch::Tensor& next_tokens = forward_output.sample_output.next_tokens;
  // A verify result is marked solely by a non-empty row layout; ordinary
  // decode emits 1-D tokens.
  if (forward_output.spec_verify_layouts.empty()) {
    CHECK(!next_tokens.defined() || next_tokens.dim() != 2)
        << "speculative verify output is missing row metadata";
    return {};
  }
  // Schedule-overlap MTP hands back a device next_tokens (the device copy is
  // consumed by the next step's input update) plus a host snapshot; prefer
  // the snapshot so metrics never issue a second D2H of the same matrix.
  const torch::Tensor& metric_tokens = forward_output.next_tokens_host.defined()
                                           ? forward_output.next_tokens_host
                                           : next_tokens;
  SpecMetrics metrics =
      make_spec_metrics(metric_tokens, forward_output.spec_verify_layouts);
  record_spec_metrics(metrics);
  return std::move(metrics.sequence_stats);
}

void WorkerService::set_worker(std::unique_ptr<Worker> worker) {
  worker_ = std::move(worker);
  initialized_ = true;
}

void WorkerService::step(const DiTForwardInput& input,
                         std::vector<torch::Tensor>& tensors,
                         std::vector<std::string>& text_output) {
  auto result = std::move(worker_->step_async(input)).get();
  if (!result.has_value()) {
    return;
  }
  const DiTForwardOutput& output = result->dit_forward_output;
  c10::StreamGuard stream_guard = stream_->set_stream_guard();
  tensors.reserve(output.tensors.size());
  for (const torch::Tensor& tensor : output.tensors) {
    tensors.emplace_back(safe_to(tensor, torch::kCPU, /*non_blocking=*/true));
  }
  text_output = output.text_output;
  stream_->synchronize();
}

template <typename Input>
void WorkerService::step(
    Input& fwd_input,
    torch::Tensor& next_tokens,
    torch::Tensor& logprobs,
    torch::Tensor& top_tokens,
    torch::Tensor& top_logprobs,
    torch::Tensor& embeddings,
    std::vector<std::vector<torch::Tensor>>& mm_embeddings,
    std::vector<SpeculativeTokenStats>& speculative_token_stats,
    std::vector<torch::Tensor>& dit_images,
    std::vector<std::string>& dit_text_output,
    torch::Tensor& expert_load_data,
    int64_t& prepared_token,
    torch::Tensor& src_seq_idxes,
    torch::Tensor& out_tokens,
    torch::Tensor& out_logprobs,
    std::vector<JsonObjectOutputError>& json_object_errors) {
  speculative_token_stats.clear();
  const bool use_default_stream =
      !options_.enable_schedule_overlap() && options_.backend() == "llm";
  const bool task_pipeline = options_.enable_task_pipeline();
  const bool cpu_ready_task_result =
      task_pipeline && !worker_->task_pipeline_uses_worker_prepare();
  const bool worker_owned_packed_input =
      task_pipeline && worker_->task_pipeline_uses_worker_prepare() &&
      fwd_input.runtime.input_host_buffer_has_layout &&
      fwd_input.runtime.input_host_buffer.defined();
  if (options_.enable_schedule_overlap() &&
      (!task_pipeline || (worker_->task_pipeline_uses_worker_prepare() &&
                          !worker_owned_packed_input))) {
    // A worker-owned task slot retains the packed host buffer through Consume.
    // Its prepare path reconstructs host views from that owner, so cloning the
    // three SHM views here would only be overwritten by task submit.
    stabilize_schedule_overlap_host_views(fwd_input);
  }
  // execute model
  auto future = worker_->step_async(fwd_input);
  if (!options_.enable_schedule_overlap()) {
    auto forward_outputs = std::move(future).get();
    // convert ForwardOutput to proto::ForwardOutput which contain Tokens.
    if (forward_outputs) {
      DCHECK(forward_outputs.has_value()) << "Failed to execute model";
      const ForwardOutput& forward_output = forward_outputs.value();
      const auto& sample_output = forward_output.sample_output;
      const auto& beam_search_output = forward_output.beam_search_output;
      const auto& dit_forward_output = forward_output.dit_forward_output;
      expert_load_data = safe_to(forward_output.expert_load_data,
                                 torch::kCPU,
                                 /*non_blocking=*/true);
      prepared_token = forward_output.prepared_token;
      json_object_errors = forward_output.json_object_errors;

      {
        auto copy_output_to_host = [&]() {
          // only driver worker (rank=0) need to fill this
          // [num_seq, ..., embed_dim] FloatTensor
          embeddings =
              safe_to(sample_output.embeddings,
                      torch::dtype(torch::kFloat32).device(torch::kCPU),
                      /*non_blocking=*/true);

          mm_embeddings.clear();
          mm_embeddings.reserve(sample_output.mm_embeddings.size());
          for (const auto& seq_mm_embeddings : sample_output.mm_embeddings) {
            std::vector<torch::Tensor> seq_out;
            seq_out.reserve(seq_mm_embeddings.size());
            for (const auto& mm_embedding : seq_mm_embeddings) {
              seq_out.emplace_back(
                  safe_to(mm_embedding, torch::kCPU, /*non_blocking=*/true));
            }
            mm_embeddings.emplace_back(std::move(seq_out));
          }

          dit_images.clear();
          dit_images.reserve(dit_forward_output.tensors.size());
          for (auto dit_image : dit_forward_output.tensors) {
            dit_images.emplace_back(
                safe_to(dit_image, torch::kCPU, /*non_blocking=*/true));
          }
          dit_text_output = dit_forward_output.text_output;

          // [num_seq]
          next_tokens = safe_to(sample_output.next_tokens,
                                torch::kCPU,
                                /*non_blocking=*/true);
          if (next_tokens.defined()) {
            // [num_seq]
            logprobs = safe_to(sample_output.logprobs,
                               torch::kCPU,
                               /*non_blocking=*/true);

            if (!beam_search_output.src_seq_idxes.defined()) {
              // beam search kernel will provide final tokens/logprobs in beam
              // search output, so keep top_tokens/top_logprobs undefined to
              // avoid returning them.
              // [num_seq, topk]
              top_tokens = safe_to(sample_output.top_tokens,
                                   torch::kCPU,
                                   /*non_blocking=*/true);
              // [num_seq, topk]
              top_logprobs = safe_to(sample_output.top_logprobs,
                                     torch::kCPU,
                                     /*non_blocking=*/true);
            }
          }

          // beam search output
          // [num_seq]
          src_seq_idxes = safe_to(beam_search_output.src_seq_idxes,
                                  torch::kCPU,
                                  /*non_blocking=*/true);
          if (src_seq_idxes.defined()) {
            // [num_seq]
            out_tokens = safe_to(beam_search_output.out_tokens,
                                 torch::kCPU,
                                 /*non_blocking=*/true);
            // [num_seq]
            out_logprobs =
                safe_to(beam_search_output.out_logprobs,
                        torch::dtype(torch::kFloat32).device(torch::kCPU),
                        /*non_blocking=*/true);
          }
        };
        if (cpu_ready_task_result) {
          // The pipeline Future completes after Consume produces CPU results.
          copy_output_to_host();
        } else {
          if (use_default_stream) {
            copy_output_to_host();
            device_.synchronize_default_stream();
          } else {
            c10::StreamGuard stream_guard = stream_->set_stream_guard();
            copy_output_to_host();
            stream_->synchronize();
          }
        }
        speculative_token_stats =
            record_speculative_metrics_from_output(forward_output);
      }
    }
  } else {
    if (task_pipeline) {
      // Never publish placeholders before the Worker accepts and owns input.
      const auto ack = std::move(future).get();
      CHECK(!ack.has_value()) << "Two-Slot Step must return PrepareAck.";
    }
    auto int_options = torch::TensorOptions().device(torch::kCPU);
    if (worker_->is_driver()) {
      // construct fake output tensor
      int32_t num_decode_seqs =
          get_num_decode_seqs_for_schedule_overlap(fwd_input);
      // Negative values identify rows in the previous sampling output.
      next_tokens = torch::arange(
          -1, -1 * (num_decode_seqs + 1), -1, int_options.dtype(torch::kInt32));
      if (!task_pipeline) {
        std::move(future).deferValue([](auto&&) {});
      }
    }
    expert_load_data = torch::zeros({1, 1}, int_options.dtype(torch::kInt64));
  }
}

void WorkerService::create_polling_shm_thread(
    std::unique_ptr<ForwardSharedMemoryManager> input_shm_manager,
    std::unique_ptr<ForwardSharedMemoryManager> output_shm_manager) {
  CHECK(!polling_thread_) << "SHM polling thread has already been created.";
  input_shm_manager_ = std::move(input_shm_manager);
  polling_thread_ = std::make_unique<std::thread>(
      [this, output_shm_manager = std::move(output_shm_manager)]() mutable {
        device_.set_device();
        Timer timer;
        while (true) {
          std::variant<LlmForwardInput,
                       VlmForwardInput,
                       RecForwardInput,
                       DiTForwardInput>
              input;
          if (options_.backend() == "dit") {
            input.emplace<DiTForwardInput>();
          } else if (options_.backend() == "rec") {
            input.emplace<RecForwardInput>();
          } else if (options_.backend() == "vlm") {
            input.emplace<VlmForwardInput>();
          }
          // NPU graph task updates cannot safely overlap an H2D enqueue from
          // the SHM polling thread. Keep scheduler overlap, but defer device
          // materialization to WorkerImpl's ordered prepare stream.
          const InputDeviceMaterializationPolicy materialization_policy =
              options_.enable_task_pipeline() ||
                      (options_.enable_schedule_overlap() &&
                       options_.enable_graph())
                  ? InputDeviceMaterializationPolicy::DEFER_TO_WORKER_PREPARE
                  : InputDeviceMaterializationPolicy::MATERIALIZE_ON_READ;
          const bool input_ready = std::visit(
              [&](auto& typed_input) {
                using Input = std::decay_t<decltype(typed_input)>;
                if constexpr (std::is_same_v<Input, DiTForwardInput>) {
                  return input_shm_manager_->input_read(typed_input);
                } else {
                  return input_shm_manager_->input_read(
                      typed_input, device_, materialization_policy);
                }
              },
              input);
          if (!input_ready) {
            break;
          }
          timer.reset();
          // model output variables
          torch::Tensor next_tokens;
          torch::Tensor logprobs;
          torch::Tensor top_tokens;
          torch::Tensor top_logprobs;
          torch::Tensor embeddings;
          std::vector<std::vector<torch::Tensor>> mm_embeddings;
          std::vector<SpeculativeTokenStats> speculative_token_stats;
          std::vector<torch::Tensor> dit_images;
          std::vector<std::string> dit_text_output;
          torch::Tensor expert_load_data;
          int64_t prepared_token = -1;

          // beam search kernel output
          torch::Tensor src_seq_idxes;
          torch::Tensor out_tokens;
          torch::Tensor out_logprobs;
          std::vector<JsonObjectOutputError> json_object_errors;

          std::visit(
              [&](auto& typed_input) {
                using Input = std::decay_t<decltype(typed_input)>;
                if constexpr (std::is_same_v<Input, DiTForwardInput>) {
                  step(typed_input, dit_images, dit_text_output);
                } else {
                  step(typed_input,
                       next_tokens,
                       logprobs,
                       top_tokens,
                       top_logprobs,
                       embeddings,
                       mm_embeddings,
                       speculative_token_stats,
                       dit_images,
                       dit_text_output,
                       expert_load_data,
                       prepared_token,
                       src_seq_idxes,
                       out_tokens,
                       out_logprobs,
                       json_object_errors);
                }
              },
              input);

          const bool shm_write_ok =
              output_shm_manager->raw_output_write(next_tokens,
                                                   logprobs,
                                                   top_tokens,
                                                   top_logprobs,
                                                   embeddings,
                                                   mm_embeddings,
                                                   speculative_token_stats,
                                                   dit_images,
                                                   dit_text_output,
                                                   expert_load_data,
                                                   prepared_token,
                                                   src_seq_idxes,
                                                   out_tokens,
                                                   out_logprobs,
                                                   json_object_errors);
          CHECK(shm_write_ok) << "Worker output shared memory write failed.";
          COUNTER_ADD(worker_service_latency_seconds, timer.elapsed_seconds());
        }
      });
  return;
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
  uint64_t batch_id = proto_to_block_transfer_info(*req, block_transfer_info);

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
  if (!proto_to_storage_prefetch_request(*req, &request)) {
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

void WorkerService::ExecuteModel(::google::protobuf::RpcController* controller,
                                 const proto::ForwardInput* pb_forward_input,
                                 proto::ForwardOutput* pb_forward_output,
                                 ::google::protobuf::Closure* done) {
  threadpool_->schedule([this,
                         controller,
                         pb_forward_input,
                         pb_forward_output,
                         done]() mutable {
    brpc::ClosureGuard done_guard(done);
    // convert proto::ForwardInput to LlmForwardInput

    Timer timer;
    std::variant<LlmForwardInput,
                 VlmForwardInput,
                 RecForwardInput,
                 DiTForwardInput>
        input;
    if (options_.backend() == "dit") {
      input.emplace<DiTForwardInput>();
    } else if (options_.backend() == "rec") {
      input.emplace<RecForwardInput>();
    } else if (options_.backend() == "vlm") {
      input.emplace<VlmForwardInput>();
    }
    if (!pb_forward_input->has_packed_input()) {
      controller->SetFailed("LlmForwardInput requires a packed input payload");
      return;
    }
    const proto::PackedForwardInput& packed_input =
        pb_forward_input->packed_input();
    // Dispatch follows the configured domain, including token-only VLM
    // decode. A mismatched domain is rejected before tensor preparation.
    const bool valid_input = std::visit(
        [&](auto& typed_input) {
          using Input = std::decay_t<decltype(typed_input)>;
          if constexpr (std::is_same_v<Input, DiTForwardInput>) {
            return packed_proto_to_dit_forward_input(packed_input, typed_input);
          } else if constexpr (std::is_same_v<Input, RecForwardInput>) {
            return packed_proto_to_rec_forward_input(
                packed_input, typed_input, device_, stream_.get());
          } else if constexpr (std::is_same_v<Input, VlmForwardInput>) {
            return packed_proto_to_vlm_forward_input(
                packed_input, typed_input, device_, stream_.get());
          } else {
            return packed_proto_to_forward_input(
                packed_input, typed_input, device_, stream_.get());
          }
        },
        input);
    if (!valid_input) {
      controller->SetFailed("Invalid forward input domain, schema, or layout");
      return;
    }

    // model output
    torch::Tensor next_tokens;
    torch::Tensor logprobs;
    torch::Tensor top_tokens;
    torch::Tensor top_logprobs;
    torch::Tensor embeddings;
    std::vector<std::vector<torch::Tensor>> mm_embeddings;
    std::vector<SpeculativeTokenStats> speculative_token_stats;
    std::vector<torch::Tensor> dit_images;
    std::vector<std::string> dit_text_output;
    torch::Tensor expert_load_data;
    int64_t prepared_token = -1;
    // beam search kernel output
    torch::Tensor src_seq_idxes;
    torch::Tensor out_tokens;
    torch::Tensor out_logprobs;
    std::vector<JsonObjectOutputError> json_object_errors;

    std::visit(
        [&](auto& typed_input) {
          using Input = std::decay_t<decltype(typed_input)>;
          if constexpr (std::is_same_v<Input, DiTForwardInput>) {
            step(typed_input, dit_images, dit_text_output);
          } else {
            step(typed_input,
                 next_tokens,
                 logprobs,
                 top_tokens,
                 top_logprobs,
                 embeddings,
                 mm_embeddings,
                 speculative_token_stats,
                 dit_images,
                 dit_text_output,
                 expert_load_data,
                 prepared_token,
                 src_seq_idxes,
                 out_tokens,
                 out_logprobs,
                 json_object_errors);
          }
        },
        input);
    // convert to proto output
    forward_output_to_proto(next_tokens,
                            logprobs,
                            top_tokens,
                            top_logprobs,
                            embeddings,
                            mm_embeddings,
                            speculative_token_stats,
                            expert_load_data,
                            prepared_token,
                            src_seq_idxes,
                            out_tokens,
                            out_logprobs,
                            dit_images,
                            dit_text_output,
                            json_object_errors,
                            pb_forward_output);
    COUNTER_ADD(worker_service_latency_seconds, timer.elapsed_seconds());
  });
}

void WorkerService::GetLastStepResult(
    ::google::protobuf::RpcController* controller,
    const proto::Empty* req,
    proto::ForwardOutput* pb_forward_output,
    ::google::protobuf::Closure* done) {
  threadpool_->schedule(
      [this, controller, req, pb_forward_output, done]() mutable {
        brpc::ClosureGuard done_guard(done);
        const bool use_default_stream =
            !options_.enable_schedule_overlap() && options_.backend() == "llm";

        const bool task_pipeline = options_.enable_task_pipeline();
        const bool worker_task_pipeline =
            task_pipeline && worker_->task_pipeline_uses_worker_prepare();
        auto future = worker_->get_last_step_result_async();
        auto forward_outputs = std::move(future).get();
        if (forward_outputs) {
          const ForwardOutput& forward_output = forward_outputs.value();
          const auto& sample_output = forward_output.sample_output;
          // Worker-owned Consume retires the producer event before fulfilling
          // this Future. A CPU-only result needs no second event wait/stream
          // fence. Device-valued optional outputs still require the D2H path.
          const bool cpu_ready_worker_result =
              worker_task_pipeline &&
              has_cpu_serialization_inputs(forward_output);
          const bool cpu_ready_task_result =
              (task_pipeline && !worker_task_pipeline) ||
              cpu_ready_worker_result;
          int64_t prepared_token = forward_output.prepared_token;
          const auto& beam_search_output = forward_output.beam_search_output;
          torch::Tensor expert_load_data;
          torch::Tensor embeddings;
          torch::Tensor next_tokens;
          torch::Tensor logprobs;
          torch::Tensor top_tokens;
          torch::Tensor top_logprobs;
          torch::Tensor src_seq_idxes;
          torch::Tensor out_tokens;
          torch::Tensor out_logprobs;
          std::vector<SpeculativeTokenStats> speculative_token_stats;
          std::vector<torch::Tensor> dit_images;
          std::vector<std::string> dit_text_output;
          auto copy_output_to_host = [&]() {
            expert_load_data = safe_to(forward_output.expert_load_data,
                                       torch::kCPU,
                                       /*non_blocking=*/true);

            // [num_seq, ..., embed_dim]
            embeddings = safe_to(sample_output.embeddings,
                                 torch::kCPU,
                                 /*non_blocking=*/true);
            embeddings = safe_to(embeddings,
                                 torch::kFloat32,
                                 /*non_blocking=*/true);

            dit_images.reserve(
                forward_output.dit_forward_output.tensors.size());
            for (auto image : forward_output.dit_forward_output.tensors) {
              dit_images.emplace_back(image);
            }
            dit_text_output = forward_output.dit_forward_output.text_output;

            // [num_seq]
            next_tokens = safe_to(forward_output.next_tokens_host.defined()
                                      ? forward_output.next_tokens_host
                                      : sample_output.next_tokens,
                                  torch::kCPU,
                                  /*non_blocking=*/true);
            if (next_tokens.defined() ||
                ::xllm::EPLBConfig::get_instance().enable_eplb()) {
              // [num_seq] FloatTensor
              logprobs = safe_to(sample_output.logprobs,
                                 torch::kCPU,
                                 /*non_blocking=*/true);
              // [num_seq, topk]
              top_tokens = safe_to(sample_output.top_tokens,
                                   torch::kCPU,
                                   /*non_blocking=*/true);
              // [num_seq, topk]
              top_logprobs = safe_to(sample_output.top_logprobs,
                                     torch::kCPU,
                                     /*non_blocking=*/true);
              // [num_seq]
              src_seq_idxes = safe_to(beam_search_output.src_seq_idxes,
                                      torch::kCPU,
                                      /*non_blocking=*/true);
              // [num_seq]
              out_tokens = safe_to(beam_search_output.out_tokens,
                                   torch::kCPU,
                                   /*non_blocking=*/true);
              // [num_seq]
              out_logprobs =
                  safe_to(beam_search_output.out_logprobs,
                          torch::dtype(torch::kFloat32).device(torch::kCPU),
                          /*non_blocking=*/true);
            }
          };

          if (cpu_ready_task_result) {
            // The pipeline Future completes after Consume produces CPU results.
            copy_output_to_host();
#if defined(USE_NPU)
            if (cpu_ready_worker_result && !use_default_stream) {
              DeviceMonitor::get_instance().update_active_activation_memory(
                  device_.index());
            }
#endif
          } else {
            if (use_default_stream) {
              copy_output_to_host();
            } else {
              c10::StreamGuard stream_guard = stream_->set_stream_guard();
              if (forward_output.ready_event != nullptr) {
                CHECK(stream_->wait_event(forward_output.ready_event))
                    << "wait forward output ready event failed.";
              }
              copy_output_to_host();
            }
            if (use_default_stream) {
              device_.synchronize_default_stream();
            } else {
              stream_->synchronize();
#if defined(USE_NPU)
              DeviceMonitor::get_instance().update_active_activation_memory(
                  device_.index());
#endif
            }
          }
          speculative_token_stats =
              record_speculative_metrics_from_output(forward_output);

          if (next_tokens.defined() || !dit_images.empty() ||
              !dit_text_output.empty() ||
              ::xllm::EPLBConfig::get_instance().enable_eplb() ||
              !forward_output.json_object_errors.empty()) {
            const std::vector<std::vector<torch::Tensor>> mm_embeddings;
            forward_output_to_proto(next_tokens,
                                    logprobs,
                                    top_tokens,
                                    top_logprobs,
                                    embeddings,
                                    mm_embeddings,
                                    speculative_token_stats,
                                    expert_load_data,
                                    prepared_token,
                                    src_seq_idxes,
                                    out_tokens,
                                    out_logprobs,
                                    dit_images,
                                    dit_text_output,
                                    forward_output.json_object_errors,
                                    pb_forward_output);
          }
        }
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
