/* Copyright 2025-2026 The xLLM Authors.

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
#include "worker_service.h"

#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <vector>

#include "common/device_monitor.h"
#include "common/global_flags.h"
#include "common/metrics.h"
#include "common/types.h"
#include "core/distributed_runtime/comm_channel.h"
#include "core/framework/config/eplb_config.h"
#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/model/model_input_params.h"
#include "framework/request/sequence.h"
#include "framework/sampling/sampling_params.h"
#include "runtime/forward_params.h"
#include "runtime/params_utils.h"
#include "util/timer.h"

namespace xllm {
namespace {

int32_t get_num_decode_seqs_for_schedule_overlap(const ForwardInput& input) {
  if (input.sampling_params.sample_idxes.defined()) {
    return static_cast<int32_t>(input.sampling_params.sample_idxes.size(0));
  }

  if (!input.input_host_buffer_has_layout) {
    return 0;
  }

  ForwardInput unpacked_input;
  const bool unpacked = detail::unpack_from_input_host_buffer(
      input, torch::Device(torch::kCPU), unpacked_input);
  if (!unpacked || !unpacked_input.sampling_params.sample_idxes.defined()) {
    return 0;
  }
  return static_cast<int32_t>(
      unpacked_input.sampling_params.sample_idxes.size(0));
}

template <typename T>
int64_t count_negative_tokens(const torch::Tensor& tokens) {
  const T* data = tokens.const_data_ptr<T>();
  const int64_t numel = tokens.numel();
  int64_t count = 0;
  for (int64_t i = 0; i < numel; ++i) {
    if (data[i] < static_cast<T>(0)) {
      ++count;
    }
  }
  return count;
}

void record_speculative_metrics_from_output(const torch::Tensor& next_tokens,
                                            const runtime::Options& options) {
  if (!options.enable_speculative_decode() || !next_tokens.defined() ||
      next_tokens.dim() != 2 || next_tokens.numel() == 0) {
    return;
  }

  const int64_t batch_size = next_tokens.size(0);
  const int64_t token_width = next_tokens.size(1);
  const int64_t num_speculative_tokens = options.num_speculative_tokens();
  if (num_speculative_tokens <= 0 ||
      token_width != num_speculative_tokens + 1) {
    return;
  }

  torch::Tensor tokens = next_tokens.contiguous();
  int64_t rejected_count = 0;
  switch (tokens.scalar_type()) {
    case torch::kInt64:
      rejected_count = count_negative_tokens<int64_t>(tokens);
      break;
    case torch::kInt32:
      rejected_count = count_negative_tokens<int32_t>(tokens);
      break;
    case torch::kInt16:
      rejected_count = count_negative_tokens<int16_t>(tokens);
      break;
    case torch::kInt8:
      rejected_count = count_negative_tokens<int8_t>(tokens);
      break;
    default:
      LOG(WARNING) << "Unsupported speculative next_tokens dtype for metrics: "
                   << tokens.scalar_type();
      return;
  }

  const int64_t num_draft_tokens = batch_size * num_speculative_tokens;
  rejected_count = std::min(rejected_count, num_draft_tokens);
  COUNTER_ADD(speculative_num_draft_tokens_total, num_draft_tokens);
  COUNTER_ADD(speculative_num_accepted_tokens_total,
              num_draft_tokens - rejected_count);
}

torch::Tensor clone_cpu_tensor_view(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return tensor;
  }
  CHECK(tensor.device().is_cpu()) << "expected a CPU tensor view";
  return tensor.contiguous().clone();
}

void stabilize_schedule_overlap_host_views(ForwardInput& input) {
  input.token_ids_host = clone_cpu_tensor_view(input.token_ids_host);
  input.positions_host = clone_cpu_tensor_view(input.positions_host);
  input.input_params.attention.host.block_tables =
      clone_cpu_tensor_view(input.input_params.attention.host.block_tables);
}

}  // namespace

WorkerService::WorkerService(runtime::Options options,
                             const torch::Device& device)
    : options_(options), device_(device), initialized_(false) {
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
      device_(device),
      worker_(std::move(worker)),
      initialized_(true) {
  device_.set_device();
  device_.init_device_context();
  stream_ = device_.get_stream_from_pool();
  threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/4,
      /*init_func=*/[this]() mutable { device_.set_device(); },
      /*cpu_binding=*/false,
      /*pool_name=*/"WorkerService.request");
}

WorkerService::~WorkerService() = default;

void WorkerService::set_worker(std::unique_ptr<Worker> worker) {
  worker_ = std::move(worker);
  initialized_ = true;
}

void WorkerService::step(ForwardInput& fwd_input,
                         torch::Tensor& next_tokens,
                         torch::Tensor& logprobs,
                         torch::Tensor& top_tokens,
                         torch::Tensor& top_logprobs,
                         torch::Tensor& embeddings,
                         std::vector<torch::Tensor>& mm_embeddings,
                         std::vector<torch::Tensor>& dit_images,
                         std::vector<std::string>& dit_text_output,
                         torch::Tensor& expert_load_data,
                         int32_t& prepared_layer_id,
                         torch::Tensor& src_seq_idxes,
                         torch::Tensor& out_tokens,
                         torch::Tensor& out_logprobs) {
  const bool use_default_stream =
      !options_.enable_schedule_overlap() && options_.backend() == "llm";
  if (options_.enable_schedule_overlap()) {
    stabilize_schedule_overlap_host_views(fwd_input);
  }
  // execute model
  auto future = worker_->step_async(fwd_input);
  if (!options_.enable_schedule_overlap()) {
    auto forward_outputs = std::move(future).get();
    // convert ForwardOutput to proto::ForwardOutput which contain Tokens.
    if (forward_outputs) {
      DCHECK(forward_outputs.has_value()) << "Failed to execute model";
      const auto& sample_output = forward_outputs.value().sample_output;
      const auto& beam_search_output =
          forward_outputs.value().beam_search_output;
      const auto& dit_forward_output =
          forward_outputs.value().dit_forward_output;
      expert_load_data = safe_to(forward_outputs.value().expert_load_data,
                                 torch::kCPU,
                                 /*non_blocking=*/true);
      prepared_layer_id = forward_outputs.value().prepared_layer_id;

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
          for (auto mm_embedding : sample_output.mm_embeddings) {
            mm_embeddings.emplace_back(
                safe_to(mm_embedding, torch::kCPU, /*non_blocking=*/true));
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
        if (use_default_stream) {
          copy_output_to_host();
        } else {
          c10::StreamGuard stream_guard = stream_->set_stream_guard();
          copy_output_to_host();
        }
        if (use_default_stream) {
          device_.synchronize_default_stream();
        } else {
          stream_->synchronize();
        }
        record_speculative_metrics_from_output(next_tokens, options_);
      }
    }
  } else {
    auto int_options = torch::TensorOptions().device(torch::kCPU);
    if (worker_->is_driver()) {
      // construct fake output tensor
      int32_t num_decode_seqs =
          get_num_decode_seqs_for_schedule_overlap(fwd_input);
      next_tokens = torch::arange(
          -1, -1 * (num_decode_seqs + 1), -1, int_options.dtype(torch::kInt32));
      std::move(future).deferValue([](auto&&) {});
    }
    expert_load_data = torch::zeros({1, 1}, int_options.dtype(torch::kInt64));
  }
}

void WorkerService::create_polling_shm_thread(
    std::unique_ptr<ForwardSharedMemoryManager> input_shm_manager,
    std::unique_ptr<ForwardSharedMemoryManager> output_shm_manager) {
  polling_thread_ = std::make_unique<std::thread>(
      [this,
       input_shm_manager = std::move(input_shm_manager),
       output_shm_manager = std::move(output_shm_manager)]() mutable {
        device_.set_device();
        Timer timer;
        while (true) {
          ForwardInput fwd_input;
          input_shm_manager->input_read(fwd_input, device_);
          timer.reset();
          // model output variables
          torch::Tensor next_tokens;
          torch::Tensor logprobs;
          torch::Tensor top_tokens;
          torch::Tensor top_logprobs;
          torch::Tensor embeddings;
          std::vector<torch::Tensor> mm_embeddings;
          std::vector<torch::Tensor> dit_images;
          std::vector<std::string> dit_text_output;
          torch::Tensor expert_load_data;
          int32_t prepared_layer_id = -1;

          // beam search kernel output
          torch::Tensor src_seq_idxes;
          torch::Tensor out_tokens;
          torch::Tensor out_logprobs;

          step(fwd_input,
               next_tokens,
               logprobs,
               top_tokens,
               top_logprobs,
               embeddings,
               mm_embeddings,
               dit_images,
               dit_text_output,
               expert_load_data,
               prepared_layer_id,
               src_seq_idxes,
               out_tokens,
               out_logprobs);

          const bool shm_write_ok =
              output_shm_manager->raw_output_write(next_tokens,
                                                   logprobs,
                                                   top_tokens,
                                                   top_logprobs,
                                                   embeddings,
                                                   mm_embeddings,
                                                   dit_images,
                                                   dit_text_output,
                                                   expert_load_data,
                                                   prepared_layer_id,
                                                   src_seq_idxes,
                                                   out_tokens,
                                                   out_logprobs);
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
    uint64_t src_cluster_id = req->cluster_id();
    std::string addr = req->addr();
    std::vector<uint64_t> src_blocks(req->src_blocks().begin(),
                                     req->src_blocks().end());
    std::vector<uint64_t> dst_blocks(req->dst_blocks().begin(),
                                     req->dst_blocks().end());
    std::vector<uint64_t> src_linear_state_ids(
        req->src_linear_state_ids().begin(), req->src_linear_state_ids().end());
    std::vector<uint64_t> dst_linear_state_ids(
        req->dst_linear_state_ids().begin(), req->dst_linear_state_ids().end());
    auto future = worker_->pull_kv_blocks_async(src_cluster_id,
                                                addr,
                                                src_blocks,
                                                dst_blocks,
                                                src_linear_state_ids,
                                                dst_linear_state_ids);
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
      worker_->transfer_kv_blocks(batch_id, std::move(block_transfer_info)));
  return;
}

void WorkerService::PrefetchFromStorage(
    google::protobuf::RpcController* controller,
    const proto::BlockTransferInfos* req,
    proto::Status* resp,
    google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  brpc::Controller* cntl = static_cast<brpc::Controller*>(controller);

  brpc::StreamId stream_id;
  brpc::StreamOptions stream_options;
  stream_options.idle_timeout_ms = 5 * options_.prefetch_batch_size();
  if (brpc::StreamAccept(&stream_id, *cntl, &stream_options) != 0) {
    resp->set_ok(false);
    LOG(ERROR) << "Failed to accept stream!";
    return;
  }

  std::vector<BlockTransferInfo> block_transfer_info;
  proto_to_block_transfer_info(*req, block_transfer_info);

  copy_threadpool_.schedule([this,
                             block_transfer_info =
                                 std::move(block_transfer_info),
                             stream_id = std::move(stream_id)]() mutable {
    Slice<BlockTransferInfo> transfer_slice{block_transfer_info};
    bool is_completed = false;

    for (size_t i = 0; i < transfer_slice.size();
         i += options_.prefetch_batch_size()) {
      auto current_slice = transfer_slice.slice(
          i,
          std::min(i + options_.prefetch_batch_size(), transfer_slice.size()));

      auto success_cnt =
          worker_->transfer_kv_blocks(UNINITIALIZED_BATCH_ID, current_slice);

      if (success_cnt != current_slice.size() ||
          (i + options_.prefetch_batch_size()) >= transfer_slice.size()) {
        is_completed = true;
      }

      butil::IOBuf buf;
      buf.append(std::to_string(success_cnt));
      if (brpc::StreamWrite(stream_id, buf) != 0) {
        brpc::StreamClose(stream_id);
        return;
      }

      if (is_completed) {
        if (success_cnt != 0) {
          butil::IOBuf buf_end;
          buf_end.append("0");
          if (brpc::StreamWrite(stream_id, buf_end) != 0) {
            brpc::StreamClose(stream_id);
            return;
          }
        }
        break;
      }
    }
  });

  resp->set_ok(true);
  return;
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

void WorkerService::UpdateWeights(::google::protobuf::RpcController* controller,
                                  const proto::UpdateWeightsRequest* req,
                                  proto::Status* resp,
                                  ::google::protobuf::Closure* done) {
  threadpool_->schedule([this, controller, req, resp, done]() mutable {
    brpc::ClosureGuard done_guard(done);
    bool status = worker_->update_weights(req->weights_path());
    resp->set_ok(status);
  });
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
  threadpool_->schedule(
      [this, controller, pb_forward_input, pb_forward_output, done]() mutable {
        brpc::ClosureGuard done_guard(done);
        // convert proto::ForwardInput to ForwardInput

        Timer timer;
        ForwardInput forward_input;
        CHECK(pb_forward_input->has_packed_input())
            << "ForwardInput must be sent via packed_input";
        packed_proto_to_forward_input(pb_forward_input->packed_input(),
                                      forward_input,
                                      device_,
                                      stream_.get());

        // model output
        torch::Tensor next_tokens;
        torch::Tensor logprobs;
        torch::Tensor top_tokens;
        torch::Tensor top_logprobs;
        torch::Tensor embeddings;
        std::vector<torch::Tensor> mm_embeddings;
        std::vector<torch::Tensor> dit_images;
        std::vector<std::string> dit_text_output;
        torch::Tensor expert_load_data;
        int32_t prepared_layer_id = -1;
        // beam search kernel output
        torch::Tensor src_seq_idxes;
        torch::Tensor out_tokens;
        torch::Tensor out_logprobs;

        step(forward_input,
             next_tokens,
             logprobs,
             top_tokens,
             top_logprobs,
             embeddings,
             mm_embeddings,
             dit_images,
             dit_text_output,
             expert_load_data,
             prepared_layer_id,
             src_seq_idxes,
             out_tokens,
             out_logprobs);
        // convert to proto output
        forward_output_to_proto(next_tokens,
                                logprobs,
                                top_tokens,
                                top_logprobs,
                                embeddings,
                                expert_load_data,
                                prepared_layer_id,
                                src_seq_idxes,
                                out_tokens,
                                out_logprobs,
                                dit_images,
                                dit_text_output,
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

        auto future = worker_->get_last_step_result_async();
        auto forward_outputs = std::move(future).get();
        if (forward_outputs) {
          const ForwardOutput& forward_output = forward_outputs.value();
          const auto& sample_output = forward_output.sample_output;
          int32_t prepared_layer_id = forward_output.prepared_layer_id;
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
          std::vector<torch::Tensor> dit_images;
          std::vector<std::string> dit_text_output;
          auto copy_output_to_host = [&]() {
            if (options_.enable_schedule_overlap()) {
              CHECK(stream_->wait_event(forward_output.ready_event))
                  << "failed to wait forward output ready event";
            }
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
            dit_text_output =
                forward_outputs.value().dit_forward_output.text_output;

            // [num_seq]
            next_tokens = safe_to(sample_output.next_tokens,
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

          if (use_default_stream) {
            copy_output_to_host();
          } else {
            c10::StreamGuard stream_guard = stream_->set_stream_guard();
            if (forward_outputs.value().ready_event != nullptr) {
              CHECK(stream_->wait_event(forward_outputs.value().ready_event))
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
          record_speculative_metrics_from_output(next_tokens, options_);

          if (next_tokens.defined() || !dit_images.empty() ||
              !dit_text_output.empty() ||
              ::xllm::EPLBConfig::get_instance().enable_eplb()) {
            forward_output_to_proto(next_tokens,
                                    logprobs,
                                    top_tokens,
                                    top_logprobs,
                                    embeddings,
                                    expert_load_data,
                                    prepared_layer_id,
                                    src_seq_idxes,
                                    out_tokens,
                                    out_logprobs,
                                    dit_images,
                                    dit_text_output,
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
