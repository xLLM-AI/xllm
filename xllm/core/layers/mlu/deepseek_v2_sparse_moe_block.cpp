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

#include "deepseek_v2_sparse_moe_block.h"

#include <utility>

#include "common/global_flags.h"
#include "core/framework/config/eplb_config.h"
#include "framework/parallel_state/parallel_state.h"
#include "platform/device.h"

namespace xllm {
namespace layer {

DeepseekV2SparseMoEBlockImpl::DeepseekV2SparseMoEBlockImpl(
    const ModelArgs& model_args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : parallel_args_(parallel_args) {
  enable_deep_ep_ =
      ::xllm::EPLBConfig::get_instance().expert_parallel_degree() == 2 &&
      parallel_args_.ep_size() > 1;
  const FusedMoEArgs moe_args{.is_gated = true,
                              .enable_result_reduction = false};
  moe_ = register_module(
      "moe",
      FusedMoE(model_args, moe_args, quant_args, parallel_args, options));
}

void DeepseekV2SparseMoEBlockImpl::load_state_dict(
    const StateDict& state_dict) {
  moe_->load_state_dict(state_dict);
}

void DeepseekV2SparseMoEBlockImpl::verify_loaded_weights() const {
  moe_->verify_loaded_weights();
}

DeepseekV2SparseMoEBlockImpl::ExecCfg DeepseekV2SparseMoEBlockImpl::plan_exec(
    const ModelInputParams& input_params) const {
  ExecCfg cfg;
  cfg.enable_all2all = enable_deep_ep_ && all_dp_ranks_are_decode(input_params);
  cfg.need_dp_gather = need_dp_moe_gather(parallel_args_, cfg.enable_all2all);
  return cfg;
}

DeepseekV2SparseMoEBlockImpl::PrepOut DeepseekV2SparseMoEBlockImpl::prep_in(
    torch::Tensor x,
    const torch::Tensor& residual,
    const ModelInputParams& input_params,
    const ExecCfg& exec,
    DeepseekV2AttentionImpl::PostAttnLayout attn_layout) const {
  PrepOut prep;
  if (exec.enable_all2all) {
    auto shard =
        shard_attn_out(x,
                       residual,
                       get_reduce_scatter_tokens(x.size(0), parallel_args_),
                       attn_layout);
    prep.ffn_in = shard.first;
    prep.skip_local = prep.ffn_in;
    prep.pad_info = shard.second;
    prep.need_tp_pad = true;
    return prep;
  }

  CHECK(exec.need_dp_gather)
      << "prep_in only supports sparse MoE special paths";
  auto shard = shard_attn_out(
      x,
      residual,
      get_dp_gather_tokens(input_params.parallel.dp_global_token_nums,
                           parallel_args_),
      attn_layout);
  prep.ffn_in = shard.first;
  prep.pad_info = shard.second;
  prep.need_dp_gather = true;

  torch::Tensor local_tokens = prep.ffn_in;
  if (parallel_args_.tp_group_ && parallel_args_.tp_group_->world_size() > 1) {
    local_tokens =
        parallel_state::gather(local_tokens, parallel_args_.tp_group_, 0);
  }

  CHECK(parallel_args_.dp_local_process_group_ != nullptr)
      << "dp gather prep requires dp_local_process_group_";
  const int64_t dp_rank = parallel_args_.dp_local_process_group_->rank();
  CHECK_GE(dp_rank, 0) << "invalid dp rank " << dp_rank;
  CHECK_LT(
      dp_rank,
      static_cast<int64_t>(input_params.parallel.dp_global_token_nums.size()))
      << "dp rank " << dp_rank << " exceeds dp_global_token_nums size "
      << input_params.parallel.dp_global_token_nums.size();
  const int64_t local_token_num =
      input_params.parallel.dp_global_token_nums[dp_rank];
  prep.skip_local = local_tokens.slice(0, 0, local_token_num);
  return prep;
}

torch::Tensor DeepseekV2SparseMoEBlockImpl::gather_in(
    const PrepOut& prep,
    const ModelInputParams& input_params) const {
  if (!prep.need_dp_gather) {
    return prep.ffn_in;
  }

  return gather_global_tokens(
      prep.ffn_in, input_params.parallel.dp_global_token_nums, parallel_args_);
}

torch::Tensor DeepseekV2SparseMoEBlockImpl::merge_out(
    torch::Tensor x,
    const PrepOut& prep,
    const ModelInputParams& input_params) const {
  if (prep.need_dp_gather) {
    x = get_dp_local_slice(x, input_params, parallel_args_);
    return x + prep.skip_local;
  }

  if (prep.need_tp_pad && parallel_args_.tp_group_ &&
      parallel_args_.tp_group_->world_size() > 1) {
    x = parallel_state::gather(x, parallel_args_.tp_group_, 0);
    x = unpad_tokens(x, prep.pad_info);
    auto skip_local =
        parallel_state::gather(prep.skip_local, parallel_args_.tp_group_, 0);
    skip_local = unpad_tokens(skip_local, prep.pad_info);
    return x + skip_local;
  }

  return x + prep.skip_local;
}

bool DeepseekV2SparseMoEBlockImpl::has_shared() const {
  return moe_->has_shared();
}

std::pair<torch::Tensor, PaddingInfo>
DeepseekV2SparseMoEBlockImpl::shard_attn_out(
    torch::Tensor x,
    const torch::Tensor& residual,
    int64_t target_tokens,
    DeepseekV2AttentionImpl::PostAttnLayout attn_layout) const {
  if (attn_layout == DeepseekV2AttentionImpl::PostAttnLayout::kTpShard) {
    return reduce_scatter_attn_input(
        x, residual, target_tokens, parallel_args_);
  }

  CHECK(attn_layout == DeepseekV2AttentionImpl::PostAttnLayout::kReplicated)
      << "unexpected post-attention layout for sparse MoE shard path: "
      << static_cast<int>(attn_layout);
  x = x + residual;
  auto pad = pad_tokens(x, target_tokens);
  return {slice_tp_tokens(pad.first), pad.second};
}

torch::Tensor DeepseekV2SparseMoEBlockImpl::slice_tp_tokens(
    torch::Tensor x) const {
  if (!parallel_args_.tp_group_ ||
      parallel_args_.tp_group_->world_size() <= 1) {
    return x;
  }

  int64_t tp_size = parallel_args_.tp_group_->world_size();
  CHECK_EQ(x.size(0) % tp_size, 0)
      << "token_num " << x.size(0) << " must be divisible by tp_size "
      << tp_size;
  int64_t shard_tokens = x.size(0) / tp_size;
  int64_t start = parallel_args_.tp_group_->rank() * shard_tokens;
  return x.slice(0, start, start + shard_tokens);
}

ProcessGroup* DeepseekV2SparseMoEBlockImpl::routed_pg() const {
  return parallel_args_.ep_size() > 1 ? parallel_args_.moe_ep_group_
                                      : parallel_args_.tp_group_;
}

DeepseekV2SparseMoEBlockImpl::ForwardResult
DeepseekV2SparseMoEBlockImpl::forward(torch::Tensor x,
                                      bool enable_moe_all2all,
                                      const CommFns& comm_fns) {
  if (enable_moe_all2all) {
    return ForwardResult{
        .output =
            moe_->forward_experts(x, /*enable_all2all_communication=*/true),
        .keep_local_output = false,
    };
  }

  ProcessGroup* routed_group = routed_pg();
  ProcessGroup* shared_group = moe_->shared_pg();
  const bool keep_local_output = comm_fns.can_keep_local(routed_group);

  const bool can_overlap_reduce = !keep_local_output && moe_->has_shared() &&
                                  static_cast<bool>(comm_fns.launch_reduce) &&
                                  static_cast<bool>(comm_fns.finish_reduce);
  if (can_overlap_reduce) {
    moe_->init_async(x);
    Stream* comm_stream = moe_->routed_stream();
    CHECK(comm_stream != nullptr) << "forward overlap requires routed stream";

    Device device(x.device());
    auto current_stream = device.current_stream();
    torch::Tensor routed_out =
        moe_->forward_experts(x, /*enable_all2all_communication=*/false);

    parallel_state::ReduceAsyncCtx reduce_handle;
    comm_stream->wait_stream(*current_stream);
    {
      torch::StreamGuard stream_guard = comm_stream->set_stream_guard();
      reduce_handle =
          comm_fns.launch_reduce(std::move(routed_out), routed_group);
    }

    torch::Tensor shared_out = moe_->forward_shared(x);
    current_stream->wait_stream(*comm_stream);
    x = comm_fns.finish_reduce(std::move(reduce_handle));
    if (shared_out.defined()) {
      x = x + shared_out;
    }
    return ForwardResult{
        .output = std::move(x),
        .keep_local_output = false,
    };
  }

  torch::Tensor shared_out = moe_->forward_shared(x);
  if (shared_out.defined() && keep_local_output) {
    // we assume that share experts use full weights for deepseek models
    // and relies on single rank process group, so there is no need to do all
    // reduce here.
    shared_out = comm_fns.comm(shared_out, shared_group);
  }

  x = moe_->forward_experts(std::move(x),
                            /*enable_all2all_communication=*/false);
  x = keep_local_output ? comm_fns.comm(x, routed_group)
                        : comm_fns.reduce(x, routed_group);
  if (shared_out.defined()) {
    x = x + shared_out;
  }
  return ForwardResult{
      .output = std::move(x),
      .keep_local_output = keep_local_output,
  };
}

DeepseekV2SparseMoEBlockImpl::ForwardResult
DeepseekV2SparseMoEBlockImpl::forward_sp(
    torch::Tensor x,
    const v32_cp::DeepseekV32CPContext& sp_ctx,
    const CommFns& comm_fns) {
  CHECK(has_shared()) << "forward_sp requires shared experts";
  ProcessGroup* routed_group = routed_pg();
  const bool keep_local_output = comm_fns.can_keep_local(routed_group);
  if (!keep_local_output) {
    auto gathered = v32_cp::finish_all_gather_across_ranks(
        v32_cp::launch_all_gather_across_ranks(x, sp_ctx));
    return forward(std::move(gathered),
                   /*enable_moe_all2all=*/false,
                   comm_fns);
  }

  moe_->init_async(x);
  Stream* comm_stream = moe_->shared_stream();
  CHECK(comm_stream != nullptr) << "forward_sp requires shared stream";

  Device device(x.device());
  auto current_stream = device.current_stream();
  comm_stream->wait_stream(*current_stream);

  v32_cp::PaddedGatherHandle gather_handle;
  {
    torch::StreamGuard stream_guard = comm_stream->set_stream_guard();
    gather_handle = v32_cp::launch_all_gather_across_ranks(x, sp_ctx);
  }

  torch::Tensor shared_out = moe_->forward_shared(x);

  current_stream->wait_stream(*comm_stream);
  auto gathered =
      v32_cp::finish_all_gather_across_ranks(std::move(gather_handle));
  torch::Tensor routed_out =
      moe_->forward_experts(std::move(gathered),
                            /*enable_all2all_communication=*/false);
  routed_out = comm_fns.comm(std::move(routed_out), routed_group);
  return ForwardResult{
      .output = routed_out + shared_out,
      .keep_local_output = true,
  };
}

}  // namespace layer
}  // namespace xllm
