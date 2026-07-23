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

#include "npu_lm_head_impl.h"

#include <glog/logging.h>

#include "core/framework/config/load_config.h"
#include "core/framework/config/parallel_config.h"

namespace xllm {
namespace layer {

void NpuLmHeadImpl::param_from_args(atb_speed::common::LmHeadParam& param,
                                    const ModelArgs& args,
                                    const ParallelArgs& parallel_args,
                                    bool isPrefill) {
  // Model-side CP (NPU/MLU): the model already all-gathers + restores global-
  // real hidden after the last decoder layer (npu_cp::gather_restore), so the
  // LM head never needs to re-gather a CP shard. Two legacy worker-CP knobs
  // must stay off here, otherwise the ATB LmHead graph nests a second CP
  // AllGather on the already-global hidden (peak O(cp_size * T * H), OOM on
  // long prefill):
  //   * outputHidden (legacy gathered-hidden output) is permanently false now
  //     that legacy worker-side CP has been removed.
  //   * contextParallelInfo is left default-constructed (rankIds empty ->
  //     IsEnabled()==false), so the ATB `gatherAhead && IsEnabled()` gate skips
  //     the CP AllGather node. See
  //     docs/pr2003_lmhead_nested_cp_allgather_bug.md.
  param.outputHidden = false;
  param.unpadInputs = true;
  param.gatherAhead = isPrefill;
  param.hiddenSizePerAttentionHead = args.hidden_size() / args.n_heads();
  param.linearParallelParam.fusionLinearParam.isBF16 =
      args.dtype() == "bfloat16";
  param.linearParallelParam.unpadInputs = true;
  param.linearParallelParam.fusionLinearParam.transposeType = 1;

  if (parallel_args.world_size() > 1) {
    if (parallel_args.mapping_data().empty()) {
      // Model-side CP: the LM head runs AFTER npu_cp::gather_restore, so it
      // sees the full global-real sequence and must be CP-unaware but
      // DP-aware. Shard across the dp-local-TP group (world / dp_size =
      // cp_size * tp_size) — the full TP width within one DP, spanning all CP
      // ranks in that DP — instead of the CP-local TP (dp_local_tp_size_,
      // size tp). This avoids replicating the lm_head weight across CP
      // ranks. When cp_size == 1 this collapses to tp_size, so non-CP runs
      // are unchanged.
      const bool use_cp_unaware_tp = (dp_size_ > 1) || (cp_size_ > 1);
      const int32_t cp_unaware_tp_size = parallel_args.world_size() / dp_size_;
      if (use_cp_unaware_tp) {
        CHECK_GT(cp_unaware_tp_size, 0);
        const int32_t cp_unaware_tp_rank =
            parallel_args.rank() % cp_unaware_tp_size;
        CHECK_GE(cp_unaware_tp_rank, 0);
        CHECK_LT(cp_unaware_tp_rank, cp_unaware_tp_size);
        param.linearParallelParam.tensorParallelInfo.rank = cp_unaware_tp_rank;
        param.linearParallelParam.tensorParallelInfo.worldSize =
            cp_unaware_tp_size;
      } else {
        param.linearParallelParam.tensorParallelInfo.rank =
            parallel_args.rank();
        param.linearParallelParam.tensorParallelInfo.worldSize =
            parallel_args.world_size();
      }
      param.linearParallelParam.parallelType =
          atb_speed::common::COLUMN_PARALLEL;
      const int32_t tp_group_id =
          use_cp_unaware_tp ? (parallel_args.rank() / cp_unaware_tp_size) : 0;
      param.linearParallelParam.tensorParallelInfo.commDomain =
          std::to_string(tp_group_id);
      param.linearParallelParam.tensorParallelInfo.backend =
          ::xllm::ParallelConfig::get_instance().communication_backend();
    } else {
      param.linearParallelParam.parallelType =
          atb_speed::common::COLUMN_PARALLEL;
      // Mapping (ATB) path: use the dedicated LM_HEAD_TP group, which
      // MappingNPU builds as the dp-local-TP group (size cp*tp). Do NOT use
      // ATTN_TP (size tp, CP-local) — that would replicate the lm_head
      // weight across CP ranks. See mapping_npu.cpp parse_parallel_info().
      atb_speed::common::ParallelInfo parallelInfo =
          parallel_args.mapping().Get(atb_speed::base::LM_HEAD_TP);
      param.linearParallelParam.tensorParallelInfo.rank = parallelInfo.rank;
      param.linearParallelParam.tensorParallelInfo.worldSize =
          parallelInfo.rankIds.size();
      param.linearParallelParam.tensorParallelInfo.backend =
          ::xllm::ParallelConfig::get_instance().communication_backend();
      parallelInfo.InitCommDomain(
          param.linearParallelParam.tensorParallelInfo.hcommInfo,
          param.linearParallelParam.tensorParallelInfo.commDomain);
      // Do NOT assign ATTN_CP to param.contextParallelInfo. Model-side CP
      // already restored global-real hidden via npu_cp::gather_restore before
      // the LM head, so every rank already holds the full sequence. Registering
      // ATTN_CP here would make IsEnabled()==true (rankIds.size()>1) and, with
      // gatherAhead=true on prefill, build a nested CP AllGather that treats
      // the full sequence as a shard and scales hidden by cp_size again ->
      // O(cp_size * T * H) peak memory and OOM on long prefill. Leave
      // contextParallelInfo default-constructed so the ATB LmHead graph skips
      // the CP AllGather node. See
      // docs/pr2003_lmhead_nested_cp_allgather_bug.md.
    }
  }
}

NpuLmHeadImpl::NpuLmHeadImpl(const ModelContext& context) : BaseLayer(context) {
  vocab_size_ = context.get_model_args().vocab_size();
  // The LM head is sharded across the dp-local-TP group (world / dp_size =
  // cp_size * tp_size, CP-unaware but DP-aware), so pad the vocab to a multiple
  // of that width — not the CP-local TP (dp_local_tp_size_, size tp). This must
  // match get_padded_vocab_size() in base_loader.cpp. When cp_size == 1 this
  // collapses to tp_size, so non-CP runs are unchanged.
  const int32_t cp_unaware_tp_size = parallel_args_.world_size() / dp_size_;
  if (vocab_size_ > 0 && cp_unaware_tp_size > 1 &&
      vocab_size_ % cp_unaware_tp_size != 0) {
    padded_vocab_size_ =
        ((vocab_size_ + cp_unaware_tp_size - 1) / cp_unaware_tp_size) *
        cp_unaware_tp_size;
  } else {
    padded_vocab_size_ = vocab_size_;
  }

  param_from_args(lm_head_param_prefill_,
                  context.get_model_args(),
                  context.get_parallel_args(),
                  true);

  param_from_args(lm_head_param_decode_,
                  context.get_model_args(),
                  context.get_parallel_args(),
                  false);

  atb_weight_tensors_.resize(1);
  atOutTensors_.resize(2);

  auto options = context.get_tensor_options();
  dtype_ = c10::typeMetaToScalarType(options.dtype());
  prefill_tensor_storage_.resize(2);
  decode_tensor_storage_.resize(2);

  torch_placeholder_ = torch::zeros({1}).to(device_).to(dtype_);
  placeholder_ = atb_speed::Utils::AtTensor2Tensor(torch_placeholder_);

  loader_ = std::make_unique<LmHeadLoader>(
      1,
      context,
      ::xllm::LoadConfig::get_instance().enable_manual_loader()
          ? LoadMode::kManual
          : LoadMode::kEager);
}

int64_t NpuLmHeadImpl::init_layer() {
  BaseLayer::name_ = "lm_head_layer";
  model_name_ = "lm";
  CHECK_OPERATION_STATUS_RETURN(
      init_node(lm_head_node_prefill_, lm_head_param_prefill_));
  CHECK_OPERATION_STATUS_RETURN(
      init_node(lm_head_node_decode_, lm_head_param_decode_));

  return atb::NO_ERROR;
}

int64_t NpuLmHeadImpl::init_node(atb_speed::Model::Node& node,
                                 atb_speed::common::LmHeadParam& param) {
  atb::Operation* operation = nullptr;
  atb::Status atbStatus = atb_speed::common::LmHead(param, &operation);
  if (atbStatus != atb::NO_ERROR) {
    return atbStatus;
  }
  node.operation.reset(operation);
  if (node.operation == nullptr) {
    LOG(ERROR) << "node.operation is null";
    return -1;
  }
  if (node.operation->GetInputNum() < 1) {
    LOG(ERROR) << "Can not resize number which is smaller than 1";
    return -1;
  }
  node.inTensors.resize(node.operation->GetInputNum());
  node.outTensors.resize(node.operation->GetOutputNum());

  node.inTensors.at(1) = &atb_weight_tensors_[0];

  node.variantPack.inTensors.reserve(node.inTensors.size());
  node.variantPack.inTensors.resize(node.inTensors.size());
  node.variantPack.outTensors.reserve(node.outTensors.size());
  node.variantPack.outTensors.resize(node.outTensors.size());

  return atb::NO_ERROR;
}

torch::Tensor NpuLmHeadImpl::forward(const torch::Tensor& hidden_states,
                                     const torch::Tensor& seleted_idxes,
                                     int nodeId) {
  torch::Tensor out_hidden;
  return forward_with_hidden(hidden_states, seleted_idxes, out_hidden, nodeId);
}

torch::Tensor NpuLmHeadImpl::forward_with_hidden(
    const torch::Tensor& hidden_states,
    const torch::Tensor& seleted_idxes,
    torch::Tensor& out_hidden,
    int nodeId) {
  atb::Status st;
  build_node_variant_pack(lm_head_node_prefill_, hidden_states, seleted_idxes);
  st = execute_node(lm_head_node_prefill_, nodeId);
  LOG_IF(FATAL, st != 0) << model_name_
                         << "execute lmhead node fail, error code: " << st;
  torch::Tensor output = atOutTensors_[0];
  if (padded_vocab_size_ > vocab_size_ && vocab_size_ > 0) {
    output = output.slice(/*dim=*/-1, /*start=*/0, /*end=*/vocab_size_);
  }
  if (atOutTensors_.size() > 1) {
    out_hidden = atOutTensors_[1];
  }
  return output;
}

void NpuLmHeadImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    const torch::Tensor& hidden_states,
    const torch::Tensor& seleted_idxes) {
  hidden_states_atb_ = atb_speed::Utils::AtTensor2Tensor(hidden_states);
  seleted_idxes_atb_ = atb_speed::Utils::AtTensor2Tensor(seleted_idxes);
  // node.outTensors[0] = &internalTensors;

  atb::SVector<atb::TensorDesc> inTensorDescs;
  inTensorDescs.reserve(node.variantPack.inTensors.size());
  inTensorDescs.resize(node.variantPack.inTensors.size());
  atb::SVector<atb::TensorDesc> outTensorDescs;
  outTensorDescs.reserve(node.operation->GetOutputNum());
  outTensorDescs.resize(node.operation->GetOutputNum());

  node.variantPack.inTensors.at(0) = hidden_states_atb_;
  inTensorDescs.at(0) = hidden_states_atb_.desc;

  node.variantPack.inTensors.at(1) = *node.inTensors.at(1);
  inTensorDescs.at(1) = node.inTensors.at(1)->desc;

  node.variantPack.inTensors.at(2) = placeholder_;
  inTensorDescs.at(2) = placeholder_.desc;

  node.variantPack.inTensors.at(3) = placeholder_;
  inTensorDescs.at(3) = placeholder_.desc;

  node.variantPack.inTensors.at(4) = placeholder_;
  inTensorDescs.at(4) = placeholder_.desc;

  node.variantPack.inTensors.at(5) = placeholder_;
  inTensorDescs.at(5) = placeholder_.desc;

  node.variantPack.inTensors.at(6) = placeholder_;
  inTensorDescs.at(6) = placeholder_.desc;

  node.variantPack.inTensors.at(7) = seleted_idxes_atb_;
  inTensorDescs.at(7) = seleted_idxes_atb_.desc;

  node.variantPack.inTensors.at(8) = placeholder_;
  inTensorDescs.at(8) = placeholder_.desc;

  atb::Status st = node.operation->InferShape(inTensorDescs, outTensorDescs);
  LOG_IF(FATAL, st != atb::NO_ERROR)
      << model_name_ << " infer lmhead shape fail, error code: " << st;

  atOutTensors_.resize(node.variantPack.outTensors.size());
  for (size_t i = 0; i < node.variantPack.outTensors.size(); ++i) {
    atOutTensors_.at(i) =
        atb_speed::Utils::CreateAtTensorFromTensorDesc(outTensorDescs.at(i));
    node.variantPack.outTensors.at(i) =
        atb_speed::Utils::AtTensor2Tensor(atOutTensors_.at(i));
  }
}

}  // namespace layer
}  // namespace xllm
