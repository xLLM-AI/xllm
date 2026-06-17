/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "npu_qwen3_decoder_layer_impl.h"

#include <glog/logging.h>
#include <mstx/ms_tools_ext.h>

#include <map>

#include "core/framework/config/eplb_config.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "util/rec_model_utils.h"

// #include "attn_mask.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace xllm {
namespace layer {

const uint64_t WEIGHT_COUNT_PER_LAYER = 56;

void NpuQwen3DecoderLayerImpl::param_from_args(
    atb_speed::qwen::QwenLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args,
    bool isPrefill) {
  param.isFA = false;
  // Enable SwiGLU activation, as used in LLaMA
  param.enableSwiGLU = true;
  // Enable LCOC for prefill phase, similar to LLaMA
  // NOTE: Currently, single-process startup requires setting enableLcoc to
  // false, which leads to performance degradation. param.enableLcoc = false;
  // //isPrefill
  param.enableLcoc = false;
  param.rmsnormQKNorm = true;
  param.isPrefill = isPrefill;
  param.isBF16 = args.dtype() == "bfloat16";
  param.enableSplitFuse =
      ::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() &&
      isPrefill;
  param.loraEnableGMM = false;
  param.enableXattention = is_rec_multi_round_mode();

  param.linearTransposeType = {static_cast<int>(TransposeType::NOT_TRANSPOSE),
                               static_cast<int>(TransposeType::INVALID),
                               static_cast<int>(TransposeType::INVALID),
                               static_cast<int>(TransposeType::NOT_TRANSPOSE),
                               static_cast<int>(TransposeType::NOT_TRANSPOSE),
                               static_cast<int>(TransposeType::INVALID),
                               static_cast<int>(TransposeType::NOT_TRANSPOSE)};
  param.quantGroupSize = 0;
  param.normEps = args.rms_norm_eps();
  param.numAttentionHeadsPerRank = args.n_heads() / parallel_args.world_size();
  param.hiddenSizePerAttentionHead = args.head_dim();
  std::optional<long int> optionalValue = args.n_kv_heads();
  param.numKeyValueHeadsPerRank =
      static_cast<int>(optionalValue.value()) / parallel_args.world_size();
  param.backend = ::xllm::ParallelConfig::get_instance().communication_backend();
  const bool flash_comm_requested =
      ::xllm::KernelConfig::get_instance().enable_flash_comm();
  const bool has_process_group =
      parallel_args.tp_group_ != nullptr ||
      parallel_args.process_group_ != nullptr;
  const bool flash_comm_runtime_supported =
      param.backend == "lccl" && parallel_args.world_size() > 1 &&
      parallel_args.dp_size() == 1 && has_process_group;
  if (flash_comm_requested) {
    if (!flash_comm_runtime_supported) {
      LOG(WARNING) << "NPU flash comm is enabled by config but skipped for "
                   << "Qwen3 decoder layer because it requires lccl backend "
                   << "and tensor parallel world_size > 1, dp_size == 1, "
                   << "and a process group for model-level sequence "
                   << "split/gather. "
                   << "backend="
                   << param.backend
                   << ", world_size=" << parallel_args.world_size()
                   << ", dp_size=" << parallel_args.dp_size()
                   << ", has_process_group=" << has_process_group;
    }
  }
  param.enableFlashComm = flash_comm_requested && flash_comm_runtime_supported;
  param.enableLogN = false;
  param.tensorParallelInfo = {
      parallel_args.rank(),
      parallel_args.world_size(),
      ::xllm::ParallelConfig::get_instance().communication_backend()};
  param.linearHasBias = {0, 0, 0, 0};
  param.useQKNorm = true;

  param.numHiddenLayers = args.n_layers();
  param.enableIntraLayerAddNorm = true;
  if (::xllm::KernelConfig::get_instance().enable_interlayer_addnorm()) {
    param.enableInterLayerAddNorm = true;
  }
  param.enablePreFetchWeight =
      ::xllm::LoadConfig::get_instance().enable_prefetch_weight();
  param.enableAclGraphPagedAttention =
      ::xllm::ExecutionConfig::get_instance().enable_graph() && !isPrefill;
  initialize_parallel_parameters(param, parallel_args);
  initialize_quantization_parameters(param);

  if (isPrefill) {
    param.enableAclnnRmsNorm =
        param.enableIntraLayerAddNorm && quantize_type_.empty()
            ? false
            : quantize_type_.empty();
    // for prefix cache without chunked prefill.
    if (::xllm::KVCacheConfig::get_instance().enable_prefix_cache() &&
        !::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() &&
        ::xllm::KVCacheConfig::get_instance().block_size() != 128) {
      LOG(ERROR) << "try to enable prefix cache without chunked prefill but "
                    "failed, because the block_size is required to be 128.";
    }
    param.isPrefixCacheWithoutChunk =
        ::xllm::KVCacheConfig::get_instance().enable_prefix_cache() &&
        !::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() &&
        ::xllm::KVCacheConfig::get_instance().block_size() == 128;
  }
  num_hidden_layers_ = args.n_layers();
  if (::xllm::KernelConfig::get_instance().enable_split_rmsnorm_rope()) {
    param.enableSplitRmsNormRope = true;
  }
}

void NpuQwen3DecoderLayerImpl::initialize_parallel_parameters(
    atb_speed::qwen::QwenLayerParam& param,
    const ParallelArgs& parallel_args) {
  param.mapping = parallel_args.mapping();
  param.tensorParallelInfo = {
      parallel_args.rank(),
      parallel_args.world_size(),
      ::xllm::ParallelConfig::get_instance().communication_backend(),
      ::xllm::EPLBConfig::get_instance().rank_tablefile(),
      nullptr,
      ""};
}

void NpuQwen3DecoderLayerImpl::initialize_quantization_parameters(
    atb_speed::qwen::QwenLayerParam& param) {
  if (quantize_type_.empty()) {
    param.linearDescs = {static_cast<int>(LinearTypeV2::BFLOAT16),
                         static_cast<int>(LinearTypeV2::INVALID),
                         static_cast<int>(LinearTypeV2::INVALID),
                         static_cast<int>(LinearTypeV2::BFLOAT16),
                         static_cast<int>(LinearTypeV2::BFLOAT16),
                         static_cast<int>(LinearTypeV2::INVALID),
                         static_cast<int>(LinearTypeV2::BFLOAT16)};
    param.packQuantType = {static_cast<int>(PackType::PACK_QUANT_UNDEFINED),
                           static_cast<int>(PackType::PACK_QUANT_UNDEFINED)};
    param.linearQuantType = {static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID)};
  } else {
    param.linearDescs = {static_cast<int>(LinearTypeV2::W8A8),
                         static_cast<int>(LinearTypeV2::INVALID),
                         static_cast<int>(LinearTypeV2::INVALID),
                         static_cast<int>(LinearTypeV2::W8A8),
                         static_cast<int>(LinearTypeV2::W8A8),
                         static_cast<int>(LinearTypeV2::INVALID),
                         static_cast<int>(LinearTypeV2::BFLOAT16)};
    param.packQuantType = {static_cast<int>(PackType::ALL_W8A8),
                           static_cast<int>(PackType::ALL_W8A8)};
    param.linearQuantType = {static_cast<int>(LinearType::INT),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INT),
                             static_cast<int>(LinearType::INT),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::FP)};
  }
}

NpuQwen3DecoderLayerImpl::NpuQwen3DecoderLayerImpl(const ModelContext& context)
    : BaseLayer(context) {
  auto model_args = context.get_model_args();
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();

  param_from_args(prefill_param_, model_args, parallel_args, true);
  param_from_args(decode_graph_param_, model_args, parallel_args, false);
  decode_eager_param_ = decode_graph_param_;
  decode_eager_param_.enableAclGraphPagedAttention = false;
  flash_comm_enabled_ = prefill_param_.enableFlashComm;
  flash_comm_rank_ =
      parallel_args.tp_group_ != nullptr ? parallel_args.tp_group_->rank()
                                         : parallel_args.rank();
  atb_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  placeholder_vec_ = {1};
  dtype_ = c10::typeMetaToScalarType(options.dtype());
  prefill_tensor_storage_.resize(4);
  decode_tensor_storage_.resize(4);
  prefill_vector_storage_.resize(1);
  decode_vector_storage_.resize(1);
  placeholder_ = atb_speed::Utils::AtTensor2Tensor(
      torch::zeros({1}).to(device_).to(dtype_));
  at_placeholder_ = torch::zeros({1}).to(device_).to(dtype_);
  loader_ = std::make_unique<Qwen3DecoderLoader>(
      WEIGHT_COUNT_PER_LAYER,
      context,
      prefill_param_.enableIntraLayerAddNorm ||
          prefill_param_.enableInterLayerAddNorm,
      ::xllm::LoadConfig::get_instance().enable_manual_loader()
          ? LoadMode::kManual
          : LoadMode::kEager);
}

int64_t NpuQwen3DecoderLayerImpl::init_layer() {
  init_attn_mask();
  name_ = "qwen3_decoder_layer";
  model_name_ = "qwen3";
  CHECK_OPERATION_STATUS_RETURN(init_node(prefill_node_, prefill_param_));
  CHECK_OPERATION_STATUS_RETURN(
      init_node(decode_graph_node_, decode_graph_param_));
  CHECK_OPERATION_STATUS_RETURN(
      init_node(decode_eager_node_, decode_eager_param_));

  return atb::NO_ERROR;
}

int64_t NpuQwen3DecoderLayerImpl::init_attn_mask() {
  torch::Dtype dtype =
      prefill_param_.isBF16 ? torch::kBFloat16 : torch::kFloat16;
  decode_attn_mask_ = torch::zeros({1}).to(device_).to(dtype);

  return atb::NO_ERROR;
}

int64_t NpuQwen3DecoderLayerImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::qwen::QwenLayerParam& param) {
  atb::Operation* operation = nullptr;
  atb_speed::qwen::QwenDecoderLayer decoder_layer(param);
  decoder_layer.BuildGraph(&operation);
  node.operation.reset(operation);
  CHECK_NOTNULL(node.operation);
  CHECK_GT(node.operation->GetInputNum(), 0);
  node.inTensors.resize(node.operation->GetInputNum());
  node.outTensors.resize(node.operation->GetOutputNum());
  size_t inTensorId = 1;

  for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER;
       ++weightTensorId) {
    node.inTensors.at(weightTensorId) = &atb_weight_tensors_[weightTensorId];
  }

  node.variantPack.inTensors.resize(node.inTensors.size());
  node.variantPack.outTensors.resize(node.outTensors.size());

  return atb::NO_ERROR;
}

namespace {

std::vector<int64_t> make_displacements(const std::vector<int64_t>& counts) {
  std::vector<int64_t> displacements(counts.size(), 0);
  int64_t offset = 0;
  for (size_t i = 0; i < counts.size(); ++i) {
    displacements[i] = offset;
    offset += counts[i];
  }
  return displacements;
}

}  // namespace

torch::Tensor NpuQwen3DecoderLayerImpl::forward(torch::Tensor& x,
                                                torch::Tensor& cos_pos,
                                                torch::Tensor& sin_pos,
                                                torch::Tensor& attn_mask,
                                                KVCache& kv_cache,
                                                ModelInputParams& input_params,
                                                aclrtEvent* event,
                                                std::atomic<bool>* event_flag,
                                                int node_id) {
  atb::Status st;
  if (!input_params.meta.batch_forward_type.is_decode()) {
    build_node_variant_pack(prefill_node_,
                            x,
                            cos_pos,
                            sin_pos,
                            attn_mask,
                            kv_cache,
                            input_params,
                            /*is_prefill=*/true,
                            node_id,
                            /*use_graph_decode_input=*/false);
    // mstxRangeEnd(id);
    st = execute_node(prefill_node_, node_id, event, event_flag);
    LOG_IF(FATAL, st != 0) << model_name_
                           << "excute prefill layer fail, error code: " << st;
  } else {
    const bool use_graph_decode_input =
        ::xllm::ExecutionConfig::get_instance().enable_graph() &&
        input_params.graph.tiling_data.defined();
    auto& decode_node =
        use_graph_decode_input ? decode_graph_node_ : decode_eager_node_;
    build_node_variant_pack(decode_node,
                            x,
                            cos_pos,
                            sin_pos,
                            decode_attn_mask_,
                            kv_cache,
                            input_params,
                            /*is_prefill=*/false,
                            node_id,
                            use_graph_decode_input);
    st = execute_node(decode_node, node_id + 1000, event, event_flag);
    LOG_IF(FATAL, st != 0) << model_name_
                           << "excute decode layer fail, error code: " << st;
  }

  return at_placeholder_;
}

void NpuQwen3DecoderLayerImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    at::Tensor& attn_mask,
    KVCache& kv_cache,
    ModelInputParams& input_params,
    bool is_prefill,
    int node_id,
    bool use_graph_decode_input) {
  internal_tensors_ = atb_speed::Utils::AtTensor2Tensor(x);
  if (::xllm::KernelConfig::get_instance().enable_interlayer_addnorm() &&
      residual_.defined()) {
    residual_tensors_ = atb_speed::Utils::AtTensor2Tensor(residual_);
  }
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER) = internal_tensors_;
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 1) =
      atb_speed::Utils::AtTensor2Tensor(cos_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 2) =
      atb_speed::Utils::AtTensor2Tensor(sin_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 3) =
      atb_speed::Utils::AtTensor2Tensor(attn_mask);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 6) =
      atb_speed::Utils::AtTensor2Tensor(
          input_params.attention.device.kv_seq_lens);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 6).hostData =
      input_params.attention.host.kv_seq_lens.data();
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 7) = placeholder_;
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 7).hostData =
      placeholder_vec_.data();
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 8) = placeholder_;
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 9) =
      atb_speed::Utils::AtTensor2Tensor(
          input_params.attention.device.block_tables);

  int input_idx = WEIGHT_COUNT_PER_LAYER + 11;
  if (is_rec_multi_round_mode()) {
    const auto* llmrec = input_params.llmrec_params();

    if (is_prefill) {
      node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 4) =
          atb_speed::Utils::AtTensor2Tensor(kv_cache.get_k_cache());
      node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 5) =
          atb_speed::Utils::AtTensor2Tensor(kv_cache.get_v_cache());
    } else {
      CHECK_LT(static_cast<size_t>(node_id), llmrec->unshared_k_caches.size());
      CHECK_LT(static_cast<size_t>(node_id), llmrec->unshared_v_caches.size());
      node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 4) =
          atb_speed::Utils::AtTensor2Tensor(llmrec->unshared_k_caches[node_id]);
      node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 5) =
          atb_speed::Utils::AtTensor2Tensor(llmrec->unshared_v_caches[node_id]);
    }
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 10) = placeholder_;
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11) =
        atb_speed::Utils::AtTensor2Tensor(llmrec->shared_k_caches[node_id]);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 12) =
        atb_speed::Utils::AtTensor2Tensor(llmrec->shared_v_caches[node_id]);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 13) =
        atb_speed::Utils::AtTensor2Tensor(llmrec->beam_width_tensor);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 14) =
        atb_speed::Utils::AtTensor2Tensor(llmrec->current_round_tensor);
    input_idx = WEIGHT_COUNT_PER_LAYER + 15;
  } else {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 4) =
        atb_speed::Utils::AtTensor2Tensor(kv_cache.get_k_cache());
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 5) =
        atb_speed::Utils::AtTensor2Tensor(kv_cache.get_v_cache());
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 10) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.attention.device.new_cache_slots);
  }

  if (is_prefill &&
      (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() ||
       ::xllm::KVCacheConfig::get_instance().enable_prefix_cache())) {
    node.variantPack.inTensors.at(input_idx++) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.attention.device.q_seq_lens);
    node.variantPack.inTensors.at(input_idx - 1).hostData =
        input_params.attention.host.q_seq_lens.data();
  }

  if (::xllm::KernelConfig::get_instance().enable_interlayer_addnorm() &&
      node_id > 0 && residual_.defined()) {
    node.variantPack.inTensors.at(input_idx++) = residual_tensors_;
  }

  if (flash_comm_enabled_) {
    CHECK(!flash_comm_token_counts_.empty())
        << "flash comm token counts must be set before Qwen3 layer forward.";
    const int64_t local_tokens = x.size(0);
    const int64_t hidden_size = x.size(-1);
    CHECK_LT(static_cast<size_t>(flash_comm_rank_),
             flash_comm_token_counts_.size());
    CHECK_EQ(local_tokens, flash_comm_token_counts_.at(flash_comm_rank_))
        << "flash comm local shard token count mismatch.";

    flash_comm_send_counts_.resize(flash_comm_token_counts_.size());
    flash_comm_recv_counts_ = flash_comm_token_counts_;
    for (size_t i = 0; i < flash_comm_token_counts_.size(); ++i) {
      flash_comm_send_counts_[i] = flash_comm_token_counts_[i] * hidden_size;
    }
    flash_comm_sdispls_ = make_displacements(flash_comm_send_counts_);
    flash_comm_rdispls_ = make_displacements(flash_comm_recv_counts_);
    flash_comm_send_count_ = {local_tokens};
    flash_comm_recv_count_ = {local_tokens * hidden_size};

    const int64_t total_tokens =
        flash_comm_rdispls_.back() + flash_comm_recv_counts_.back();
    fake_rs_shape_ = torch::empty({local_tokens, hidden_size}, x.options());
    fake_ag_shape_ = torch::empty({total_tokens, hidden_size}, x.options());
    fake_rs_shape_tensor_ =
        atb_speed::Utils::AtTensor2Tensor(fake_rs_shape_);
    fake_ag_shape_tensor_ =
        atb_speed::Utils::AtTensor2Tensor(fake_ag_shape_);

    node.variantPack.inTensors.at(input_idx++) = placeholder_;
    node.variantPack.inTensors.at(input_idx - 1).hostData =
        flash_comm_send_counts_.data();
    node.variantPack.inTensors.at(input_idx++) = placeholder_;
    node.variantPack.inTensors.at(input_idx - 1).hostData =
        flash_comm_sdispls_.data();
    node.variantPack.inTensors.at(input_idx++) = placeholder_;
    node.variantPack.inTensors.at(input_idx - 1).hostData =
        flash_comm_send_count_.data();
    node.variantPack.inTensors.at(input_idx++) = placeholder_;
    node.variantPack.inTensors.at(input_idx - 1).hostData =
        flash_comm_recv_counts_.data();
    node.variantPack.inTensors.at(input_idx++) = placeholder_;
    node.variantPack.inTensors.at(input_idx - 1).hostData =
        flash_comm_rdispls_.data();
    node.variantPack.inTensors.at(input_idx++) = placeholder_;
    node.variantPack.inTensors.at(input_idx - 1).hostData =
        flash_comm_recv_count_.data();
    node.variantPack.inTensors.at(input_idx++) = fake_rs_shape_tensor_;
    node.variantPack.inTensors.at(input_idx++) = fake_ag_shape_tensor_;
  }

  if (!is_prefill && use_graph_decode_input &&
      input_params.graph.tiling_data.defined()) {
    node.variantPack.inTensors.at(input_idx++) =
        atb_speed::Utils::AtTensor2Tensor(input_params.graph.tiling_data);
  }

  for (size_t i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    CHECK_THROW(node.inTensors.at(i) == nullptr,
                model_name_ << "inTensor " << i << "is NULL");
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
  }

  node.variantPack.outTensors.at(0) = internal_tensors_;
  if (::xllm::KernelConfig::get_instance().enable_interlayer_addnorm() &&
      (node_id < num_hidden_layers_ - 1) && residual_.defined()) {
    node.variantPack.outTensors.at(1) = residual_tensors_;
  }
}

}  // namespace layer
}  // namespace xllm
