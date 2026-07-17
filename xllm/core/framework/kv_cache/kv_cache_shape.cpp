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

#include "framework/kv_cache/kv_cache_shape.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <utility>

#include "framework/kv_cache/kv_cache_utils.h"
#include "util/utils.h"
#include "worker.pb.h"

namespace xllm {

namespace {
constexpr int32_t kNzAlignment = 16;

int64_t get_local_head_count(int64_t total_head_count, int64_t world_size) {
  CHECK_GT(world_size, 0) << "world_size must be positive.";
  return std::max<int64_t>(1, total_head_count / world_size);
}

std::vector<int64_t> repeated_field_to_vector(
    const google::protobuf::RepeatedField<int64_t>& field) {
  return std::vector<int64_t>(field.begin(), field.end());
}

void add_shape_to_proto(const std::vector<int64_t>& shape,
                        google::protobuf::RepeatedField<int64_t>* proto_shape) {
  CHECK(proto_shape != nullptr) << "proto_shape must not be nullptr.";
  proto_shape->Reserve(static_cast<int32_t>(shape.size()));
  for (const int64_t dim : shape) {
    proto_shape->Add(dim);
  }
}

void transpose_dim_1_and_2(std::vector<int64_t>* shape) {
  CHECK(shape != nullptr) << "shape must not be nullptr.";
  if (shape->size() <= 2) {
    return;
  }
  std::swap((*shape)[1], (*shape)[2]);
}

}  // namespace

KVCacheShape::KVCacheShape(const KVCacheCapacity& kv_cache_cap,
                           const ModelArgs& model_args,
                           int64_t world_size) {
  CHECK_GT(world_size, 0) << "world_size must be positive.";
  CHECK_GT(kv_cache_cap.block_size(), 0) << "block_size must be positive.";

  if (util::is_target_model_type(model_args.model_type(),
                                 /*target_type=*/"deepseek_v4",
                                 /*match_mtp=*/true)) {
    init_dsv4_pool_shape(kv_cache_cap);
    return;
  }

  const bool enable_lighting_indexer = model_args.index_n_heads() > 0;
  const bool enable_linear_attention = has_linear_attention_layers(model_args);
  CHECK(!(enable_lighting_indexer && enable_linear_attention))
      << "KVCacheShape does not support index_cache_shape with "
      << "conv_cache_shape/ssm_cache_shape simultaneously.";

  init_key_cache_shape(kv_cache_cap, model_args, world_size);
  init_value_cache_shape(kv_cache_cap, model_args, world_size);

  if (enable_lighting_indexer) {
    init_index_cache_shape(kv_cache_cap, model_args);
  }

  if (enable_linear_attention) {
    init_conv_cache_shape(kv_cache_cap, model_args, world_size);
    init_ssm_cache_shape(kv_cache_cap, model_args, world_size);
  }

  apply_device_layout(model_args);

  if (enable_lighting_indexer && kv_cache_cap.enable_indexer_cache_quant()) {
    init_index_cache_scale_shape();
  }
}

const std::vector<int64_t>& KVCacheShape::key_cache_shape() const {
  CHECK(key_cache_shape_.has_value()) << "key_cache_shape is not initialized.";
  return *key_cache_shape_;
}

const std::vector<int64_t>& KVCacheShape::value_cache_shape() const {
  if (!value_cache_shape_.has_value()) {
    return empty_shape();
  }
  return *value_cache_shape_;
}

const std::vector<int64_t>& KVCacheShape::index_cache_shape() const {
  if (!index_cache_shape_.has_value()) {
    return empty_shape();
  }
  return *index_cache_shape_;
}

const std::vector<int64_t>& KVCacheShape::index_cache_scale_shape() const {
  if (!index_cache_scale_shape_.has_value()) {
    return empty_shape();
  }
  return *index_cache_scale_shape_;
}

const std::vector<int64_t>& KVCacheShape::conv_cache_shape() const {
  if (!conv_cache_shape_.has_value()) {
    return empty_shape();
  }
  return *conv_cache_shape_;
}

const std::vector<int64_t>& KVCacheShape::ssm_cache_shape() const {
  if (!ssm_cache_shape_.has_value()) {
    return empty_shape();
  }
  return *ssm_cache_shape_;
}

bool KVCacheShape::has_key_cache_shape() const {
  return key_cache_shape_.has_value();
}

bool KVCacheShape::has_value_cache_shape() const {
  return value_cache_shape_.has_value();
}

bool KVCacheShape::has_index_cache_shape() const {
  return index_cache_shape_.has_value();
}

bool KVCacheShape::has_index_cache_scale_shape() const {
  return index_cache_scale_shape_.has_value();
}

bool KVCacheShape::has_conv_cache_shape() const {
  return conv_cache_shape_.has_value();
}

bool KVCacheShape::has_ssm_cache_shape() const {
  return ssm_cache_shape_.has_value();
}

void KVCacheShape::print_shapes() const {
  if (shape_kind_ == ShapeKind::GROUPED_POOL) {
    print_dsv4_pool_shape();
    return;
  }

  if (has_key_cache_shape()) {
    LOG(INFO) << "Initializing k cache with shape: [" << key_cache_shape()
              << "]";
  }
  if (has_value_cache_shape()) {
    LOG(INFO) << "Initializing v cache with shape: [" << value_cache_shape()
              << "]";
  }
  if (has_index_cache_shape()) {
    LOG(INFO) << "Initializing indexer cache with shape: ["
              << index_cache_shape() << "]";
  }
  if (has_index_cache_scale_shape()) {
    LOG(INFO) << "Initializing indexer cache scale with shape: ["
              << index_cache_scale_shape() << "]";
  }
  if (has_conv_cache_shape()) {
    LOG(INFO) << "Initializing conv cache with shape: [" << conv_cache_shape()
              << "]";
  }
  if (has_ssm_cache_shape()) {
    LOG(INFO) << "Initializing ssm cache with shape: [" << ssm_cache_shape()
              << "]";
  }
}

void KVCacheShape::init_dsv4_pool_shape(const KVCacheCapacity& kv_cache_cap) {
  shape_kind_ = ShapeKind::GROUPED_POOL;
  key_cache_shape_ = std::vector<int64_t>{kv_cache_cap.swa_count(),
                                          kv_cache_cap.c4_count(),
                                          kv_cache_cap.c128_count()};
}

void KVCacheShape::print_dsv4_pool_shape() const {
  CHECK(has_key_cache_shape())
      << "DeepSeek V4 cache shape must contain cache pool counts.";
  const std::vector<int64_t>& pool_counts = key_cache_shape();
  CHECK_GE(pool_counts.size(), 3)
      << "DeepSeek V4 cache shape must be [swa_count, c4_count, c128_count].";
  LOG(INFO) << "Initializing DSV4 kv cache with shape: [swa_count="
            << pool_counts[0] << ", c4_count=" << pool_counts[1]
            << ", c128_count=" << pool_counts[2] << "]";
}

void KVCacheShape::to_proto(proto::KVCacheShape* proto_shape) const {
  CHECK(proto_shape != nullptr) << "proto_shape must not be nullptr.";
  proto_shape->Clear();
  proto_shape->set_grouped_cache_layout(has_grouped_cache_layout());
  add_shape_to_proto(key_cache_shape(), proto_shape->mutable_key_cache_shape());
  if (has_value_cache_shape()) {
    add_shape_to_proto(value_cache_shape(),
                       proto_shape->mutable_value_cache_shape());
  }
  if (has_index_cache_shape()) {
    add_shape_to_proto(index_cache_shape(),
                       proto_shape->mutable_index_cache_shape());
  }
  if (has_index_cache_scale_shape()) {
    add_shape_to_proto(index_cache_scale_shape(),
                       proto_shape->mutable_index_cache_scale_shape());
  }
  if (has_conv_cache_shape()) {
    add_shape_to_proto(conv_cache_shape(),
                       proto_shape->mutable_conv_cache_shape());
    add_shape_to_proto(ssm_cache_shape(),
                       proto_shape->mutable_ssm_cache_shape());
  }
}

KVCacheShape KVCacheShape::from_proto(const proto::KVCacheShape& proto_shape) {
  KVCacheShape kv_cache_shape;
  if (proto_shape.grouped_cache_layout()) {
    kv_cache_shape.shape_kind_ = ShapeKind::GROUPED_POOL;
  }
  kv_cache_shape.key_cache_shape_ =
      repeated_field_to_vector(proto_shape.key_cache_shape());
  if (proto_shape.value_cache_shape_size() > 0) {
    kv_cache_shape.value_cache_shape_ =
        repeated_field_to_vector(proto_shape.value_cache_shape());
  }
  if (proto_shape.index_cache_shape_size() > 0) {
    kv_cache_shape.index_cache_shape_ =
        repeated_field_to_vector(proto_shape.index_cache_shape());
  }
  if (proto_shape.index_cache_scale_shape_size() > 0) {
    kv_cache_shape.index_cache_scale_shape_ =
        repeated_field_to_vector(proto_shape.index_cache_scale_shape());
  }
  if (proto_shape.conv_cache_shape_size() > 0) {
    kv_cache_shape.conv_cache_shape_ =
        repeated_field_to_vector(proto_shape.conv_cache_shape());
  }
  if (proto_shape.ssm_cache_shape_size() > 0) {
    kv_cache_shape.ssm_cache_shape_ =
        repeated_field_to_vector(proto_shape.ssm_cache_shape());
  }
  return kv_cache_shape;
}

void KVCacheShape::init_key_cache_shape(const KVCacheCapacity& kv_cache_cap,
                                        const ModelArgs& model_args,
                                        int64_t world_size) {
  if (model_args.enable_mla()) {
#if defined(USE_NPU)
    if (use_npu_nz_kv_cache_layout(model_args.model_type())) {
      key_cache_shape_ = std::vector<int64_t>{
          kv_cache_cap.n_blocks(),
          util::ceil_div(model_args.kv_lora_rank(), kNzAlignment),
          kv_cache_cap.block_size(),
          kNzAlignment};
      return;
    }
#endif
    key_cache_shape_ = std::vector<int64_t>{kv_cache_cap.n_blocks(),
                                            kv_cache_cap.block_size(),
                                            1,
                                            model_args.kv_lora_rank()};
    return;
  }

  const int64_t total_kv_head_count =
      model_args.n_kv_heads().value_or(model_args.n_heads());
  const int64_t local_kv_head_count =
      get_local_head_count(total_kv_head_count, world_size);
  key_cache_shape_ = std::vector<int64_t>{kv_cache_cap.n_blocks(),
                                          kv_cache_cap.block_size(),
                                          local_kv_head_count,
                                          model_args.head_dim()};
}

void KVCacheShape::init_value_cache_shape(const KVCacheCapacity& kv_cache_cap,
                                          const ModelArgs& model_args,
                                          int64_t world_size) {
  if (model_args.enable_mla()) {
#if defined(USE_NPU)
    if (use_npu_nz_kv_cache_layout(model_args.model_type())) {
      value_cache_shape_ = std::vector<int64_t>{
          kv_cache_cap.n_blocks(),
          util::ceil_div(model_args.qk_rope_head_dim(), kNzAlignment),
          kv_cache_cap.block_size(),
          kNzAlignment};
      return;
    }
#endif
    value_cache_shape_ = std::vector<int64_t>{kv_cache_cap.n_blocks(),
                                              kv_cache_cap.block_size(),
                                              1,
                                              model_args.qk_rope_head_dim()};
    return;
  }

  const int64_t total_kv_head_count =
      model_args.n_kv_heads().value_or(model_args.n_heads());
  const int64_t local_kv_head_count =
      get_local_head_count(total_kv_head_count, world_size);
  value_cache_shape_ = std::vector<int64_t>{kv_cache_cap.n_blocks(),
                                            kv_cache_cap.block_size(),
                                            local_kv_head_count,
                                            model_args.head_dim()};
}

void KVCacheShape::init_index_cache_shape(const KVCacheCapacity& kv_cache_cap,
                                          const ModelArgs& model_args) {
  index_cache_shape_ = std::vector<int64_t>{kv_cache_cap.n_blocks(),
                                            kv_cache_cap.block_size(),
                                            1,
                                            model_args.index_head_dim()};
}

void KVCacheShape::init_index_cache_scale_shape() {
  CHECK(index_cache_shape_.has_value())
      << "index_cache_shape must be initialized before index cache scale.";
  CHECK_EQ(index_cache_shape_->size(), 4)
      << "index_cache_shape must be [blocks, heads, block_size, head_dim].";
  index_cache_scale_shape_ = std::vector<int64_t>{(*index_cache_shape_)[0],
                                                  (*index_cache_shape_)[1],
                                                  (*index_cache_shape_)[2]};
}

void KVCacheShape::init_conv_cache_shape(const KVCacheCapacity& kv_cache_cap,
                                         const ModelArgs& model_args,
                                         int64_t world_size) {
  const int64_t local_linear_k_head_count =
      get_local_head_count(model_args.linear_num_key_heads(), world_size);
  const int64_t local_linear_v_head_count =
      get_local_head_count(model_args.linear_num_value_heads(), world_size);
  const int64_t conv_state_len = kv_cache_cap.linear_conv_state_len() > 0
                                     ? kv_cache_cap.linear_conv_state_len()
                                     : model_args.linear_conv_kernel_dim() - 1;

  conv_cache_shape_ = std::vector<int64_t>{
      kv_cache_cap.num_linear_state_blocks(),
      conv_state_len,
      model_args.linear_key_head_dim() * local_linear_k_head_count * 2 +
          model_args.linear_key_head_dim() * local_linear_v_head_count};
}

void KVCacheShape::init_ssm_cache_shape(const KVCacheCapacity& kv_cache_cap,
                                        const ModelArgs& model_args,
                                        int64_t world_size) {
  const int64_t local_linear_v_head_count =
      get_local_head_count(model_args.linear_num_value_heads(), world_size);
  const int64_t checkpoint_stride =
      std::max<int64_t>(kv_cache_cap.linear_ssm_checkpoint_stride(), 1);
  ssm_cache_shape_ = std::vector<int64_t>{
      kv_cache_cap.num_linear_state_blocks() * checkpoint_stride,
      local_linear_v_head_count,
      model_args.linear_key_head_dim(),
      model_args.linear_value_head_dim()};
}

void KVCacheShape::apply_device_layout(const ModelArgs& model_args) {
  // default k/v cache layout: [n_blocks, block_size, n_head, head_dim]
  // => mlu/ilu k/v cache layout: [n_blocks, n_head, block_size, head_dim]
#if defined(USE_MLU) || defined(USE_ILU)
  if (key_cache_shape_.has_value()) {
    transpose_dim_1_and_2(&*key_cache_shape_);
  }
  if (value_cache_shape_.has_value()) {
    transpose_dim_1_and_2(&*value_cache_shape_);
  }
  if (index_cache_shape_.has_value()) {
    transpose_dim_1_and_2(&*index_cache_shape_);
  }
#endif

#if defined(USE_MLU) || defined(USE_DCU)
  if (model_args.enable_mla()) {
    CHECK(key_cache_shape_.has_value())
        << "key_cache_shape is not initialized.";
    CHECK_GE(key_cache_shape_->size(), 4) << "invalid mla key_cache_shape.";
    (*key_cache_shape_)[3] =
        model_args.kv_lora_rank() + model_args.qk_rope_head_dim();
    value_cache_shape_.reset();
  }
#else
  static_cast<void>(model_args);
#endif
}

const std::vector<int64_t>& KVCacheShape::empty_shape() {
  static const std::vector<int64_t> kEmptyShape;
  return kEmptyShape;
}

}  // namespace xllm
