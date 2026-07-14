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

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/model/model_args.h"

namespace xllm {

namespace proto {
class KVCacheShape;
}

class KVCacheShape final {
 public:
  KVCacheShape() = default;

  KVCacheShape(const KVCacheCapacity& kv_cache_cap,
               const ModelArgs& model_args,
               int64_t world_size);

  const std::vector<int64_t>& key_cache_shape() const;
  const std::vector<int64_t>& value_cache_shape() const;
  const std::vector<int64_t>& index_cache_shape() const;
  const std::vector<int64_t>& index_cache_scale_shape() const;
  const std::vector<int64_t>& conv_cache_shape() const;
  const std::vector<int64_t>& ssm_cache_shape() const;

  bool has_key_cache_shape() const;
  bool has_value_cache_shape() const;
  bool has_index_cache_shape() const;
  bool has_index_cache_scale_shape() const;
  bool has_conv_cache_shape() const;
  bool has_ssm_cache_shape() const;

  void print_shapes() const;

  void to_proto(proto::KVCacheShape* proto_shape) const;
  static KVCacheShape from_proto(const proto::KVCacheShape& proto_shape);

 private:
  enum class ShapeKind : int8_t {
    NORMAL = 0,
    DSV4_POOL = 1,
  };

  void init_dsv4_pool_shape(const KVCacheCapacity& kv_cache_cap);
  void init_key_cache_shape(const KVCacheCapacity& kv_cache_cap,
                            const ModelArgs& model_args,
                            int64_t world_size);
  void init_value_cache_shape(const KVCacheCapacity& kv_cache_cap,
                              const ModelArgs& model_args,
                              int64_t world_size);
  void init_index_cache_shape(const KVCacheCapacity& kv_cache_cap,
                              const ModelArgs& model_args);
  void init_index_cache_scale_shape();
  void init_conv_cache_shape(const KVCacheCapacity& kv_cache_cap,
                             const ModelArgs& model_args,
                             int64_t world_size);
  void init_ssm_cache_shape(const KVCacheCapacity& kv_cache_cap,
                            const ModelArgs& model_args,
                            int64_t world_size);
  void apply_device_layout(const ModelArgs& model_args);
  void print_dsv4_pool_shape() const;

  static const std::vector<int64_t>& empty_shape();

 private:
  ShapeKind shape_kind_ = ShapeKind::NORMAL;
  std::optional<std::vector<int64_t>> key_cache_shape_;
  std::optional<std::vector<int64_t>> value_cache_shape_;

  // for index cache
  std::optional<std::vector<int64_t>> index_cache_shape_;
  std::optional<std::vector<int64_t>> index_cache_scale_shape_;

  // for linear attention
  std::optional<std::vector<int64_t>> conv_cache_shape_;
  std::optional<std::vector<int64_t>> ssm_cache_shape_;
};

}  // namespace xllm
