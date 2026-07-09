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

#pragma once
#include <glog/logging.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/global_flags.h"
#include "core/framework/config/dit_config.h"

namespace xllm {

/*
group_ranks: the rank sizes of the sub groups
group_order: the priority of the sub groups, the group with a
    higher priority will be assigned closer rank ids.
world_size: the global world_size
*/
class RankGenerator {
 public:
  explicit RankGenerator(int32_t world_size = 1, int32_t rank_offset = 0)
      : rank_offset_(rank_offset), world_size_(world_size) {}

  std::unordered_map<std::string, std::vector<std::vector<int32_t>>>
  get_ranks_mapping(std::vector<int32_t>& group_ranks,
                    std::vector<std::string>& group_order) {
    CHECK(!group_ranks.empty() && group_ranks.size() != 0)
        << "The RankGenerator expected to initialize with group_ranks that "
           "contains the ranks of sub groups"
        << ", but got an empty group_ranks";

    CHECK(!group_order.empty() && group_order.size() != 0)
        << "The RankGenerator expected to initialize with group_order that "
           "indicates the priority of sub groups"
        << ", but got empty string";

    int32_t product_size = 1;
    for (const auto& group_rank : group_ranks) {
      product_size *= group_rank;
    }

    bool is_single_group = (group_ranks.size() == 1);
    if (is_single_group && group_ranks[0] != world_size_) {
      if (world_size_ % group_ranks[0] != 0) {
        LOG(FATAL) << "The world_size could not be divided by "
                   << group_order[0] << "_size, "
                   << "got world_size: " << world_size_ << ", "
                   << group_order[0] << "_size: " << group_ranks[0] << ".";
      }
      LOG(WARNING) << "The sub group size does not equal world_size"
                   << ", we will assign the " << group_order[0]
                   << ", with sub group size: " << group_ranks[0];
      group_ranks.emplace_back(world_size_ / group_ranks[0]);
      group_order.emplace_back("place_holder");
    } else if (world_size_ != product_size) {
      LOG(FATAL) << "The world_size does not equals the product of sub "
                    "group sizes, "
                 << "got world_size: " << world_size_
                 << ", sub groups: " << group_order[0]
                 << "sub groups sizes: " << group_ranks[0];
    }

    CHECK(group_order.size() == group_ranks.size())
        << "The size of group_ranks does not equals the size of group_order.";

    std::stringstream ss;
    for (size_t i = 0; i < group_ranks.size(); i++) {
      ss << group_order[i] << "=" << group_ranks[i] << ", ";
    }

    LOG(INFO) << "RankGenerator initialized with " << ss.str()
              << "world_size=" << world_size_;

    auto group_mapping =
        std::unordered_map<std::string, std::vector<std::vector<int32_t>>>();
    group_mapping.reserve(group_order.size());

    for (auto& group_name : group_order) {
      auto sub_group_ranks = get_ranks(group_name, group_ranks, group_order);
      if (::xllm::DiTConfig::get_instance().dit_debug_print()) {
        print_ranks(group_name, sub_group_ranks);
      }
      group_mapping.insert({group_name, sub_group_ranks});
    }
    return group_mapping;
  }

  int32_t get_world_size() const { return world_size_; }

 private:
  std::vector<std::vector<int32_t>> get_ranks(
      const std::string& group_query,
      const std::vector<int32_t>& group_ranks,
      const std::vector<std::string>& group_order) {
    std::vector<bool> mask = get_mask(group_query, group_order);
    std::vector<std::vector<int32_t>> ranks =
        generate_masked_orthogonal_rank_groups(world_size_, group_ranks, mask);
    if (rank_offset_ > 0) {
      for (auto& rank_group : ranks) {
        for (size_t i = 0; i < rank_group.size(); i++) {
          rank_group[i] += rank_offset_;
        }
      }
    }

    return ranks;
  }

  void print_ranks(const std::string& group_query,
                   const std::vector<std::vector<int32_t>>& ranks) {
    std::stringstream ss;
    ss << "Ranks for query '" << group_query << "':" << std::endl;
    for (size_t i = 0; i < ranks.size(); i++) {
      ss << "  Group " << i << ": [";
      for (size_t j = 0; j < ranks[i].size(); j++) {
        ss << ranks[i][j];
        if (j < ranks[i].size() - 1) ss << ", ";
      }
      ss << "]" << std::endl;
    }
    LOG(INFO) << ss.str();
  }

  std::vector<int32_t> prefix_product(const std::vector<int32_t>& group_size,
                                      int32_t init = 1) {
    std::vector<int32_t> prefix_product_sizes;
    prefix_product_sizes.push_back(init);
    for (int32_t size : group_size) {
      init = init * size;
      prefix_product_sizes.push_back(init);
    }
    return prefix_product_sizes;
  }

  int32_t inner_product(const std::vector<int32_t>& a,
                        const std::vector<int32_t>& b) {
    int32_t result = 0;
    for (size_t i = 0; i < a.size(); i++) {
      result += a[i] * b[i];
    }
    return result;
  }

  std::vector<int32_t> decompose(int32_t index,
                                 const std::vector<int32_t>& shape,
                                 const std::vector<int32_t>& stride = {}) {
    std::vector<int32_t> idx;
    std::vector<int32_t> actual_stride;

    if (stride.empty()) {
      actual_stride = prefix_product(shape);
    } else {
      actual_stride = stride;
    }

    for (size_t i = 0; i < shape.size(); i++) {
      int32_t d = actual_stride[i];
      int32_t s = shape[i];
      idx.push_back((index / d) % s);
    }

    int32_t sum = 0;
    for (size_t i = 0; i < idx.size(); i++) {
      sum += idx[i] * actual_stride[i];
    }

    if (sum != index) {
      std::stringstream ss;
      ss << "idx " << index << " with shape [";
      for (size_t i = 0; i < shape.size(); i++) {
        ss << shape[i];
        if (i < shape.size() - 1) ss << ", ";
      }
      ss << "] mismatch the return idx [";
      for (size_t i = 0; i < idx.size(); i++) {
        ss << idx[i];
        if (i < idx.size() - 1) ss << ", ";
      }
      ss << "]";
      LOG(INFO) << ss.str();
    }

    return idx;
  }

  std::vector<std::vector<int32_t>> generate_masked_orthogonal_rank_groups(
      int32_t world_size,
      const std::vector<int32_t>& parallel_size,
      const std::vector<bool>& mask) {
    std::vector<int32_t> queried_group_size;
    std::vector<int32_t> unqueried_group_size;
    for (size_t i = 0; i < parallel_size.size(); i++) {
      if (mask[i]) {
        queried_group_size.push_back(parallel_size[i]);
      } else {
        unqueried_group_size.push_back(parallel_size[i]);
      }
    }
    std::vector<int32_t> global_group_stride = prefix_product(parallel_size);
    std::vector<int32_t> queried_group_stride;
    std::vector<int32_t> unqueried_group_stride;
    for (size_t i = 0; i < parallel_size.size(); i++) {
      if (mask[i]) {
        queried_group_stride.push_back(global_group_stride[i]);
      } else {
        unqueried_group_stride.push_back(global_group_stride[i]);
      }
    }
    std::vector<int32_t> queried_group_prefix =
        prefix_product(queried_group_size);
    // group size equals to the product of queryed group type sizes;
    int32_t group_size = queried_group_prefix.back();
    int32_t num_of_group = world_size / group_size;
    std::vector<std::vector<int32_t>> ranks;
    for (int32_t group_index = 0; group_index < num_of_group; group_index++) {
      std::vector<int32_t> decomposed_group_idx =
          decompose(group_index, unqueried_group_size);
      std::vector<int32_t> rank;
      for (int32_t rank_in_group = 0; rank_in_group < group_size;
           rank_in_group++) {
        std::vector<int32_t> decomposed_rank_idx =
            decompose(rank_in_group, queried_group_size);
        int32_t calculated_rank =
            inner_product(decomposed_rank_idx, queried_group_stride) +
            inner_product(decomposed_group_idx, unqueried_group_stride);
        rank.push_back(calculated_rank);
      }
      ranks.push_back(rank);
    }

    return ranks;
  }

  std::vector<bool> get_mask(const std::string& group_query,
                             const std::vector<std::string>& group_order) {
    auto split = [](const std::string& s,
                    char delimiter) -> std::vector<std::string> {
      std::vector<std::string> tokens;
      std::string token;
      std::istringstream tokenStream(s);
      while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
      }
      return tokens;
    };
    std::vector<std::string> query_group_name = split(group_query, '-');
    std::vector<bool> mask(group_order.size(), false);

    for (const std::string& group_name : query_group_name) {
      auto it = std::find(group_order.begin(), group_order.end(), group_name);
      if (it != group_order.end()) {
        size_t index = std::distance(group_order.begin(), it);
        mask[index] = true;
      }
    }

    return mask;
  }

  std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
      tokens.push_back(token);
    }
    return tokens;
  }

  int32_t rank_offset_;
  int32_t world_size_;
};

}  // namespace xllm
