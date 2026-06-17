#include <glog/logging.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/global_flags.h"
#include "core/framework/config/dit_config.h"

class RankGenerator {
 public:
  RankGenerator(int32_t tp,
                int32_t sp,
                int32_t cfg,
                int32_t dp,
                const std::string& group_order = "tp-sp-cfg-dp",
                int32_t rank_offset = 0)
      : tp_(tp), sp_(sp), cfg_(cfg), dp_(dp), rank_offset_(rank_offset) {
    world_size_ = tp * sp * cfg * dp;

    group_size_map_["tp"] = tp;
    group_size_map_["sp"] = sp;
    group_size_map_["cfg"] = cfg;
    group_size_map_["dp"] = dp;

    auto full_order = group_order;
    for (const auto& group_size_pair : group_size_map_) {
      const std::string& group_name = group_size_pair.first;
      int32_t group_size = group_size_pair.second;

      if (full_order.find(group_name) == std::string::npos) {
        if (group_size != 1) {
          LOG(FATAL) << "The size of (" << group_name << ") is (" << group_size
                     << "), but you haven't specified it in order ("
                     << full_order << ").";
        } else {
          full_order = full_order + "-" + group_name;
        }
      }
    }

    group_order_ = full_order;

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

    ordered_group_name_ = split(group_order_, '-');
    for (const std::string& token : ordered_group_name_) {
      auto it = group_size_map_.find(token);
      if (it != group_size_map_.end()) {
        ordered_group_size_.push_back(it->second);
      }
    }

    LOG(INFO) << "RankGenerator initialized with tp=" << tp << ", sp=" << sp
              << ", cfg=" << cfg << ", dp=" << dp << ", order=" << group_order_
              << ", world_size=" << world_size_;

    if (::xllm::DiTConfig::get_instance().dit_debug_print()) {
      debug_print();
    }
  }

  std::vector<std::vector<int32_t>> get_ranks(const std::string& group_query) {
    std::vector<bool> mask = get_mask(group_query);
    std::vector<std::vector<int32_t>> ranks =
        generate_masked_orthogonal_rank_groups(
            world_size_, ordered_group_size_, mask);
    if (rank_offset_ > 0) {
      for (auto& rank_group : ranks) {
        for (size_t i = 0; i < rank_group.size(); i++) {
          rank_group[i] += rank_offset_;
        }
      }
    }

    return ranks;
  }

  int32_t get_world_size() const { return world_size_; }
  const std::string& get_order() const { return group_order_; }
  int32_t get_tp() const { return tp_; }
  int32_t get_sp() const { return sp_; }
  int32_t get_cfg() const { return cfg_; }
  int32_t get_dp() const { return dp_; }

  void debug_print() {
    print_ranks("cfg");
    print_ranks("tp");
    print_ranks("sp");
    print_ranks("dp");
  }

  void print_ranks(const std::string& group_query) {
    auto ranks = get_ranks(group_query);

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

 private:
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

  std::vector<bool> get_mask(const std::string& group_query) {
    std::vector<std::string> query_group_name = split(group_query, '-');
    std::vector<bool> mask(ordered_group_name_.size(), false);

    for (const std::string& group_name : query_group_name) {
      auto it = std::find(
          ordered_group_name_.begin(), ordered_group_name_.end(), group_name);
      if (it != ordered_group_name_.end()) {
        size_t index = std::distance(ordered_group_name_.begin(), it);
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

 private:
  int32_t tp_;
  int32_t sp_;
  int32_t cfg_;
  int32_t dp_;
  int32_t rank_offset_;
  int32_t world_size_;
  std::string group_order_;
  std::vector<int32_t> ordered_group_size_;
  std::vector<std::string> ordered_group_name_;
  std::unordered_map<std::string, int32_t> group_size_map_;
};
