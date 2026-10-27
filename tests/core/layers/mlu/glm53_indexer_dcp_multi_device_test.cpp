/* Copyright 2026 The xLLM Authors.

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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/layers/mlu/dcp_decode_context.h"
#include "core/layers/mlu/indexer.h"
#include "core/layers/mlu/multi_device_test_utils.h"
#include "core/platform/device.h"
#include "framework/parallel_state/collective_communicator.h"
#include "framework/parallel_state/parallel_state.h"

namespace xllm::layer {
namespace {
constexpr int32_t kRanks = 8;
constexpr int32_t kCp = 4;
constexpr int32_t kTp = 2;
constexpr int32_t kTokens = 8192;
constexpr int32_t kHeads = 32;
constexpr int32_t kDim = 128;
constexpr int32_t kTopk = 2048;
constexpr int32_t kPage = 16;
constexpr int32_t kStripe = kPage * kCp;
constexpr int32_t kHidden = 6144;
constexpr int32_t kLora = 2048;
constexpr int32_t kQueryTile = 32;
constexpr int32_t kKeyTile = 2048;
constexpr float kScale = 1.0f / 64.0f;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

struct Stats {
  int64_t rows = 0;
  int64_t columns = 0;
  int64_t ordered_equal = 0;
  int64_t set_equal = 0;
  int64_t cutoff_swaps = 0;
  int64_t score_count = 0;
  int64_t score_bits = 0;
  uint64_t max_ulp = 0;
  double max_abs = 0;
  double max_rel = 0;
  double max_normalized = 0;
  int64_t final_differences = 0;
  int64_t cutoff_final_diff = 0;
  int64_t rounding_final_diff = 0;
  int64_t intermediate_final_diff = 0;
  int64_t merge_tie_diff = 0;
};

class Pages final {
 public:
  Pages(int32_t batch, int32_t history, int32_t seed)
      : batch_(batch),
        history_(history),
        query_(kTokens / batch),
        per_req_((history + query_ + kStripe - 1) / kStripe),
        count_(batch * per_req_ + 7),
        pages_(count_),
        requests_(count_, /*value=*/-1),
        logical_(count_, /*value=*/-1) {
    std::iota(pages_.begin(), pages_.end(), 0);
    std::mt19937 generator(seed);
    std::shuffle(pages_.begin(), pages_.end(), generator);
    for (int32_t req = 0; req < batch_; ++req) {
      for (int32_t page = 0; page < per_req_; ++page) {
        const int32_t physical = pages_[req * per_req_ + page];
        requests_[physical] = req;
        logical_[physical] = page;
      }
    }
  }

  int32_t slot(int32_t req, int32_t position) const {
    return pages_[req * per_req_ + position / kStripe] * kStripe +
           position % kStripe;
  }

  int32_t visible(int32_t row, int32_t shard) const {
    const int32_t end = history_ + row % query_ + 1;
    return end / kStripe * kPage +
           std::clamp(end % kStripe - shard * kPage, 0, kPage);
  }

  // Returns the request-local ordinal, or -1 for a foreign/noncausal slot.
  int32_t ordinal(int32_t row, int32_t global, int32_t shard) const {
    if (global < 0 || global >= count_ * kStripe ||
        global % kStripe / kPage != shard) {
      return -1;
    }
    const int32_t page = global / kStripe;
    const int32_t position = logical_[page] * kStripe + global % kStripe;
    if (requests_[page] != row / query_ || position < 0 ||
        position > history_ + row % query_) {
      return -1;
    }
    return logical_[page] * kPage + global % kPage;
  }

  torch::Tensor table() const {
    return torch::tensor(
               std::vector<int32_t>(pages_.begin(),
                                    pages_.begin() + batch_ * per_req_),
               torch::kInt32)
        .view({batch_, per_req_});
  }

  torch::Tensor new_slots() const {
    std::vector<int32_t> slots;
    slots.reserve(kTokens);
    for (int32_t row = 0; row < kTokens; ++row) {
      slots.emplace_back(slot(row / query_, history_ + row % query_));
    }
    return torch::tensor(slots, torch::kInt32);
  }

  torch::Tensor local_slots(int32_t req, int32_t shard) const {
    std::vector<int64_t> slots;
    slots.reserve(per_req_ * kPage);
    for (int32_t ordinal = 0; ordinal < per_req_ * kPage; ++ordinal) {
      const int32_t position =
          ordinal / kPage * kStripe + shard * kPage + ordinal % kPage;
      const int32_t global = slot(req, position);
      slots.emplace_back(global / kStripe * kPage + global % kPage);
    }
    return torch::tensor(slots, torch::kInt64);
  }

  int32_t batch() const { return batch_; }
  int32_t history() const { return history_; }
  int32_t query() const { return query_; }
  int32_t count() const { return count_; }
  int32_t local_length() const { return per_req_ * kPage; }

 private:
  int32_t batch_;
  int32_t history_;
  int32_t query_;
  int32_t per_req_;
  int32_t count_;
  std::vector<int32_t> pages_;
  std::vector<int32_t> requests_;
  std::vector<int32_t> logical_;
};

AttentionMetadata make_metadata(const Pages& pages,
                                const torch::Device& device) {
  const auto ints = torch::TensorOptions().device(device).dtype(torch::kInt32);
  AttentionMetadata metadata{};
  metadata.q_cu_seq_lens =
      torch::arange(/*start=*/0, kTokens + 1, pages.query(), ints);
  metadata.kv_cu_seq_lens =
      torch::arange(/*start=*/0,
                    pages.batch() * (pages.history() + pages.query()) + 1,
                    pages.history() + pages.query(),
                    ints);
  metadata.kv_seq_lens =
      torch::full({pages.batch()}, pages.history() + pages.query(), ints);
  metadata.block_table = pages.table().to(device);
  metadata.slot_mapping = pages.new_slots().to(device);
  metadata.max_query_len = pages.query();
  metadata.max_seq_len = pages.history() + pages.query();
  metadata.total_kv_len = pages.batch() * metadata.max_seq_len;
  metadata.is_prefill = true;
  metadata.is_chunked_prefill = pages.history() > 0;
  metadata.compute_dtype = "bfloat16";
  return metadata;
}

torch::Tensor random_bf16(torch::IntArrayRef shape) {
  return (torch::randn(shape, torch::kFloat32) * 0.02).to(torch::kBFloat16);
}

Indexer make_indexer(const ParallelArgs& parallel,
                     const torch::TensorOptions& options) {
  auto rope = std::make_shared<RotaryEmbeddingImpl>(
      /*rotary_dim=*/64,
      /*max_position_embeddings=*/73728,
      /*base=*/10000,
      /*interleaved=*/true,
      options);
  Indexer indexer(kHidden,
                  kHeads,
                  kDim,
                  /*qk_rope_head_dim=*/64,
                  kTopk,
                  kLora,
                  /*enable_fused_qk=*/true,
                  rope,
                  QuantArgs(),
                  parallel,
                  options);
  std::unordered_map<std::string, torch::Tensor> weights;
  weights.emplace("wq_b.weight", random_bf16({kHeads * kDim, kLora}));
  weights.emplace("wk.weight", random_bf16({kDim, kHidden}));
  weights.emplace("weights_proj.weight", random_bf16({kHeads, kHidden}));
  weights.emplace("k_norm.weight", torch::ones({kDim}, torch::kFloat32));
  weights.emplace("k_norm.bias", torch::randn({kDim}, torch::kFloat32) * 0.02);
  indexer->load_state_dict(StateDict(std::move(weights)));
  return indexer;
}

void write_reference_cache(torch::Tensor& cache,
                           const torch::Tensor& keys,
                           const Pages& pages,
                           int32_t shard) {
  std::vector<int64_t> rows;
  std::vector<int64_t> slots;
  rows.reserve(kTokens / kCp);
  slots.reserve(kTokens / kCp);
  for (int32_t row = 0; row < kTokens; ++row) {
    const int32_t global =
        pages.slot(row / pages.query(), pages.history() + row % pages.query());
    if (global % kStripe / kPage != shard) {
      continue;
    }
    rows.emplace_back(row);
    slots.emplace_back(global / kStripe * kPage + global % kPage);
  }
  const auto row_index = torch::tensor(rows, torch::kInt64).to(cache.device());
  const auto slot_index =
      torch::tensor(slots, torch::kInt64).to(cache.device());
  cache.view({-1, kDim})
      .index_copy_(/*dim=*/0,
                   slot_index,
                   keys.view({-1, kDim}).index_select(/*dim=*/0, row_index));
}

torch::Tensor scale_weights(const torch::Tensor& weights) {
  return (weights.to(torch::kFloat32).view({-1, kHeads}) * kScale)
      .to(torch::kBFloat16);
}

// Both reductions accumulate FP32. Only their outputs and the scaled weights
// round to BF16; rounding each product in the head reduction changes TopK.
torch::Tensor score_keys(const torch::Tensor& query,
                         const torch::Tensor& keys,
                         const torch::Tensor& scaled_weights) {
  const int64_t rows = query.size(/*dim=*/0);
  // Native BF16 matmul uses FP32 accumulation and rounds once on output.
  // Promoting operands to FP32 selects a different reduction order that can
  // cross an intermediate BF16 midpoint and change the strict cutoff set.
  auto qk =
      torch::relu(
          torch::matmul(
              keys,
              query.reshape({-1, kDim}).transpose(/*dim0=*/0, /*dim1=*/1)))
          .transpose(/*dim0=*/0, /*dim1=*/1)
          .contiguous()
          .view({rows, kHeads, keys.size(/*dim=*/0)});
  return torch::bmm(qk.transpose(/*dim0=*/1, /*dim1=*/2).contiguous(),
                    scaled_weights.unsqueeze(/*dim=*/2))
      .squeeze(/*dim=*/2);
}

torch::Tensor selector_scores(const torch::Tensor& query,
                              const torch::Tensor& keys,
                              const torch::Tensor& weights) {
  return score_keys(query, keys, scale_weights(weights));
}

struct ReferenceKeys {
  torch::Tensor local_slots;
  torch::Tensor values;
};

ReferenceKeys gather_keys(const torch::Tensor& cache,
                          const Pages& pages,
                          int32_t req,
                          int32_t shard) {
  auto local = pages.local_slots(req, shard).to(cache.device());
  auto values = cache.view({-1, kDim}).index_select(/*dim=*/0, local);
  return {std::move(local), std::move(values)};
}

torch::Tensor gather_scores(const torch::Tensor& pool,
                            const torch::Tensor& ordinals) {
  CHECK_EQ(pool.scalar_type(), torch::kBFloat16);
  CHECK_EQ(pool.dim(), 2);
  CHECK_EQ(ordinals.dim(), 2);
  CHECK_EQ(pool.size(/*dim=*/0), ordinals.size(/*dim=*/0));
  CHECK(ordinals.ge(/*other=*/-1).all().item<bool>());
  CHECK(ordinals.lt(pool.size(/*dim=*/1)).all().item<bool>());
  return pool.gather(/*dim=*/1, ordinals.clamp_min(/*min=*/0))
      .masked_fill(ordinals < 0, kNegInf);
}

torch::Tensor reuse_scores(const torch::Tensor& pool,
                           const torch::Tensor& slots,
                           const Pages& pages,
                           int32_t shard,
                           int32_t start) {
  auto cpu = slots.cpu().contiguous();
  const auto values = cpu.accessor<int32_t, 2>();
  std::vector<int64_t> ordinals;
  ordinals.reserve(slots.numel());
  for (int32_t row = 0; row < slots.size(/*dim=*/0); ++row) {
    for (int32_t col = 0; col < slots.size(/*dim=*/1); ++col) {
      const int32_t slot = values[row][col];
      const int32_t ordinal = pages.ordinal(start + row, slot, shard);
      CHECK(slot == -1 || ordinal >= 0) << "Invalid candidate slot=" << slot;
      ordinals.emplace_back(ordinal);
    }
  }
  return gather_scores(pool,
                       torch::tensor(ordinals, torch::kInt64)
                           .view(slots.sizes())
                           .to(pool.device()));
}

bool equal_reused_scores(const torch::Tensor& actual,
                         const torch::Tensor& expected) {
  return actual.scalar_type() == torch::kBFloat16 &&
         expected.scalar_type() == torch::kBFloat16 &&
         actual.sizes() == expected.sizes() && torch::equal(actual, expected);
}

DcpIndexerLocalCandidates select_reference(const IndexerCPPreOut& pre,
                                           const ReferenceKeys& keys,
                                           const Pages& pages,
                                           int32_t shard,
                                           int32_t start,
                                           int32_t rows,
                                           torch::Tensor& all_scores) {
  const auto device = keys.values.device();
  const auto& local = keys.local_slots;
  auto q = pre.q.narrow(/*dim=*/0, start, rows).view({rows, kHeads, kDim});
  auto scaled_weights =
      scale_weights(pre.weights.narrow(/*dim=*/0, start, rows));
  all_scores = torch::empty({rows, pages.local_length()},
                            keys.values.options().dtype(torch::kBFloat16));
  for (int32_t key = 0; key < pages.local_length(); key += kKeyTile) {
    const int32_t count = std::min(kKeyTile, pages.local_length() - key);
    all_scores.narrow(/*dim=*/1, key, count)
        .copy_(score_keys(
            q, keys.values.narrow(/*dim=*/0, key, count), scaled_weights));
  }
  std::vector<int32_t> counts;
  counts.reserve(rows);
  for (int32_t row = start; row < start + rows; ++row) {
    counts.emplace_back(pages.visible(row, shard));
  }
  auto visible =
      torch::tensor(counts, torch::kInt32).to(device).unsqueeze(/*dim=*/1);
  auto ordinal = torch::arange(pages.local_length(), local.options())
                     .unsqueeze(/*dim=*/0)
                     .expand({rows, -1});
  all_scores.masked_fill_(ordinal >= visible, kNegInf);
  const int32_t width = std::min(kTopk, pages.local_length());
  auto selected = ordinal.narrow(/*dim=*/1, /*start=*/0, width);
  if (std::any_of(counts.begin(), counts.end(), [](int32_t count) {
        return count > kTopk;
      })) {
    auto top = torch::topk(all_scores,
                           width,
                           /*dim=*/1,
                           /*largest=*/true,
                           /*sorted=*/true);
    selected = torch::where(visible <= kTopk, selected, std::get<1>(top));
  }
  auto chosen =
      local.index_select(/*dim=*/0, selected.flatten()).view({rows, width});
  auto global = (torch::floor_divide(chosen, kPage) * kStripe + shard * kPage +
                 chosen.remainder(kPage))
                    .to(torch::kInt32);
  auto slots = torch::full({rows, kTopk}, /*fill_value=*/-1, global.options());
  slots.narrow(/*dim=*/1, /*start=*/0, width).copy_(global);
  auto valid =
      torch::arange(kTopk, visible.options()).unsqueeze(/*dim=*/0) < visible;
  slots.masked_fill_(~valid, /*value=*/-1);
  auto scores = torch::full({rows, kTopk}, kNegInf, all_scores.options());
  scores.narrow(/*dim=*/1, /*start=*/0, width)
      .copy_(all_scores.gather(/*dim=*/1, selected));
  scores.masked_fill_(~valid, kNegInf);
  CHECK(equal_reused_scores(
      scores, reuse_scores(all_scores, slots, pages, shard, start)));
  return {std::move(scores), std::move(slots)};
}

bool valid_selection(const std::vector<int32_t>& selected,
                     const std::vector<float>& scores,
                     int32_t visible,
                     int32_t topk) {
  const int32_t count = std::min(visible, topk);
  if (static_cast<int32_t>(selected.size()) != count) {
    return false;
  }
  std::unordered_set<int32_t> unique(selected.begin(), selected.end());
  if (unique.size() != selected.size()) {
    return false;
  }
  for (int32_t index : selected) {
    if (index < 0 || index >= visible) {
      return false;
    }
  }
  if (visible <= topk) {
    for (int32_t index = 0; index < count; ++index) {
      if (selected[index] != index) {
        return false;
      }
    }
    return true;
  }
  std::vector<float> ordered(scores.begin(), scores.begin() + visible);
  std::nth_element(ordered.begin(),
                   ordered.begin() + topk - 1,
                   ordered.end(),
                   std::greater<float>());
  const float cutoff = ordered[topk - 1];
  int32_t required = 0;
  for (int32_t index = 0; index < visible; ++index) {
    required += scores[index] > cutoff;
  }
  int32_t found = 0;
  for (int32_t index : selected) {
    if (scores[index] < cutoff) {
      return false;
    }
    found += scores[index] > cutoff;
  }
  return found == required;
}

uint32_t ordered_bits(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  return (bits & 0x80000000U) != 0 ? ~bits : bits | 0x80000000U;
}

void compare_scores(const torch::Tensor& actual,
                    const torch::Tensor& ref,
                    int32_t start,
                    int32_t rank,
                    Stats& stats) {
  CHECK_EQ(actual.scalar_type(), torch::kFloat32);
  CHECK_EQ(ref.scalar_type(), torch::kBFloat16);
  CHECK(actual.sizes() == ref.sizes());
  auto scaled = actual * kScale;
  auto rounded = scaled.to(torch::kBFloat16);
  auto actual_cpu = rounded.to(torch::kFloat32).cpu().contiguous();
  auto scaled_cpu = scaled.cpu().contiguous();
  auto ref_cpu = ref.to(torch::kFloat32).cpu().contiguous();
  const auto before = scaled_cpu.accessor<float, 2>();
  const auto a = actual_cpu.accessor<float, 2>();
  const auto r = ref_cpu.accessor<float, 2>();
  for (int32_t row = 0; row < actual.size(/*dim=*/0); ++row) {
    for (int32_t col = 0; col < kTopk; ++col) {
      if (r[row][col] == kNegInf) {
        CHECK_EQ(a[row][col], kNegInf);
        continue;
      }
      CHECK(std::isfinite(a[row][col]));
      CHECK(std::isfinite(r[row][col]));
      const double error =
          std::abs(static_cast<double>(a[row][col]) - r[row][col]);
      const double relative =
          error / std::max(std::abs(static_cast<double>(r[row][col])), 1e-30);
      const uint32_t ab = ordered_bits(a[row][col]) >> 16;
      const uint32_t rb = ordered_bits(r[row][col]) >> 16;
      const uint64_t ulp = ab > rb ? ab - rb : rb - ab;
      stats.max_ulp = std::max(stats.max_ulp, ulp);
      stats.max_abs = std::max(stats.max_abs, error);
      stats.max_rel = std::max(stats.max_rel, relative);
      stats.max_normalized =
          std::max(stats.max_normalized, error / (1.0 + std::abs(r[row][col])));
      ++stats.score_count;
      const bool same = std::bit_cast<uint32_t>(a[row][col]) ==
                        std::bit_cast<uint32_t>(r[row][col]);
      if (!same && stats.score_count - stats.score_bits <= 8) {
        LOG(INFO) << "score_diff rank=" << rank << " row=" << start + row
                  << " col=" << col << " scaled=" << before[row][col]
                  << " rounded=" << a[row][col] << " ref=" << r[row][col]
                  << " ulp=" << ulp;
      }
      stats.score_bits += same;
    }
  }
}

torch::Tensor compare_selection(const torch::Tensor& actual,
                                const torch::Tensor& ref,
                                const torch::Tensor& selector,
                                const Pages& pages,
                                int32_t shard,
                                int32_t start,
                                Stats& stats) {
  auto actual_cpu = actual.cpu().contiguous();
  auto ref_cpu = ref.cpu().contiguous();
  CHECK_EQ(selector.scalar_type(), torch::kBFloat16);
  auto scores_cpu = selector.to(torch::kFloat32).cpu().contiguous();
  auto ordinals = torch::full(actual_cpu.sizes(),
                              /*fill_value=*/-1,
                              actual_cpu.options().dtype(torch::kInt64));
  auto mapped = ordinals.accessor<int64_t, 2>();
  const auto a = actual_cpu.accessor<int32_t, 2>();
  const auto r = ref_cpu.accessor<int32_t, 2>();
  const auto scores = scores_cpu.accessor<float, 2>();
  for (int32_t row = 0; row < actual.size(/*dim=*/0); ++row) {
    const int32_t visible = pages.visible(start + row, shard);
    const int32_t count = std::min(visible, kTopk);
    std::vector<int32_t> selected;
    selected.reserve(count);
    std::unordered_set<int32_t> reference;
    reference.reserve(count);
    for (int32_t col = 0; col < count; ++col) {
      const int32_t ordinal = pages.ordinal(start + row, a[row][col], shard);
      CHECK_GE(ordinal, 0) << "row=" << start + row << " col=" << col
                           << " slot=" << a[row][col] << " shard=" << shard;
      mapped[row][col] = ordinal;
      selected.emplace_back(ordinal);
      reference.emplace(r[row][col]);
      stats.ordered_equal += a[row][col] == r[row][col];
    }
    const std::vector<float> row_scores(
        scores[row].data(), scores[row].data() + pages.local_length());
    if (!valid_selection(selected, row_scores, visible, kTopk)) {
      auto ordered = row_scores;
      std::nth_element(ordered.begin(),
                       ordered.begin() + count - 1,
                       ordered.begin() + visible,
                       std::greater<float>());
      const float cutoff = ordered[count - 1];
      int32_t shown = 0;
      for (int32_t col = 0; col < count && shown < 8; ++col) {
        if (row_scores[selected[col]] < cutoff) {
          LOG(ERROR) << "below_cutoff shard=" << shard << " row=" << start + row
                     << " ordinal=" << selected[col]
                     << " score=" << row_scores[selected[col]]
                     << " cutoff=" << cutoff;
          ++shown;
        }
      }
      LOG(FATAL) << "Invalid BF16 TopK shard=" << shard
                 << " row=" << start + row;
    }
    int32_t overlap = 0;
    for (int32_t col = 0; col < count; ++col) {
      overlap += reference.count(a[row][col]);
    }
    stats.set_equal += overlap;
    stats.cutoff_swaps += count - overlap;
    stats.columns += count;
    ++stats.rows;
    for (int32_t col = count; col < kTopk; ++col) {
      CHECK_EQ(a[row][col], -1);
    }
  }
  return ordinals;
}

DsaTopkState merge_reference(const torch::Tensor& scores,
                             const torch::Tensor& slots,
                             const torch::Tensor& mapping) {
  const int64_t rows = scores.size(/*dim=*/1);
  auto flat_scores =
      scores.to(torch::kFloat32).permute({1, 0, 2}).reshape({rows, -1});
  auto flat_slots = slots.permute({1, 0, 2}).reshape({rows, -1});
  auto valid = flat_slots >= 0;
  // Order equal scores by physical slot. Packing the monotone FP32
  // bits and an inverse nonnegative int32 slot into 63 bits gives a unique
  // Torch TopK key, independent of either branch's candidate ordering.
  auto bits = flat_scores.contiguous().view(torch::kInt32).to(torch::kInt64);
  bits = torch::where(flat_scores == 0, /*self=*/0, bits);
  auto ordered =
      torch::where(bits < 0,
                   bits.bitwise_not().bitwise_and(/*other=*/0xffffffffLL),
                   bits.bitwise_or(/*other=*/0x80000000LL));
  auto keys = ordered * (int64_t{1} << 31) +
              std::numeric_limits<int32_t>::max() -
              flat_slots.to(torch::kInt64);
  keys.masked_fill_(~valid, /*value=*/-1);
  auto order = std::get<1>(torch::topk(keys,
                                       kTopk,
                                       /*dim=*/1,
                                       /*largest=*/true,
                                       /*sorted=*/true));
  auto output = flat_slots.gather(/*dim=*/1, order);
  auto selected_valid =
      valid.gather(/*dim=*/1, order) & (mapping.unsqueeze(/*dim=*/1) >= 0);
  output.masked_fill_(~selected_valid, /*value=*/-1);
  return DsaTopkState(output, selected_valid.sum(/*dim=*/1).to(torch::kInt32));
}

// FP32 merge ties may reorder or exchange equal-scoring candidates. Require
// every score above the cutoff, descending score order and exact valid tails.
void compare_merge(const DsaTopkState& actual,
                   const torch::Tensor& gathered_scores,
                   const torch::Tensor& gathered_slots,
                   const Pages& pages,
                   const DsaTopkState& reference) {
  CHECK_EQ(actual.block_tables().scalar_type(), torch::kInt32);
  CHECK_EQ(actual.context_lens().scalar_type(), torch::kInt32);
  CHECK(actual.block_tables().sizes() == torch::IntArrayRef({kTokens, kTopk}));
  CHECK(actual.context_lens().sizes() == torch::IntArrayRef({kTokens}));
  CHECK(torch::equal(actual.context_lens().cpu(), reference.context_lens()));
  auto output = actual.block_tables().cpu().contiguous();
  auto score_cpu =
      gathered_scores.cpu().permute({1, 0, 2}).contiguous().view({kTokens, -1});
  auto slot_cpu =
      gathered_slots.cpu().permute({1, 0, 2}).contiguous().view({kTokens, -1});
  const auto out = output.accessor<int32_t, 2>();
  const auto scores = score_cpu.accessor<float, 2>();
  const auto slots = slot_cpu.accessor<int32_t, 2>();
  for (int32_t row = 0; row < kTokens; ++row) {
    const int32_t count =
        std::min(kTopk, pages.history() + row % pages.query() + 1);
    std::unordered_map<int32_t, int32_t> indices;
    indices.reserve(kCp * kTopk);
    std::vector<float> valid_scores;
    valid_scores.reserve(kCp * kTopk);
    for (int32_t col = 0; col < kCp * kTopk; ++col) {
      if (slots[row][col] < 0) {
        continue;
      }
      CHECK(indices.emplace(slots[row][col], valid_scores.size()).second);
      valid_scores.emplace_back(scores[row][col]);
    }
    std::vector<int32_t> selected;
    selected.reserve(count);
    float previous = std::numeric_limits<float>::infinity();
    for (int32_t col = 0; col < count; ++col) {
      auto found = indices.find(out[row][col]);
      CHECK(found != indices.end());
      const int32_t shard = out[row][col] % kStripe / kPage;
      CHECK_GE(pages.ordinal(row, out[row][col], shard), 0);
      selected.emplace_back(found->second);
      CHECK_LE(valid_scores[found->second], previous);
      previous = valid_scores[found->second];
    }
    // Unlike the local selector, the merge sorts even when fewer than K exist.
    if (static_cast<int32_t>(valid_scores.size()) <= kTopk) {
      CHECK_EQ(
          std::unordered_set<int32_t>(selected.begin(), selected.end()).size(),
          valid_scores.size());
    } else {
      CHECK(
          valid_selection(selected, valid_scores, valid_scores.size(), kTopk));
    }
    for (int32_t col = count; col < kTopk; ++col) {
      CHECK_EQ(out[row][col], -1);
    }
  }
}

void report_stats(const Stats& stats,
                  int32_t rank,
                  int32_t batch,
                  int32_t history,
                  int32_t seed) {
  LOG(INFO) << "GLM53_INDEXER rank=" << rank << " batch=" << batch
            << " history=" << history << " seed=" << seed
            << " rows=" << stats.rows << " valid=" << stats.columns
            << " ordered_equal=" << stats.ordered_equal
            << " set_equal=" << stats.set_equal << " ordered_ratio="
            << static_cast<double>(stats.ordered_equal) / stats.columns
            << " set_ratio="
            << static_cast<double>(stats.set_equal) / stats.columns
            << " cutoff_swaps=" << stats.cutoff_swaps
            << " bf16_count=" << stats.score_count
            << " bf16_bits=" << stats.score_bits
            << " max_bf16_ulp=" << stats.max_ulp << " bf16_bit_ratio="
            << static_cast<double>(stats.score_bits) / stats.score_count
            << " max_abs=" << stats.max_abs << " max_rel=" << stats.max_rel
            << " E=" << stats.max_normalized
            << " final_set_diff=" << stats.final_differences
            << " cutoff_final_diff=" << stats.cutoff_final_diff
            << " rounding_final_diff=" << stats.rounding_final_diff
            << " intermediate_final_diff=" << stats.intermediate_final_diff
            << " merge_tie_diff=" << stats.merge_tie_diff;
}

void run_case(int32_t rank,
              const ParallelArgs& parallel,
              const torch::Device& device,
              int32_t batch,
              int32_t history,
              int32_t seed) {
  torch::manual_seed(seed);
  const Pages pages(batch, history, seed);
  const int32_t shard = rank / kTp;
  const KVShardLayout layout(kPage, kCp, shard);
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  auto indexer = make_indexer(parallel, options);
  auto metadata = make_metadata(pages, device);
  auto context = v32_cp::build_deepseek_v32_cp_context(
      kCp,
      metadata,
      history > 0 ? BatchForwardType::CHUNKED_PREFILL
                  : BatchForwardType::PREFILL,
      torch::arange(kTokens, options.dtype(torch::kInt32)),
      parallel.cp_group_,
      shard,
      kCp,
      layout);
  CHECK(context.has_value());
  auto x = torch::randn({kTokens, kHidden}, torch::kFloat32)
               .to(torch::kBFloat16)
               .to(device);
  auto q_norm = torch::randn({kTokens, kLora}, torch::kFloat32)
                    .to(torch::kBFloat16)
                    .to(device);
  auto positions =
      (torch::arange(kTokens, torch::kInt32).remainder(pages.query()) + history)
          .to(device);
  const auto local_pre =
      indexer->cp_pre(v32_cp::reorder_to_local_shard(x, *context),
                      v32_cp::reorder_to_local_shard(q_norm, *context),
                      v32_cp::reorder_to_local_shard(positions, *context),
                      context->local_attn_metadata);
  IndexerCPPreOut pre;
  pre.q = v32_cp::gather_and_restore_global(local_pre.q, *context);
  {
    const auto gather_handle =
        v32_cp::launch_all_gather_across_ranks(local_pre.k_local, *context);
    pre.k_local = v32_cp::restore_gathered_to_global_order(
        parallel_state::finish_gather(gather_handle), *context);
  }
  pre.weights = v32_cp::gather_and_restore_global(local_pre.weights, *context);
  CHECK(pre.q.sizes() == torch::IntArrayRef({kTokens, kHeads, kDim}));
  CHECK_EQ(pre.q.scalar_type(), torch::kBFloat16);
  CHECK_EQ(pre.k_local.scalar_type(), torch::kBFloat16);
  CHECK_EQ(pre.weights.scalar_type(), torch::kBFloat16);
  CHECK(!pre.q_scale.has_value());
  // CPU-generated history is deterministic and distinct on each KV owner.
  torch::manual_seed(seed + 100 + shard);
  auto initial = torch::randn({pages.count(), 1, kPage, kDim}, torch::kFloat32)
                     .to(torch::kBFloat16)
                     .to(device);
  auto actual_cache = initial.clone();
  auto ref_cache = initial.clone();
  write_reference_cache(ref_cache, pre.k_local, pages, shard);
  auto causal = build_kv_shard_causal_selector_metadata(metadata, layout);
  auto prefill = metadata;
  prefill.slot_mapping = localize_kv_shard_slots(metadata.slot_mapping, layout);
  auto selector = prefill;
  selector.q_cu_seq_lens = causal.q_cu_seq_lens;
  selector.kv_cu_seq_lens = causal.q_cu_seq_lens;
  selector.kv_seq_lens = causal.local_context_lens;
  selector.block_table = causal.block_table;
  selector.max_query_len = 1;
  selector.max_seq_len = kTopk;
  selector.total_kv_len = 0;
  selector.is_prefill = false;
  selector.is_chunked_prefill = false;
  std::vector<int32_t> expected_counts;
  expected_counts.reserve(kTokens);
  for (int32_t row = 0; row < kTokens; ++row) {
    expected_counts.emplace_back(pages.visible(row, shard));
  }
  CHECK(torch::equal(selector.kv_seq_lens.cpu(),
                     torch::tensor(expected_counts, torch::kInt32)));
  const auto actual = indexer->forward_dcp_local_prefill_from_cp(
      pre, actual_cache, prefill, selector);
  CHECK_EQ(actual.scores.scalar_type(), torch::kFloat32);
  CHECK_EQ(actual.global_slots.scalar_type(), torch::kInt32);
  CHECK(actual.scores.sizes() == torch::IntArrayRef({kTokens, kTopk}));
  CHECK(actual.global_slots.sizes() == actual.scores.sizes());
  CHECK(torch::equal(actual_cache, ref_cache)) << "Paged cache writes differ";

  auto ref_slots = torch::empty_like(actual.global_slots);
  auto ref_scores = torch::empty_like(actual.scores, torch::kBFloat16);
  auto reused_actual = torch::empty_like(ref_scores);
  Stats stats;
  for (int32_t req = 0; req < pages.batch(); ++req) {
    const auto keys = gather_keys(ref_cache, pages, req, shard);
    const int32_t end = (req + 1) * pages.query();
    for (int32_t start = req * pages.query(); start < end;
         start += kQueryTile) {
      const int32_t rows = std::min(kQueryTile, end - start);
      torch::Tensor all_scores;
      auto ref =
          select_reference(pre, keys, pages, shard, start, rows, all_scores);
      const auto slots = actual.global_slots.narrow(/*dim=*/0, start, rows);
      auto ordinals = compare_selection(
          slots, ref.global_slots, all_scores, pages, shard, start, stats);
      auto reused = gather_scores(all_scores, ordinals.to(device));
      reused_actual.narrow(/*dim=*/0, start, rows).copy_(reused);
      compare_scores(actual.scores.narrow(/*dim=*/0, start, rows),
                     reused,
                     start,
                     rank,
                     stats);
      ref_slots.narrow(/*dim=*/0, start, rows).copy_(ref.global_slots);
      ref_scores.narrow(/*dim=*/0, start, rows).copy_(ref.scores);
    }
  }

  // Use the synchronous collective as a separate witness for the async wire.
  auto witness_scores = parallel.dcp_group_->allgather_base_sync(actual.scores);
  auto witness_slots =
      parallel.dcp_group_->allgather_base_sync(actual.global_slots);
  DcpDecodeContext dcp(layout, parallel.dcp_group_);
  auto gather =
      dcp.launch_indexer_candidate_gather(actual.scores, actual.global_slots);
  const auto wire_scores = gather.gathered_scores;
  const auto wire_slots = gather.gathered_global_slots;
  const auto merged = dcp.finish_indexer_candidate_merge(
      std::move(gather), kTopk, metadata.slot_mapping);
  CHECK(torch::equal(wire_scores.view(torch::kInt32),
                     witness_scores.view(torch::kInt32)));
  CHECK(torch::equal(wire_slots, witness_slots));
  auto witness_scores_cpu = witness_scores.cpu();
  auto witness_slots_cpu = witness_slots.cpu();
  const DsaTopkState merged_cpu(merged.block_tables().cpu(),
                                merged.context_lens().cpu());
  const auto mapping = pages.new_slots();
  const auto same_scores =
      merge_reference(witness_scores_cpu, witness_slots_cpu, mapping);
  compare_merge(
      merged_cpu, witness_scores_cpu, witness_slots_cpu, pages, same_scores);
  auto packed = v32_cp::reorder_to_local_shard(merged.block_tables(), *context);
  auto local_index = context->gathered_reorder_index
                         .narrow(/*dim=*/0,
                                 context->comm_plan.token_num_offset,
                                 context->comm_plan.tokens_per_rank.at(shard))
                         .cpu();
  CHECK(torch::equal(
      packed.cpu(),
      merged_cpu.block_tables().index_select(/*dim=*/0, local_index)));
  CHECK(torch::equal(v32_cp::gather_and_restore_global(packed, *context),
                     merged.block_tables()));
  auto packed_lens =
      v32_cp::reorder_to_local_shard(merged.context_lens(), *context);
  CHECK(torch::equal(v32_cp::gather_and_restore_global(packed_lens, *context),
                     merged.context_lens()));

  auto full_scores = parallel.dcp_group_->allgather_base_sync(ref_scores).cpu();
  auto full_slots = parallel.dcp_group_->allgather_base_sync(ref_slots).cpu();
  const auto full = merge_reference(full_scores, full_slots, mapping);
  auto gathered_reused =
      parallel.dcp_group_->allgather_base_sync(reused_actual).cpu();
  auto scaled_scores = witness_scores_cpu * kScale;
  auto rounded_scores = scaled_scores.to(torch::kBFloat16);
  const auto rounded =
      merge_reference(rounded_scores, witness_slots_cpu, mapping);
  const auto same_candidates =
      merge_reference(gathered_reused, witness_slots_cpu, mapping);
  auto diff_sets = [](const torch::Tensor& left, const torch::Tensor& right) {
    auto l = left.cpu().contiguous();
    auto r = right.cpu().contiguous();
    const auto la = l.accessor<int32_t, 2>();
    const auto ra = r.accessor<int32_t, 2>();
    int64_t differences = 0;
    for (int32_t row = 0; row < l.size(/*dim=*/0); ++row) {
      std::unordered_set<int32_t> candidates;
      candidates.reserve(kTopk);
      for (int32_t col = 0; col < kTopk; ++col) {
        candidates.emplace(ra[row][col]);
      }
      for (int32_t col = 0; col < kTopk; ++col) {
        differences += candidates.count(la[row][col]) == 0;
      }
    }
    return differences;
  };
  stats.final_differences =
      diff_sets(merged_cpu.block_tables(), full.block_tables());
  stats.cutoff_final_diff =
      diff_sets(same_candidates.block_tables(), full.block_tables());
  stats.rounding_final_diff =
      diff_sets(same_scores.block_tables(), rounded.block_tables());
  stats.intermediate_final_diff =
      diff_sets(rounded.block_tables(), same_candidates.block_tables());
  stats.merge_tie_diff =
      diff_sets(merged_cpu.block_tables(), same_scores.block_tables());
  CHECK_LE(stats.final_differences,
           stats.cutoff_final_diff + stats.rounding_final_diff +
               stats.intermediate_final_diff + stats.merge_tie_diff);
  report_stats(stats, rank, batch, history, seed);
}

void run_edges(int32_t rank,
               const ParallelArgs& parallel,
               const torch::Device& device) {
  const std::vector<int32_t> counts = {0, 1, 15, 16, 17, 2047, 2048, 2049};
  const int32_t rows = static_cast<int32_t>(counts.size());
  constexpr int32_t kPages = 129;
  const int32_t shard = rank / kTp;
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  torch::manual_seed(/*seed=*/42);
  auto indexer = make_indexer(parallel, options);
  // Odd physical pages leave untouched holes between every populated page.
  auto table = torch::arange(kPages, torch::kInt32) * 2 + 1;
  auto logical = torch::zeros({kPages * kPage, kDim}, torch::kBFloat16);
  for (int32_t index = 0; index < kPages * kPage; ++index) {
    logical[index][0] = index % 3 == 0 ? 2.0f : -4.0f;
    logical[index][1] = index % 3 == 1 ? 3.0f : 0.0f;
  }
  auto cache = torch::zeros({kPages * 2 + 1, 1, kPage, kDim}, options);
  auto local = (table.to(torch::kInt64).unsqueeze(/*dim=*/1) * kPage +
                torch::arange(kPage, torch::kInt64))
                   .flatten()
                   .to(device);
  cache.view({-1, kDim}).index_copy_(/*dim=*/0, local, logical.to(device));
  auto initial = cache.clone();
  auto query = torch::zeros({rows, kHeads, kDim}, torch::kBFloat16);
  query.select(/*dim=*/1, /*index=*/0)
      .select(/*dim=*/1, /*index=*/0)
      .fill_(/*value=*/1);
  query.select(/*dim=*/1, /*index=*/1)
      .select(/*dim=*/1, /*index=*/1)
      .fill_(/*value=*/1);
  auto weights = torch::zeros({rows, kHeads}, torch::kBFloat16);
  weights.select(/*dim=*/1, /*index=*/0).fill_(/*value=*/1);
  weights.select(/*dim=*/1, /*index=*/1).fill_(/*value=*/-1);
  weights[7].neg_();
  IndexerCPPreOut pre{query.to(device),
                      torch::zeros({rows, kDim}, options),
                      weights.to(device),
                      std::nullopt};
  const auto ints = options.dtype(torch::kInt32);
  AttentionMetadata prefill{};
  prefill.slot_mapping = torch::full({rows}, /*fill_value=*/-1, ints);
  prefill.is_prefill = true;
  AttentionMetadata selector{};
  selector.q_cu_seq_lens = torch::arange(rows + 1, ints);
  selector.kv_cu_seq_lens = selector.q_cu_seq_lens;
  selector.kv_seq_lens = torch::tensor(counts, torch::kInt32).to(device);
  selector.block_table =
      table.to(device).unsqueeze(/*dim=*/0).repeat({rows, 1});
  selector.max_query_len = 1;
  selector.max_seq_len = kTopk;
  selector.is_prefill = false;
  auto actual =
      indexer->forward_dcp_local_prefill_from_cp(pre, cache, prefill, selector);
  CHECK(torch::equal(initial, cache));
  auto all = selector_scores(pre.q, logical.to(device), pre.weights)
                 .to(torch::kFloat32)
                 .cpu()
                 .contiguous();
  auto slots_cpu = actual.global_slots.cpu().contiguous();
  const auto scores = all.accessor<float, 2>();
  const auto slots = slots_cpu.accessor<int32_t, 2>();
  for (int32_t row = 0; row < rows; ++row) {
    const int32_t count = std::min(counts[row], kTopk);
    std::vector<int32_t> selected;
    selected.reserve(count);
    for (int32_t col = 0; col < count; ++col) {
      const int32_t global = slots[row][col];
      CHECK_GE(global, 0);
      CHECK_EQ(global % kStripe / kPage, shard);
      const int32_t physical = global / kStripe;
      CHECK_EQ(physical % 2, 1);
      selected.emplace_back((physical - 1) / 2 * kPage + global % kPage);
    }
    CHECK(
        valid_selection(selected,
                        std::vector<float>(scores[row].data(),
                                           scores[row].data() + kPages * kPage),
                        counts[row],
                        kTopk));
    for (int32_t col = count; col < kTopk; ++col) {
      CHECK_EQ(slots[row][col], -1);
    }
  }
  auto ordinal_tensor = slots_cpu.to(torch::kInt64);
  ordinal_tensor =
      torch::floor_divide(ordinal_tensor.clamp_min(/*min=*/0), kStripe)
              .sub(/*other=*/1)
              .div(/*other=*/2, /*rounding_mode=*/"floor") *
          kPage +
      ordinal_tensor.clamp_min(/*min=*/0).remainder(kPage);
  ordinal_tensor.masked_fill_(slots_cpu < 0, /*value=*/-1);
  auto reused = gather_scores(all.to(torch::kBFloat16), ordinal_tensor);
  Stats stats;
  compare_scores(actual.scores, reused.to(device), /*start=*/0, rank, stats);
  DcpDecodeContext dcp(KVShardLayout(kPage, kCp, shard), parallel.dcp_group_);
  auto merged = dcp.finish_indexer_candidate_merge(
      dcp.launch_indexer_candidate_gather(actual.scores, actual.global_slots),
      kTopk,
      prefill.slot_mapping);
  CHECK(merged.block_tables().eq(/*other=*/-1).all().item<bool>());
  CHECK(merged.context_lens().eq(/*other=*/0).all().item<bool>());
  LOG(INFO) << "GLM53_INDEXER_EDGES rank=" << rank
            << " counts=0,1,15,16,17,2047,2048,2049 signed_weights=1 padding=1";
}

void run_rank(int32_t rank, int32_t port, int32_t batch, int32_t history) {
  torch::set_num_threads(1);
  torch::NoGradGuard guard;
  Device dev(rank);
  dev.set_device();
  const auto device = dev.unwrap();
  ParallelConfig::get_instance().kv_split_size(kCp);
  KVCacheConfig::get_instance().block_size(kPage);
  CollectiveCommunicator comm(rank,
                              kRanks,
                              /*dp_size=*/1,
                              /*ep_size=*/kRanks,
                              /*cp_size=*/kCp);
  comm.create_process_groups("127.0.0.1:" + std::to_string(port), device);
  const ParallelArgs& parallel = *comm.parallel_args();
  CHECK_EQ(parallel.tp_group_->world_size(), kTp);
  CHECK_EQ(parallel.tp_group_->rank(), rank % kTp);
  CHECK_EQ(parallel.cp_group_->world_size(), kCp);
  CHECK_EQ(parallel.cp_group_->rank(), rank / kTp);
  CHECK_EQ(parallel.dcp_group_, parallel.cp_group_);
  CHECK_EQ(parallel.kv_split_rank(), rank / kTp);
  run_edges(rank, parallel, device);
  for (int32_t seed : {42, 2026}) {
    run_case(rank, parallel, device, batch, history, seed);
    CHECK_EQ(dev.synchronize_default_stream(), 0);
  }
}

void run_distributed(int32_t batch, int32_t history) {
  xllm::test::run_multi_device(kRanks,
                               [batch, history](int32_t rank, int32_t port) {
                                 run_rank(rank, port, batch, history);
                               });
}

TEST(Glm53IndexerDcpMultiDeviceTest, ChunkedBatch1) {
  run_distributed(/*batch=*/1, /*history=*/65536);
}
TEST(Glm53IndexerDcpMultiDeviceTest, ChunkedBatch16) {
  run_distributed(/*batch=*/16, /*history=*/65536);
}
TEST(Glm53IndexerDcpMultiDeviceTest, PrefillBatch1) {
  run_distributed(/*batch=*/1, /*history=*/0);
}
TEST(Glm53IndexerDcpMultiDeviceTest, PrefillBatch16) {
  run_distributed(/*batch=*/16, /*history=*/0);
}

// CPU-only fixtures leave the parent MLU runtime untouched before fork.
TEST(Glm53IndexerReferenceDeathTest, RejectsForeignAndNoncausalSlots) {
  const Pages pages(/*batch=*/16, /*history=*/0, /*seed=*/42);
  auto pool = torch::zeros({1, pages.local_length()}, torch::kBFloat16);
  for (int32_t slot : {pages.slot(/*req=*/1, /*position=*/0),
                       pages.slot(/*req=*/0, /*position=*/1),
                       pages.slot(/*req=*/0, /*position=*/16),
                       -2}) {
    auto slots = torch::tensor({{slot}}, torch::kInt32);
    EXPECT_DEATH(reuse_scores(pool, slots, pages, /*shard=*/0, /*start=*/0),
                 "Invalid candidate");
  }
  EXPECT_DEATH(
      gather_scores(pool,
                    torch::tensor({{pages.local_length()}}, torch::kInt64)),
      "Check failed");
}

TEST(Glm53IndexerReferenceTest, ReusesVisiblePrefixAndCutoffScores) {
  const Pages pages(/*batch=*/1, /*history=*/64, /*seed=*/42);
  IndexerCPPreOut pre;
  pre.q = torch::ones({1, kHeads, kDim}, torch::kBFloat16)
              .expand({kTokens, kHeads, kDim});
  pre.weights =
      torch::ones({1, kHeads}, torch::kBFloat16).expand({kTokens, kHeads});
  auto cache = torch::ones({pages.count(), 1, kPage, kDim}, torch::kBFloat16);
  const auto keys = gather_keys(cache, pages, /*req=*/0, /*shard=*/0);
  for (int32_t start : {8078, 8079, 8128}) {
    torch::Tensor pool;
    const auto ref = select_reference(
        pre, keys, pages, /*shard=*/0, start, /*rows=*/1, pool);
    EXPECT_EQ(pages.visible(start, /*shard=*/0),
              start == 8078 ? 2047 : (start == 8079 ? 2048 : 2049));
    EXPECT_EQ(pool.scalar_type(), torch::kBFloat16);
    EXPECT_TRUE(equal_reused_scores(
        ref.scores, reuse_scores(pool, ref.global_slots, pages, 0, start)));
    Stats stats;
    auto ordinals = compare_selection(ref.global_slots,
                                      ref.global_slots,
                                      pool,
                                      pages,
                                      /*shard=*/0,
                                      start,
                                      stats);
    EXPECT_TRUE(equal_reused_scores(gather_scores(pool, ordinals), ref.scores));
    EXPECT_EQ(stats.columns, std::min(kTopk, pages.visible(start, 0)));
    EXPECT_TRUE(ref.scores[0][0].eq(/*other=*/64).item<bool>());
  }
  const Pages empty_pages(/*batch=*/16, /*history=*/0, /*seed=*/42);
  auto empty_cache =
      torch::ones({empty_pages.count(), 1, kPage, kDim}, torch::kBFloat16);
  const auto empty_keys =
      gather_keys(empty_cache, empty_pages, /*req=*/0, /*shard=*/1);
  torch::Tensor pool;
  const auto empty = select_reference(pre,
                                      empty_keys,
                                      empty_pages,
                                      /*shard=*/1,
                                      /*start=*/0,
                                      /*rows=*/1,
                                      pool);
  EXPECT_TRUE(empty.global_slots.eq(/*other=*/-1).all().item<bool>());
  EXPECT_TRUE(empty.scores.isneginf().all().item<bool>());
}

TEST(Glm53IndexerReferenceTest, ShardedTiesPreserveFullCutoff) {
  constexpr int32_t kCount = kTopk + 3;
  auto all = torch::ones({2, 1, kCount}, torch::kBFloat16);
  all[0][0][0] = 2;
  all[1][0][0] = 3;
  all[0][0][kCount - 1] = -1;
  auto slots = torch::arange(2 * kCount, torch::kInt32).view({2, 1, kCount});
  auto local_indices = std::get<1>(all.to(torch::kFloat32).topk(kTopk));
  auto local_scores = all.gather(/*dim=*/2, local_indices);
  auto local_slots = slots.gather(/*dim=*/2, local_indices);
  auto mapping = torch::zeros({1}, torch::kInt32);
  const auto sharded = merge_reference(local_scores, local_slots, mapping);
  const auto full = merge_reference(all, slots, mapping);
  auto cpu_scores = all.to(torch::kFloat32).flatten().contiguous();
  const auto data = cpu_scores.accessor<float, 1>();
  const std::vector<float> scores(data.data(), data.data() + 2 * kCount);
  for (const auto& result : {sharded, full}) {
    auto cpu_slots = result.block_tables().flatten().contiguous();
    const auto indices = cpu_slots.accessor<int32_t, 1>();
    EXPECT_TRUE(valid_selection(
        std::vector<int32_t>(indices.data(), indices.data() + kTopk),
        scores,
        /*visible=*/2 * kCount,
        kTopk));
  }
}

TEST(Glm53IndexerReferenceTest, RejectsInvalidCandidatesAndScores) {
  EXPECT_TRUE(valid_selection({}, {}, /*visible=*/0, kTopk));
  for (int32_t count : {2047, 2048, 2049}) {
    std::vector<float> scores(count, /*value=*/1.0f);
    scores.front() = 2.0f;
    std::vector<int32_t> selected(std::min(count, kTopk));
    std::iota(selected.begin(), selected.end(), 0);
    EXPECT_TRUE(valid_selection(selected, scores, count, kTopk));
    auto invalid = selected;
    invalid.back() = count;
    EXPECT_FALSE(valid_selection(invalid, scores, count, kTopk));
    invalid = selected;
    invalid.back() = invalid.front();
    EXPECT_FALSE(valid_selection(invalid, scores, count, kTopk));
    if (count > kTopk) {
      selected.back() = count - 1;
      EXPECT_TRUE(valid_selection(selected, scores, count, kTopk));
      selected.front() = count - 2;
      EXPECT_FALSE(valid_selection(selected, scores, count, kTopk));
    }
  }
}

TEST(Glm53IndexerReferenceTest, MergesTiesIndependentlyOfCandidateOrder) {
  auto scores = torch::full({1, 2, kTopk}, kNegInf, torch::kFloat32);
  auto slots = torch::full({1, 2, kTopk}, /*fill_value=*/-1, torch::kInt32);
  scores[0][0]
      .narrow(/*dim=*/0, /*start=*/0, /*length=*/7)
      .copy_(torch::tensor({2.0f,
                            2.0f,
                            -3.0f,
                            0.0f,
                            -0.0f,
                            std::numeric_limits<float>::infinity(),
                            kNegInf}));
  slots[0][0]
      .narrow(/*dim=*/0, /*start=*/0, /*length=*/7)
      .copy_(torch::tensor({19, 7, 3, 11, 5, 23, 29}, torch::kInt32));
  auto mapping = torch::tensor({0, -1}, torch::kInt32);
  const auto merged = merge_reference(scores, slots, mapping);
  EXPECT_TRUE(torch::equal(
      merged.block_tables()[0].narrow(/*dim=*/0, /*start=*/0, /*length=*/7),
      torch::tensor({23, 7, 19, 5, 11, 3, 29}, torch::kInt32)));
  EXPECT_TRUE(merged.block_tables()[0]
                  .narrow(/*dim=*/0, /*start=*/7, kTopk - 7)
                  .eq(/*other=*/-1)
                  .all()
                  .item<bool>());
  EXPECT_TRUE(merged.block_tables()[1].eq(/*other=*/-1).all().item<bool>());
  EXPECT_TRUE(torch::equal(merged.context_lens(),
                           torch::tensor({7, 0}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(merge_reference(scores.flip({2}), slots.flip({2}), mapping)
                       .block_tables(),
                   merged.block_tables()));
}

TEST(Glm53IndexerReferenceTest, PreservesMap64RoundingBoundaries) {
  auto query = torch::zeros({1, kHeads, kDim}, torch::kBFloat16);
  query[0][0][0] = 1.0f;
  query[0][1][0] = 1.0078125f;
  auto keys = torch::zeros({1, kDim}, torch::kBFloat16);
  keys[0][0] = 1.0078125f;
  auto weights = torch::zeros({1, kHeads}, torch::kBFloat16);
  weights[0][0] = 1.0078125f;
  weights[0][1] = -1.0f;
  // BF16 QK rounding leaves exactly 2^-14 before scaling. Rounding each
  // head product to BF16 would instead cancel the result to zero.
  EXPECT_EQ(selector_scores(query, keys, weights).item<float>(), 0x1p-20f);
}

TEST(Glm53IndexerReferenceTest, CoversPageBoundariesAndSignedWeights) {
  const Pages pages(/*batch=*/16, /*history=*/0, /*seed=*/42);
  EXPECT_EQ(pages.visible(/*row=*/0, /*shard=*/1), 0);
  EXPECT_EQ(pages.visible(/*row=*/14, /*shard=*/0), 15);
  EXPECT_EQ(pages.visible(/*row=*/15, /*shard=*/0), 16);
  EXPECT_EQ(pages.visible(/*row=*/16, /*shard=*/1), 1);
  EXPECT_EQ(pages.visible(/*row=*/63, /*shard=*/3), 16);
  EXPECT_EQ(pages.visible(/*row=*/64, /*shard=*/0), 17);
  EXPECT_EQ(
      pages.ordinal(
          /*row=*/16, pages.slot(/*req=*/0, /*position=*/16), /*shard=*/1),
      0);
  EXPECT_EQ(
      pages.ordinal(
          /*row=*/15, pages.slot(/*req=*/0, /*position=*/16), /*shard=*/1),
      -1);
  EXPECT_EQ(
      pages.ordinal(
          /*row=*/512, pages.slot(/*req=*/0, /*position=*/0), /*shard=*/0),
      -1);
  EXPECT_EQ(pages.ordinal(/*row=*/0, /*global=*/-1, /*shard=*/0), -1);
  auto q = torch::zeros({2, kHeads, kDim}, torch::kBFloat16);
  q[0][0][0] = 1;
  q[0][1][1] = 1;
  q[1].copy_(q[0]);
  auto keys = torch::zeros({3, kDim}, torch::kBFloat16);
  keys[0][0] = 2;
  keys[1][1] = 3;
  keys[2][0] = -4;
  auto w = torch::zeros({2, kHeads}, torch::kBFloat16);
  w[0][0] = 1;
  w[0][1] = -1;
  w[1][0] = -1;
  w[1][1] = 1;
  EXPECT_TRUE(torch::equal(
      selector_scores(q, keys, w),
      torch::tensor({{2.0f, -3.0f, 0.0f}, {-2.0f, 3.0f, 0.0f}}) * kScale));
  auto pool = selector_scores(q, keys, w);
  auto ordinals = torch::full({2, kTopk}, /*fill_value=*/-1, torch::kInt64);
  ordinals.narrow(/*dim=*/1, /*start=*/0, /*length=*/3)
      .copy_(torch::tensor({0, 1, 2}, torch::kInt64).expand({2, 3}));
  auto scores = gather_scores(pool, ordinals);
  EXPECT_EQ(scores.scalar_type(), torch::kBFloat16);
  EXPECT_TRUE(
      torch::equal(scores.narrow(/*dim=*/1, /*start=*/0, /*length=*/3), pool));
  EXPECT_TRUE(scores.narrow(/*dim=*/1, /*start=*/3, kTopk - 3)
                  .isneginf()
                  .all()
                  .item<bool>());
  EXPECT_TRUE(equal_reused_scores(scores, scores.clone()));
  auto wrong = scores.clone();
  wrong[0][0] = 0;
  EXPECT_FALSE(equal_reused_scores(wrong, scores));
  EXPECT_FALSE(equal_reused_scores(scores.to(torch::kFloat32), scores));
  wrong = scores.clone();
  wrong[0][3] = 0;
  EXPECT_FALSE(equal_reused_scores(wrong, scores));
}
}  // namespace
}  // namespace xllm::layer
