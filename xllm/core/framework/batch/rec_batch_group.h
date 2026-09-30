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

#pragma once

#include <cstddef>
#include <vector>

#include "core/framework/batch/rec_batch.h"

namespace xllm {

// DP-rank-local Rec batches for one scheduler step.
class RecBatchGroup final {
 public:
  using Container = std::vector<RecBatch>;
  using iterator = Container::iterator;
  using const_iterator = Container::const_iterator;

  RecBatchGroup() = default;
  RecBatchGroup(size_t dp_size, BatchInputType input_type);

  RecBatchGroup(const RecBatchGroup&) = delete;
  RecBatchGroup& operator=(const RecBatchGroup&) = delete;
  RecBatchGroup(RecBatchGroup&&) noexcept = default;
  RecBatchGroup& operator=(RecBatchGroup&&) noexcept = default;

  size_t size() const { return batches_.size(); }
  bool empty() const { return batches_.empty(); }
  RecBatch& front() { return batches_.front(); }
  const RecBatch& front() const { return batches_.front(); }
  RecBatch& back() { return batches_.back(); }
  const RecBatch& back() const { return batches_.back(); }
  RecBatch& operator[](size_t rank) { return batches_[rank]; }
  const RecBatch& operator[](size_t rank) const { return batches_[rank]; }
  RecBatch& at(size_t rank) { return batches_.at(rank); }
  const RecBatch& at(size_t rank) const { return batches_.at(rank); }
  iterator begin() { return batches_.begin(); }
  iterator end() { return batches_.end(); }
  const_iterator begin() const { return batches_.begin(); }
  const_iterator end() const { return batches_.end(); }
  const_iterator cbegin() const { return batches_.cbegin(); }
  const_iterator cend() const { return batches_.cend(); }

 private:
  Container batches_;
};

}  // namespace xllm
