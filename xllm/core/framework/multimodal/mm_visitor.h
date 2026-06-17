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

#pragma once

#include <torch/torch.h>

#include <unordered_set>
#include <vector>

#include "mm_batch_data.h"
#include "mm_data.h"
#include "mm_input.h"

namespace xllm {

class MMInputGatherVisitor final : public MMInputItem::IVisitor {
 public:
  MMInputGatherVisitor() = default;

  bool visit(const MMInputItem& item) override;
  MMItemVec finish(std::vector<MMDataItem>& image_items,
                   std::vector<MMDataItem>& video_items,
                   std::vector<MMDataItem>& audio_items);

  uint32_t data_type_ = MMType::NONE;
  std::vector<uint32_t> item_types_;
  std::vector<torch::Tensor> images_;
  std::vector<torch::Tensor> videos_;
  std::vector<torch::Tensor> audios_;
  std::vector<VideoMetadata> video_metadata_;
  std::vector<AudioMetadata> audio_metadata_;
  std::vector<MMDataItem> image_embedding_items_;
};

class CollectItemTensorVisitor : public MMData::IItemVisitor {
 public:
  CollectItemTensorVisitor(
      std::unordered_map<MMKey, std::vector<torch::Tensor>>& datas,
      const std::unordered_set<MMKey>& black_keys = {},
      const std::unordered_set<MMKey>& white_keys = {})
      : datas_(datas), black_keys_(black_keys), white_keys_(white_keys) {};

  CollectItemTensorVisitor(const std::unordered_set<MMKey>& black_keys = {},
                           const std::unordered_set<MMKey>& white_keys = {})
      : datas_(stub_), black_keys_(black_keys), white_keys_(white_keys) {};

  bool visit(MMDataItem& item) override;
  bool visit(const MMKey& key, MMValue& value) override;

 public:
  std::unordered_map<MMKey, std::vector<torch::Tensor>> stub_;
  std::unordered_map<MMKey, std::vector<torch::Tensor>>& datas_;

  std::unordered_set<MMKey> black_keys_;
  std::unordered_set<MMKey> white_keys_;
};

class CollectMMDataTensorVisitor : public MMData::IVisitor {
 public:
  CollectMMDataTensorVisitor(const std::unordered_set<MMKey>& black_keys = {},
                             const std::unordered_set<MMKey>& white_keys = {})
      : item_visitor_(datas_, black_keys, white_keys) {};

  bool visit(MMData& data) override;

 public:
  uint32_t type_ = MMType::NONE;
  std::unordered_map<MMKey, std::vector<torch::Tensor>> datas_;

  CollectItemTensorVisitor item_visitor_;
};

class MMTokenNumVisitor final : public MMDataItem::IVisitor {
 public:
  explicit MMTokenNumVisitor(MMType type) : type_(type) {}

  bool visit(MMDataItem& item) override;

  const std::vector<int32_t>& token_nums() const { return token_nums_; }

 private:
  MMType type_;
  std::vector<int32_t> token_nums_;
};

class EncoderInputGatherVisitor : public MMDataItem::IVisitor {
 public:
  EncoderInputGatherVisitor() = default;

  bool visit(MMDataItem& item) override;
  bool finish(MMBatchData& mm_data);

 public:
  std::unordered_map<MMKey, std::vector<torch::Tensor>> datas_;
  std::string filter_prefix_ = "embedding";
};

class EncoderOutputScatterVisitor : public MMDataItem::IVisitor {
 public:
  EncoderOutputScatterVisitor(const MMDict& data) : data_(data) {}

  bool visit(MMDataItem& data) override;
  bool finish() const;

 public:
  const MMDict& data_;
  int32_t image_idx = 0;
  int32_t video_idx = 0;
  int32_t audio_idx = 0;
};

class EncoderEmbeddingGatherVisitor : public MMDataItem::IVisitor {
 public:
  EncoderEmbeddingGatherVisitor(const torch::Device& device,
                                uint32_t mm_type,
                                const std::vector<int32_t>& seq_lens,
                                const std::vector<int32_t>& scheduled_seq_lens);
  bool visit(MMDataItem& data) override;
  bool finish(MMBatchData& mm_data);

 public:
  torch::Device device_;
  std::string gather_prefix_ = "embedding";
  std::vector<int32_t> per_seq_context_lens_;
  std::vector<int32_t> per_seq_scheduled_lens_;
  std::vector<int32_t> per_seq_scheduled_offsets_;
  int32_t total_scheduled_tokens_ = 0;
  torch::Tensor image_mask_;
  torch::Tensor video_mask_;
  torch::Tensor audio_mask_;
  std::unordered_map<MMKey, std::vector<torch::Tensor>> datas_;
};

class UpdateMMItemScheduleStateVisitor : public MMDataItem::IVisitor {
 public:
  UpdateMMItemScheduleStateVisitor(int32_t computed_token_num = 0,
                                   int32_t q_seq_len = 0,
                                   int32_t seq_idx = 0)
      : computed_token_num_(computed_token_num),
        q_seq_len_(q_seq_len),
        seq_idx_(seq_idx) {}

  bool visit(MMDataItem& item) override;

 public:
  std::vector<MMDataItem> mm_data_items_;
  uint32_t scheduled_type_ = MMType::NONE;
  int32_t computed_token_num_ = 0;
  int32_t q_seq_len_ = 0;
  int32_t seq_idx_ = 0;
};

class EncoderCache;

class EncoderCacheLookupVisitor : public MMDataItem::IVisitor {
 public:
  explicit EncoderCacheLookupVisitor(EncoderCache* cache);
  bool visit(MMDataItem& item) override;

 private:
  EncoderCache* cache_;
};

class EncoderCacheInsertVisitor : public MMDataItem::IVisitor {
 public:
  explicit EncoderCacheInsertVisitor(EncoderCache* cache);
  bool visit(MMDataItem& item) override;

 private:
  EncoderCache* cache_;
};

}  // namespace xllm
