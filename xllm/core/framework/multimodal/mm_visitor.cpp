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

#include "mm_visitor.h"

#include <absl/strings/match.h>
#include <glog/logging.h>

#include <numeric>
#include <optional>
#include <utility>

#include "framework/encoder_cache/encoder_cache.h"

namespace xllm {

namespace {
MMDict build_embedding_data(MMType type, const EmbeddingOutput& embedding) {
  MMDict data;
  data[get_embedding_key(type)] = embedding.embedding;
  for (const auto& [key, value] : embedding.metadata) {
    data[key] = value;
  }
  return data;
}

std::pair<int32_t, int32_t> compute_emb_range(int32_t start_pos,
                                              int32_t end_pos,
                                              const torch::Tensor& mask) {
  if (!mask.defined() || mask.numel() == 0) {
    return {start_pos, end_pos};
  }
  auto mask_cpu = mask.to(torch::kCPU);
  auto upto_start_pos =
      mask_cpu.slice(/*dim*/ 0, /*start*/ 0, /*end*/ start_pos)
          .sum()
          .item<int32_t>();
  auto upto_end_pos = mask_cpu.slice(/*dim*/ 0, /*start*/ 0, /*end*/ end_pos)
                          .sum()
                          .item<int32_t>();
  return {upto_start_pos, upto_end_pos};
}

std::vector<int32_t> normalize_to_per_seq_lens(
    const std::vector<int32_t>& seq_lens) {
#if defined(USE_NPU) || defined(USE_MUSA)
  return seq_lens;
#else
  // Other backends pass cumulative sequence lengths.
  std::vector<int32_t> per_seq_lens;
  per_seq_lens.reserve(seq_lens.empty() ? 0 : seq_lens.size() - 1);
  for (size_t i = 1; i < seq_lens.size(); ++i) {
    per_seq_lens.push_back(seq_lens[i] - seq_lens[i - 1]);
  }
  return per_seq_lens;
#endif
}
}  // namespace

bool MMInputGatherVisitor::visit(const MMInputItem& item) {
  if (item.has_type(MMType::IMAGE)) {
    data_type_ |= MMType::IMAGE;
    if (item.is_embedding()) {
      item_types_.push_back(MMType::IMAGE | MMType::EMBEDDING);
      MMDataItem data_item(MMType::IMAGE);
      data_item.set_data(build_embedding_data(MMType::IMAGE, item.embedding));
      image_embedding_items_.push_back(std::move(data_item));
    } else {
      item_types_.push_back(MMType::IMAGE);
      images_.push_back(item.decode_image);
    }
  }
  if (item.has_type(MMType::VIDEO)) {
    data_type_ |= MMType::VIDEO;
    item_types_.push_back(MMType::VIDEO);
    videos_.push_back(item.decode_video);
    video_metadata_.push_back(item.video_meta);
  }
  if (item.has_type(MMType::AUDIO)) {
    data_type_ |= MMType::AUDIO;
    item_types_.push_back(MMType::AUDIO);
    audios_.push_back(item.decode_audio);
    audio_metadata_.push_back(item.audio_meta);
  }
  return true;
}

MMItemVec MMInputGatherVisitor::finish(std::vector<MMDataItem>& image_items,
                                       std::vector<MMDataItem>& video_items,
                                       std::vector<MMDataItem>& audio_items) {
  size_t image_embedding_idx = 0;
  size_t image_idx = 0;
  size_t video_idx = 0;
  size_t audio_idx = 0;
  MMItemVec output_items;

  auto take_next = [&output_items](std::vector<MMDataItem>& items,
                                   size_t& index) {
    CHECK(index < items.size())
        << "Multimodal item count does not match input.";
    output_items.push_back(std::move(items[index]));
    ++index;
  };

  output_items.reserve(item_types_.size());
  for (uint32_t item_type : item_types_) {
    if ((item_type & MMType::IMAGE) && (item_type & MMType::EMBEDDING)) {
      take_next(image_embedding_items_, image_embedding_idx);
    } else if (item_type & MMType::IMAGE) {
      take_next(image_items, image_idx);
    } else if (item_type & MMType::VIDEO) {
      take_next(video_items, video_idx);
    } else if (item_type & MMType::AUDIO) {
      take_next(audio_items, audio_idx);
    } else {
      LOG(FATAL) << "Invalid multimodal item type: " << item_type;
    }
  }

  CHECK(image_embedding_idx == image_embedding_items_.size() &&
        image_idx == image_items.size() && video_idx == video_items.size() &&
        audio_idx == audio_items.size())
      << "Multimodal item count does not match input.";
  return output_items;
}

bool CollectItemTensorVisitor::visit(MMDataItem& item) {
  for (const auto& pair : item.data()) {
    const auto& key = pair.first;

    if (!black_keys_.empty() && black_keys_.count(key)) {
      continue;
    }

    if (!white_keys_.empty() && !white_keys_.count(key)) {
      continue;
    }

    auto& tar = datas_[pair.first];
    if (std::holds_alternative<torch::Tensor>(pair.second)) {
      tar.emplace_back(std::get<torch::Tensor>(pair.second));
    } else if (std::holds_alternative<std::vector<torch::Tensor>>(
                   pair.second)) {
      const auto& lst = std::get<std::vector<torch::Tensor>>(pair.second);
      tar.insert(tar.end(), lst.begin(), lst.end());
    }
  }
  return true;
}

bool CollectItemTensorVisitor::visit(const MMKey& key, MMValue& value) {
  if (!black_keys_.empty() && black_keys_.count(key)) {
    return true;
  }

  if (!white_keys_.empty() && !white_keys_.count(key)) {
    return true;
  }

  auto& tar = datas_[key];
  if (std::holds_alternative<torch::Tensor>(value)) {
    tar.push_back(std::get<torch::Tensor>(value));
  } else if (std::holds_alternative<std::vector<torch::Tensor>>(value)) {
    const auto& lst = std::get<std::vector<torch::Tensor>>(value);
    tar.insert(tar.end(), lst.begin(), lst.end());
  }

  return true;
}

bool CollectMMDataTensorVisitor::visit(MMData& data) {
  type_ |= data.type();
  data.foreach (item_visitor_);
  return true;
}

bool MMTokenNumVisitor::visit(MMDataItem& item) {
  if (item.type() == type_ && !item.is_embedded()) {
    token_nums_.push_back(item.state().mm_token_num());
  }
  return true;
}

bool EncoderInputGatherVisitor::visit(MMDataItem& item) {
  if (item.is_embedded()) return true;

  for (const auto& [key, value] : item.data()) {
    if (absl::StartsWith(key, filter_prefix_)) continue;
    auto& tar = datas_[key];
    if (std::holds_alternative<torch::Tensor>(value)) {
      tar.push_back(std::get<torch::Tensor>(value));
    } else if (std::holds_alternative<std::vector<torch::Tensor>>(value)) {
      const auto& vec = std::get<std::vector<torch::Tensor>>(value);
      tar.insert(tar.end(), vec.begin(), vec.end());
    }
  }
  return true;
}

bool EncoderInputGatherVisitor::finish(MMBatchData& mm_data) {
  MMDict dict;
  for (const auto& pair : datas_) {
    torch::Tensor tar;
    if (safe_concat(pair.second, tar)) {
      dict[pair.first] = tar;
    } else {
      dict[pair.first] = std::move(pair.second);
    }
  }
  mm_data.replace(dict);
  return true;
}

bool EncoderOutputScatterVisitor::visit(MMDataItem& item) {
  if (item.is_embedded()) return true;

  int32_t* idx = nullptr;

  if (item.type() == MMType::IMAGE) {
    idx = &image_idx;
  } else if (item.type() == MMType::VIDEO) {
    idx = &video_idx;
  } else if (item.type() == MMType::AUDIO) {
    idx = &audio_idx;
  } else {
    LOG(FATAL) << " mm data item type invalid, type is " << item.type();
    return true;
  }

  const std::string embedding_key = get_embedding_key(item.type());
  const auto& vec =
      std::get<std::vector<torch::Tensor>>(data_.at(embedding_key));
  item.add(embedding_key, vec[*idx]);
  ++(*idx);
  return true;
}

bool EncoderOutputScatterVisitor::finish() const {
  for (const auto& [key, value] : data_) {
    std::string name = key.substr(0, key.find("|"));
    uint32_t idx = 0;
    if (name == "image") {
      idx = image_idx;
    } else if (name == "video") {
      idx = video_idx;
    } else if (name == "audio") {
      idx = audio_idx;
    } else {
      LOG(FATAL) << "invalid modality key: " << key;
    }
    if (idx != std::get<std::vector<torch::Tensor>>(value).size()) {
      return false;
    }
  }
  return true;
}

EncoderEmbeddingGatherVisitor::EncoderEmbeddingGatherVisitor(
    const torch::Device& device,
    uint32_t mm_type,
    const std::vector<int32_t>& seq_lens,
    const std::vector<int32_t>& scheduled_seq_lens)
    : device_(device),
      per_seq_context_lens_(normalize_to_per_seq_lens(seq_lens)),
      per_seq_scheduled_lens_(normalize_to_per_seq_lens(scheduled_seq_lens)),
      per_seq_scheduled_offsets_(per_seq_scheduled_lens_.size(), 0),
      total_scheduled_tokens_(std::accumulate(per_seq_scheduled_lens_.begin(),
                                              per_seq_scheduled_lens_.end(),
                                              0)) {
  if (per_seq_scheduled_lens_.size() > 1) {
    std::partial_sum(per_seq_scheduled_lens_.begin(),
                     per_seq_scheduled_lens_.end() - 1,
                     per_seq_scheduled_offsets_.begin() + 1);
  }
  if (mm_type & MMType::IMAGE) {
    image_mask_ = torch::zeros({total_scheduled_tokens_},
                               torch::dtype(torch::kBool).device(device_));
  }
  if (mm_type & MMType::VIDEO) {
    video_mask_ = torch::zeros({total_scheduled_tokens_},
                               torch::dtype(torch::kBool).device(device_));
  }
  if (mm_type & MMType::AUDIO) {
    audio_mask_ = torch::zeros({total_scheduled_tokens_},
                               torch::dtype(torch::kBool).device(device_));
  }
}

bool EncoderEmbeddingGatherVisitor::visit(MMDataItem& item) {
  const auto& state = item.state();

  int32_t seq_index = state.seq_index();
  CHECK_GE(seq_index, 0);

  auto token_pos = item.state().token_pos();
  int32_t start_pos = state.schedule_data().start_pos;
  int32_t end_pos = state.schedule_data().end_pos;
  // start_pos / end_pos select the scheduled subrange inside this multimodal
  // item's token span in the prompt. Not every token in that subrange is a real
  // multimodal token, so mm_token_mask is used to filter out non-multimodal
  // positions when slicing multimodal embeddings.
  auto [emb_start, emb_end] =
      compute_emb_range(start_pos, end_pos, state.mm_token_mask());
  int32_t schedule_tokens_num = per_seq_scheduled_lens_[seq_index];
  int32_t context_tokens_num = per_seq_context_lens_[seq_index];

  int32_t computed_tokens_num = context_tokens_num - schedule_tokens_num;
  int32_t req_start_idx_ = per_seq_scheduled_offsets_[seq_index];
  int32_t req_start_pos =
      req_start_idx_ + token_pos.offset - computed_tokens_num + start_pos;
  int32_t req_end_pos =
      req_start_idx_ + token_pos.offset - computed_tokens_num + end_pos;

  if (item.type() == MMType::IMAGE) {
    image_mask_.slice(/*dim*/ 0,
                      /*start*/ req_start_pos,
                      /*end*/ req_end_pos) =
        state.mm_token_mask().slice(
            /*dim*/ 0, /*start*/ start_pos, /*end*/ end_pos);
  } else if (item.type() == MMType::VIDEO) {
    video_mask_.slice(/*dim*/ 0,
                      /*start*/ req_start_pos,
                      /*end*/ req_end_pos) =
        state.mm_token_mask().slice(
            /*dim*/ 0, /*start*/ start_pos, /*end*/ end_pos);
  } else if (item.type() == MMType::AUDIO) {
    audio_mask_.slice(/*dim*/ 0,
                      /*start*/ req_start_pos,
                      /*end*/ req_end_pos) =
        state.mm_token_mask().slice(
            /*dim*/ 0, /*start*/ start_pos, /*end*/ end_pos);
  }
  const std::string key = get_embedding_key(item.type());
  auto emb = item.get<torch::Tensor>(key);
  if (!emb.has_value()) {
    LOG(ERROR) << "embedding not found for key: " << key;
    return false;
  }
  torch::Tensor embedding = safe_to(emb.value(), device_, true);
  datas_[key].push_back(
      embedding.slice(/*dim*/ 0, /*start*/ emb_start, /*end*/ emb_end));
  return true;
}

bool EncoderEmbeddingGatherVisitor::finish(MMBatchData& mm_data) {
  MMDict data;
  torch::Tensor tar;
  for (auto& [key, value] : datas_) {
    if (safe_concat(value, tar)) {
      data[key] = tar;
    } else {
      LOG(ERROR) << "safe concat failed.";
      return false;
    }
  }
  // mask for merge multimodal embeddings into text.
  if (image_mask_.defined()) {
    data["image|mask"] = image_mask_;
  }
  if (video_mask_.defined()) {
    data["video|mask"] = video_mask_;
  }
  if (audio_mask_.defined()) {
    data["audio|mask"] = audio_mask_;
  }
  mm_data.replace(data);
  return true;
}

bool UpdateMMItemScheduleStateVisitor::visit(MMDataItem& item) {
  auto& schedule_data = item.mutable_state().mutable_schedule_data();
  auto& token_pos = item.state().token_pos();
  int32_t mm_end_idx = token_pos.offset + token_pos.length - 1;
  int32_t schedule_token_num = q_seq_len_;
  if (mm_end_idx < computed_token_num_) {
    return true;
  }
  if (token_pos.offset >= computed_token_num_ + schedule_token_num) {
    return true;
  }
  schedule_data.start_pos = std::max(computed_token_num_ - token_pos.offset, 0);
  schedule_data.end_pos =
      std::min(computed_token_num_ - token_pos.offset + schedule_token_num,
               token_pos.length);
  item.mutable_state().mutable_seq_index() = seq_idx_;
  scheduled_type_ |= item.type();
  mm_data_items_.push_back(item);
  return true;
}

EncoderCacheLookupVisitor::EncoderCacheLookupVisitor(EncoderCache* cache)
    : cache_(cache) {}

bool EncoderCacheLookupVisitor::visit(MMDataItem& item) {
  if (item.is_embedded()) {
    return true;
  }
  std::optional<torch::Tensor> cached =
      cache_->lookup(item.state().schedule_data().key);
  if (!cached.has_value()) {
    return true;
  }
  item.add(get_embedding_key(item.type()), cached.value());
  return true;
}

EncoderCacheInsertVisitor::EncoderCacheInsertVisitor(EncoderCache* cache)
    : cache_(cache) {}

bool EncoderCacheInsertVisitor::visit(MMDataItem& item) {
  if (!item.is_embedded()) {
    return true;
  }
  const XXH3Key& key = item.state().schedule_data().key;
  std::optional<torch::Tensor> embedding =
      item.get<torch::Tensor>(get_embedding_key(item.type()));
  if (!embedding.has_value()) {
    return true;
  }
  cache_->insert(key, std::move(embedding.value()));
  return true;
}

}  // namespace xllm
