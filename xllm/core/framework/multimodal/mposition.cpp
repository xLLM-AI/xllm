/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/framework/multimodal/mposition.h"

#include <glog/logging.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace xllm {
namespace {

std::vector<std::tuple<std::string, int32_t, int32_t>> group_by_token_type(
    const std::vector<std::string>& token_types) {
  std::vector<std::tuple<std::string, int32_t, int32_t>> groups;
  if (token_types.empty()) {
    return groups;
  }

  std::string current_key = token_types[0];
  int32_t start = 0;

  for (size_t i = 1; i < token_types.size(); ++i) {
    if (token_types[i] != current_key) {
      groups.emplace_back(current_key, start, i);
      current_key = token_types[i];
      start = static_cast<int32_t>(i);
    }
  }
  groups.emplace_back(
      current_key, start, static_cast<int32_t>(token_types.size()));
  return groups;
}

torch::Tensor get_mm_tensor(const MMData& mm_data, const std::string& key) {
  torch::Tensor tensor;
  if (std::optional<torch::Tensor> result = mm_data.get<torch::Tensor>(key)) {
    tensor = result.value();
  }
  return tensor;
}

}  // namespace

torch::Tensor build_image_mrope_positions(const torch::Tensor& input_ids,
                                          const torch::Tensor& attention_mask,
                                          const torch::Tensor& image_grid_thw,
                                          int64_t spatial_merge_size,
                                          int32_t image_token_id,
                                          int32_t vision_start_token_id) {
  CHECK_GT(spatial_merge_size, 0);
  const auto long_opts =
      torch::TensorOptions().dtype(/*device=*/torch::kLong).device(/*dtype=*/torch::kCPU);
  const auto ids = input_ids.to(/*device=*/torch::kCPU, /*dtype=*/torch::kLong);
  const auto mask = attention_mask.to(/*device=*/torch::kCPU, /*dtype=*/torch::kBool);
  const auto grids = image_grid_thw.defined()
                         ? image_grid_thw.to(/*device=*/torch::kCPU, /*dtype=*/torch::kLong)
                         : torch::Tensor();
  const int64_t num_image_grids = grids.defined() ? grids.size(/*dim=*/0) : 0;
  const int64_t batch_size = ids.size(/*dim=*/0);
  const int64_t seq_len = ids.size(/*dim=*/1);
  auto position_ids = torch::ones({3, batch_size, seq_len}, long_opts);

  int64_t global_image_index = 0;
  for (int64_t b = 0; b < batch_size; ++b) {
    const auto mask_b = mask[b];
    const auto ids_b = ids[b].index({mask_b});
    const auto* tokens = ids_b.const_data_ptr<int64_t>();
    const int64_t token_count = ids_b.numel();

    int64_t image_nums = 0;
    for (int64_t i = 0; i + 1 < token_count; ++i) {
      if (tokens[i] == vision_start_token_id &&
          tokens[i + 1] == image_token_id) {
        image_nums++;
      }
    }

    std::vector<torch::Tensor> llm_pos_ids_list;
    llm_pos_ids_list.reserve(static_cast<size_t>(image_nums) * 2 + 1);
    int64_t st = 0;
    int64_t next_pos = 0;
    for (int64_t n = 0; n < image_nums; ++n) {
      int64_t ed = token_count + 1;
      for (int64_t p = st; p < token_count; ++p) {
        if (tokens[p] == image_token_id) {
          ed = p;
          break;
        }
      }

      CHECK_LT(global_image_index, num_image_grids)
          << "image_grid_thw count is smaller than required image tokens";
      const auto grid = grids[global_image_index++];
      const int64_t llm_grid_t = grid[0].item<int64_t>();
      const int64_t llm_grid_h = grid[1].item<int64_t>() / spatial_merge_size;
      const int64_t llm_grid_w = grid[2].item<int64_t>() / spatial_merge_size;
      int64_t text_len = ed - st;

      auto text_pos =
          torch::arange(text_len, long_opts).view(/*size=*/{1, -1}).expand(/*size=*/{3, -1}) +
          next_pos;
      llm_pos_ids_list.push_back(text_pos);

      auto t_index = torch::arange(llm_grid_t, long_opts)
                         .view(/*size=*/{-1, 1})
                         .expand({-1, llm_grid_h * llm_grid_w})
                         .flatten();
      auto h_index = torch::arange(llm_grid_h, long_opts)
                         .view(/*size=*/{1, -1, 1})
                         .expand({llm_grid_t, -1, llm_grid_w})
                         .flatten();
      auto w_index = torch::arange(llm_grid_w, long_opts)
                         .view(/*size=*/{1, 1, -1})
                         .expand({llm_grid_t, llm_grid_h, -1})
                         .flatten();
      next_pos += text_len;
      auto vision_pos = torch::stack({t_index, h_index, w_index}) + next_pos;
      llm_pos_ids_list.push_back(vision_pos);
      next_pos += std::max({llm_grid_t, llm_grid_h, llm_grid_w});
      st = ed + llm_grid_t * llm_grid_h * llm_grid_w;
    }

    if (st < token_count) {
      int64_t text_len = token_count - st;
      auto text_pos =
          torch::arange(text_len, long_opts).view(/*size=*/{1, -1}).expand(/*size=*/{3, -1}) +
          next_pos;
      llm_pos_ids_list.push_back(text_pos);
    }

    if (!llm_pos_ids_list.empty()) {
      auto llm_positions = torch::cat(llm_pos_ids_list, /*dim=*/1);
      position_ids.index_put_({torch::indexing::Slice(), b, mask_b},
                              llm_positions);
    }
  }

  return position_ids.view(/*size=*/{3, -1}).to(input_ids.device());
}

MPositionGeneratorFactory& MPositionGeneratorFactory::get_instance() {
  static MPositionGeneratorFactory instance;
  return instance;
}

bool MPositionGeneratorFactory::register_creator(const std::string& name,
                                                 Creator creator) {
  auto [it, inserted] = creators_.emplace(name, std::move(creator));
  return inserted;
}

std::unique_ptr<MPositionGenerator>
MPositionGeneratorFactory::create_mposition_generator(const std::string& name) {
  auto it = creators_.find(name);
  if (it == creators_.end()) {
    LOG(FATAL) << "No mposition generator found: " << name;
  }

  return it->second();
}

std::tuple<torch::Tensor, int32_t> Glm4VMPositionGenerator::generate(
    Slice<int32_t> input_tokens,
    const MMData& mm_data,
    const ModelArgs& model_args) const {
  torch::Tensor image_grid_thw = get_mm_tensor(mm_data, "image_grid_thw");
  torch::Tensor video_grid_thw = get_mm_tensor(mm_data, "video_grid_thw");
  int32_t spatial_merge_size = model_args.mm_spatial_merge_size();
  int32_t image_token_id = model_args.image_token_id();
  int32_t video_token_id = model_args.video_token_id();
  int32_t video_start_token_id = model_args.video_start_token_id();
  int32_t video_end_token_id = model_args.video_end_token_id();

  torch::Dtype dtype = torch::kInt32;

  std::vector<std::string> input_token_type;
  bool in_video = false;
  int32_t num_tokens = static_cast<int32_t>(input_tokens.size());

  for (int32_t index = 0; index < num_tokens; ++index) {
    int32_t token = input_tokens[index];
    if (token == video_start_token_id) {
      in_video = true;
    } else if (token == video_end_token_id) {
      in_video = false;
    }

    if (token == image_token_id && !in_video) {
      input_token_type.push_back("image");
    } else if (token == image_token_id && in_video) {
      input_token_type.push_back("video");
    } else {
      input_token_type.push_back("text");
    }
  }
  auto input_type_group = group_by_token_type(input_token_type);
  int32_t image_index = 0;
  int32_t video_index = 0;
  int32_t video_group_index = 0;

  std::vector<torch::Tensor> llm_pos_ids_list;
  int32_t video_frame_num = 1;
  for (const auto& group : input_type_group) {
    const std::string& modality_type = std::get<0>(group);
    int32_t start_idx = std::get<1>(group);
    int32_t end_idx = std::get<2>(group);
    int32_t st_idx = 0;
    if (!llm_pos_ids_list.empty()) {
      st_idx = llm_pos_ids_list.back().max().item<int32_t>() + 1;
    }

    if (modality_type == "image") {
      torch::Tensor grid = image_grid_thw[image_index];
      int32_t t = grid[0].item<int32_t>();
      int32_t h = grid[1].item<int32_t>() / spatial_merge_size;
      int32_t w = grid[2].item<int32_t>() / spatial_merge_size;

      torch::Tensor t_arange =
          torch::arange(t, dtype).view({-1, 1}).expand({-1, h * w}).flatten();
      torch::Tensor h_arange =
          torch::arange(h, dtype).view({1, -1, 1}).expand({t, -1, w}).flatten();
      torch::Tensor w_arange =
          torch::arange(w, dtype).view({1, 1, -1}).expand({t, h, -1}).flatten();

      torch::Tensor pos = torch::stack({t_arange, h_arange, w_arange}) + st_idx;
      llm_pos_ids_list.push_back(pos);
      video_frame_num = 1;
      image_index++;
    } else if (modality_type == "video") {
      int32_t t = video_frame_num;
      int32_t h =
          video_grid_thw[video_index][1].item<int32_t>() / spatial_merge_size;
      int32_t w =
          video_grid_thw[video_index][2].item<int32_t>() / spatial_merge_size;

      for (int32_t t_idx = 0; t_idx < t; ++t_idx) {
        torch::Tensor t_tensor =
            torch::full({1, h * w}, t_idx, dtype).flatten();
        torch::Tensor h_tensor = torch::arange(h, dtype)
                                     .view({1, -1, 1})
                                     .expand({1, -1, w})
                                     .flatten();
        torch::Tensor w_tensor = torch::arange(w, dtype)
                                     .view({1, 1, -1})
                                     .expand({1, h, -1})
                                     .flatten();

        torch::Tensor pos =
            torch::stack({t_tensor, h_tensor, w_tensor}) + st_idx;
        llm_pos_ids_list.push_back(pos);
      }

      video_group_index++;
      if (video_group_index >= video_grid_thw[video_index][0].item<int32_t>()) {
        video_index++;
        video_group_index = 0;
      }
      video_frame_num++;
    } else {
      int32_t text_len = end_idx - start_idx;
      torch::Tensor arange =
          torch::arange(text_len, dtype).view({1, -1}).expand({3, -1}) + st_idx;
      llm_pos_ids_list.push_back(arange);
      video_frame_num = 1;
    }
  }

  torch::Tensor llm_positions =
      torch::cat(llm_pos_ids_list, /*dim=*/1).reshape({3, -1});
  int32_t mrope_position_delta = llm_positions.max().item<int32_t>() + 1 -
                                 static_cast<int32_t>(input_tokens.size());

  return std::make_pair(llm_positions, mrope_position_delta);
}

std::tuple<torch::Tensor, int32_t> QwenVLMPositionGenerator::generate(
    Slice<int32_t> input_tokens,
    const MMData& mm_data,
    const ModelArgs& model_args) const {
  torch::Tensor image_grid_thw = get_mm_tensor(mm_data, "image_grid_thw");
  torch::Tensor video_grid_thw = get_mm_tensor(mm_data, "video_grid_thw");
  torch::Tensor second_per_grid_ts =
      get_mm_tensor(mm_data, "second_per_grid_ts");
  int32_t image_token_id = model_args.image_token_id();
  int32_t video_token_id = model_args.video_token_id();
  int32_t vision_start_token_id = model_args.vision_start_token_id();
  int32_t spatial_merge_size = model_args.mm_spatial_merge_size();
  float tokens_per_second = model_args.mm_tokens_per_second();

  torch::Tensor input_tokens_tensor =
      torch::tensor(std::vector<int32_t>(input_tokens), torch::kInt32);
  torch::Tensor vision_start_indices =
      torch::argwhere(input_tokens_tensor == vision_start_token_id).squeeze(1);

  torch::Tensor vision_tokens =
      input_tokens_tensor.index({vision_start_indices + 1});
  int32_t image_nums =
      torch::sum(vision_tokens == image_token_id).item<int32_t>();
  int32_t video_nums =
      torch::sum(vision_tokens == video_token_id).item<int32_t>();

  std::vector<torch::Tensor> llm_pos_ids_list;
  int32_t st = 0;
  int32_t remain_images = image_nums;
  int32_t remain_videos = video_nums;
  int32_t image_index = 0;
  int32_t video_index = 0;

  for (int32_t i = 0; i < image_nums + video_nums; ++i) {
    float video_second_per_grid_t = 1.0f;
    int32_t ed_image = static_cast<int32_t>(input_tokens.size()) + 1;
    int32_t ed_video = static_cast<int32_t>(input_tokens.size()) + 1;

    if (remain_images > 0) {
      auto it = std::find(
          input_tokens.begin() + st, input_tokens.end(), image_token_id);
      if (it != input_tokens.end()) {
        ed_image =
            static_cast<int32_t>(std::distance(input_tokens.begin(), it));
      }
    }

    if (remain_videos > 0) {
      auto it = std::find(
          input_tokens.begin() + st, input_tokens.end(), video_token_id);
      if (it != input_tokens.end()) {
        ed_video =
            static_cast<int32_t>(std::distance(input_tokens.begin(), it));
      }
    }

    int32_t t = 0;
    int32_t h = 0;
    int32_t w = 0;
    int32_t ed = 0;
    if (ed_image < ed_video) {
      t = image_grid_thw[image_index][0].item<int32_t>();
      h = image_grid_thw[image_index][1].item<int32_t>();
      w = image_grid_thw[image_index][2].item<int32_t>();
      image_index++;
      remain_images--;
      ed = ed_image;
    } else {
      t = video_grid_thw[video_index][0].item<int32_t>();
      h = video_grid_thw[video_index][1].item<int32_t>();
      w = video_grid_thw[video_index][2].item<int32_t>();

      video_second_per_grid_t = second_per_grid_ts[video_index].item<float>();

      video_index++;
      remain_videos--;
      ed = ed_video;
    }

    int32_t llm_grid_t = t;
    int32_t llm_grid_h = h / spatial_merge_size;
    int32_t llm_grid_w = w / spatial_merge_size;
    int32_t text_len = ed - st;

    int32_t st_idx = 0;
    if (!llm_pos_ids_list.empty()) {
      st_idx = llm_pos_ids_list.back().max().item<int32_t>() + 1;
    }

    if (text_len > 0) {
      torch::Tensor text_pos =
          torch::arange(text_len, torch::kInt32).view({1, -1}).expand({3, -1}) +
          st_idx;
      llm_pos_ids_list.push_back(text_pos);
    }

    torch::Tensor t_index = (torch::arange(llm_grid_t, torch::kInt32)
                                 .view({-1, 1})
                                 .expand({-1, llm_grid_h * llm_grid_w}) *
                             video_second_per_grid_t * tokens_per_second)
                                .to(torch::kInt32)
                                .flatten();

    torch::Tensor h_index = torch::arange(llm_grid_h, torch::kInt32)
                                .view({1, -1, 1})
                                .expand({llm_grid_t, -1, llm_grid_w})
                                .flatten();

    torch::Tensor w_index = torch::arange(llm_grid_w, torch::kInt32)
                                .view({1, 1, -1})
                                .expand({llm_grid_t, llm_grid_h, -1})
                                .flatten();

    torch::Tensor visual_pos =
        torch::stack({t_index, h_index, w_index}) + text_len + st_idx;
    llm_pos_ids_list.push_back(visual_pos);

    st = ed + llm_grid_t * llm_grid_h * llm_grid_w;
  }

  if (st < static_cast<int32_t>(input_tokens.size())) {
    int32_t st_idx = 0;
    if (!llm_pos_ids_list.empty()) {
      st_idx = llm_pos_ids_list.back().max().item<int32_t>() + 1;
    }

    int32_t text_len = static_cast<int32_t>(input_tokens.size()) - st;
    torch::Tensor text_pos =
        torch::arange(text_len, torch::kInt32).view({1, -1}).expand({3, -1}) +
        st_idx;
    llm_pos_ids_list.push_back(text_pos);
  }

  torch::Tensor llm_positions =
      torch::cat(llm_pos_ids_list, 1).reshape({3, -1});
  int32_t mrope_position_delta = llm_positions.max().item<int32_t>() + 1 -
                                 static_cast<int32_t>(input_tokens.size());
  return std::make_tuple(llm_positions, mrope_position_delta);
}

std::tuple<torch::Tensor, int32_t> Qwen3VLMPositionGenerator::generate(
    Slice<int32_t> input_tokens,
    const MMData& mm_data,
    const ModelArgs& model_args) const {
  torch::Tensor image_grid_thw = get_mm_tensor(mm_data, "image_grid_thw");
  torch::Tensor video_grid_thw = get_mm_tensor(mm_data, "video_grid_thw");
  int32_t image_token_id = model_args.image_token_id();
  int32_t video_token_id = model_args.video_token_id();
  int32_t vision_start_token_id = model_args.vision_start_token_id();
  int32_t spatial_merge_size = model_args.mm_spatial_merge_size();

  if (video_grid_thw.defined() && video_grid_thw.numel() > 0) {
    torch::Tensor t_counts =
        video_grid_thw.index({torch::indexing::Slice(), 0}).to(torch::kLong);
    video_grid_thw =
        torch::repeat_interleave(video_grid_thw, t_counts, /*dim=*/0);
    video_grid_thw.index_put_({torch::indexing::Slice(), 0}, 1);
  }

  torch::Tensor input_tokens_tensor =
      torch::tensor(std::vector<int32_t>(input_tokens), torch::kInt32);
  torch::Tensor vision_start_indices =
      torch::argwhere(input_tokens_tensor == vision_start_token_id).squeeze(1);
  torch::Tensor vision_tokens =
      input_tokens_tensor.index({vision_start_indices + 1});

  int32_t image_nums =
      torch::sum(vision_tokens == image_token_id).item<int32_t>();
  int32_t video_nums =
      torch::sum(vision_tokens == video_token_id).item<int32_t>();

  std::vector<torch::Tensor> llm_pos_ids_list;
  int32_t st = 0;
  int32_t remain_images = image_nums;
  int32_t remain_videos = video_nums;
  int32_t image_index = 0;
  int32_t video_index = 0;

  for (int32_t i = 0; i < image_nums + video_nums; ++i) {
    int32_t ed_image = static_cast<int32_t>(input_tokens.size()) + 1;
    int32_t ed_video = static_cast<int32_t>(input_tokens.size()) + 1;

    if (remain_images > 0) {
      auto it = std::find(
          input_tokens.begin() + st, input_tokens.end(), image_token_id);
      if (it != input_tokens.end()) {
        ed_image =
            static_cast<int32_t>(std::distance(input_tokens.begin(), it));
      }
    }

    if (remain_videos > 0) {
      auto it = std::find(
          input_tokens.begin() + st, input_tokens.end(), video_token_id);
      if (it != input_tokens.end()) {
        ed_video =
            static_cast<int32_t>(std::distance(input_tokens.begin(), it));
      }
    }

    int32_t t = 0;
    int32_t h = 0;
    int32_t w = 0;
    int32_t ed = 0;
    if (ed_image < ed_video) {
      t = image_grid_thw[image_index][0].item<int32_t>();
      h = image_grid_thw[image_index][1].item<int32_t>();
      w = image_grid_thw[image_index][2].item<int32_t>();
      image_index++;
      remain_images--;
      ed = ed_image;
    } else {
      t = video_grid_thw[video_index][0].item<int32_t>();
      h = video_grid_thw[video_index][1].item<int32_t>();
      w = video_grid_thw[video_index][2].item<int32_t>();
      video_index++;
      remain_videos--;
      ed = ed_video;
    }

    int32_t llm_grid_t = t;
    int32_t llm_grid_h = h / spatial_merge_size;
    int32_t llm_grid_w = w / spatial_merge_size;
    int32_t text_len = ed - st;

    int32_t st_idx = 0;
    if (!llm_pos_ids_list.empty()) {
      st_idx = llm_pos_ids_list.back().max().item<int32_t>() + 1;
    }

    if (text_len > 0) {
      torch::Tensor text_pos =
          torch::arange(text_len, torch::kInt32).view({1, -1}).expand({3, -1}) +
          st_idx;
      llm_pos_ids_list.push_back(text_pos);
    }

    torch::Tensor t_index = torch::arange(llm_grid_t, torch::kInt32)
                                .view({-1, 1})
                                .expand({-1, llm_grid_h * llm_grid_w})
                                .flatten();

    torch::Tensor h_index = torch::arange(llm_grid_h, torch::kInt32)
                                .view({1, -1, 1})
                                .expand({llm_grid_t, -1, llm_grid_w})
                                .flatten();

    torch::Tensor w_index = torch::arange(llm_grid_w, torch::kInt32)
                                .view({1, 1, -1})
                                .expand({llm_grid_t, llm_grid_h, -1})
                                .flatten();

    torch::Tensor visual_pos =
        torch::stack({t_index, h_index, w_index}) + text_len + st_idx;
    llm_pos_ids_list.push_back(visual_pos);

    st = ed + llm_grid_t * llm_grid_h * llm_grid_w;
  }

  if (st < static_cast<int32_t>(input_tokens.size())) {
    int32_t st_idx = 0;
    if (!llm_pos_ids_list.empty()) {
      st_idx = llm_pos_ids_list.back().max().item<int32_t>() + 1;
    }

    int32_t text_len = static_cast<int32_t>(input_tokens.size()) - st;
    torch::Tensor text_pos =
        torch::arange(text_len, torch::kInt32).view({1, -1}).expand({3, -1}) +
        st_idx;
    llm_pos_ids_list.push_back(text_pos);
  }

  torch::Tensor llm_positions =
      torch::cat(llm_pos_ids_list, 1).reshape({3, -1});
  int32_t mrope_position_delta = llm_positions.max().item<int32_t>() + 1 -
                                 static_cast<int32_t>(input_tokens.size());
  return std::make_tuple(llm_positions, mrope_position_delta);
}

}  // namespace xllm
