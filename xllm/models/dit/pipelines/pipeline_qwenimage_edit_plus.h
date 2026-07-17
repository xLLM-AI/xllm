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

#include <torch/torch.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "core/common/global_flags.h"
#include "core/framework/config/dit_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/dit_cache/dit_cache.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/request/dit_request_state.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "framework/parallel_state/parallel_state.h"
#include "models/dit/autoencoders/autoencoder_kl.h"
#include "models/dit/autoencoders/autoencoder_kl_qwenimage.h"
#include "models/dit/schedulers/flowmatch_euler_discrete_scheduler.h"
#include "models/dit/transformers/transformer_qwen_image.h"
#include "models/dit/utils/util.h"
#include "models/model_registry.h"
#if defined(USE_DCU)
#include "models/vlm/qwen2_5_vl.h"
#endif
#include "processors/qwen2_vl_image_processor.h"
#include "util/tensor_helper.h"

#define CONDITION_IMAGE_SIZE 147456

namespace xllm {

#if defined(USE_DCU)
using QwenImageEditTextEncoder = Qwen2_5_VLForConditionalGeneration;
#endif

inline int32_t get_qwen_image_edit_vlm_tp_port(int32_t rank) {
  constexpr int32_t kQwenImageEditVlmTpBasePort = 29502;
  const int32_t port = kQwenImageEditVlmTpBasePort + rank;
  CHECK_LE(port, 65535)
      << "Qwen-Image-Edit VLM ProcessGroup port overflow: base_port="
      << kQwenImageEditVlmTpBasePort << ", rank=" << rank;
  return port;
}

inline int64_t get_qwen_image_edit_vae_target_area(int64_t height,
                                                   int64_t width) {
  constexpr int64_t kDefaultVaeTargetArea = 1024 * 1024;
  int64_t request_area =
      height > 0 && width > 0 ? height * width : kDefaultVaeTargetArea;

  const int64_t config_area =
      ::xllm::DiTConfig::get_instance().dit_vae_image_size();
  if (config_area > 0 && config_area != kDefaultVaeTargetArea) {
    return config_area;
  }
  return request_area;
}

class QwenImageEditPlusPipelineImpl : public torch::nn::Module {
 public:
  QwenImageEditPlusPipelineImpl(const DiTModelContext& context)
      : context_(context),
        parallel_args_(context.get_parallel_args()),
#if defined(USE_DCU)
        vl_image_processor_(context.get_model_args("processor")),
#endif
        vae_model_args_(context.get_model_args("vae")) {
    options_ = context.get_tensor_options();
    dtype_ = options_.dtype().toScalarType();
    device_ = options_.device();
    LOG(INFO) << "model info " << dtype_ << " ; " << options_.device();

    in_channels_ = context.get_model_args("transformer").in_channels();
    num_layers_ = context.get_model_args("transformer").num_layers();

    vae_scale_factor_ = static_cast<int64_t>(
        std::pow(2, vae_model_args_.temperal_downsample().size()));
    latent_channels_ = vae_model_args_.z_dim();
    tokenizer_max_length_ = 1024;

    prompt_template_encode_ =
        "<|im_start|>system\nDescribe the key features of the input image "
        "(color, shape, size, texture, objects, background), then explain how "
        "the user's text instruction should alter or modify the image. "
        "Generate a new image that meets the user's requirements while "
        "maintaining consistency with the original input where "
        "appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>"
        "assistant\n";
    prompt_template_encode_start_idx_ = 64;
    default_sample_size_ = 128;

    vae_ = AutoencoderKLQwenImage(context.get_model_context("vae"));
    if (::xllm::DiTConfig::get_instance().dit_enable_vae_tiling()) {
      vae_->enable_tiling();
    }
    transformer_ = QwenImageTransformer2DModel(
        context.get_model_context("transformer"), parallel_args_);
    scheduler_ =
        FlowMatchEulerDiscreteScheduler(context.get_model_context("scheduler"));

#if defined(USE_DCU)
    const auto& original_parallel_args = context.get_parallel_args();
    ParallelArgs vlm_parallel_args = original_parallel_args;
    if (original_parallel_args.tp_group_ == nullptr) {
      if (original_parallel_args.dit_tp_group_ != nullptr) {
        vlm_parallel_args.tp_group_ = original_parallel_args.dit_tp_group_;
      } else {
        const int32_t vlm_tp_port =
            get_qwen_image_edit_vlm_tp_port(original_parallel_args.rank());
        LOG(INFO) << "Creating single-rank Qwen-Image-Edit VLM ProcessGroup on "
                  << "127.0.0.1:" << vlm_tp_port;
        vlm_tp_group_ =
            create_process_group(/*rank=*/0,
                                 /*world_size=*/1,
                                 /*rank_size=*/1,
                                 vlm_tp_port,
                                 /*trans=*/false,
                                 /*host=*/"127.0.0.1",
                                 /*group_name=*/"vlm_tp_group_qwen_image_edit",
                                 options_.device());
        vlm_parallel_args.tp_group_ = vlm_tp_group_.get();
      }
    }
    text_encoder_ = QwenImageEditTextEncoder(
        ModelContext(vlm_parallel_args,
                     context.get_model_args("text_encoder"),
                     context.get_quant_args("text_encoder"),
                     context.get_tensor_options()));
#endif

    vae_image_processor_ =
        xllm::VAEImageProcessor(context.get_model_context("vae"),
                                /*do_resize=*/true,
                                /*do_normalize=*/true,
                                /*do_binarize=*/false,
                                /*do_convert_rgb=*/false,
                                /*do_convert_grayscale=*/false,
                                latent_channels_,
                                /*scale_factor=*/vae_scale_factor_ * 2);

    register_module("vae", vae_);
    register_module("scheduler", scheduler_);
    register_module("transformer", transformer_);
#if defined(USE_DCU)
    register_module("text_encoder", text_encoder_);
#endif
    register_module("vae_image_processor", vae_image_processor_);

    use_layer3d_rope_ = context.get_model_context("transformer")
                            .get_model_args()
                            .use_layer3d_rope();
    std::vector<int64_t> axes_dims_rope =
        context.get_model_context("transformer")
            .get_model_args()
            .axes_dims_rope();
    if (use_layer3d_rope_) {
      pos_embed_3d_rope_ = register_module(
          "pos_embed",
          QwenEmbedLayer3DRope(context.get_model_context("transformer"),
                               /*theta=*/10000,
                               axes_dims_rope,
                               true));
    } else {
      pos_embed_ = register_module(
          "pos_embed",
          QwenEmbedRope(context.get_model_context("transformer"),
                        /*theta=*/10000,
                        axes_dims_rope,
                        true));
    }
  }

#if defined(USE_DCU)
  torch::Tensor build_qwen2_5_vl_mrope_positions(
      const torch::Tensor& input_ids,
      const torch::Tensor& attention_mask,
      const torch::Tensor& image_grid_thw) {
    CHECK(input_ids.dim() == 2) << "input_ids must be [B, S]";
    CHECK(attention_mask.dim() == 2) << "attention_mask must be [B, S]";

    auto long_opts =
        torch::TensorOptions().dtype(torch::kLong).device(input_ids.device());
    torch::Tensor position_ids =
        torch::ones({3, input_ids.size(0), input_ids.size(1)}, long_opts);

    const auto& text_encoder_args = context_.get_model_args("text_encoder");
    int64_t spatial_merge_size = text_encoder_args.mm_spatial_merge_size();
    CHECK_GT(spatial_merge_size, 0)
        << "QwenImageEditPlus text_encoder mm_spatial_merge_size must be > 0";
    int64_t image_token_id = text_encoder_args.image_token_id();
    int64_t vision_start_token_id = text_encoder_args.vision_start_token_id();

    int64_t global_image_index = 0;
    int64_t num_image_grids =
        image_grid_thw.defined() ? image_grid_thw.size(0) : 0;

    for (int64_t b = 0; b < input_ids.size(0); ++b) {
      auto mask_b = attention_mask[b].to(torch::kBool);
      auto ids_b = input_ids[b].index({mask_b}).to(torch::kLong).cpu();
      std::vector<int64_t> tokens(ids_b.data_ptr<int64_t>(),
                                  ids_b.data_ptr<int64_t>() + ids_b.numel());

      int64_t image_nums = 0;
      for (int64_t i = 0; i + 1 < static_cast<int64_t>(tokens.size()); ++i) {
        if (tokens[i] == vision_start_token_id &&
            tokens[i + 1] == image_token_id) {
          image_nums++;
        }
      }

      std::vector<torch::Tensor> llm_pos_ids_list;
      llm_pos_ids_list.reserve(static_cast<size_t>(image_nums) * 2 + 1);
      int64_t st = 0;
      int64_t remain_images = image_nums;
      for (int64_t n = 0; n < image_nums; ++n) {
        int64_t ed = static_cast<int64_t>(tokens.size()) + 1;
        if (remain_images > 0) {
          for (int64_t p = st; p < static_cast<int64_t>(tokens.size()); ++p) {
            if (tokens[p] == image_token_id) {
              ed = p;
              break;
            }
          }
        }

        CHECK(num_image_grids > 0)
            << "image_grid_thw is required when image tokens exist";
        CHECK(global_image_index < num_image_grids)
            << "image_grid_thw count is smaller than required image tokens";
        int64_t t = image_grid_thw[global_image_index][0].item<int64_t>();
        int64_t h = image_grid_thw[global_image_index][1].item<int64_t>();
        int64_t w = image_grid_thw[global_image_index][2].item<int64_t>();

        int64_t llm_grid_t = t;
        int64_t llm_grid_h = h / spatial_merge_size;
        int64_t llm_grid_w = w / spatial_merge_size;
        int64_t text_len = ed - st;

        int64_t st_idx = 0;
        if (!llm_pos_ids_list.empty()) {
          st_idx = llm_pos_ids_list.back().max().item<int64_t>() + 1;
        }

        auto text_pos =
            torch::arange(text_len, long_opts).view({1, -1}).expand({3, -1}) +
            st_idx;
        llm_pos_ids_list.push_back(text_pos);

        auto t_index = torch::arange(llm_grid_t, long_opts)
                           .view({-1, 1})
                           .expand({-1, llm_grid_h * llm_grid_w})
                           .flatten();
        auto h_index = torch::arange(llm_grid_h, long_opts)
                           .view({1, -1, 1})
                           .expand({llm_grid_t, -1, llm_grid_w})
                           .flatten();
        auto w_index = torch::arange(llm_grid_w, long_opts)
                           .view({1, 1, -1})
                           .expand({llm_grid_t, llm_grid_h, -1})
                           .flatten();
        auto vision_pos =
            torch::stack({t_index, h_index, w_index}, 0) + text_len + st_idx;
        llm_pos_ids_list.push_back(vision_pos);

        st = ed + llm_grid_t * llm_grid_h * llm_grid_w;
        global_image_index++;
        remain_images--;
      }

      if (st < static_cast<int64_t>(tokens.size())) {
        int64_t st_idx = 0;
        if (!llm_pos_ids_list.empty()) {
          st_idx = llm_pos_ids_list.back().max().item<int64_t>() + 1;
        }
        int64_t text_len = static_cast<int64_t>(tokens.size()) - st;
        auto text_pos =
            torch::arange(text_len, long_opts).view({1, -1}).expand({3, -1}) +
            st_idx;
        llm_pos_ids_list.push_back(text_pos);
      }

      if (!llm_pos_ids_list.empty()) {
        auto llm_positions = torch::cat(llm_pos_ids_list, 1).reshape({3, -1});
        position_ids.index_put_({torch::indexing::Slice(), b, mask_b},
                                llm_positions.to(position_ids.device()));
      }
    }

    return position_ids.reshape({3, -1}).contiguous();
  }

  ModelInputParams build_qwen_vl_input_params(
      const torch::Tensor& tokens,
      const torch::Tensor& attention_mask,
      MMBatchData mm_batch) {
    ModelInputParams params;
    CHECK(attention_mask.defined() && attention_mask.dim() == 2)
        << "QwenImageEditPlus text encoder requires a [B, S] attention mask";
    int64_t batch_size = attention_mask.size(0);
    if (tokens.numel() > 0) {
      auto lengths_cpu =
          attention_mask.to(torch::kCPU).to(torch::kInt64).sum(/*dim=*/1);
      const int64_t* lengths_data = lengths_cpu.data_ptr<int64_t>();
      std::vector<int32_t> seq_lens;
      seq_lens.reserve(static_cast<size_t>(batch_size));
      std::vector<int32_t> cu_seqlens_vec;
      cu_seqlens_vec.reserve(static_cast<size_t>(batch_size) + 1);
      cu_seqlens_vec.emplace_back(0);
      int32_t max_seq_len = 0;
      for (int64_t i = 0; i < batch_size; ++i) {
        CHECK_GT(lengths_data[i], 0)
            << "QwenImageEditPlus text encoder sequence length must be > 0";
        CHECK_LE(lengths_data[i], std::numeric_limits<int32_t>::max())
            << "QwenImageEditPlus text encoder sequence length exceeds int32";
        int32_t seq_len = static_cast<int32_t>(lengths_data[i]);
        seq_lens.emplace_back(seq_len);
        cu_seqlens_vec.emplace_back(cu_seqlens_vec.back() + seq_len);
        max_seq_len = std::max(max_seq_len, seq_len);
      }
      CHECK_EQ(tokens.numel(), cu_seqlens_vec.back())
          << "packed token count does not match attention mask";
      params.meta.num_sequences = static_cast<int32_t>(batch_size);
      params.meta.actual_num_sequences = static_cast<int32_t>(batch_size);
      params.meta.q_max_seq_len = max_seq_len;
      params.meta.kv_max_seq_len = max_seq_len;
      auto cu_seqlens =
          torch::tensor(cu_seqlens_vec, torch::kInt).to(tokens.device());
      params.attention.device.q_seq_lens = cu_seqlens;
      params.attention.device.kv_seq_lens = cu_seqlens;
      params.attention.device.q_cu_seq_lens = cu_seqlens;
      params.attention.host.q_seq_lens = seq_lens;
      params.attention.host.q_cu_seq_lens = cu_seqlens_vec;
      params.attention.host.kv_seq_lens = seq_lens;
      params.meta.batch_forward_type = BatchForwardType::PREFILL;
    }

    ModelInputParams mm_params;
    mm_params.multimodal.mm_data = MMBatchData::to(mm_batch, tokens.device());
    MMDict multimodal_embeds =
        text_encoder_->get_multimodal_embeddings(mm_params);

    auto image_embedding_value = multimodal_embeds.find("image|embedding");
    CHECK(image_embedding_value != multimodal_embeds.end())
        << "Qwen image encoder did not produce image embeddings";
    torch::Tensor image_embeddings;
    if (std::holds_alternative<torch::Tensor>(image_embedding_value->second)) {
      image_embeddings = std::get<torch::Tensor>(image_embedding_value->second);
    } else {
      const auto& image_embedding_list =
          std::get<std::vector<torch::Tensor>>(image_embedding_value->second);
      image_embeddings = torch::cat(image_embedding_list, 0);
    }

    torch::Tensor image_mask =
        tokens.eq(context_.get_model_args("text_encoder").image_token_id());
    CHECK_EQ(image_mask.sum().item<int64_t>(), image_embeddings.size(0))
        << "image token count does not match image embedding count";

    MMDict embedding_data;
    embedding_data["image|embedding"] = image_embeddings;
    embedding_data["image|mask"] = image_mask;
    params.multimodal.mm_data = MMBatchData(MMType::IMAGE, embedding_data);

    params.embedding.input_embedding =
        text_encoder_->get_input_embeddings(tokens, params);
    return params;
  }
#endif

  std::pair<torch::Tensor, torch::Tensor> _get_qwen_prompt_embeds(
      const std::vector<std::string>& prompt,
      const std::vector<torch::Tensor>& image,
      torch::TensorOptions& options,
      int64_t max_sequence_length) {
#if defined(USE_DCU)
    CHECK(tokenizer_ != nullptr) << "Tokenizer not loaded";
    CHECK(!text_encoder_.is_empty()) << "Text encoder not loaded";
    CHECK(!image.empty()) << "QwenImageEditPlus requires input images";

    const auto& text_encoder_args = context_.get_model_args("text_encoder");
    const auto& processor_args = context_.get_model_args("processor");
    int64_t merge_size = processor_args.mm_image_merge_size();
    CHECK_GT(merge_size, 0)
        << "QwenImageEditPlus processor mm_image_merge_size must be > 0";
    int64_t merge_length = merge_size * merge_size;

    int64_t batch_size = static_cast<int64_t>(prompt.size());
    if (batch_size == 0) {
      for (const auto& img : image) {
        if (img.dim() == 4) {
          batch_size = img.size(0);
          break;
        }
      }
      if (batch_size == 0) {
        batch_size = 1;
      }
    }

    std::vector<std::vector<torch::Tensor>> per_sample_images(
        static_cast<size_t>(batch_size));
    for (const auto& img : image) {
      if (img.dim() == 3) {
        CHECK_EQ(batch_size, 1)
            << "unbatched image input can only be used with batch_size=1";
        per_sample_images[0].push_back(img);
      } else {
        CHECK_EQ(img.dim(), 4)
            << "QwenImageEditPlus image input must be [C,H,W] or [B,C,H,W]";
        CHECK_EQ(img.size(0), batch_size)
            << "batched image input size must match prompt batch size";
        for (int64_t b = 0; b < batch_size; ++b) {
          per_sample_images[static_cast<size_t>(b)].push_back(img[b]);
        }
      }
    }

    std::vector<MMData> mm_data_list;
    mm_data_list.reserve(static_cast<size_t>(batch_size));
    std::vector<torch::Tensor> grid_tensors;
    std::vector<torch::Tensor> first_sample_grids;
    std::vector<MMDataItem> first_sample_mm_items;
    for (int64_t b = 0; b < batch_size; ++b) {
      std::vector<torch::Tensor> vl_images;
      vl_images.reserve(per_sample_images[static_cast<size_t>(b)].size());
      for (auto img : per_sample_images[static_cast<size_t>(b)]) {
        img = img.to(torch::kFloat32);
        float max_val = img.max().item<float>();
        if (max_val <= 1.1f) {
          img = img.clamp(0.0f, 1.0f) * 255.0f;
        } else {
          img = img.clamp(0.0f, 255.0f);
        }
        vl_images.push_back(img);
      }

      std::vector<MMDataItem> mm_items;
      CHECK(vl_image_processor_.process(vl_images, mm_items))
          << "VL image processor failed";
      CHECK(!mm_items.empty()) << "VL image processor produced no image items";
      std::vector<torch::Tensor> sample_grids;
      sample_grids.reserve(mm_items.size());
      for (auto& item : mm_items) {
        sample_grids.push_back(
            item.get<torch::Tensor>("image_grid_thw").value().to(torch::kCPU));
      }
      if (b == 0) {
        first_sample_mm_items = mm_items;
        first_sample_grids = sample_grids;
      } else {
        CHECK_EQ(sample_grids.size(), first_sample_grids.size())
            << "QwenImageEditPlus batch currently requires every sample to "
               "have the same number of input images";
        for (size_t i = 0; i < sample_grids.size(); ++i) {
          CHECK(torch::equal(sample_grids[i], first_sample_grids[i]))
              << "QwenImageEditPlus batch currently requires identical "
                 "image_grid_thw across samples. Different input image sizes "
                 "or aspect ratios should be scheduled in separate batches.";
        }
      }
      for (auto& sample_grid : sample_grids) {
        grid_tensors.push_back(sample_grid);
      }
      MMData mm_data;
      mm_data.set(MMType::IMAGE, mm_items);
      mm_data_list.emplace_back(std::move(mm_data));
    }
    torch::Tensor image_grid_thw = torch::cat(grid_tensors, 0);

    std::string base_img_prompt;
    for (size_t i = 0; i < first_sample_mm_items.size(); ++i) {
      torch::Tensor grid =
          first_sample_mm_items[i].get<torch::Tensor>("image_grid_thw").value();
      int64_t num_image_tokens = grid.prod().item<int64_t>() / merge_length;
      base_img_prompt +=
          "Picture " + std::to_string(i + 1) + ": <|vision_start|>";
      for (int64_t j = 0; j < num_image_tokens; ++j) {
        base_img_prompt += "<|image_pad|>";
      }
      base_img_prompt += "<|vision_end|>\n";
    }

    size_t slot = prompt_template_encode_.find("{}");
    CHECK(slot != std::string::npos)
        << "QwenImageEditPlus prompt template must contain {}";
    std::string prefix_str =
        prompt_template_encode_.substr(0, slot) + base_img_prompt;
    std::string suffix_str =
        prompt_template_encode_.substr(slot + std::string("{}").size());

    std::vector<int32_t> prefix_tokens;
    CHECK(tokenizer_->encode(prefix_str, &prefix_tokens, false));
    std::vector<int32_t> suffix_tokens;
    CHECK(tokenizer_->encode(suffix_str, &suffix_tokens, false));
    int64_t suffix_len = suffix_tokens.size();

    CHECK(prompt.empty() || static_cast<int64_t>(prompt.size()) == batch_size)
        << "prompt batch size must match image batch size";
    std::vector<std::vector<int32_t>> batch_tokens;
    batch_tokens.reserve(batch_size);
    int64_t prefix_len = prompt_template_encode_start_idx_;
    CHECK_GE(prefix_len, 0)
        << "QwenImageEditPlus prompt_template_encode_start_idx_ must be >= 0";

    for (const auto& each_prompt : prompt) {
      std::vector<int32_t> tokens;
      if (!tokenizer_->encode(each_prompt, &tokens, false)) {
        tokens.clear();
      }

      if (static_cast<int64_t>(tokens.size()) > max_sequence_length) {
        LOG(WARNING) << "Input truncated from " << tokens.size() << " to "
                     << max_sequence_length << " tokens";
        tokens.resize(max_sequence_length);
      }
      batch_tokens.push_back(std::move(tokens));
    }

    if (batch_tokens.empty()) {
      batch_tokens.resize(static_cast<size_t>(batch_size));
    }

    int32_t pad_token_id = text_encoder_args.pad_token_id();
    if (pad_token_id == 0) {
      auto pad_id = tokenizer_->token_to_id("<|endoftext|>");
      pad_token_id = pad_id.value_or(151643);
    }

    for (auto& tokens : batch_tokens) {
      int64_t pad_len =
          max_sequence_length - static_cast<int64_t>(tokens.size());
      if (pad_len > 0) {
        tokens.insert(tokens.end(), pad_len, pad_token_id);
      }
    }

    int64_t prefix_full_len = prefix_tokens.size();
    CHECK_LT(prefix_len, prefix_full_len)
        << "QwenImageEditPlus prompt_template_encode_start_idx_ must be "
           "inside the encoded prefix. prefix_len="
        << prefix_len << ", prefix_full_len=" << prefix_full_len;
    int64_t total_seq_len = prefix_full_len + max_sequence_length + suffix_len;
    std::vector<int32_t> input_ids_flat;
    std::vector<int32_t> attention_mask_flat;
    input_ids_flat.reserve(batch_size * total_seq_len);
    attention_mask_flat.reserve(batch_size * total_seq_len);

    for (int64_t i = 0; i < batch_size; ++i) {
      input_ids_flat.insert(
          input_ids_flat.end(), prefix_tokens.begin(), prefix_tokens.end());
      attention_mask_flat.insert(
          attention_mask_flat.end(), static_cast<size_t>(prefix_full_len), 1);

      input_ids_flat.insert(
          input_ids_flat.end(), batch_tokens[i].begin(), batch_tokens[i].end());
      int64_t non_pad_count = 0;
      for (const auto& token : batch_tokens[i]) {
        if (token != pad_token_id) {
          non_pad_count++;
        } else {
          break;
        }
      }
      for (int64_t j = 0; j < max_sequence_length; ++j) {
        attention_mask_flat.push_back(j < non_pad_count ? 1 : 0);
      }

      input_ids_flat.insert(
          input_ids_flat.end(), suffix_tokens.begin(), suffix_tokens.end());
      attention_mask_flat.insert(attention_mask_flat.end(), suffix_len, 1);
    }

    torch::Tensor input_ids =
        torch::tensor(input_ids_flat, torch::dtype(torch::kLong))
            .view({batch_size, total_seq_len})
            .to(device_);
    torch::Tensor attention_mask =
        torch::tensor(attention_mask_flat, torch::dtype(torch::kLong))
            .view({batch_size, total_seq_len})
            .to(device_);

    torch::Tensor positions_2d = build_qwen2_5_vl_mrope_positions(
        input_ids, attention_mask, image_grid_thw);
    torch::Tensor valid_token_mask =
        attention_mask.view({-1}).to(torch::kBool).contiguous();
    torch::Tensor valid_token_indices =
        torch::nonzero(valid_token_mask).squeeze(1).to(device_);
    torch::Tensor tokens_flat =
        input_ids.view({-1}).index_select(0, valid_token_indices);
    torch::Tensor positions_packed =
        positions_2d.index_select(1, valid_token_indices);

    MMBatchData mm_batch(std::move(mm_data_list));

    std::vector<KVCache> kv_caches(text_encoder_args.n_layers());
    ModelInputParams input_params = build_qwen_vl_input_params(
        tokens_flat, attention_mask, std::move(mm_batch));
#if defined(USE_DCU)
    input_params.graph.use_dense_flash_attention = true;
#endif
    auto model_output = text_encoder_->forward(
        tokens_flat, positions_packed, kv_caches, input_params);
    torch::Tensor hidden_states_flat = model_output.hidden_states;
    int64_t hidden_size = hidden_states_flat.size(-1);
    torch::Tensor padded_hidden_states_flat =
        torch::zeros({batch_size * total_seq_len, hidden_size},
                     hidden_states_flat.options());
    padded_hidden_states_flat.index_copy_(
        0, valid_token_indices, hidden_states_flat);
    torch::Tensor hidden_states_last = padded_hidden_states_flat.view(
        {batch_size, total_seq_len, hidden_size});

    int64_t end = total_seq_len - suffix_len;
    CHECK_LT(prefix_len, end)
        << "QwenImageEditPlus prompt embedding slice is invalid: prefix_len="
        << prefix_len << ", end=" << end << ", total_seq_len=" << total_seq_len
        << ", suffix_len=" << suffix_len;
    torch::Tensor prompt_embeds =
        hidden_states_last.slice(1, prefix_len, end).to(options);
    torch::Tensor prompt_embeds_mask =
        attention_mask.slice(1, prefix_len, end).to(device_);
    return std::make_pair(prompt_embeds, prompt_embeds_mask);
#else
    (void)prompt;
    (void)image;
    (void)options;
    (void)max_sequence_length;
    return std::make_pair(torch::Tensor(), torch::Tensor());
#endif
  }

  void _encode_prompt(const std::vector<torch::Tensor>& image,
                      const std::vector<std::string>& prompt,
                      torch::Tensor& prompt_embeds,
                      torch::Tensor& prompt_embeds_mask,
                      torch::TensorOptions& options,
                      int64_t num_images_per_prompt = 1,
                      int64_t max_sequence_length = 1024) {
    int64_t batch_size =
        prompt_embeds.defined() ? prompt_embeds.size(0) : prompt.size();
    if (batch_size == 0) {
      batch_size = 1;
    }
    if (!prompt_embeds.defined()) {
      std::tie(prompt_embeds, prompt_embeds_mask) =
          _get_qwen_prompt_embeds(prompt, image, options, max_sequence_length);
    }

    CHECK(prompt_embeds.defined())
        << "currently, the prompt input is not supported for qwen image, "
        << "expected a valid prompt_embeds input, but got empty tensor ";

    auto seq_len = prompt_embeds.size(1);
    prompt_embeds = prompt_embeds.repeat({1, num_images_per_prompt, 1});
    prompt_embeds =
        prompt_embeds.view({batch_size * num_images_per_prompt, seq_len, -1});
    if (prompt_embeds_mask.defined()) {
      prompt_embeds_mask =
          prompt_embeds_mask.repeat({1, num_images_per_prompt, 1});
      prompt_embeds_mask = prompt_embeds_mask.view(
          {batch_size * num_images_per_prompt, seq_len});
    } else {
      prompt_embeds_mask = torch::ones(
          {prompt_embeds.size(0), prompt_embeds.size(1)},
          torch::TensorOptions().device(device_).dtype(torch::kLong));
    }
  }

  torch::Tensor _retrieve_latents(const AutoencoderKLOutput& encoder_output,
                                  int64_t seed = 42,
                                  const std::string& sample_mode = "sample") {
    if (sample_mode == "sample") {
      return encoder_output.latent_dist.sample(seed);
    } else if (sample_mode == "argmax") {
      return encoder_output.latent_dist.mode();
    } else {
      CHECK(false)
          << "sample_mode is expected to be 'sample' or 'argmax', but get: "
          << sample_mode;
      return torch::Tensor();
    }
  }

  torch::Tensor _pack_latents(torch::Tensor latents,
                              int64_t batch_size,
                              int64_t num_channels_latents,
                              int64_t height,
                              int64_t width) {
    latents = latents.view(
        {batch_size, num_channels_latents, height / 2, 2, width / 2, 2});
    latents = latents.permute({0, 2, 4, 1, 3, 5});
    latents = latents.reshape(
        {batch_size, (height / 2) * (width / 2), num_channels_latents * 4});

    return latents;
  }

  torch::Tensor _unpack_latents(torch::Tensor latents,
                                int64_t height,
                                int64_t width,
                                int64_t vae_scale_factor) {
    auto sizes = latents.sizes();
    int64_t batch_size = sizes[0];
    int64_t num_patches = sizes[1];
    int64_t channels = sizes[2];

    height = 2 * (height / (vae_scale_factor * 2));
    width = 2 * (width / (vae_scale_factor * 2));

    latents =
        latents.view({batch_size, height / 2, width / 2, channels / 4, 2, 2});
    latents = latents.permute({0, 3, 1, 4, 2, 5});
    latents =
        latents.reshape({batch_size, channels / (2 * 2), 1, height, width});

    return latents;
  }

  torch::Tensor _encode_vae_image(torch::Tensor image,
                                  int64_t seed,
                                  torch::Device device) {
    auto image_latents = _retrieve_latents(vae_->encode(image), seed, "argmax");
    auto latents_mean =
        torch::tensor(vae_model_args_.latents_mean(), torch::kDouble);
    latents_mean = latents_mean.view({1, latent_channels_, 1, 1, 1})
                       .to(device, image_latents.dtype());
    auto latents_std =
        torch::tensor(vae_model_args_.latents_std(), torch::kDouble);
    latents_std = latents_std.view({1, latent_channels_, 1, 1, 1})
                      .to(device, image_latents.dtype());
    image_latents = (image_latents - latents_mean) / latents_std;
    return image_latents;
  }

  std::pair<torch::Tensor, torch::Tensor> _prepare_latents(
      const std::vector<torch::Tensor>& images,
      int64_t request_batch_size,
      int64_t num_images_per_prompt,
      int64_t num_channels_latents,
      int64_t height,
      int64_t width,
      torch::TensorOptions& options,
      int64_t seed,
      torch::Tensor latents = torch::Tensor()) {
    height = 2 * (height / (vae_scale_factor_ * 2));
    width = 2 * (width / (vae_scale_factor_ * 2));
    const int64_t total_batch_size = request_batch_size * num_images_per_prompt;

    std::vector<int64_t> shape = {
        total_batch_size, 1, num_channels_latents, height, width};

    torch::Tensor image_latents;
    if (!images.empty()) {
      std::vector<torch::Tensor> all_image_latents;
      for (const auto& image : images) {
        auto current_image = image.to(options);
        torch::Tensor current_image_latents;

        if (current_image.size(1) != latent_channels_) {
          current_image_latents =
              _encode_vae_image(current_image, seed, device_);
        } else {
          current_image_latents = current_image;
        }

        current_image_latents = torch::cat({current_image_latents}, 0);
        CHECK_EQ(current_image_latents.size(0), request_batch_size)
            << "image latent batch size must match request batch size";
        if (num_images_per_prompt > 1) {
          auto sizes = current_image_latents.sizes().vec();
          sizes[0] = total_batch_size;
          current_image_latents =
              current_image_latents.unsqueeze(1)
                  .repeat({1, num_images_per_prompt, 1, 1, 1, 1})
                  .view(sizes);
        }
        int64_t image_latent_height = current_image_latents.size(3);
        int64_t image_latent_width = current_image_latents.size(4);

        current_image_latents = _pack_latents(current_image_latents,
                                              total_batch_size,
                                              num_channels_latents,
                                              image_latent_height,
                                              image_latent_width);
        all_image_latents.emplace_back(current_image_latents);
      }

      image_latents = torch::cat(all_image_latents, 1);
    }

    if (!latents.defined()) {
      latents = xllm::dit::randn_tensor(shape, seed, options);
      latents = _pack_latents(
          latents, total_batch_size, num_channels_latents, height, width);
    } else {
      latents = latents.to(options);
    }
    return std::make_pair(latents, image_latents);
  }

  DiTForwardOutput forward(const DiTForwardInput& input) {
    torch::NoGradGuard no_grad;
    const auto& generation_params = input.generation_params;
    auto height = generation_params.height;
    auto width = generation_params.width;
    auto true_cfg_scale = generation_params.true_cfg_scale;
    auto num_inference_steps = generation_params.num_inference_steps;
    DiTCache::get_instance().set_infer_steps(num_inference_steps);
    DiTCache::get_instance().set_num_blocks(num_layers_);
    auto max_sequence_length = generation_params.max_sequence_length;
    auto seed = generation_params.seed >= 0 ? generation_params.seed : 42;

    auto prompts = input.prompts;
    auto negative_prompts = input.negative_prompts;
    auto latents = input.latents;
    if (latents.defined()) {
      latents = latents.to(options_.device(), dtype_);
    }

    auto prompt_embeds = input.prompt_embeds;
    if (prompt_embeds.defined()) {
      prompt_embeds = prompt_embeds.to(options_.device(), dtype_);
    }
    torch::Tensor prompt_embeds_mask;

    auto negative_prompt_embeds = input.negative_prompt_embeds;
    if (negative_prompt_embeds.defined()) {
      negative_prompt_embeds =
          negative_prompt_embeds.to(options_.device(), dtype_);
    }
    torch::Tensor negative_prompt_embeds_mask;

    std::vector<torch::Tensor> raw_image_inputs;
    if (!input.images_list.empty()) {
      raw_image_inputs = input.images_list;
    } else if (input.images.defined()) {
      raw_image_inputs.emplace_back(input.images);
    } else {
      LOG(FATAL) << "QwenImageEditPlus pipeline expected to have image inputs "
                    "in images or images_list. batch_size="
                 << input.batch_size << ", prompts=" << input.prompts.size()
                 << ", prompt_embeds_defined=" << prompt_embeds.defined()
                 << ", images_defined=" << input.images.defined()
                 << ", images_list_size=" << input.images_list.size();
    }

    std::vector<torch::Tensor> image_list;
    image_list.reserve(raw_image_inputs.size());

    for (const auto& images : raw_image_inputs) {
      auto img = images.to(options_.device(), dtype_);
      if (img.dim() != 4) {
        LOG(ERROR)
            << "image inputs are expected to be a 4 dim tensor, but got: "
            << img.dim() << "d tensor";
        continue;
      }
      image_list.emplace_back(img);
    }

    if (image_list.empty()) {
      LOG(FATAL) << "No valid images found in images or images_list. ";
    }

    int64_t batch_size = input.batch_size;
    if (batch_size == 0 && !prompts.empty()) {
      batch_size = static_cast<int64_t>(prompts.size());
    }
    if (batch_size == 0 && prompt_embeds.defined()) {
      batch_size = prompt_embeds.size(0);
    }
    if (batch_size == 0) {
      batch_size = image_list[0].size(0);
    }
    CHECK_GT(batch_size, 0) << "QwenImageEditPlus batch_size must be > 0";
    for (const auto& image : image_list) {
      CHECK_EQ(image.size(0), batch_size)
          << "all image inputs must have the same batch size";
    }

    double height_size = static_cast<double>(image_list[0].size(2));
    double width_size = static_cast<double>(image_list[0].size(3));
    int64_t num_images_per_prompt = generation_params.num_images_per_prompt;
    CHECK_GT(num_images_per_prompt, 0)
        << "QwenImageEditPlus num_images_per_prompt must be > 0";

    double aspect_ratio = width_size / height_size;
    auto [calculated_width, calculated_height] =
        xllm::dit::calculate_dimensions(1024 * 1024, aspect_ratio);

    height = (height == 0) ? calculated_height : height;
    width = (width == 0) ? calculated_width : width;

    int64_t multiple_of = vae_scale_factor_ * 2;
    width = (width / multiple_of) * multiple_of;
    height = (height / multiple_of) * multiple_of;

    current_timestep_ = torch::Tensor();

    std::vector<torch::Tensor> condition_images;
    std::vector<torch::Tensor> vae_images;
    std::vector<std::pair<int64_t, int64_t>> vae_image_sizes;
    if (!image_list.empty() && image_list[0].size(1) != latent_channels_) {
      for (size_t i = 0; i < image_list.size(); i++) {
        aspect_ratio =
            static_cast<double>(image_list[i].size(3)) / image_list[i].size(2);
        auto [condition_width, condition_height] =
            xllm::dit::calculate_dimensions(CONDITION_IMAGE_SIZE, aspect_ratio);
        int64_t vae_target_area =
            get_qwen_image_edit_vae_target_area(height, width);
        auto [vae_width, vae_height] =
            xllm::dit::calculate_dimensions(vae_target_area, aspect_ratio);
        CHECK_GT(vae_width, 0) << "QwenImageEditPlus vae_width must be > 0";
        CHECK_GT(vae_height, 0) << "QwenImageEditPlus vae_height must be > 0";
        vae_image_sizes.push_back({vae_width, vae_height});
        auto img = image_list[i];
        auto condition_img = vae_image_processor_->resize(
            img,
            {condition_height, condition_width},
            /*resample=*/3,  // BICUBIC (approximate LANCZOS)
            /*antialias=*/true);
        auto vae_img = vae_image_processor_
                           ->preprocess(img,
                                        vae_height,
                                        vae_width,
                                        /*resize_mode=*/"default")
                           .unsqueeze(2);

        condition_images.push_back(condition_img);
        vae_images.push_back(vae_img);
      }
    }

    bool has_neg_prompt =
        negative_prompts.size() > 0 || negative_prompt_embeds.defined();

    bool do_true_cfg = (true_cfg_scale > 1.0) && has_neg_prompt;
    // inplace update prompt_embeds and prompt_embeds_mask
    _encode_prompt(condition_images,
                   prompts,
                   prompt_embeds,
                   prompt_embeds_mask,
                   options_,
                   num_images_per_prompt,
                   max_sequence_length);

    if (do_true_cfg) {
      // inplace update negative_prompt_embeds and negative_prompt_embeds_mask
      _encode_prompt(condition_images,
                     negative_prompts,
                     negative_prompt_embeds,
                     negative_prompt_embeds_mask,
                     options_,
                     num_images_per_prompt,
                     max_sequence_length);
    }

    int64_t num_channels_latents = in_channels_ / 4;
    torch::Tensor final_latents;
    torch::Tensor image_latents;

    std::tie(final_latents, image_latents) =
        _prepare_latents(vae_images,
                         batch_size,
                         num_images_per_prompt,
                         num_channels_latents,
                         height,
                         width,
                         options_,
                         seed,
                         latents);

    std::vector<std::vector<int64_t>> main_shape = {
        {1, height / vae_scale_factor_ / 2, width / vae_scale_factor_ / 2}};

    for (const auto& [vae_width, vae_height] : vae_image_sizes) {
      main_shape.push_back({1,
                            vae_height / vae_scale_factor_ / 2,
                            vae_width / vae_scale_factor_ / 2});
    }

    std::vector<float> new_sigmas;
    double start = 1.0;
    double end = 1.0 / static_cast<double>(num_inference_steps);
    for (int64_t i = 0; i < num_inference_steps; ++i) {
      double v = start + (end - start) * static_cast<double>(i) /
                             (num_inference_steps - 1);
      new_sigmas.push_back(static_cast<float>(v));
    }

    int64_t image_seq_len = final_latents.size(1);
    float mu = xllm::dit::calculate_shift(image_seq_len,
                                          scheduler_->base_image_seq_len(),
                                          scheduler_->max_image_seq_len(),
                                          scheduler_->base_shift(),
                                          scheduler_->max_shift());
    auto timesteps_result = xllm::dit::retrieve_timesteps(
        scheduler_, num_inference_steps, device_, new_sigmas, mu);
    auto timesteps = std::get<0>(timesteps_result);
    num_timesteps_ = timesteps.size(0);
    torch::Tensor txt_seq_lens;
    if (prompt_embeds_mask.defined()) {
      txt_seq_lens = prompt_embeds_mask.sum(1);
    }
    torch::Tensor negative_txt_seq_lens;
    if (do_true_cfg && negative_prompt_embeds_mask.defined()) {
      negative_txt_seq_lens = negative_prompt_embeds_mask.sum(1);
    }
    scheduler_->set_begin_index(0);

    auto get_image_rotary_emb =
        [&](int64_t text_seq_len) -> std::tuple<torch::Tensor, torch::Tensor> {
      if (use_layer3d_rope_) {
        auto image_rotary_emb = pos_embed_3d_rope_->forward(
            main_shape, text_seq_len, prompt_embeds.device());
        return std::make_tuple(image_rotary_emb.first, image_rotary_emb.second);
      }
      return pos_embed_->forward(main_shape,
                                 text_seq_len,
                                 prompt_embeds.device(),
                                 /*max_txt_seq_len=*/std::nullopt);
    };
    std::tuple<torch::Tensor, torch::Tensor> image_rotary_emb_pos =
        get_image_rotary_emb(prompt_embeds.size(1));
    std::tuple<torch::Tensor, torch::Tensor> image_rotary_emb_neg =
        do_true_cfg ? get_image_rotary_emb(negative_prompt_embeds.size(1))
                    : image_rotary_emb_pos;

    for (int64_t i = 0; i < timesteps.size(0); ++i) {
      auto t = timesteps[i];
      current_timestep_ = t;

      auto latent_model_input = final_latents;
      if (image_latents.defined()) {
        latent_model_input = torch::cat({final_latents, image_latents}, 1);
      }

      auto timestep_expanded =
          t.expand({final_latents.size(0)}).to(final_latents.dtype());

      torch::Tensor noise_pred;
      torch::Tensor neg_noise_pred;
      torch::Tensor pos_neg_noise_preds;
      if (::xllm::ParallelConfig::get_instance().cfg_size() == 2 &&
          do_true_cfg) {
        auto rank = parallel_args_.dit_cfg_group_->rank();
        if (rank == 0) {
          noise_pred = transformer_->forward(latent_model_input,
                                             prompt_embeds,
                                             prompt_embeds_mask,
                                             timestep_expanded / 1000.0,
                                             main_shape,
                                             txt_seq_lens,
                                             image_rotary_emb_pos,
                                             /*use_cfg=*/false,
                                             /*step_index=*/i);
          noise_pred = noise_pred.slice(1, 0, final_latents.size(1));
          pos_neg_noise_preds =
              xllm::parallel_state::gather(noise_pred,
                                           parallel_args_.dit_cfg_group_,
                                           /*dim=*/0);
        } else {
          neg_noise_pred = transformer_->forward(latent_model_input,
                                                 negative_prompt_embeds,
                                                 negative_prompt_embeds_mask,
                                                 timestep_expanded / 1000.0,
                                                 main_shape,
                                                 negative_txt_seq_lens,
                                                 image_rotary_emb_neg,
                                                 /*use_cfg=*/true,
                                                 /*step_index=*/i);

          neg_noise_pred = neg_noise_pred.slice(1, 0, final_latents.size(1));
          pos_neg_noise_preds =
              xllm::parallel_state::gather(neg_noise_pred,
                                           parallel_args_.dit_cfg_group_,
                                           /*dim=*/0);
        }
        auto noise_preds = torch::chunk(pos_neg_noise_preds, 2, 0);
        auto comb_pred =
            noise_preds[1] + true_cfg_scale * (noise_preds[0] - noise_preds[1]);
        auto cond_norm = torch::norm(noise_preds[0], 2, -1, true);
        auto noise_norm = torch::norm(comb_pred, 2, -1, true);
        noise_pred = comb_pred * (cond_norm / noise_norm);

      } else {
        noise_pred = transformer_->forward(latent_model_input,
                                           prompt_embeds,
                                           prompt_embeds_mask,
                                           timestep_expanded / 1000.0,
                                           main_shape,
                                           txt_seq_lens,
                                           image_rotary_emb_pos,
                                           /*use_cfg=*/false,
                                           /*step_index=*/i);
        noise_pred = noise_pred.slice(1, 0, final_latents.size(1));
        if (do_true_cfg) {
          neg_noise_pred = transformer_->forward(latent_model_input,
                                                 negative_prompt_embeds,
                                                 negative_prompt_embeds_mask,
                                                 timestep_expanded / 1000.0,
                                                 main_shape,
                                                 negative_txt_seq_lens,
                                                 image_rotary_emb_neg,
                                                 /*use_cfg=*/true,
                                                 /*step_index=*/i);

          neg_noise_pred = neg_noise_pred.slice(1, 0, final_latents.size(1));

          auto comb_pred =
              neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred);
          auto cond_norm = torch::norm(noise_pred, 2, -1, true);
          auto noise_norm = torch::norm(comb_pred, 2, -1, true);
          noise_pred = comb_pred * (cond_norm / noise_norm);
        }
      }

      auto latents_dtype = final_latents.dtype();
      final_latents = scheduler_->step(noise_pred, t, final_latents);
      if (final_latents.dtype() != latents_dtype) {
        final_latents = final_latents.to(latents_dtype);
      }
    }
    current_timestep_ = torch::Tensor();

    torch::Tensor output_image;

    auto unpacked_latents =
        _unpack_latents(final_latents, height, width, vae_scale_factor_)
            .to(dtype_);
    auto latents_mean =
        torch::tensor(vae_model_args_.latents_mean(), torch::kDouble);
    latents_mean = latents_mean.view({1, latent_channels_, 1, 1, 1})
                       .to(device_, image_latents.dtype());
    auto latents_std =
        torch::tensor(vae_model_args_.latents_std(), torch::kDouble);
    latents_std = 1.0 / latents_std.view({1, latent_channels_, 1, 1, 1})
                            .to(device_, image_latents.dtype());

    unpacked_latents = unpacked_latents / latents_std + latents_mean;
    output_image = vae_->decode(unpacked_latents).sample.squeeze(2);
    output_image = vae_image_processor_->postprocess(output_image);
    auto output_chunks = torch::chunk(output_image, batch_size, /*dim=*/0);
    DiTForwardOutput out;
    out.tensors = std::move(output_chunks);
    return out;
  }

  void load_model(std::unique_ptr<DiTModelLoader> loader) {
    LOG(INFO) << "QwenImageEditPlusPipeline loading model from"
              << loader->model_root_path();
    auto transformer_loader = loader->take_component_loader("transformer");
    auto vae_loader = loader->take_component_loader("vae");
#if defined(USE_DCU)
    auto clip_loader = loader->take_component_loader("text_encoder");
    auto tokenizer_loader = loader->take_component_loader("tokenizer");
#endif
    LOG(INFO) << " QwenImageEditplus model components loaded, start to load "
                 "weights to sub models";

    LOG(INFO) << "QwenImageEditPlusPipeline loading vae weights";
    vae_->load_model(std::move(vae_loader));
    LOG(INFO) << "QwenImageEditPlusPipeline moving vae to device";
    vae_->to(options_.device(), dtype_);
    LOG(INFO) << "QwenImageEditPlusPipeline vae loaded";

    LOG(INFO) << "QwenImageEditPlusPipeline loading transformer weights";
    transformer_->load_model(std::move(transformer_loader));

    LOG(INFO) << "QwenImageEditPlusPipeline moving transformer to device";
    transformer_->to(options_.device());
    LOG(INFO) << "QwenImageEditPlusPipeline transformer loaded";

#if defined(USE_DCU)
    LOG(INFO) << "QwenImageEditPlusPipeline loading text_encoder weights";
    text_encoder_->load_model(std::move(clip_loader));
    LOG(INFO) << "QwenImageEditPlusPipeline moving text_encoder to device";
    text_encoder_->to(options_.device(), dtype_);
    LOG(INFO) << "QwenImageEditPlusPipeline text_encoder loaded";

    tokenizer_ = tokenizer_loader->tokenizer();
    LOG(INFO) << "QwenImageEditPlusPipeline tokenizer loaded";
#endif
  }

 private:
  DiTModelContext context_;
  // members from former QwenImagePipelineBaseImpl base class
  torch::Device device_ = torch::kCPU;
  torch::ScalarType dtype_;
  torch::TensorOptions options_;
  AutoencoderKLQwenImage vae_{nullptr};
  xllm::VAEImageProcessor vae_image_processor_{nullptr};
  std::unique_ptr<Qwen2VLImageProcessor> qwen_image_processor_{nullptr};
  QwenImageTransformer2DModel transformer_{nullptr};
#if defined(USE_DCU)
  QwenImageEditTextEncoder text_encoder_{nullptr};
  std::unique_ptr<ProcessGroup> vlm_tp_group_;
  Qwen2VLImageProcessor vl_image_processor_;
#endif
  std::unique_ptr<Tokenizer> qwen_tokenizer_;
  std::unique_ptr<Tokenizer> tokenizer_;
  xllm::FlowMatchEulerDiscreteScheduler scheduler_{nullptr};

  int64_t vae_scale_factor_;
  int64_t latent_channels_;
  int64_t tokenizer_max_length_;
  int64_t prompt_template_encode_start_idx_;
  int64_t default_sample_size_;
  int64_t in_channels_;
  int64_t num_timesteps_;
  int64_t num_layers_;
  const ParallelArgs parallel_args_;
  torch::Tensor current_timestep_;
  std::string prompt_template_encode_;
  const ModelArgs& vae_model_args_;
  bool use_layer3d_rope_;
  QwenEmbedRope pos_embed_{nullptr};
  QwenEmbedLayer3DRope pos_embed_3d_rope_{nullptr};
};

REGISTER_MODEL_ARGS(Qwen2Tokenizer, [&] {});
TORCH_MODULE(QwenImageEditPlusPipeline);

REGISTER_DIT_MODEL(QwenImageEditPlusPipeline, QwenImageEditPlusPipeline);
}  // namespace xllm
