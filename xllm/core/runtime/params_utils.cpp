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

#include "runtime/params_utils.h"

#include <torch/torch.h>

#include "common/global_flags.h"
#include "common/macros.h"
#include "common/metrics.h"
#include "core/framework/config/eplb_config.h"
#include "core/framework/multimodal/mm_batch_data.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {
void proto_to_forward_output(const proto::ForwardOutput& pb_output,
                             RawForwardOutput& raw_forward_output) {
  Timer timer;
  size_t seq_nums = pb_output.outputs().size();
  raw_forward_output.outputs.reserve(seq_nums);
  size_t expert_load_data_size = pb_output.expert_load_data().size();
  raw_forward_output.expert_load_data.reserve(expert_load_data_size);
  raw_forward_output.expert_load_data.assign(
      pb_output.expert_load_data().begin(), pb_output.expert_load_data().end());
  raw_forward_output.src_seq_idxes.reserve(pb_output.src_seq_idxes().size());
  raw_forward_output.src_seq_idxes.assign(pb_output.src_seq_idxes().begin(),
                                          pb_output.src_seq_idxes().end());
  raw_forward_output.out_tokens.reserve(pb_output.out_tokens().size());
  raw_forward_output.out_tokens.assign(pb_output.out_tokens().begin(),
                                       pb_output.out_tokens().end());
  raw_forward_output.out_logprobs.reserve(pb_output.out_logprobs().size());
  raw_forward_output.out_logprobs.assign(pb_output.out_logprobs().begin(),
                                         pb_output.out_logprobs().end());
  raw_forward_output.prepared_layer_id = pb_output.prepared_layer_id();
  for (size_t i = 0; i < seq_nums; ++i) {
    proto::SquenceOutput pb_seq_out = pb_output.outputs()[i];
    RawSampleOutput s;
    size_t token_nums = pb_seq_out.tokens().size();
    s.tokens.reserve(token_nums);
    for (size_t j = 0; j < token_nums; ++j) {
      RawToken t;
      t.id = pb_seq_out.tokens()[j].id();
      switch (pb_seq_out.tokens()[j].lp_case()) {
        case proto::Token::kEmpty:
          break;
        case proto::Token::kLogprob:
          t.logprob = pb_seq_out.tokens()[j].logprob();
          break;
        default:
          break;
      }
      t.top_tokens.assign(pb_seq_out.tokens()[j].top_tokens().begin(),
                          pb_seq_out.tokens()[j].top_tokens().end());
      t.top_logprobs.assign(pb_seq_out.tokens()[j].top_logprobs().begin(),
                            pb_seq_out.tokens()[j].top_logprobs().end());
      t.embeddings.assign(pb_seq_out.tokens()[j].embeddings().vals().begin(),
                          pb_seq_out.tokens()[j].embeddings().vals().end());
      s.tokens.emplace_back(t);
    }
    s.mm_embeddings.reserve(pb_seq_out.mm_embeddings().tensors_size());
    for (const auto& pb_tensor : pb_seq_out.mm_embeddings().tensors()) {
      s.mm_embeddings.emplace_back(util::proto_to_torch(pb_tensor));
    }
    raw_forward_output.outputs.emplace_back(s);
  }
  proto_to_dit_forward_output(pb_output.dit_forward_output(),
                              raw_forward_output.dit_forward_output);
  COUNTER_ADD(proto_latency_seconds_proto2o, timer.elapsed_seconds());
}

void forward_output_to_proto(
    const torch::Tensor& next_tokens,
    const torch::Tensor& logprobs,
    const torch::Tensor& top_tokens,
    const torch::Tensor& top_logprobs,
    const torch::Tensor& embeddings,
    const std::vector<std::vector<torch::Tensor>>& mm_embeddings,
    const torch::Tensor& expert_load_data,
    int32_t prepared_layer_id,
    const torch::Tensor& src_seq_idxes,
    const torch::Tensor& out_tokens,
    const torch::Tensor& out_logprobs,
    const std::vector<torch::Tensor>& dit_images,
    const std::vector<std::string>& dit_text_output,
    proto::ForwardOutput* pb_forward_output) {
  Timer timer;
  // LLM decode fills next_tokens; DiT text diffusion (e.g. Cola-DLM) may leave
  // it undefined and only populate dit_text_output. Guard before
  // .size()/.dim().
  int32_t num_seqs =
      next_tokens.defined() ? static_cast<int32_t>(next_tokens.size(0)) : 0;
  if (embeddings.defined() && embeddings.numel() > 0) {
    num_seqs = std::max(num_seqs, static_cast<int32_t>(embeddings.size(0)));
  }
  if (!mm_embeddings.empty()) {
    num_seqs = std::max(num_seqs, static_cast<int32_t>(mm_embeddings.size()));
  }
  pb_forward_output->mutable_outputs()->Reserve(num_seqs);
  for (int32_t output_idx = 0; output_idx < num_seqs; ++output_idx) {
    if (next_tokens.defined() && next_tokens.dim() == 2) {
      const auto curr_idx = output_idx;
      const auto curr_next_tokens = next_tokens[curr_idx];
      const auto curr_logprobs =
          logprobs.defined() ? logprobs[curr_idx] : logprobs;
      const auto curr_top_tokens =
          top_tokens.defined() ? top_tokens[curr_idx] : top_tokens;
      const auto curr_top_logprobs =
          top_logprobs.defined() ? top_logprobs[curr_idx] : top_logprobs;
      const auto curr_embeddings =
          embeddings.defined() ? embeddings[curr_idx] : embeddings;

      int32_t num_tokens = curr_next_tokens.size(0);
      std::vector<Token> tokens;
      tokens.reserve(num_tokens);
      for (int32_t i = 0; i < num_tokens; ++i) {
        const auto token = build_token(i,
                                       curr_next_tokens,
                                       curr_logprobs,
                                       curr_top_tokens,
                                       curr_top_logprobs);
        if (token.id == -1) {
          break;
        }
        tokens.push_back(token);
      }
      num_tokens = tokens.size();
      proto::SquenceOutput pb_seq_out;
      pb_seq_out.mutable_tokens()->Reserve(num_tokens);
      for (int32_t i = 0; i < num_tokens; ++i) {
        const auto& token = tokens[i];
        proto::Token pb_token;
        pb_token.set_id(token.id);
        if (token.logprob.has_value()) {
          pb_token.set_logprob(token.logprob.value());
        } else {
          pb_token.set_empty(true);
        }
        pb_token.mutable_top_tokens()->Reserve(token.top_tokens.size());
        for (auto it = token.top_tokens.cbegin(); it != token.top_tokens.cend();
             ++it) {
          pb_token.add_top_tokens(*it);
        }
        pb_token.mutable_top_logprobs()->Reserve(token.top_logprobs.size());
        for (auto it = token.top_logprobs.cbegin();
             it != token.top_logprobs.cend();
             ++it) {
          pb_token.add_top_logprobs(*it);
        }
        const auto token_embeddings =
            curr_embeddings.defined() ? curr_embeddings[i] : curr_embeddings;
        if (token_embeddings.defined()) {
          Slice<float> embedding_slice = {
              token_embeddings.data_ptr<float>(),
              static_cast<size_t>(token_embeddings.size(0))};
          ADD_VECTOR_TO_PROTO(pb_token.mutable_embeddings()->mutable_vals(),
                              embedding_slice);
        }
        *pb_seq_out.mutable_tokens()->Add() = pb_token;
      }
      *pb_forward_output->mutable_outputs()->Add() = pb_seq_out;
    } else {
      proto::SquenceOutput pb_seq_out;
      pb_seq_out.mutable_tokens()->Reserve(1);
      proto::Token pb_token;

      // Handle case where next_tokens might be empty but embeddings have data
      if (next_tokens.defined() && next_tokens.numel() > 0) {
        const auto token = build_token(
            output_idx, next_tokens, logprobs, top_tokens, top_logprobs);
        pb_token.set_id(token.id);
        if (token.logprob.has_value()) {
          pb_token.set_logprob(token.logprob.value());
        } else {
          pb_token.set_empty(true);
        }
        pb_token.mutable_top_tokens()->Reserve(token.top_tokens.size());
        for (auto it = token.top_tokens.cbegin(); it != token.top_tokens.cend();
             ++it) {
          pb_token.add_top_tokens(*it);
        }
        pb_token.mutable_top_logprobs()->Reserve(token.top_logprobs.size());
        for (auto it = token.top_logprobs.cbegin();
             it != token.top_logprobs.cend();
             ++it) {
          pb_token.add_top_logprobs(*it);
        }
      } else {
        // For embedding-only requests, set a placeholder token ID
        pb_token.set_id(-1);
        pb_token.set_empty(true);
      }

      const auto token_embeddings =
          embeddings.defined() ? embeddings[output_idx] : embeddings;
      if (token_embeddings.defined()) {
        Slice<float> embedding_slice = {
            token_embeddings.data_ptr<float>(),
            static_cast<size_t>(token_embeddings.size(0))};
        ADD_VECTOR_TO_PROTO(pb_token.mutable_embeddings()->mutable_vals(),
                            embedding_slice);
      }
      *pb_seq_out.mutable_tokens()->Add() = pb_token;
      if (output_idx < static_cast<int32_t>(mm_embeddings.size())) {
        for (const auto& tensor : mm_embeddings[output_idx]) {
          torch_tensor_to_proto_tensor(
              tensor, pb_seq_out.mutable_mm_embeddings()->add_tensors());
        }
      }
      *pb_forward_output->mutable_outputs()->Add() = pb_seq_out;
    }
  }

  if (::xllm::EPLBConfig::get_instance().enable_eplb()) {
    pb_forward_output->set_prepared_layer_id(prepared_layer_id);

    if (expert_load_data.defined()) {
      torch::Tensor expert_load_data_flattened =
          expert_load_data.view({-1}).contiguous();
      Slice<int64_t> expert_load_data_flattened_slice = {
          expert_load_data_flattened.data_ptr<int64_t>(),
          static_cast<size_t>(expert_load_data_flattened.size(0))};
      ADD_VECTOR_TO_PROTO(pb_forward_output->mutable_expert_load_data(),
                          expert_load_data_flattened_slice);
    }
  }

  if (src_seq_idxes.defined() && src_seq_idxes.numel() > 0) {
    Slice<int32_t> src_seq_idxes_slice = {
        src_seq_idxes.data_ptr<int32_t>(),
        static_cast<size_t>(src_seq_idxes.numel())};
    ADD_VECTOR_TO_PROTO(pb_forward_output->mutable_src_seq_idxes(),
                        src_seq_idxes_slice);
  }
  if (out_tokens.defined() && out_tokens.numel() > 0) {
    Slice<int32_t> out_tokens_slice = {out_tokens.data_ptr<int32_t>(),
                                       static_cast<size_t>(out_tokens.numel())};
    ADD_VECTOR_TO_PROTO(pb_forward_output->mutable_out_tokens(),
                        out_tokens_slice);
  }
  if (out_logprobs.defined() && out_logprobs.numel() > 0) {
    Slice<float> out_logprobs_slice = {
        out_logprobs.data_ptr<float>(),
        static_cast<size_t>(out_logprobs.numel())};
    ADD_VECTOR_TO_PROTO(pb_forward_output->mutable_out_logprobs(),
                        out_logprobs_slice);
  }
  if (!dit_images.empty()) {
    TORCH_TENSOR_VEC_TO_PROTO_TENSOR_LIST(
        pb_forward_output->mutable_dit_forward_output()->mutable_tensors(),
        dit_images);
  }
  if (!dit_text_output.empty()) {
    // DiT text-only path: serialized even when next_tokens is undefined above.
    auto* pb_dit_output = pb_forward_output->mutable_dit_forward_output();
    for (const auto& text : dit_text_output) {
      pb_dit_output->add_text_output(text);
    }
  }
  COUNTER_ADD(proto_latency_seconds_o2proto, timer.elapsed_seconds());
  return;
}

Token build_token(int64_t index,
                  torch::Tensor token_ids,
                  torch::Tensor logprobs,
                  torch::Tensor top_tokens,
                  torch::Tensor top_logprobs) {
  Token token(token_ids[index].item<int64_t>());
  if (logprobs.defined()) {
    token.logprob = logprobs[index].item<float>();
    if (top_tokens.defined() && top_logprobs.defined()) {
      auto topk_tokens = top_tokens[index];
      auto topk_logprobs = top_logprobs[index];
      const size_t size = topk_tokens.numel();
      token.top_tokens = {topk_tokens.const_data_ptr<int64_t>(), size};
      token.top_logprobs = {topk_logprobs.const_data_ptr<float>(), size};
    }
  }
  return token;
}

uint64_t proto_to_block_transfer_info(
    const proto::BlockTransferInfos& pb_block_transfer_info,
    std::vector<BlockTransferInfo>& block_transfer_info) {
  block_transfer_info.reserve(pb_block_transfer_info.transfer_infos_size());

  for (int i = 0; i < pb_block_transfer_info.transfer_infos_size(); ++i) {
    block_transfer_info.emplace_back(
        pb_block_transfer_info.transfer_infos(i).src_block_id(),
        pb_block_transfer_info.transfer_infos(i).dst_block_id(),
        reinterpret_cast<const uint8_t*>(
            pb_block_transfer_info.transfer_infos(i).hash_key().data()),
        TransferType(pb_block_transfer_info.transfer_type()),
        static_cast<BlockType>(
            pb_block_transfer_info.transfer_infos(i).block_type()));
  }

  return pb_block_transfer_info.batch_id();
}

bool block_transfer_info_to_proto(
    const std::vector<BlockTransferInfo>& block_transfer_info,
    proto::BlockTransferInfos* pb_block_transfer_info) {
  pb_block_transfer_info->mutable_transfer_infos()->Reserve(
      block_transfer_info.size());
  auto transfer_type = block_transfer_info[0].transfer_type;
  for (const BlockTransferInfo info : block_transfer_info) {
    if (transfer_type != info.transfer_type) {
      LOG(ERROR) << "Convert to BlockTransferInfos fail, TransferType must be "
                    "same, but got "
                 << uint8_t(transfer_type) << " and "
                 << uint8_t(info.transfer_type);
      return false;
    }

    proto::BlockTransferInfo pb_cache;
    pb_cache.set_src_block_id(info.src_block_id);
    pb_cache.set_dst_block_id(info.dst_block_id);
    pb_cache.set_hash_key(info.hash_key, XXH3_128BITS_HASH_VALUE_LEN);
    pb_cache.set_block_type(
        static_cast<proto::BlockType>(static_cast<int8_t>(info.block_type)));

    *pb_block_transfer_info->mutable_transfer_infos()->Add() =
        std::move(pb_cache);
  }
  pb_block_transfer_info->set_transfer_type(proto::TransferType(transfer_type));

  return true;
}

bool block_transfer_info_to_proto(
    const uint64_t batch_id,
    const std::vector<BlockTransferInfo>& block_transfer_info,
    proto::BlockTransferInfos* pb_block_transfer_info) {
  if (!block_transfer_info_to_proto(block_transfer_info,
                                    pb_block_transfer_info)) {
    return false;
  }
  pb_block_transfer_info->set_batch_id(batch_id);
  return true;
}

bool dit_forward_input_to_proto(const DiTForwardInput& dit_inputs,
                                proto::DiTForwardInput* pb_dit_inputs) {
  pb_dit_inputs->set_batch_size(dit_inputs.batch_size);

  ADD_VECTOR_TO_PROTO(pb_dit_inputs->mutable_prompts(), dit_inputs.prompts);

  ADD_VECTOR_TO_PROTO(pb_dit_inputs->mutable_prompts_2(), dit_inputs.prompts_2);

  ADD_VECTOR_TO_PROTO(pb_dit_inputs->mutable_negative_prompts(),
                      dit_inputs.negative_prompts);

  ADD_VECTOR_TO_PROTO(pb_dit_inputs->mutable_negative_prompts_2(),
                      dit_inputs.negative_prompts_2);

  torch_tensor_to_proto_tensor(dit_inputs.images,
                               pb_dit_inputs->mutable_images());

  auto* pb_images_list =
      pb_dit_inputs->mutable_images_list()->mutable_tensors();
  for (const auto& tensor : dit_inputs.images_list) {
    torch_tensor_to_proto_tensor(tensor, pb_images_list->Add());
  }

  torch_tensor_to_proto_tensor(dit_inputs.mask_images,
                               pb_dit_inputs->mutable_mask_images());

  torch_tensor_to_proto_tensor(dit_inputs.control_image,
                               pb_dit_inputs->mutable_control_image());

  torch_tensor_to_proto_tensor(dit_inputs.masked_image_latents,
                               pb_dit_inputs->mutable_masked_image_latents());

  torch_tensor_to_proto_tensor(dit_inputs.prompt_embeds,
                               pb_dit_inputs->mutable_prompt_embeds());

  torch_tensor_to_proto_tensor(dit_inputs.pooled_prompt_embeds,
                               pb_dit_inputs->mutable_pooled_prompt_embeds());

  torch_tensor_to_proto_tensor(dit_inputs.negative_prompt_embeds,
                               pb_dit_inputs->mutable_negative_prompt_embeds());

  torch_tensor_to_proto_tensor(
      dit_inputs.negative_pooled_prompt_embeds,
      pb_dit_inputs->mutable_negative_pooled_prompt_embeds());

  torch_tensor_to_proto_tensor(dit_inputs.latents,
                               pb_dit_inputs->mutable_latents());
  torch_tensor_to_proto_tensor(dit_inputs.last_images,
                               pb_dit_inputs->mutable_last_images());

  torch_tensor_to_proto_tensor(dit_inputs.prompt_audio,
                               pb_dit_inputs->mutable_prompt_audio());

  if (!dit_inputs.audio_prompt_text.empty()) {
    pb_dit_inputs->set_audio_prompt_text(dit_inputs.audio_prompt_text);
  }

  if (!generation_params_to_proto(dit_inputs.generation_params,
                                  pb_dit_inputs->mutable_generation_params())) {
    LOG(ERROR) << "Failed to convert generation_params";
    return false;
  }

  return true;
}

bool generation_params_to_proto(
    const DiTGenerationParams& dit_generation_params,
    proto::DiTGenerationParams* pb_dit_generation_params) {
  pb_dit_generation_params->set_width(dit_generation_params.width);
  pb_dit_generation_params->set_height(dit_generation_params.height);
  pb_dit_generation_params->set_num_inference_steps(
      dit_generation_params.num_inference_steps);
  pb_dit_generation_params->set_true_cfg_scale(
      dit_generation_params.true_cfg_scale);
  pb_dit_generation_params->set_guidance_scale(
      dit_generation_params.guidance_scale);
  pb_dit_generation_params->set_num_images_per_prompt(
      dit_generation_params.num_images_per_prompt);
  pb_dit_generation_params->set_seed_is_set(dit_generation_params.seed_is_set);
  if (dit_generation_params.seed_is_set) {
    pb_dit_generation_params->set_seed(dit_generation_params.seed);
  }
  pb_dit_generation_params->set_max_sequence_length(
      dit_generation_params.max_sequence_length);
  pb_dit_generation_params->set_strength(dit_generation_params.strength);
  pb_dit_generation_params->set_enable_cfg_renorm(
      dit_generation_params.enable_cfg_renorm);
  pb_dit_generation_params->set_cfg_renorm_min(
      dit_generation_params.cfg_renorm_min);
  pb_dit_generation_params->set_num_frames(dit_generation_params.num_frames);
  pb_dit_generation_params->set_force_video_output(
      dit_generation_params.force_video_output);
  pb_dit_generation_params->set_video_fps(dit_generation_params.video_fps);
  pb_dit_generation_params->set_guidance_scale_2(
      dit_generation_params.guidance_scale_2);
  pb_dit_generation_params->set_seconds(dit_generation_params.seconds);
  pb_dit_generation_params->set_boundary_ratio(
      dit_generation_params.boundary_ratio);
  pb_dit_generation_params->set_flow_shift(dit_generation_params.flow_shift);
  pb_dit_generation_params->set_num_videos_per_prompt(
      dit_generation_params.num_videos_per_prompt);
  // Text diffusion params
  if (dit_generation_params.max_new_tokens > 0) {
    pb_dit_generation_params->set_max_new_tokens(
        dit_generation_params.max_new_tokens);
  }
  if (dit_generation_params.diffusion_steps > 0) {
    pb_dit_generation_params->set_diffusion_steps(
        dit_generation_params.diffusion_steps);
  }
  pb_dit_generation_params->set_temperature(dit_generation_params.temperature);
  pb_dit_generation_params->set_top_k(dit_generation_params.top_k);
  pb_dit_generation_params->set_top_p(dit_generation_params.top_p);
  pb_dit_generation_params->set_repetition_penalty(
      dit_generation_params.repetition_penalty);
  return true;
}

bool proto_to_dit_forward_input(const proto::DiTForwardInput& pb_dit_inputs,
                                DiTForwardInput& dit_inputs) {
  dit_inputs.batch_size = pb_dit_inputs.batch_size();

  std::vector<std::string> prompts = std::vector<std::string>(
      pb_dit_inputs.prompts().begin(), pb_dit_inputs.prompts().end());
  std::vector<std::string> prompts_2 = std::vector<std::string>(
      pb_dit_inputs.prompts_2().begin(), pb_dit_inputs.prompts_2().end());
  std::vector<std::string> negative_prompts =
      std::vector<std::string>(pb_dit_inputs.negative_prompts().begin(),
                               pb_dit_inputs.negative_prompts().end());
  std::vector<std::string> negative_prompts_2 =
      std::vector<std::string>(pb_dit_inputs.negative_prompts_2().begin(),
                               pb_dit_inputs.negative_prompts_2().end());
  dit_inputs.prompts = std::move(prompts);

  dit_inputs.prompts_2 = std::move(prompts_2);

  dit_inputs.negative_prompts = std::move(negative_prompts);

  dit_inputs.negative_prompts_2 = std::move(negative_prompts_2);

  if (pb_dit_inputs.has_images()) {
    dit_inputs.images = util::proto_to_torch(pb_dit_inputs.images());
  }

  if (pb_dit_inputs.has_images_list()) {
    dit_inputs.images_list.reserve(
        pb_dit_inputs.images_list().tensors().size());
    for (const auto& pb_tensor : pb_dit_inputs.images_list().tensors()) {
      dit_inputs.images_list.emplace_back(util::proto_to_torch(pb_tensor));
    }
  }

  if (pb_dit_inputs.has_mask_images()) {
    dit_inputs.mask_images = util::proto_to_torch(pb_dit_inputs.mask_images());
  }

  if (pb_dit_inputs.has_control_image()) {
    dit_inputs.control_image =
        util::proto_to_torch(pb_dit_inputs.control_image());
  }

  if (pb_dit_inputs.has_masked_image_latents()) {
    dit_inputs.masked_image_latents =
        util::proto_to_torch(pb_dit_inputs.masked_image_latents());
  }

  if (pb_dit_inputs.has_prompt_embeds()) {
    dit_inputs.prompt_embeds =
        util::proto_to_torch(pb_dit_inputs.prompt_embeds());
  }

  if (pb_dit_inputs.has_pooled_prompt_embeds()) {
    dit_inputs.pooled_prompt_embeds =
        util::proto_to_torch(pb_dit_inputs.pooled_prompt_embeds());
  }

  if (pb_dit_inputs.has_negative_prompt_embeds()) {
    dit_inputs.negative_prompt_embeds =
        util::proto_to_torch(pb_dit_inputs.negative_prompt_embeds());
  }

  if (pb_dit_inputs.has_negative_pooled_prompt_embeds()) {
    dit_inputs.negative_pooled_prompt_embeds =
        util::proto_to_torch(pb_dit_inputs.negative_pooled_prompt_embeds());
  }

  if (pb_dit_inputs.has_latents()) {
    dit_inputs.latents = util::proto_to_torch(pb_dit_inputs.latents());
  }
  if (pb_dit_inputs.has_last_images()) {
    dit_inputs.last_images = util::proto_to_torch(pb_dit_inputs.last_images());
  }

  if (!proto_to_generation_params(pb_dit_inputs.generation_params(),
                                  dit_inputs.generation_params)) {
    LOG(ERROR) << "Failed to convert generation_params";
    return false;
  }

  if (pb_dit_inputs.has_prompt_audio()) {
    dit_inputs.prompt_audio =
        util::proto_to_torch(pb_dit_inputs.prompt_audio());
  }
  if (pb_dit_inputs.has_audio_prompt_text()) {
    dit_inputs.audio_prompt_text = pb_dit_inputs.audio_prompt_text();
  }

  return true;
}

bool proto_to_generation_params(
    const proto::DiTGenerationParams& pb_dit_generation_params,
    DiTGenerationParams& dit_generation_params) {
  LOG(INFO) << "start brpc transfer";
  dit_generation_params.width = pb_dit_generation_params.width();
  dit_generation_params.height = pb_dit_generation_params.height();
  dit_generation_params.num_inference_steps =
      pb_dit_generation_params.num_inference_steps();
  dit_generation_params.true_cfg_scale =
      pb_dit_generation_params.true_cfg_scale();
  dit_generation_params.guidance_scale =
      pb_dit_generation_params.guidance_scale();
  dit_generation_params.num_images_per_prompt =
      pb_dit_generation_params.num_images_per_prompt();
  if (pb_dit_generation_params.seed_is_set()) {
    dit_generation_params.seed_is_set = true;
    if (pb_dit_generation_params.has_seed()) {
      dit_generation_params.seed = pb_dit_generation_params.seed();
    }
  } else if (pb_dit_generation_params.has_seed()) {
    // Backward compat: messages sent before seed_is_set existed.
    dit_generation_params.seed = pb_dit_generation_params.seed();
    dit_generation_params.seed_is_set = true;
  } else {
    dit_generation_params.seed_is_set = false;
  }
  dit_generation_params.max_sequence_length =
      pb_dit_generation_params.max_sequence_length();
  dit_generation_params.strength = pb_dit_generation_params.strength();
  dit_generation_params.enable_cfg_renorm =
      pb_dit_generation_params.enable_cfg_renorm();
  dit_generation_params.cfg_renorm_min =
      pb_dit_generation_params.cfg_renorm_min();
  dit_generation_params.num_frames = pb_dit_generation_params.num_frames();
  dit_generation_params.force_video_output =
      pb_dit_generation_params.force_video_output();
  dit_generation_params.video_fps = pb_dit_generation_params.video_fps();
  dit_generation_params.guidance_scale_2 =
      pb_dit_generation_params.guidance_scale_2();
  dit_generation_params.seconds = pb_dit_generation_params.seconds();
  dit_generation_params.boundary_ratio =
      pb_dit_generation_params.boundary_ratio();
  dit_generation_params.flow_shift = pb_dit_generation_params.flow_shift();
  dit_generation_params.num_videos_per_prompt =
      pb_dit_generation_params.num_videos_per_prompt();
  // Text diffusion params
  if (pb_dit_generation_params.has_max_new_tokens()) {
    dit_generation_params.max_new_tokens =
        pb_dit_generation_params.max_new_tokens();
  }
  if (pb_dit_generation_params.has_diffusion_steps()) {
    dit_generation_params.diffusion_steps =
        pb_dit_generation_params.diffusion_steps();
  }
  if (pb_dit_generation_params.has_temperature()) {
    dit_generation_params.temperature = pb_dit_generation_params.temperature();
  }
  if (pb_dit_generation_params.has_top_k()) {
    dit_generation_params.top_k = pb_dit_generation_params.top_k();
  }
  if (pb_dit_generation_params.has_top_p()) {
    dit_generation_params.top_p = pb_dit_generation_params.top_p();
  }
  if (pb_dit_generation_params.has_repetition_penalty()) {
    dit_generation_params.repetition_penalty =
        pb_dit_generation_params.repetition_penalty();
  }
  return true;
}

bool proto_to_dit_forward_output(const proto::DiTForwardOutput& pb_dit_outputs,
                                 DiTForwardOutput& dit_outputs) {
  const auto& pb_tensor_list = pb_dit_outputs.tensors();
  std::vector<torch::Tensor> torch_tensor_vec;
  torch_tensor_vec.reserve(pb_tensor_list.tensors_size());
  for (const auto& pb_tensor : pb_tensor_list.tensors()) {
    torch::Tensor torch_tensor = util::proto_to_torch(pb_tensor);
    if (!torch_tensor.defined()) {
      LOG(ERROR) << "Failed to convert PB Tensor to torch Tensor (list item)";
      return false;
    }
    torch_tensor_vec.emplace_back(std::move(torch_tensor));
  }
  dit_outputs.tensors = std::move(torch_tensor_vec);

  // Deserialize text_output for text diffusion models
  dit_outputs.text_output.assign(pb_dit_outputs.text_output().begin(),
                                 pb_dit_outputs.text_output().end());

  return true;
}

bool torch_tensor_to_proto_tensor(const torch::Tensor& torch_tensor,
                                  proto::Tensor* proto_tensor) {
  if (torch_tensor.defined()) {
    if (!util::torch_to_proto(torch_tensor, proto_tensor)) {
      LOG(ERROR) << "Failed to convert torch Tensor to Pb Tensor ";
      return false;
    }
  }
  return true;
}

}  // namespace xllm
