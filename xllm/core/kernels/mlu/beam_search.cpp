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

#include <array>
#include <cstdint>
#include <string_view>

#include "common/global_flags.h"
#include "mlu_ops_api.h"

namespace xllm::kernel::mlu {
namespace {

using NamedTensor = std::pair<std::string_view, const torch::Tensor*>;

void check_tensor(const torch::Tensor& tensor,
                  std::string_view name,
                  torch::ScalarType dtype,
                  const torch::Device& device) {
  TORCH_CHECK(tensor.defined(), "beam_search: ", name, " must be defined");
  TORCH_CHECK(tensor.device() == device,
              "beam_search: ",
              name,
              " must be on ",
              device,
              ", got ",
              tensor.device());
  TORCH_CHECK(tensor.scalar_type() == dtype,
              "beam_search: ",
              name,
              " has invalid dtype");
  TORCH_CHECK(
      tensor.is_contiguous(), "beam_search: ", name, " must be contiguous");
}

void check_shape(const torch::Tensor& tensor,
                 std::string_view name,
                 at::IntArrayRef expected) {
  TORCH_CHECK(tensor.sizes() == expected,
              "beam_search: ",
              name,
              " must have shape ",
              expected,
              ", got ",
              tensor.sizes());
}

void check_aliases(const std::array<NamedTensor, 9>& tensors) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    for (size_t j = i + 1; j < tensors.size(); ++j) {
      TORCH_CHECK(!tensors[i].second->is_alias_of(*tensors[j].second),
                  "beam_search: ",
                  tensors[i].first,
                  " and ",
                  tensors[j].first,
                  " must not share storage");
    }
  }
}

void check_args(torch::Tensor acc_logprob,
                torch::Tensor in_sequence_group,
                torch::Tensor top_tokens,
                torch::Tensor top_logprobs,
                torch::Tensor out_acc_logprob,
                torch::Tensor out_token_ids,
                torch::Tensor out_token_index,
                torch::Tensor out_beam_count_prefix_sums,
                torch::Tensor out_sequence_group,
                uint32_t batch_size,
                uint32_t num_return_sequences,
                uint32_t current_step) {
  TORCH_CHECK(batch_size > 0, "beam_search: batch_size must be positive");
  TORCH_CHECK(in_sequence_group.defined(),
              "beam_search: in_sequence_group must be defined");
  TORCH_CHECK(in_sequence_group.dim() == 3,
              "beam_search: in_sequence_group must be 3D");

  const int64_t beam_size = in_sequence_group.size(1);
  const int64_t total_rounds = in_sequence_group.size(2);
  TORCH_CHECK(beam_size > 0, "beam_search: beam_size must be positive");
  TORCH_CHECK(total_rounds > 0, "beam_search: total_rounds must be positive");
  TORCH_CHECK(current_step < static_cast<uint64_t>(total_rounds),
              "beam_search: current_step must be less than total_rounds");
  TORCH_CHECK(!(current_step == static_cast<uint64_t>(total_rounds - 1) &&
                num_return_sequences > static_cast<uint64_t>(beam_size)),
              "beam_search: widened final results are not supported");

  TORCH_CHECK(acc_logprob.defined(),
              "beam_search: acc_logprob must be defined");
  const auto device = acc_logprob.device();
  TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1,
              "beam_search: tensors must be on an MLU device");

  check_tensor(acc_logprob, "acc_logprob", torch::kFloat32, device);
  check_tensor(in_sequence_group, "in_sequence_group", torch::kInt32, device);
  check_tensor(top_tokens, "top_tokens", torch::kInt32, device);
  check_tensor(top_logprobs, "top_logprobs", torch::kFloat32, device);
  check_tensor(out_acc_logprob, "out_acc_logprob", torch::kFloat32, device);
  check_tensor(out_token_ids, "out_token_ids", torch::kInt32, device);
  check_tensor(out_token_index, "out_token_index", torch::kInt32, device);
  check_tensor(out_beam_count_prefix_sums,
               "out_beam_count_prefix_sums",
               torch::kInt32,
               device);
  check_tensor(out_sequence_group, "out_sequence_group", torch::kInt32, device);

  const int64_t batch = batch_size;
  const int64_t rows = batch * beam_size;
  check_shape(acc_logprob, "acc_logprob", {rows, 1});
  check_shape(
      in_sequence_group, "in_sequence_group", {batch, beam_size, total_rounds});
  const std::vector<int64_t> candidate_shape =
      current_step == 0 ? std::vector<int64_t>{batch, beam_size}
                        : std::vector<int64_t>{rows, beam_size};
  check_shape(top_tokens, "top_tokens", candidate_shape);
  check_shape(top_logprobs, "top_logprobs", candidate_shape);
  TORCH_CHECK(top_tokens.size(1) == beam_size,
              "beam_search: top_k must equal beam_size");
  check_shape(out_acc_logprob, "out_acc_logprob", {rows, 1});
  check_shape(out_token_ids, "out_token_ids", {rows, 1});
  check_shape(out_token_index, "out_token_index", {rows, 1});
  check_shape(
      out_beam_count_prefix_sums, "out_beam_count_prefix_sums", {rows, 1});
  check_shape(out_sequence_group,
              "out_sequence_group",
              {batch, beam_size, total_rounds});

  check_aliases(
      {NamedTensor{"acc_logprob", &acc_logprob},
       NamedTensor{"in_sequence_group", &in_sequence_group},
       NamedTensor{"top_tokens", &top_tokens},
       NamedTensor{"top_logprobs", &top_logprobs},
       NamedTensor{"out_acc_logprob", &out_acc_logprob},
       NamedTensor{"out_token_ids", &out_token_ids},
       NamedTensor{"out_token_index", &out_token_index},
       NamedTensor{"out_beam_count_prefix_sums", &out_beam_count_prefix_sums},
       NamedTensor{"out_sequence_group", &out_sequence_group}});
}

void init_beams(torch::Tensor top_tokens,
                torch::Tensor top_logprobs,
                torch::Tensor out_acc_logprob,
                torch::Tensor out_token_ids,
                torch::Tensor out_token_index,
                torch::Tensor out_sequence_group,
                int64_t batch_size,
                int64_t beam_size) {
  out_token_ids.view({batch_size, beam_size}).copy_(top_tokens);
  out_acc_logprob.view({batch_size, beam_size}).copy_(top_logprobs);
  auto indices = torch::arange(beam_size,
                               torch::TensorOptions()
                                   .dtype(torch::kInt32)
                                   .device(top_tokens.device()))
                     .unsqueeze(0)
                     .expand({batch_size, beam_size});
  out_token_index.view({batch_size, beam_size}).copy_(indices);
  out_sequence_group.select(2, 0).copy_(top_tokens);
}

void select_beams(torch::Tensor acc_logprob,
                  torch::Tensor in_sequence_group,
                  torch::Tensor top_tokens,
                  torch::Tensor top_logprobs,
                  torch::Tensor out_acc_logprob,
                  torch::Tensor out_token_ids,
                  torch::Tensor out_token_index,
                  torch::Tensor out_sequence_group,
                  int64_t batch_size,
                  int64_t beam_size,
                  int64_t total_rounds,
                  uint32_t current_step) {
  auto combined =
      (acc_logprob + top_logprobs).view({batch_size, beam_size * beam_size});
  auto topk = torch::topk(combined,
                          beam_size,
                          /*dim=*/1,
                          /*largest=*/true,
                          /*sorted=*/FLAGS_enable_topk_sorted);
  auto scores = std::get<0>(topk);
  auto flat_idx = std::get<1>(topk);

  if (current_step < static_cast<uint64_t>(total_rounds - 1)) {
    auto order = flat_idx.argsort(/*dim=*/1, /*descending=*/false);
    scores = scores.gather(1, order);
    flat_idx = flat_idx.gather(1, order);
  }

  auto flat_tokens = top_tokens.view({batch_size, beam_size * beam_size});
  auto selected_tokens = flat_tokens.gather(1, flat_idx);
  auto parent = torch::div(flat_idx, beam_size, "floor");
  auto history_idx =
      parent.unsqueeze(-1).expand({batch_size, beam_size, total_rounds});
  auto selected_history = in_sequence_group.gather(1, history_idx);

  out_acc_logprob.view({batch_size, beam_size}).copy_(scores);
  out_token_ids.view({batch_size, beam_size}).copy_(selected_tokens);
  // The index is local to each batch and flattens
  // [parent_beam, candidate_index].
  out_token_index.view({batch_size, beam_size})
      .copy_(flat_idx.to(torch::kInt32));
  out_sequence_group.slice(2, 0, current_step)
      .copy_(selected_history.slice(2, 0, current_step));
  out_sequence_group.select(2, current_step).copy_(selected_tokens);
}

}  // namespace

void beam_search(torch::Tensor acc_logprob,
                 torch::Tensor in_sequence_group,
                 torch::Tensor top_tokens,
                 torch::Tensor top_logprobs,
                 torch::Tensor out_acc_logprob,
                 torch::Tensor out_token_ids,
                 torch::Tensor out_token_index,
                 torch::Tensor out_beam_count_prefix_sums,
                 torch::Tensor out_sequence_group,
                 uint32_t batch_size,
                 uint32_t num_return_sequences,
                 uint32_t current_step) {
  check_args(acc_logprob,
             in_sequence_group,
             top_tokens,
             top_logprobs,
             out_acc_logprob,
             out_token_ids,
             out_token_index,
             out_beam_count_prefix_sums,
             out_sequence_group,
             batch_size,
             num_return_sequences,
             current_step);

  const int64_t beam_size = in_sequence_group.size(1);
  const int64_t total_rounds = in_sequence_group.size(2);
  if (current_step == 0) {
    init_beams(top_tokens,
               top_logprobs,
               out_acc_logprob,
               out_token_ids,
               out_token_index,
               out_sequence_group,
               batch_size,
               beam_size);
    return;
  }

  select_beams(acc_logprob,
               in_sequence_group,
               top_tokens,
               top_logprobs,
               out_acc_logprob,
               out_token_ids,
               out_token_index,
               out_sequence_group,
               batch_size,
               beam_size,
               total_rounds,
               current_step);
}

}  // namespace xllm::kernel::mlu
