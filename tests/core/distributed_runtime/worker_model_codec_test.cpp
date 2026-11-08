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

#include "core/distributed_runtime/worker_model_codec.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace xllm {
namespace {

class LlmCodec final {
 public:
  using Response = proto::LLMForwardResponse;
  static constexpr auto encode = encode_llm_output;
  static constexpr auto decode = decode_llm_output;
};

class VlmCodec final {
 public:
  using Response = proto::VLMForwardResponse;
  static constexpr auto encode = encode_vlm_output;
  static constexpr auto decode = decode_vlm_output;
};

class RecCodec final {
 public:
  using Response = proto::RECForwardResponse;
  static constexpr auto encode = encode_rec_output;
  static constexpr auto decode = decode_rec_output;
};

template <typename Codec>
class WorkerModelTokenCodecTest : public ::testing::Test {};

using TokenCodecs = ::testing::Types<LlmCodec, VlmCodec, RecCodec>;
TYPED_TEST_SUITE(WorkerModelTokenCodecTest, TokenCodecs);

RawForwardOutput original_output() {
  RawForwardOutput output;
  output.prepared_token = 81;
  output.outputs.resize(1);
  output.outputs[0].tokens.resize(1);
  output.outputs[0].tokens[0].id = 19;
  output.dit_forward_output.text_output = {"original"};
  return output;
}

void expect_original_output(const RawForwardOutput& output) {
  EXPECT_EQ(output.prepared_token, 81);
  ASSERT_EQ(output.outputs.size(), 1);
  ASSERT_EQ(output.outputs[0].tokens.size(), 1);
  EXPECT_EQ(output.outputs[0].tokens[0].id, 19);
  EXPECT_EQ(output.dit_forward_output.text_output,
            (std::vector<std::string>{"original"}));
}

TYPED_TEST(WorkerModelTokenCodecTest,
           RoundTripPreservesTokensEmbeddingsStatsBeamAndJson) {
  ForwardOutput source;
  source.sample_output.next_tokens = torch::tensor({7, 8}, torch::kInt32);
  source.sample_output.logprobs = torch::tensor({-0.5f, -1.0f});
  source.sample_output.top_tokens =
      torch::tensor({{7, 9}, {8, 10}}, torch::kInt64);
  source.sample_output.top_logprobs =
      torch::tensor({{-0.5f, -1.5f}, {-1.0f, -2.0f}});
  source.sample_output.embeddings = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
  source.sample_output.mm_embeddings = {{torch::tensor({1.0f, 2.0f})},
                                        {torch::tensor({3.0f})}};
  source.beam_search_output.src_seq_idxes =
      torch::tensor({1, 0}, torch::kInt32);
  source.beam_search_output.out_tokens = torch::tensor({8, 7}, torch::kInt32);
  source.beam_search_output.out_logprobs = torch::tensor({-1.0f, -0.5f});
  source.json_object_errors = {{"sequence-0", "invalid JSON"}};
  // Token domains do not serialize diffusion output.
  source.dit_forward_output.tensors = {torch::Tensor()};
  source.dit_forward_output.text_output = {"unused"};
  typename TypeParam::Response encoded;
  ASSERT_TRUE(TypeParam::encode(source, {{1, 2}, {2, 3}}, &encoded));
  encoded.set_prepared_token(42);
  encoded.add_expert_load_data(9);
  RawForwardOutput decoded = original_output();
  ASSERT_TRUE(TypeParam::decode(encoded, decoded));

  ASSERT_EQ(decoded.outputs.size(), 2);
  ASSERT_EQ(decoded.outputs[0].tokens.size(), 1);
  const RawToken& token = decoded.outputs[0].tokens[0];
  EXPECT_EQ(token.id, 7);
  ASSERT_TRUE(token.logprob.has_value());
  EXPECT_FLOAT_EQ(*token.logprob, -0.5f);
  EXPECT_EQ(token.top_tokens, (std::vector<int64_t>{7, 9}));
  EXPECT_EQ(token.top_logprobs, (std::vector<float>{-0.5f, -1.5f}));
  EXPECT_EQ(token.embeddings, (std::vector<float>{1.0f, 2.0f}));
  ASSERT_EQ(decoded.outputs[0].mm_embeddings.size(), 1);
  EXPECT_TRUE(torch::equal(decoded.outputs[0].mm_embeddings[0],
                           source.sample_output.mm_embeddings[0][0]));
  EXPECT_EQ(decoded.outputs[0].speculative_token_stats.accepted_tokens, 1);
  EXPECT_EQ(decoded.outputs[1].speculative_token_stats.proposed_tokens, 3);
  EXPECT_EQ(decoded.prepared_token, 42);
  EXPECT_EQ(decoded.expert_load_data, (std::vector<int64_t>{9}));
  EXPECT_EQ(decoded.src_seq_idxes, (std::vector<int32_t>{1, 0}));
  EXPECT_EQ(decoded.out_tokens, (std::vector<int32_t>{8, 7}));
  EXPECT_EQ(decoded.out_logprobs, (std::vector<float>{-1.0f, -0.5f}));
  ASSERT_EQ(decoded.json_object_errors.size(), 1);
  EXPECT_EQ(decoded.json_object_errors[0].sample_sequence_id, "sequence-0");
  EXPECT_EQ(decoded.json_object_errors[0].message, "invalid JSON");
  EXPECT_TRUE(decoded.dit_forward_output.tensors.empty());
  EXPECT_TRUE(decoded.dit_forward_output.text_output.empty());
}

TYPED_TEST(WorkerModelTokenCodecTest,
           InvalidTensorLayoutsAndBeamCountsPreserveEncodedDestination) {
  typename TypeParam::Response destination;
  destination.add_outputs()->add_tokens()->set_id(19);
  const std::string original = destination.SerializeAsString();
  ForwardOutput valid;
  valid.sample_output.next_tokens = torch::tensor({7, 8}, torch::kInt32);
  std::vector<ForwardOutput> invalid_inputs(12, valid);
  invalid_inputs[0].sample_output.next_tokens = torch::tensor({7.0f, 8.0f});
  invalid_inputs[1].sample_output.logprobs = torch::tensor({-0.5f});
  invalid_inputs[2].sample_output.top_tokens =
      torch::tensor({{7}, {8}}, torch::kInt64);
  invalid_inputs[3].sample_output.embeddings = torch::tensor({1.0f, 2.0f});
  invalid_inputs[4].sample_output.next_tokens =
      torch::tensor({{7, 8}, {9, 10}}, torch::kInt32).transpose(0, 1);
  invalid_inputs[5].sample_output.mm_embeddings = {{torch::Tensor()}};
  invalid_inputs[6].beam_search_output.src_seq_idxes =
      torch::tensor({1, 0}, torch::kInt64);
  invalid_inputs[7].beam_search_output.src_seq_idxes =
      torch::tensor({1, 0}, torch::kInt32);
  invalid_inputs[8].beam_search_output.src_seq_idxes =
      torch::tensor({1, 0}, torch::kInt32);
  invalid_inputs[8].beam_search_output.out_tokens =
      torch::tensor({8}, torch::kInt32);
  invalid_inputs[8].beam_search_output.out_logprobs =
      torch::tensor({-1.0f, -0.5f});
  invalid_inputs[9].beam_search_output.out_tokens =
      torch::tensor({8}, torch::kInt32);
  invalid_inputs[9].beam_search_output.src_seq_idxes =
      torch::empty({0}, torch::kInt32);
  invalid_inputs[10].sample_output.mm_embeddings = {{torch::tensor({1.0f})},
                                                    {torch::tensor({2.0f})},
                                                    {torch::tensor({3.0f})}};
  invalid_inputs[11].sample_output.next_tokens =
      torch::tensor({{7, 8}, {9, 10}}, torch::kInt32);
  invalid_inputs[11].sample_output.mm_embeddings = {{torch::Tensor()}};

  for (const ForwardOutput& invalid : invalid_inputs) {
    EXPECT_FALSE(TypeParam::encode(invalid, {}, &destination));
    EXPECT_EQ(destination.SerializeAsString(), original);
  }
  EXPECT_FALSE(TypeParam::encode(valid, {{1, 2}}, &destination));
  EXPECT_EQ(destination.SerializeAsString(), original);
  EXPECT_FALSE(TypeParam::encode(valid, {}, nullptr));
}

TYPED_TEST(WorkerModelTokenCodecTest,
           MalformedPayloadsPreserveDecodedDestination) {
  std::vector<typename TypeParam::Response> invalid_inputs(3);
  invalid_inputs[0].add_src_seq_idxes(0);
  invalid_inputs[0].add_out_tokens(7);
  invalid_inputs[1].add_outputs()->add_tokens()->add_top_tokens(7);
  invalid_inputs[2].add_outputs()->mutable_mm_embeddings()->add_tensors();
  for (const typename TypeParam::Response& invalid : invalid_inputs) {
    RawForwardOutput destination = original_output();
    EXPECT_FALSE(TypeParam::decode(invalid, destination));
    expect_original_output(destination);
  }
}

TYPED_TEST(WorkerModelTokenCodecTest,
           SpeculativePaddingPreservesPerTokenPayloads) {
  ForwardOutput source;
  source.sample_output.next_tokens =
      torch::tensor({{7, -1, 99}, {8, 9, -1}}, torch::kInt64);
  source.sample_output.logprobs =
      torch::tensor({{-0.5f, 0.0f, 0.0f}, {-1.0f, -1.5f, 0.0f}});
  source.sample_output.top_tokens =
      torch::tensor({{{7}, {0}, {99}}, {{8}, {9}, {0}}}, torch::kInt64);
  source.sample_output.top_logprobs =
      source.sample_output.logprobs.unsqueeze(/*dim=*/2).contiguous();
  source.sample_output.embeddings =
      torch::tensor({{{1.0f}, {2.0f}, {3.0f}}, {{4.0f}, {5.0f}, {6.0f}}});
  source.sample_output.mm_embeddings = {{torch::tensor({1.0f, 2.0f})},
                                        {torch::tensor({3.0f})}};
  typename TypeParam::Response encoded;
  ASSERT_TRUE(TypeParam::encode(source, {{1, 2}, {2, 2}}, &encoded));
  RawForwardOutput decoded;
  ASSERT_TRUE(TypeParam::decode(encoded, decoded));
  ASSERT_EQ(decoded.outputs.size(), 2);
  ASSERT_EQ(decoded.outputs[0].tokens.size(), 1);
  EXPECT_EQ(decoded.outputs[0].tokens[0].id, 7);
  ASSERT_EQ(decoded.outputs[1].tokens.size(), 2);
  const RawToken& token = decoded.outputs[1].tokens[1];
  EXPECT_EQ(token.id, 9);
  ASSERT_TRUE(token.logprob.has_value());
  EXPECT_FLOAT_EQ(*token.logprob, -1.5f);
  EXPECT_EQ(token.top_tokens, (std::vector<int64_t>{9}));
  EXPECT_EQ(token.top_logprobs, (std::vector<float>{-1.5f}));
  EXPECT_EQ(token.embeddings, (std::vector<float>{5.0f}));
  EXPECT_EQ(decoded.outputs[1].speculative_token_stats.accepted_tokens, 2);
  for (size_t output_idx = 0; output_idx < decoded.outputs.size();
       ++output_idx) {
    ASSERT_EQ(decoded.outputs[output_idx].mm_embeddings.size(), 1);
    EXPECT_TRUE(
        torch::equal(decoded.outputs[output_idx].mm_embeddings[0],
                     source.sample_output.mm_embeddings[output_idx][0]));
  }
}

TYPED_TEST(WorkerModelTokenCodecTest,
           ZeroSpeculativeRowsRejectEmbeddingRowsWithoutChangingDestination) {
  typename TypeParam::Response destination;
  destination.add_outputs()->add_tokens()->set_id(19);
  const std::string original = destination.SerializeAsString();
  ForwardOutput source;
  source.sample_output.next_tokens = torch::empty({0, 2}, torch::kInt64);
  source.sample_output.mm_embeddings = {{torch::tensor({1.0f})}};
  EXPECT_FALSE(TypeParam::encode(source, {}, &destination));
  EXPECT_EQ(destination.SerializeAsString(), original);

  source.sample_output.mm_embeddings.clear();
  source.sample_output.embeddings = torch::ones({1, 2, 1}, torch::kFloat32);
  EXPECT_FALSE(TypeParam::encode(source, {}, &destination));
  EXPECT_EQ(destination.SerializeAsString(), original);
}

TYPED_TEST(WorkerModelTokenCodecTest,
           EmptySpeculativeColumnsPreserveMultimodalEmbeddingRows) {
  ForwardOutput source;
  source.sample_output.next_tokens = torch::empty({1, 0}, torch::kInt64);
  source.sample_output.mm_embeddings = {{torch::tensor({1.0f, 2.0f})}};
  typename TypeParam::Response encoded;
  ASSERT_TRUE(TypeParam::encode(source, {}, &encoded));
  RawForwardOutput decoded;
  ASSERT_TRUE(TypeParam::decode(encoded, decoded));
  ASSERT_EQ(decoded.outputs.size(), 1);
  EXPECT_TRUE(decoded.outputs[0].tokens.empty());
  ASSERT_EQ(decoded.outputs[0].mm_embeddings.size(), 1);
  EXPECT_TRUE(torch::equal(decoded.outputs[0].mm_embeddings[0],
                           source.sample_output.mm_embeddings[0][0]));
}

TYPED_TEST(WorkerModelTokenCodecTest,
           EmbeddingOnlyAndEmptyOutputReplacePreviousPayloads) {
  ForwardOutput source;
  source.sample_output.embeddings = torch::tensor({{1.0f, 2.0f}});
  typename TypeParam::Response encoded;
  encoded.add_src_seq_idxes(0);
  encoded.add_out_tokens(7);
  encoded.add_out_logprobs(-0.5f);
  encoded.add_json_object_errors()->set_message("stale");
  ASSERT_TRUE(TypeParam::encode(source, {}, &encoded));
  RawForwardOutput decoded = original_output();
  ASSERT_TRUE(TypeParam::decode(encoded, decoded));
  ASSERT_EQ(decoded.outputs.size(), 1);
  ASSERT_EQ(decoded.outputs[0].tokens.size(), 1);
  EXPECT_EQ(decoded.outputs[0].tokens[0].id, -1);
  EXPECT_FALSE(decoded.outputs[0].tokens[0].logprob.has_value());
  EXPECT_EQ(decoded.outputs[0].tokens[0].embeddings,
            (std::vector<float>{1.0f, 2.0f}));
  EXPECT_TRUE(decoded.src_seq_idxes.empty());
  EXPECT_TRUE(decoded.out_tokens.empty());
  EXPECT_TRUE(decoded.out_logprobs.empty());
  EXPECT_TRUE(decoded.json_object_errors.empty());
  EXPECT_TRUE(decoded.dit_forward_output.text_output.empty());

  ASSERT_TRUE(TypeParam::encode(ForwardOutput{}, {}, &encoded));
  ASSERT_TRUE(TypeParam::decode(encoded, decoded));
  EXPECT_TRUE(decoded.outputs.empty());
}

TEST(WorkerModelDitCodecTest,
     TensorAudioAndTextResultsReplacePreviousPayloads) {
  ForwardOutput source;
  source.dit_forward_output.tensors = {torch::tensor({5.0f, 6.0f})};
  source.dit_forward_output.audio_tensors = {
      torch::tensor({{0.25f, 0.5f}, {0.75f, 1.0f}})};
  source.dit_forward_output.text_output = {"generated"};
  // Diffusion domains do not serialize token output.
  source.sample_output.next_tokens = torch::tensor({7.0f});
  source.json_object_errors = {{"unused", "unused"}};
  proto::DITForwardResponse encoded;
  ASSERT_TRUE(encode_dit_output(source, &encoded));
  RawForwardOutput decoded = original_output();
  ASSERT_TRUE(decode_dit_output(encoded, decoded));
  ASSERT_EQ(decoded.dit_forward_output.tensors.size(), 1);
  EXPECT_TRUE(torch::equal(decoded.dit_forward_output.tensors[0],
                           source.dit_forward_output.tensors[0]));
  ASSERT_EQ(decoded.dit_forward_output.audio_tensors.size(), 1);
  EXPECT_TRUE(torch::equal(decoded.dit_forward_output.audio_tensors[0],
                           source.dit_forward_output.audio_tensors[0]));
  EXPECT_EQ(decoded.dit_forward_output.text_output,
            (std::vector<std::string>{"generated"}));
  EXPECT_TRUE(decoded.outputs.empty());
  EXPECT_TRUE(decoded.json_object_errors.empty());

  source.dit_forward_output.tensors.clear();
  source.dit_forward_output.text_output.clear();
  ASSERT_TRUE(encode_dit_output(source, &encoded));
  ASSERT_TRUE(decode_dit_output(encoded, decoded));
  EXPECT_TRUE(decoded.dit_forward_output.tensors.empty());
  EXPECT_TRUE(decoded.dit_forward_output.text_output.empty());
  ASSERT_EQ(decoded.dit_forward_output.audio_tensors.size(), 1);
  EXPECT_TRUE(torch::equal(decoded.dit_forward_output.audio_tensors[0],
                           source.dit_forward_output.audio_tensors[0]));

  source.dit_forward_output.audio_tensors.clear();
  source.dit_forward_output.text_output = {"text only"};
  ASSERT_TRUE(encode_dit_output(source, &encoded));
  ASSERT_TRUE(decode_dit_output(encoded, decoded));
  EXPECT_TRUE(decoded.dit_forward_output.tensors.empty());
  EXPECT_TRUE(decoded.dit_forward_output.audio_tensors.empty());
  EXPECT_EQ(decoded.dit_forward_output.text_output,
            (std::vector<std::string>{"text only"}));

  ASSERT_TRUE(encode_dit_output(ForwardOutput{}, &encoded));
  ASSERT_TRUE(decode_dit_output(encoded, decoded));
  EXPECT_TRUE(decoded.dit_forward_output.tensors.empty());
  EXPECT_TRUE(decoded.dit_forward_output.audio_tensors.empty());
  EXPECT_TRUE(decoded.dit_forward_output.text_output.empty());
}

TEST(WorkerModelDitCodecTest, InvalidTensorEncodePreservesDestination) {
  proto::DITForwardResponse destination;
  destination.mutable_dit_forward_output()->add_text_output("original");
  const std::string original = destination.SerializeAsString();
  ForwardOutput source;
  source.dit_forward_output.tensors = {torch::Tensor()};
  EXPECT_FALSE(encode_dit_output(source, &destination));
  EXPECT_EQ(destination.SerializeAsString(), original);
  source.dit_forward_output.tensors = {
      torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}}).transpose(0, 1)};
  EXPECT_FALSE(encode_dit_output(source, &destination));
  EXPECT_EQ(destination.SerializeAsString(), original);
  EXPECT_FALSE(encode_dit_output(source, nullptr));
}

TEST(WorkerModelDitCodecTest, InvalidTensorDecodePreservesDestination) {
  proto::DITForwardResponse source;
  source.mutable_dit_forward_output()->mutable_tensors()->add_tensors();
  RawForwardOutput destination = original_output();
  EXPECT_FALSE(decode_dit_output(source, destination));
  expect_original_output(destination);
}

TEST(WorkerModelDitCodecTest, InvalidAudioTensorEncodePreservesDestination) {
  ForwardOutput source;
  source.dit_forward_output.tensors = {torch::tensor({5.0f, 6.0f})};
  source.dit_forward_output.audio_tensors = {torch::tensor({0.25f, 0.5f})};
  proto::DITForwardResponse destination;
  ASSERT_TRUE(encode_dit_output(source, &destination));
  const std::string original = destination.SerializeAsString();

  source.dit_forward_output.audio_tensors = {torch::Tensor()};
  EXPECT_FALSE(encode_dit_output(source, &destination));
  EXPECT_EQ(destination.SerializeAsString(), original);
  source.dit_forward_output.audio_tensors = {
      torch::tensor({{0.25f, 0.5f}, {0.75f, 1.0f}}).transpose(0, 1)};
  EXPECT_FALSE(encode_dit_output(source, &destination));
  EXPECT_EQ(destination.SerializeAsString(), original);
}

TEST(WorkerModelDitCodecTest, InvalidAudioTensorDecodePreservesDestination) {
  ForwardOutput source;
  source.dit_forward_output.tensors = {torch::tensor({5.0f, 6.0f})};
  source.dit_forward_output.audio_tensors = {torch::tensor({0.25f, 0.5f})};
  proto::DITForwardResponse encoded;
  ASSERT_TRUE(encode_dit_output(source, &encoded));
  encoded.mutable_dit_forward_output()->mutable_audio_tensors()->add_tensors();

  RawForwardOutput destination = original_output();
  const torch::Tensor original_audio = torch::tensor({0.75f, 1.0f});
  destination.dit_forward_output.audio_tensors = {original_audio};
  EXPECT_FALSE(decode_dit_output(encoded, destination));
  expect_original_output(destination);
  ASSERT_EQ(destination.dit_forward_output.audio_tensors.size(), 1);
  EXPECT_TRUE(torch::equal(destination.dit_forward_output.audio_tensors[0],
                           original_audio));
}

}  // namespace
}  // namespace xllm
