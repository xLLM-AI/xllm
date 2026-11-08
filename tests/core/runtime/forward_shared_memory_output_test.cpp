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

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/config/eplb_config.h"
#include "core/framework/config/execution_config.h"
#include "core/runtime/forward_shared_memory_transport.h"

namespace xllm {
namespace {

constexpr std::array kTokenOutputMethods = {
    std::pair{&ForwardSharedMemoryTransport::llm_output_write,
              &ForwardSharedMemoryTransport::llm_output_read},
    std::pair{&ForwardSharedMemoryTransport::vlm_output_write,
              &ForwardSharedMemoryTransport::vlm_output_read},
    std::pair{&ForwardSharedMemoryTransport::rec_output_write,
              &ForwardSharedMemoryTransport::rec_output_read}};

template <typename Value>
void append_wire_value(std::string& frame, const Value& value) {
  frame.append(reinterpret_cast<const char*>(&value), sizeof(value));
}

std::string make_metadata_frame() {
  std::string frame;
  append_wire_value(frame, uint64_t{1});
  const std::string name = "token_tags";
  append_wire_value(frame, static_cast<uint64_t>(name.size()));
  frame.append(name);
  append_wire_value(frame, uint64_t{1});  // tensor rank
  append_wire_value(frame, uint64_t{2});  // tensor dimension
  append_wire_value(frame, static_cast<int8_t>(torch::kInt64));
  append_wire_value(frame, uint64_t{2 * sizeof(int64_t)});
  append_wire_value(frame, int64_t{0});
  append_wire_value(frame, int64_t{1});
  return frame;
}

class ForwardSharedMemoryOutputTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto& execution_config = ExecutionConfig::get_instance();
    previous_contiguous_ = execution_config.use_contiguous_input_buffer();
    execution_config.use_contiguous_input_buffer(false);
    previous_eplb_ = EPLBConfig::get_instance().enable_eplb();
    EPLBConfig::get_instance().enable_eplb(false);
    const std::string prefix =
        "forward_output_" + std::to_string(getpid()) + "_" +
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    const std::string name = ForwardSharedMemoryTransport::create_unique_name(
        prefix, /*dp_group=*/0, ForwardType::RAW_OUTPUT, /*rank=*/0);
    bool is_creator = false;
    writer_ = std::make_unique<ForwardSharedMemoryTransport>(
        name, kSegmentBytes, is_creator, ForwardType::RAW_OUTPUT);
    ASSERT_TRUE(is_creator);
    reader_ = std::make_unique<ForwardSharedMemoryTransport>(
        name, kSegmentBytes, is_creator, ForwardType::RAW_OUTPUT);
    ASSERT_FALSE(is_creator);
  }

  void TearDown() override {
    reader_.reset();
    writer_.reset();
    ExecutionConfig::get_instance().use_contiguous_input_buffer(
        previous_contiguous_);
    EPLBConfig::get_instance().enable_eplb(previous_eplb_);
  }

  uint64_t published_version() const {
    return static_cast<const ControlMetadata*>(writer_->base_address())
        ->version;
  }

  void replace_empty_token_metadata(const std::string& metadata_frame) {
    ASSERT_GE(metadata_frame.size(), sizeof(uint64_t));
    ASSERT_LE(kTokenMetadataOffset + metadata_frame.size(), kSegmentBytes);
    char* metadata =
        static_cast<char*>(writer_->base_address()) + kTokenMetadataOffset;
    std::memmove(metadata + metadata_frame.size(),
                 metadata + sizeof(uint64_t),
                 kSegmentBytes - kTokenMetadataOffset - metadata_frame.size());
    std::memcpy(metadata, metadata_frame.data(), metadata_frame.size());
  }

  // One ordinary token without probabilities, embeddings, or MM tensors.
  static constexpr size_t kTokenMetadataOffset =
      sizeof(ControlMetadata) + 5 * sizeof(uint64_t) + sizeof(int64_t) +
      sizeof(bool) + sizeof(int32_t);
  static constexpr size_t kSegmentBytes = 4096;
  bool previous_contiguous_ = false;
  bool previous_eplb_ = false;
  std::unique_ptr<ForwardSharedMemoryTransport> writer_;
  std::unique_ptr<ForwardSharedMemoryTransport> reader_;
};

TEST_F(ForwardSharedMemoryOutputTest, LlmPreservesDsparkHostFeedback) {
  ForwardOutput source;
  source.sample_output.next_tokens = torch::tensor({7, 9}, torch::kInt64);
  source.dspark_feedback.confidence.assign(14, 0.5F);
  source.dspark_feedback.rounds_since_full_width = {3, 4};
  ASSERT_TRUE(
      writer_->llm_output_write(source, /*speculative_token_stats=*/{}));
  RawForwardOutput decoded;
  ASSERT_TRUE(reader_->llm_output_read(decoded));
  EXPECT_EQ(decoded.dspark_feedback.confidence,
            source.dspark_feedback.confidence);
  EXPECT_EQ(decoded.dspark_feedback.rounds_since_full_width,
            source.dspark_feedback.rounds_since_full_width);
  const uint64_t version = published_version();
  source.dspark_feedback.confidence.pop_back();
  EXPECT_FALSE(
      writer_->llm_output_write(source, /*speculative_token_stats=*/{}));
  EXPECT_EQ(published_version(), version);
}

TEST_F(ForwardSharedMemoryOutputTest,
       TokenDomainsKeepDistinctDsparkFeedbackLayouts) {
  ForwardOutput source;
  source.sample_output.next_tokens = torch::tensor({7}, torch::kInt64);
  source.dspark_feedback.confidence.assign(7, 0.5F);
  source.dspark_feedback.rounds_since_full_width = {3};
  ASSERT_TRUE(writer_->llm_output_write(source, {}));
  RawForwardOutput decoded;
  ASSERT_TRUE(reader_->llm_output_read(decoded));
  EXPECT_EQ(decoded.dspark_feedback.confidence,
            source.dspark_feedback.confidence);

  // Only the LLM wire format carries and validates adaptive DSpark feedback.
  source.dspark_feedback.confidence.pop_back();
  const uint64_t version = published_version();
  EXPECT_FALSE(writer_->llm_output_write(source, {}));
  EXPECT_EQ(published_version(), version);
  const std::array non_llm_methods = {
      std::pair{&ForwardSharedMemoryTransport::vlm_output_write,
                &ForwardSharedMemoryTransport::vlm_output_read},
      std::pair{&ForwardSharedMemoryTransport::rec_output_write,
                &ForwardSharedMemoryTransport::rec_output_read}};
  for (const auto& [write, read] : non_llm_methods) {
    ASSERT_TRUE((writer_.get()->*write)(source, {}));
    ASSERT_TRUE((reader_.get()->*read)(decoded));
    ASSERT_EQ(decoded.outputs.size(), 1);
    ASSERT_EQ(decoded.outputs[0].tokens.size(), 1);
    EXPECT_EQ(decoded.outputs[0].tokens[0].id, 7);
    EXPECT_TRUE(decoded.dspark_feedback.confidence.empty());
    EXPECT_TRUE(decoded.dspark_feedback.rounds_since_full_width.empty());
  }
}

TEST_F(ForwardSharedMemoryOutputTest,
       LlmPreservesSpeculativeTokensAndJsonErrors) {
  ForwardOutput source;
  source.sample_output.next_tokens =
      torch::tensor({{10, 11, -1}, {20, 21, 22}}, torch::kInt64);
  source.sample_output.logprobs =
      torch::tensor({{-0.1f, -0.2f, -0.3f}, {-0.4f, -0.5f, -0.6f}});
  source.sample_output.embeddings =
      torch::arange(12, torch::kFloat32).reshape({2, 3, 2});
  source.json_object_errors = {{"request#0", "invalid prior token"}};
  ASSERT_TRUE(writer_->llm_output_write(source, {{1, 3}, {2, 3}}));

  RawForwardOutput output;
  ASSERT_TRUE(reader_->llm_output_read(output));
  ASSERT_EQ(output.outputs.size(), 2);
  ASSERT_EQ(output.outputs[0].tokens.size(), 2);
  EXPECT_EQ(output.outputs[0].tokens[1].id, 11);
  EXPECT_FLOAT_EQ(output.outputs[0].tokens[1].logprob.value(), -0.2f);
  EXPECT_EQ(output.outputs[0].tokens[1].embeddings,
            (std::vector<float>{2.0f, 3.0f}));
  EXPECT_EQ(output.outputs[1].tokens.size(), 3);
  EXPECT_EQ(output.outputs[0].speculative_token_stats.accepted_tokens, 1);
  EXPECT_EQ(output.outputs[1].speculative_token_stats.proposed_tokens, 3);
  ASSERT_EQ(output.json_object_errors.size(), 1);
  EXPECT_EQ(output.json_object_errors[0].sample_sequence_id, "request#0");
  EXPECT_EQ(output.json_object_errors[0].message, "invalid prior token");
}

TEST_F(ForwardSharedMemoryOutputTest,
       VlmPreservesMultimodalOutputAndTopTokens) {
  ForwardOutput source;
  source.sample_output.next_tokens = torch::tensor({7}, torch::kInt32);
  source.sample_output.logprobs = torch::tensor({-0.5f});
  source.sample_output.top_tokens = torch::tensor({{7, 9}}, torch::kInt64);
  source.sample_output.top_logprobs = torch::tensor({{-0.5f, -0.8f}});
  source.sample_output.mm_embeddings = {{torch::tensor({1.0f, 2.0f})}};
  ASSERT_TRUE(writer_->vlm_output_write(source, {}));

  RawForwardOutput output;
  ASSERT_TRUE(reader_->vlm_output_read(output));
  ASSERT_EQ(output.outputs.size(), 1);
  ASSERT_EQ(output.outputs[0].tokens.size(), 1);
  EXPECT_EQ(output.outputs[0].tokens[0].id, 7);
  EXPECT_EQ(output.outputs[0].tokens[0].top_tokens,
            (std::vector<int64_t>{7, 9}));
  EXPECT_EQ(output.outputs[0].tokens[0].top_logprobs,
            (std::vector<float>{-0.5f, -0.8f}));
  ASSERT_EQ(output.outputs[0].mm_embeddings.size(), 1);
  EXPECT_TRUE(torch::equal(output.outputs[0].mm_embeddings[0],
                           source.sample_output.mm_embeddings[0][0]));
}

TEST_F(ForwardSharedMemoryOutputTest, RecPreservesBeamOutput) {
  ForwardOutput source;
  source.sample_output.next_tokens = torch::tensor({7, 8}, torch::kInt32);
  source.beam_search_output.src_seq_idxes =
      torch::tensor({1, 0}, torch::kInt32);
  source.beam_search_output.out_tokens = torch::tensor({8, 7}, torch::kInt32);
  source.beam_search_output.out_logprobs = torch::tensor({-0.2f, -0.3f});
  ASSERT_TRUE(writer_->rec_output_write(source, {}));

  RawForwardOutput output;
  ASSERT_TRUE(reader_->rec_output_read(output));
  EXPECT_EQ(output.src_seq_idxes, (std::vector<int32_t>{1, 0}));
  EXPECT_EQ(output.out_tokens, (std::vector<int32_t>{8, 7}));
  EXPECT_EQ(output.out_logprobs, (std::vector<float>{-0.2f, -0.3f}));
  ASSERT_EQ(output.outputs.size(), 2);
  EXPECT_EQ(output.outputs[1].tokens[0].id, 8);
}

TEST_F(ForwardSharedMemoryOutputTest,
       TokenDomainsPreserveCombinedOutputAfterSourceMutation) {
  EPLBConfig::get_instance().enable_eplb(true);
  for (const auto& [write, read] : kTokenOutputMethods) {
    ForwardOutput source;
    source.sample_output.next_tokens =
        torch::tensor({{10, -1, 99}, {20, 21, 22}}, torch::kInt64);
    source.sample_output.logprobs =
        torch::tensor({{-0.1f, -0.2f, -0.3f}, {-0.4f, -0.5f, -0.6f}});
    source.sample_output.top_tokens =
        torch::arange(12, torch::kInt64).reshape({2, 3, 2});
    source.sample_output.top_logprobs =
        torch::arange(12, torch::kFloat32).reshape({2, 3, 2}) * -0.1f;
    source.sample_output.embeddings =
        torch::arange(18, torch::kFloat32).reshape({2, 3, 3});
    source.sample_output.mm_embeddings = {
        {torch::tensor({1.0f, 2.0f}), torch::empty({0, 3})}};
    source.expert_load_data = torch::tensor({{1, 2}, {3, 4}}, torch::kInt64);
    source.beam_search_output.src_seq_idxes =
        torch::tensor({1, 0}, torch::kInt32);
    source.beam_search_output.out_tokens = torch::tensor({8, 7}, torch::kInt32);
    source.beam_search_output.out_logprobs = torch::tensor({-0.2f, -0.3f});
    source.json_object_errors = {{"request#0", "invalid prior token"}};
    source.prepared_token = 45;
    ASSERT_TRUE((writer_.get()->*write)(source, {{0, 3}, {2, 3}}));

    source.sample_output.next_tokens.fill_(-1);
    source.sample_output.top_tokens.zero_();
    source.sample_output.top_logprobs.zero_();
    source.sample_output.embeddings.zero_();
    source.sample_output.mm_embeddings.clear();
    source.expert_load_data.zero_();
    source.beam_search_output.out_tokens.zero_();
    source.json_object_errors.clear();

    RawForwardOutput output;
    ASSERT_TRUE((reader_.get()->*read)(output));
    ASSERT_EQ(output.outputs.size(), 2);
    ASSERT_EQ(output.outputs[0].tokens.size(), 1);
    EXPECT_EQ(output.outputs[0].tokens[0].id, 10);
    ASSERT_EQ(output.outputs[1].tokens.size(), 3);
    const RawToken& token = output.outputs[1].tokens[1];
    EXPECT_EQ(token.id, 21);
    EXPECT_FLOAT_EQ(token.logprob.value(), -0.5f);
    EXPECT_EQ(token.top_tokens, (std::vector<int64_t>{8, 9}));
    ASSERT_EQ(token.top_logprobs.size(), 2);
    EXPECT_FLOAT_EQ(token.top_logprobs[0], -0.8f);
    EXPECT_FLOAT_EQ(token.top_logprobs[1], -0.9f);
    EXPECT_EQ(token.embeddings, (std::vector<float>{12.0f, 13.0f, 14.0f}));
    ASSERT_EQ(output.outputs[0].mm_embeddings.size(), 2);
    EXPECT_TRUE(torch::equal(output.outputs[0].mm_embeddings[0],
                             torch::tensor({1.0f, 2.0f})));
    EXPECT_EQ(output.outputs[0].mm_embeddings[1].sizes().vec(),
              (std::vector<int64_t>{0, 3}));
    EXPECT_TRUE(output.outputs[1].mm_embeddings.empty());
    EXPECT_EQ(output.outputs[1].speculative_token_stats.accepted_tokens, 2);
    EXPECT_EQ(output.outputs[1].speculative_token_stats.proposed_tokens, 3);
    EXPECT_EQ(output.expert_load_data, (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(output.src_seq_idxes, (std::vector<int32_t>{1, 0}));
    EXPECT_EQ(output.out_tokens, (std::vector<int32_t>{8, 7}));
    EXPECT_EQ(output.out_logprobs, (std::vector<float>{-0.2f, -0.3f}));
    ASSERT_EQ(output.json_object_errors.size(), 1);
    EXPECT_EQ(output.json_object_errors[0].message, "invalid prior token");
    EXPECT_EQ(output.prepared_token, 45);
  }
}

TEST_F(ForwardSharedMemoryOutputTest,
       OversizedTokenFieldsPreserveEveryPublishedPayloadByte) {
  EPLBConfig::get_instance().enable_eplb(true);
  ForwardOutput source;
  source.sample_output.next_tokens = torch::tensor({7}, torch::kInt32);
  ASSERT_TRUE(writer_->llm_output_write(source, {}));
  const std::string published(static_cast<const char*>(writer_->base_address()),
                              kSegmentBytes);
  for (const auto& methods : kTokenOutputMethods) {
    const auto write = methods.first;
    ForwardOutput overflow;
    overflow.sample_output.embeddings =
        torch::ones({1, static_cast<int64_t>(kSegmentBytes)}, torch::kFloat32);
    EXPECT_FALSE((writer_.get()->*write)(overflow, {}));
    EXPECT_EQ(
        std::memcmp(writer_->base_address(), published.data(), kSegmentBytes),
        0);

    overflow = ForwardOutput();
    overflow.sample_output.next_tokens = torch::tensor({7}, torch::kInt32);
    overflow.sample_output.logprobs = torch::tensor({-0.5f});
    overflow.sample_output.top_tokens =
        torch::zeros({1, static_cast<int64_t>(kSegmentBytes)}, torch::kInt64);
    overflow.sample_output.top_logprobs =
        torch::zeros({1, static_cast<int64_t>(kSegmentBytes)}, torch::kFloat32);
    EXPECT_FALSE((writer_.get()->*write)(overflow, {}));
    EXPECT_EQ(
        std::memcmp(writer_->base_address(), published.data(), kSegmentBytes),
        0);

    overflow = ForwardOutput();
    overflow.expert_load_data =
        torch::ones({static_cast<int64_t>(kSegmentBytes)}, torch::kInt64);
    EXPECT_FALSE((writer_.get()->*write)(overflow, {}));
    EXPECT_EQ(
        std::memcmp(writer_->base_address(), published.data(), kSegmentBytes),
        0);

    overflow = ForwardOutput();
    overflow.beam_search_output.src_seq_idxes =
        torch::zeros({static_cast<int64_t>(kSegmentBytes)}, torch::kInt32);
    overflow.beam_search_output.out_tokens =
        torch::zeros({static_cast<int64_t>(kSegmentBytes)}, torch::kInt32);
    overflow.beam_search_output.out_logprobs =
        torch::zeros({static_cast<int64_t>(kSegmentBytes)}, torch::kFloat32);
    EXPECT_FALSE((writer_.get()->*write)(overflow, {}));
    EXPECT_EQ(
        std::memcmp(writer_->base_address(), published.data(), kSegmentBytes),
        0);
  }
  RawForwardOutput output;
  ASSERT_TRUE(reader_->llm_output_read(output));
  ASSERT_EQ(output.outputs.size(), 1);
  EXPECT_EQ(output.outputs[0].tokens[0].id, 7);
}

TEST_F(ForwardSharedMemoryOutputTest,
       TokenDomainsReadOwnedMetadataWithCompatibleRowFraming) {
  RawForwardOutput output;
  const torch::Tensor expected_tags = torch::tensor({0, 1}, torch::kInt64);
  for (const auto& [write, read] : kTokenOutputMethods) {
    ForwardOutput source;
    source.sample_output.next_tokens = torch::tensor({7}, torch::kInt32);
    source.prepared_token = 45;
    ASSERT_TRUE((writer_.get()->*write)(source, {{2, 3}}));
    uint64_t metadata_count = 1;
    std::memcpy(&metadata_count,
                static_cast<const char*>(writer_->base_address()) +
                    kTokenMetadataOffset,
                sizeof(metadata_count));
    EXPECT_EQ(metadata_count, 0);
    replace_empty_token_metadata(make_metadata_frame());
    ASSERT_TRUE((reader_.get()->*read)(output));
    ASSERT_EQ(output.outputs.size(), 1);
    EXPECT_EQ(output.outputs[0].tokens[0].id, 7);
    ASSERT_EQ(output.outputs[0].mm_embeddings_metadata.size(), 1);
    EXPECT_TRUE(
        torch::equal(output.outputs[0].mm_embeddings_metadata.at("token_tags"),
                     expected_tags));
    EXPECT_EQ(output.outputs[0].speculative_token_stats.accepted_tokens, 2);
    EXPECT_EQ(output.outputs[0].speculative_token_stats.proposed_tokens, 3);
    EXPECT_EQ(output.prepared_token, 45);

    ASSERT_TRUE((writer_.get()->*write)(ForwardOutput(), {}));
    RawForwardOutput empty;
    ASSERT_TRUE((reader_.get()->*read)(empty));
    EXPECT_TRUE(empty.outputs.empty());
    EXPECT_TRUE(
        torch::equal(output.outputs[0].mm_embeddings_metadata.at("token_tags"),
                     expected_tags));
  }
  reader_.reset();
  writer_.reset();
  EXPECT_TRUE(
      torch::equal(output.outputs[0].mm_embeddings_metadata.at("token_tags"),
                   expected_tags));
}

TEST_F(ForwardSharedMemoryOutputTest,
       MalformedMetadataFieldsFailWithoutMutatingOutput) {
  const std::array<size_t, 3> malformed_offsets = {
      0, sizeof(uint64_t), make_metadata_frame().size() - 3 * sizeof(uint64_t)};
  for (const auto& [write, read] : kTokenOutputMethods) {
    for (size_t offset : malformed_offsets) {
      ForwardOutput source;
      source.sample_output.next_tokens = torch::tensor({7}, torch::kInt32);
      ASSERT_TRUE((writer_.get()->*write)(source, {}));
      std::string metadata_frame = make_metadata_frame();
      const uint64_t invalid_size = std::numeric_limits<uint64_t>::max();
      std::memcpy(
          metadata_frame.data() + offset, &invalid_size, sizeof(invalid_size));
      replace_empty_token_metadata(metadata_frame);
      RawForwardOutput output;
      output.prepared_token = 81;
      EXPECT_FALSE((reader_.get()->*read)(output));
      EXPECT_TRUE(output.outputs.empty());
      EXPECT_EQ(output.prepared_token, 81);
    }
  }
}

TEST_F(ForwardSharedMemoryOutputTest,
       MetadataFramingIsIncludedInTokenCapacityChecks) {
  // This embedding fits only when the required metadata count is omitted.
  constexpr size_t kFixedFrameBytes = 11 * sizeof(uint64_t) +
                                      4 * sizeof(int64_t) + 2 * sizeof(bool) +
                                      sizeof(int32_t);
  const int64_t embedding_width = static_cast<int64_t>(
      (kSegmentBytes - sizeof(ControlMetadata) - kFixedFrameBytes) /
          sizeof(float) +
      1);
  for (const auto& methods : kTokenOutputMethods) {
    const auto write = methods.first;
    ForwardOutput valid;
    valid.sample_output.next_tokens = torch::tensor({7}, torch::kInt32);
    ASSERT_TRUE((writer_.get()->*write)(valid, {}));
    const std::string published(
        static_cast<const char*>(writer_->base_address()), kSegmentBytes);
    ForwardOutput overflow;
    overflow.sample_output.embeddings =
        torch::ones({1, embedding_width}, torch::kFloat32);
    EXPECT_FALSE((writer_.get()->*write)(overflow, {}));
    EXPECT_EQ(
        std::memcmp(writer_->base_address(), published.data(), kSegmentBytes),
        0);
  }
}

TEST_F(ForwardSharedMemoryOutputTest,
       TokenDomainsRejectAudioWithoutPublishing) {
  ForwardOutput source;
  source.sample_output.next_tokens = torch::tensor({7}, torch::kInt32);
  ASSERT_TRUE(writer_->llm_output_write(source, {}));
  const std::string published(static_cast<const char*>(writer_->base_address()),
                              kSegmentBytes);
  source.dit_forward_output.audio_tensors = {torch::ones({2})};
  for (const auto& methods : kTokenOutputMethods) {
    EXPECT_FALSE((writer_.get()->*methods.first)(source, {}));
    EXPECT_EQ(
        std::memcmp(writer_->base_address(), published.data(), kSegmentBytes),
        0);
  }
}

TEST_F(ForwardSharedMemoryOutputTest,
       VlmPreservesSpeculativeTokensWithMultimodalOutput) {
  ForwardOutput source;
  source.sample_output.next_tokens = torch::tensor({{7, 8, -1}}, torch::kInt64);
  source.sample_output.mm_embeddings = {{torch::tensor({1.0f, 2.0f})}};
  ASSERT_TRUE(writer_->vlm_output_write(source, {{1, 3}}));
  RawForwardOutput output;
  ASSERT_TRUE(reader_->vlm_output_read(output));
  ASSERT_EQ(output.outputs.size(), 1);
  ASSERT_EQ(output.outputs[0].tokens.size(), 2);
  EXPECT_EQ(output.outputs[0].tokens[1].id, 8);
  EXPECT_EQ(output.outputs[0].speculative_token_stats.proposed_tokens, 3);
  ASSERT_EQ(output.outputs[0].mm_embeddings.size(), 1);
  EXPECT_TRUE(torch::equal(output.outputs[0].mm_embeddings[0],
                           source.sample_output.mm_embeddings[0][0]));
}

TEST_F(ForwardSharedMemoryOutputTest,
       VlmEmptySpeculativeColumnsPreserveMultimodalRows) {
  ForwardOutput source;
  source.sample_output.next_tokens = torch::empty({1, 0}, torch::kInt64);
  source.sample_output.mm_embeddings = {{torch::tensor({1.0f, 2.0f})}};
  ASSERT_TRUE(writer_->vlm_output_write(source, {}));
  RawForwardOutput output;
  ASSERT_TRUE(reader_->vlm_output_read(output));
  ASSERT_EQ(output.outputs.size(), 1);
  EXPECT_TRUE(output.outputs[0].tokens.empty());
  ASSERT_EQ(output.outputs[0].mm_embeddings.size(), 1);
  EXPECT_TRUE(torch::equal(output.outputs[0].mm_embeddings[0],
                           source.sample_output.mm_embeddings[0][0]));
}

TEST_F(ForwardSharedMemoryOutputTest,
       VlmEmptyTokenVectorPreservesMultimodalRows) {
  ForwardOutput source;
  source.sample_output.next_tokens = torch::empty({0}, torch::kInt32);
  source.sample_output.mm_embeddings = {{torch::tensor({1.0f, 2.0f})}};
  ASSERT_TRUE(writer_->vlm_output_write(source, {}));
  RawForwardOutput output;
  ASSERT_TRUE(reader_->vlm_output_read(output));
  ASSERT_EQ(output.outputs.size(), 1);
  ASSERT_EQ(output.outputs[0].tokens.size(), 1);
  EXPECT_EQ(output.outputs[0].tokens[0].id, -1);
  ASSERT_EQ(output.outputs[0].mm_embeddings.size(), 1);
  EXPECT_TRUE(torch::equal(output.outputs[0].mm_embeddings[0],
                           source.sample_output.mm_embeddings[0][0]));
}

TEST_F(ForwardSharedMemoryOutputTest,
       ZeroSpeculativeRowsRejectMultimodalRowsWithoutPublishing) {
  ForwardOutput source;
  source.sample_output.next_tokens = torch::empty({0, 2}, torch::kInt64);
  source.sample_output.mm_embeddings = {{torch::tensor({1.0f})}};
  EXPECT_FALSE(writer_->llm_output_write(source, {}));
  EXPECT_FALSE(writer_->vlm_output_write(source, {}));
  EXPECT_FALSE(writer_->rec_output_write(source, {}));
  EXPECT_EQ(published_version(), 0);
}

TEST_F(ForwardSharedMemoryOutputTest, RecBeamOnlyOutputNeedsNoSampleRows) {
  ForwardOutput source;
  source.beam_search_output.src_seq_idxes =
      torch::tensor({1, 0}, torch::kInt32);
  source.beam_search_output.out_tokens = torch::tensor({8, 7}, torch::kInt32);
  source.beam_search_output.out_logprobs = torch::tensor({-0.2f, -0.3f});
  ASSERT_TRUE(writer_->rec_output_write(source, {}));
  RawForwardOutput output;
  ASSERT_TRUE(reader_->rec_output_read(output));
  EXPECT_TRUE(output.outputs.empty());
  EXPECT_EQ(output.src_seq_idxes, (std::vector<int32_t>{1, 0}));
  EXPECT_EQ(output.out_tokens, (std::vector<int32_t>{8, 7}));
  EXPECT_EQ(output.out_logprobs, (std::vector<float>{-0.2f, -0.3f}));
}

TEST_F(ForwardSharedMemoryOutputTest, EmptyTokenOutputClearsPreviousPayload) {
  ForwardOutput source;
  source.sample_output.next_tokens = torch::tensor({7}, torch::kInt32);
  ASSERT_TRUE(writer_->llm_output_write(source, {}));
  RawForwardOutput output;
  ASSERT_TRUE(reader_->llm_output_read(output));
  ASSERT_FALSE(output.outputs.empty());
  ASSERT_TRUE(writer_->llm_output_write(ForwardOutput(), {}));
  ASSERT_TRUE(reader_->llm_output_read(output));
  EXPECT_TRUE(output.outputs.empty());
  EXPECT_TRUE(output.json_object_errors.empty());
  EXPECT_TRUE(output.expert_load_data.empty());
  EXPECT_EQ(output.prepared_token, -1);
}

TEST_F(ForwardSharedMemoryOutputTest, DiTPreservesTextOnlyOutput) {
  DiTForwardOutput source;
  source.text_output = {"diffusion text", ""};
  ASSERT_TRUE(writer_->dit_output_write(source));

  RawForwardOutput output;
  ASSERT_TRUE(reader_->dit_output_read(output));
  EXPECT_TRUE(output.outputs.empty());
  EXPECT_TRUE(output.dit_forward_output.tensors.empty());
  EXPECT_EQ(output.dit_forward_output.text_output, source.text_output);
}

TEST_F(ForwardSharedMemoryOutputTest,
       DiTPreservesAudioOnlyOutputAfterSourceMutation) {
  DiTForwardOutput source;
  source.audio_tensors = {torch::tensor({0.25f, -0.5f})};
  ASSERT_TRUE(writer_->dit_output_write(source));
  source.audio_tensors[0].zero_();
  RawForwardOutput output;
  ASSERT_TRUE(reader_->dit_output_read(output));
  EXPECT_TRUE(output.dit_forward_output.tensors.empty());
  EXPECT_TRUE(output.dit_forward_output.text_output.empty());
  ASSERT_EQ(output.dit_forward_output.audio_tensors.size(), 1);
  EXPECT_TRUE(torch::equal(output.dit_forward_output.audio_tensors[0],
                           torch::tensor({0.25f, -0.5f})));
}

TEST_F(ForwardSharedMemoryOutputTest,
       LlmAllowsMissingExpertDataWhenEplbIsEnabled) {
  EPLBConfig::get_instance().enable_eplb(true);
  ForwardOutput source;
  source.sample_output.next_tokens = torch::tensor({7}, torch::kInt32);
  source.prepared_token = 81;
  ASSERT_TRUE(writer_->llm_output_write(source, {}));

  RawForwardOutput output;
  ASSERT_TRUE(reader_->llm_output_read(output));
  EXPECT_TRUE(output.expert_load_data.empty());
  EXPECT_EQ(output.prepared_token, 81);
}

TEST_F(ForwardSharedMemoryOutputTest, VlmPreservesEplbExpertLoads) {
  EPLBConfig::get_instance().enable_eplb(true);
  ForwardOutput source;
  source.expert_load_data = torch::tensor({{1, 2}, {3, 4}}, torch::kInt64);
  source.prepared_token = 19;
  ASSERT_TRUE(writer_->vlm_output_write(source, {}));

  RawForwardOutput output;
  ASSERT_TRUE(reader_->vlm_output_read(output));
  EXPECT_EQ(output.expert_load_data, (std::vector<int64_t>{1, 2, 3, 4}));
  EXPECT_EQ(output.prepared_token, 19);
}

TEST_F(ForwardSharedMemoryOutputTest, TokenReadsClearPreviousDiTAndBeamData) {
  ForwardOutput source;
  source.sample_output.next_tokens = torch::tensor({7}, torch::kInt32);
  ASSERT_TRUE(writer_->llm_output_write(source, {}));
  RawForwardOutput output;
  output.dit_forward_output.text_output = {"old text"};
  output.dit_forward_output.tensors = {torch::ones({2})};
  output.src_seq_idxes = {1};
  output.out_tokens = {3};
  output.out_logprobs = {-0.5f};
  output.json_object_errors = {{"old", "old error"}};
  ASSERT_TRUE(reader_->llm_output_read(output));
  EXPECT_TRUE(output.dit_forward_output.text_output.empty());
  EXPECT_TRUE(output.dit_forward_output.tensors.empty());
  EXPECT_TRUE(output.src_seq_idxes.empty());
  EXPECT_TRUE(output.out_tokens.empty());
  EXPECT_TRUE(output.out_logprobs.empty());
  EXPECT_TRUE(output.json_object_errors.empty());
}

TEST_F(ForwardSharedMemoryOutputTest, EmptyDiTReadClearsPreviousOutput) {
  DiTForwardOutput source;
  source.text_output = {"first output"};
  source.tensors = {torch::ones({2})};
  source.audio_tensors = {torch::ones({3})};
  ASSERT_TRUE(writer_->dit_output_write(source));
  RawForwardOutput output;
  ASSERT_TRUE(reader_->dit_output_read(output));
  ASSERT_FALSE(output.dit_forward_output.tensors.empty());
  ASSERT_TRUE(writer_->dit_output_write(DiTForwardOutput()));
  ASSERT_TRUE(reader_->dit_output_read(output));
  EXPECT_TRUE(output.dit_forward_output.text_output.empty());
  EXPECT_TRUE(output.dit_forward_output.tensors.empty());
  EXPECT_TRUE(output.dit_forward_output.audio_tensors.empty());
}

TEST_F(ForwardSharedMemoryOutputTest,
       DiTTensorsSurviveOverwriteAndSegmentRelease) {
  const torch::Tensor expected = torch::tensor({1.0f, 2.0f, 3.0f});
  const torch::Tensor expected_audio = torch::tensor({0.25f, -0.5f});
  DiTForwardOutput source;
  source.tensors = {expected};
  source.audio_tensors = {expected_audio};
  source.text_output = {"first"};
  ASSERT_TRUE(writer_->dit_output_write(source));
  RawForwardOutput output;
  ASSERT_TRUE(reader_->dit_output_read(output));
  ASSERT_EQ(output.dit_forward_output.tensors.size(), 1);
  source.tensors = {torch::zeros({3})};
  source.audio_tensors = {torch::zeros({2})};
  source.text_output = {"other"};
  ASSERT_TRUE(writer_->dit_output_write(source));
  reader_.reset();
  writer_.reset();
  EXPECT_TRUE(torch::equal(output.dit_forward_output.tensors[0], expected));
  ASSERT_EQ(output.dit_forward_output.audio_tensors.size(), 1);
  EXPECT_TRUE(
      torch::equal(output.dit_forward_output.audio_tensors[0], expected_audio));
  EXPECT_EQ(output.dit_forward_output.text_output,
            (std::vector<std::string>{"first"}));
}

TEST_F(ForwardSharedMemoryOutputTest, VlmTensorsSurviveSubsequentOutputWrites) {
  const torch::Tensor expected = torch::tensor({1.0f, 2.0f});
  ForwardOutput source;
  source.sample_output.mm_embeddings = {{expected}};
  ASSERT_TRUE(writer_->vlm_output_write(source, {}));
  RawForwardOutput output;
  ASSERT_TRUE(reader_->vlm_output_read(output));
  ASSERT_EQ(output.outputs.size(), 1);
  ASSERT_EQ(output.outputs[0].mm_embeddings.size(), 1);
  source.sample_output.mm_embeddings = {{torch::zeros({2})}};
  ASSERT_TRUE(writer_->vlm_output_write(source, {}));
  EXPECT_TRUE(torch::equal(output.outputs[0].mm_embeddings[0], expected));
}

TEST_F(ForwardSharedMemoryOutputTest,
       InvalidTokenWritesPreservePublishedOutput) {
  ForwardOutput valid;
  valid.sample_output.next_tokens = torch::tensor({7}, torch::kInt32);
  ASSERT_TRUE(writer_->llm_output_write(valid, {}));
  const uint64_t version = published_version();
  ForwardOutput invalid = valid;
  invalid.sample_output.logprobs = torch::tensor({-0.5f, -0.2f});
  EXPECT_FALSE(writer_->llm_output_write(invalid, {}));
  invalid = valid;
  invalid.sample_output.mm_embeddings = {{torch::Tensor()}};
  EXPECT_FALSE(writer_->vlm_output_write(invalid, {}));
  invalid = valid;
  invalid.beam_search_output.out_tokens = torch::tensor({8}, torch::kInt32);
  EXPECT_FALSE(writer_->rec_output_write(invalid, {}));
  EXPECT_EQ(published_version(), version);
  RawForwardOutput output;
  ASSERT_TRUE(reader_->llm_output_read(output));
  EXPECT_EQ(output.outputs[0].tokens[0].id, 7);
}

TEST_F(ForwardSharedMemoryOutputTest,
       OverflowDiTWritePreservesPublishedOutput) {
  DiTForwardOutput valid;
  valid.text_output = {"first"};
  ASSERT_TRUE(writer_->dit_output_write(valid));
  const uint64_t version = published_version();
  const std::string published(static_cast<const char*>(writer_->base_address()),
                              kSegmentBytes);
  DiTForwardOutput overflow;
  overflow.text_output = {std::string(kSegmentBytes, 'x')};
  EXPECT_FALSE(writer_->dit_output_write(overflow));
  overflow = DiTForwardOutput();
  overflow.audio_tensors = {
      torch::ones({static_cast<int64_t>(kSegmentBytes)}, torch::kFloat32)};
  EXPECT_FALSE(writer_->dit_output_write(overflow));
  EXPECT_EQ(
      std::memcmp(writer_->base_address(), published.data(), kSegmentBytes), 0);
  EXPECT_EQ(published_version(), version);
  RawForwardOutput output;
  ASSERT_TRUE(reader_->dit_output_read(output));
  EXPECT_EQ(output.dit_forward_output.text_output, valid.text_output);
}

TEST_F(ForwardSharedMemoryOutputTest, InvalidDiTTensorDoesNotPublishOutput) {
  DiTForwardOutput source;
  source.tensors = {torch::Tensor()};
  EXPECT_FALSE(writer_->dit_output_write(source));
  source.tensors = {torch::tensor(1.0f)};
  EXPECT_FALSE(writer_->dit_output_write(source));
  EXPECT_EQ(published_version(), 0);
}

TEST_F(ForwardSharedMemoryOutputTest,
       InvalidDiTAudioTensorDoesNotPublishOutput) {
  DiTForwardOutput source;
  source.audio_tensors = {torch::Tensor()};
  EXPECT_FALSE(writer_->dit_output_write(source));
  source.audio_tensors = {torch::tensor(1.0f)};
  EXPECT_FALSE(writer_->dit_output_write(source));
  source.audio_tensors = {torch::ones({2, 2}).transpose(0, 1)};
  EXPECT_FALSE(writer_->dit_output_write(source));
  EXPECT_EQ(published_version(), 0);
}

TEST_F(ForwardSharedMemoryOutputTest,
       MalformedAudioTensorSizeFailsWithoutMutation) {
  DiTForwardOutput source;
  source.audio_tensors = {torch::ones({2})};
  ASSERT_TRUE(writer_->dit_output_write(source));
  // Empty image/text vectors precede the audio vector and its tensor header.
  constexpr size_t kAudioTensorBytesOffset =
      sizeof(ControlMetadata) + 6 * sizeof(uint64_t) + sizeof(int64_t) +
      sizeof(bool) + 2 * sizeof(int32_t) + 2 * sizeof(uint64_t) +
      sizeof(int64_t) + sizeof(int8_t);
  const uint64_t invalid_bytes = kSegmentBytes;
  std::memcpy(
      static_cast<char*>(writer_->base_address()) + kAudioTensorBytesOffset,
      &invalid_bytes,
      sizeof(invalid_bytes));
  RawForwardOutput output;
  output.dit_forward_output.text_output = {"previous"};
  EXPECT_FALSE(reader_->dit_output_read(output));
  EXPECT_EQ(output.dit_forward_output.text_output,
            (std::vector<std::string>{"previous"}));
}

TEST_F(ForwardSharedMemoryOutputTest,
       DiTReaderRejectsTokenOutputWithoutMutation) {
  ForwardOutput source;
  source.sample_output.next_tokens = torch::tensor({7}, torch::kInt32);
  ASSERT_TRUE(writer_->rec_output_write(source, {}));
  RawForwardOutput output;
  output.dit_forward_output.text_output = {"previous"};
  EXPECT_FALSE(reader_->dit_output_read(output));
  EXPECT_EQ(output.dit_forward_output.text_output,
            (std::vector<std::string>{"previous"}));
}

TEST_F(ForwardSharedMemoryOutputTest,
       TokenReaderRejectsDiTOutputWithoutMutation) {
  DiTForwardOutput source;
  source.text_output = {"diffusion"};
  ASSERT_TRUE(writer_->dit_output_write(source));
  RawForwardOutput output;
  output.prepared_token = 81;
  EXPECT_FALSE(reader_->vlm_output_read(output));
  EXPECT_EQ(output.prepared_token, 81);
}

TEST_F(ForwardSharedMemoryOutputTest, MalformedCountFailsWithoutMutation) {
  ForwardOutput source;
  ASSERT_TRUE(writer_->rec_output_write(source, {}));
  const uint64_t invalid_count = std::numeric_limits<uint64_t>::max();
  char* payload =
      static_cast<char*>(writer_->base_address()) + sizeof(ControlMetadata);
  std::memcpy(payload, &invalid_count, sizeof(invalid_count));
  RawForwardOutput output;
  output.prepared_token = 81;
  EXPECT_FALSE(reader_->rec_output_read(output));
  EXPECT_EQ(output.prepared_token, 81);
}

TEST_F(ForwardSharedMemoryOutputTest,
       MalformedReadAllowsSubsequentOutputToBeConsumed) {
  ASSERT_TRUE(writer_->llm_output_write(ForwardOutput(), {}));
  const uint64_t invalid_count = std::numeric_limits<uint64_t>::max();
  char* payload =
      static_cast<char*>(writer_->base_address()) + sizeof(ControlMetadata);
  std::memcpy(payload, &invalid_count, sizeof(invalid_count));
  RawForwardOutput output;
  EXPECT_FALSE(reader_->llm_output_read(output));
  ForwardOutput source;
  source.sample_output.next_tokens = torch::tensor({7}, torch::kInt32);
  ASSERT_TRUE(writer_->llm_output_write(source, {}));
  ASSERT_TRUE(reader_->llm_output_read(output));
  ASSERT_EQ(output.outputs.size(), 1);
  EXPECT_EQ(output.outputs[0].tokens[0].id, 7);
}

TEST_F(ForwardSharedMemoryOutputTest, MalformedTensorSizeFailsWithoutMutation) {
  DiTForwardOutput source;
  source.tensors = {torch::ones({2})};
  ASSERT_TRUE(writer_->dit_output_write(source));
  // RAW_OUTPUT framing, tensor vector count, rank, shape, and dtype precede
  // the tensor byte count. Corrupt only that count in an otherwise valid frame.
  constexpr size_t kTensorBytesOffset =
      sizeof(ControlMetadata) + 6 * sizeof(uint64_t) + sizeof(int64_t) +
      sizeof(bool) + sizeof(int32_t) + sizeof(uint64_t) + sizeof(int64_t) +
      sizeof(int8_t);
  char* byte_count =
      static_cast<char*>(writer_->base_address()) + kTensorBytesOffset;
  const uint64_t invalid_bytes = kSegmentBytes;
  std::memcpy(byte_count, &invalid_bytes, sizeof(invalid_bytes));
  RawForwardOutput output;
  output.dit_forward_output.text_output = {"previous"};
  EXPECT_FALSE(reader_->dit_output_read(output));
  EXPECT_EQ(output.dit_forward_output.text_output,
            (std::vector<std::string>{"previous"}));
}

}  // namespace
}  // namespace xllm
