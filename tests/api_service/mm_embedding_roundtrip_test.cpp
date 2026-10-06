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

#include <absl/cleanup/cleanup.h>
#include <gtest/gtest.h>
#include <openssl/crypto.h>
#include <openssl/err.h>
#include <openssl/provider.h>
#include <openssl/rand.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "api_service/embedding_output_builder.h"
#include "core/framework/config/model_config.h"
#include "core/framework/multimodal/mm_embedding_handler.h"
#include "core/framework/multimodal/mm_input.h"
#include "core/framework/prefix_cache/block_hasher.h"
#include "core/framework/request/sequence.h"
#include "core/util/hash_util.h"
#include "processors/multimodal_processor.h"
#include "processors/qwen3_vl_prompt_processor.h"

namespace xllm {
namespace {

constexpr int32_t kVisionStart = 151652;
constexpr int32_t kVisionEnd = 151653;
constexpr int32_t kImageToken = 151655;

EmbeddingOutput make_embedding() {
  EmbeddingOutput output;
  output.embedding = torch::zeros({4, 8}, torch::kFloat32);
  output.metadata.emplace(
      "image_grid_thw",
      torch::tensor({1, 4, 4}, torch::kInt64).reshape({1, 3}));
  return output;
}

bool load_embedding(const EmbeddingOutput& output,
                    MMInputItem& input,
                    bool binary = false) {
  EmbeddingOutputBuilder builder(binary,
                                 /*metadata_use_binary_encoding=*/false);
  proto::Embedding wire;
  std::string binary_payload;
  if (!builder.build_embedding_output(output, wire, binary_payload)) {
    return false;
  }
  MMEmbeddingHandler handler(MMType::IMAGE);
  MMPayload payload(std::move(binary_payload));
  return handler.process(MMContent("image_embedding", wire), input, payload) ==
         MMErrCode::SUCCESS;
}

std::vector<int32_t> make_tokens(size_t num_images) {
  std::vector<int32_t> tokens;
  tokens.reserve(num_images * 8);
  for (size_t image_index = 0; image_index < num_images; ++image_index) {
    tokens.insert(tokens.end(),
                  {kVisionStart,
                   kImageToken,
                   kImageToken,
                   kImageToken,
                   kImageToken,
                   kVisionEnd,
                   17,
                   18});
  }
  return tokens;
}

MMData make_mm_data(const std::vector<XXH3Key>& keys) {
  MMItemVec items;
  items.reserve(keys.size());
  for (size_t image_index = 0; image_index < keys.size(); ++image_index) {
    MMDataItem item(MMType::IMAGE);
    item.add("image_grid_thw",
             torch::tensor({1, 4, 4}, torch::kInt64).reshape({1, 3}));
    item.add("pixel_values", torch::zeros({1, 12}));
    item.mutable_state().mutable_schedule_data().key = keys[image_index];
    item.mutable_state().mutable_token_pos() = {
        static_cast<int32_t>(image_index * 8 + 1), 4};
    items.emplace_back(std::move(item));
  }
  return MMData(MMType::IMAGE, items);
}

class MMEmbeddingRoundtripTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ModelConfig& config = ModelConfig::get_instance();
    original_task_ = config.task();
    original_return_full_embeddings_ =
        config.enable_return_mm_full_embeddings();
    config.task("mm_embed").enable_return_mm_full_embeddings(false);
  }

  void TearDown() override {
    ModelConfig::get_instance()
        .task(std::move(original_task_))
        .enable_return_mm_full_embeddings(original_return_full_embeddings_);
  }

  SequenceOutput generate_output(const std::vector<XXH3Key>& keys,
                                 const std::vector<torch::Tensor>& embeddings) {
    sampling_param_.is_embeddings = true;
    const std::vector<int32_t> tokens =
        keys.empty() ? std::vector<int32_t>{17, 18} : make_tokens(keys.size());
    SequenceParams params;
    params.seq_capacity = tokens.size() + 8;
    params.sampling_param = &sampling_param_;
    params.stopping_checker = &stopping_checker_;
    IncrementalDecoder decoder("",
                               tokens.size(),
                               /*echo=*/false,
                               /*skip_special_tokens=*/true);
    Sequence sequence(/*index=*/0,
                      tokens,
                      torch::Tensor(),
                      keys.empty() ? MMData() : make_mm_data(keys),
                      decoder,
                      params);
    sequence.update_mm_embeddings(embeddings);
    return sequence.generate_output(Tokenizer());
  }

  SequenceOutput generate_output(const std::vector<XXH3Key>& keys) {
    std::vector<torch::Tensor> embeddings;
    embeddings.reserve(keys.size());
    for (size_t image_index = 0; image_index < keys.size(); ++image_index) {
      embeddings.emplace_back(
          torch::full({4, 8}, image_index, torch::kFloat32));
    }
    return generate_output(keys, embeddings);
  }

 private:
  std::string original_task_;
  bool original_return_full_embeddings_ = false;
  RequestSamplingParam sampling_param_;
  StoppingChecker stopping_checker_;
};

TEST_F(MMEmbeddingRoundtripTest,
       FullSequenceEmbeddingsDoNotRequirePerImageKeys) {
  ModelConfig::get_instance().task("embed").enable_return_mm_full_embeddings(
      true);
  const std::vector<XXH3Key> all_keys = {hash_string("first"),
                                         hash_string("second")};
  for (size_t image_count = 0; image_count <= all_keys.size(); ++image_count) {
    SCOPED_TRACE(image_count);
    const std::vector<XXH3Key> keys(all_keys.begin(),
                                    all_keys.begin() + image_count);
    const int64_t token_count =
        image_count == 0 ? 2 : static_cast<int64_t>(image_count) * 8;
    const torch::Tensor full_embeddings =
        torch::arange(token_count * 8, torch::kFloat32)
            .reshape({token_count, 8});
    const SequenceOutput output = generate_output(keys, {full_embeddings});
    ASSERT_TRUE(output.mm_embeddings.has_value());
    ASSERT_EQ(output.mm_embeddings->size(), 1u);
    EXPECT_TRUE(output.mm_embeddings->front().embedding.equal(full_embeddings));
    EXPECT_TRUE(output.mm_embeddings->front().hash_key.empty());
  }
}

TEST_F(MMEmbeddingRoundtripTest, RoundTripsPreserveImageKeysAndPrefixHashes) {
  const std::vector<XXH3Key> keys = {hash_string("first"),
                                     hash_string("second")};
  const SequenceOutput output = generate_output(keys);
  ASSERT_TRUE(output.mm_embeddings.has_value());
  ASSERT_EQ(output.mm_embeddings->size(), keys.size());
  const std::vector<int32_t> tokens = make_tokens(keys.size());
  std::vector<XXH3Key> expected_hashes;
  extend_prefix_hashes(BlockHasherType::MM,
                       make_mm_data(keys),
                       tokens,
                       /*block_size=*/4,
                       tokens.size() / 4,
                       expected_hashes);
  ModelArgs args;
  args.vision_start_token_id(kVisionStart)
      .vision_end_token_id(kVisionEnd)
      .image_token_id(kImageToken);
  MultimodalProcessor<Qwen3VLPromptProcessor> processor(
      args, nullptr, TokenizerArgs{});
  Qwen3VLPromptProcessor prompt_processor(args);
  for (const bool binary : {false, true}) {
    std::vector<MMInputItem> items;
    items.reserve(keys.size());
    for (size_t image_index = 0; image_index < keys.size(); ++image_index) {
      const auto& embedding = (*output.mm_embeddings)[image_index];
      EXPECT_EQ(embedding.hash_key.size(), 2 * XXH3_128BITS_HASH_VALUE_LEN);
      EXPECT_EQ(embedding.metadata.count("hash_key"), 0u);
      EmbeddingOutputBuilder builder(binary, false);
      proto::Embedding wire;
      std::string binary_payload;
      ASSERT_TRUE(
          builder.build_embedding_output(embedding, wire, binary_payload));
      EXPECT_EQ(wire.hash_key(), embedding.hash_key);
      MMInputItem item;
      MMEmbeddingHandler handler(MMType::IMAGE);
      MMPayload payload(std::move(binary_payload));
      ASSERT_EQ(
          handler.process(MMContent("image_embedding", wire), item, payload),
          MMErrCode::SUCCESS);
      ASSERT_TRUE(item.hash_key.has_value());
      EXPECT_EQ(*item.hash_key, keys[image_index]);
      EXPECT_EQ(item.embedding.hash_key, embedding.hash_key);
      items.emplace_back(std::move(item));
    }
    MMInput inputs;
    inputs.insert(items);
    MMData received;
    ASSERT_TRUE(processor.process_multimodal(inputs, received));
    prompt_processor.find_mm_spans(tokens, received);
    std::vector<XXH3Key> actual_hashes;
    extend_prefix_hashes(BlockHasherType::MM,
                         received,
                         tokens,
                         /*block_size=*/4,
                         tokens.size() / 4,
                         actual_hashes);
    EXPECT_EQ(actual_hashes, expected_hashes);
  }
}

TEST_F(MMEmbeddingRoundtripTest, ReusedInputsDoNotKeepPreviousImageState) {
  const uint8_t key_bytes[] = {0x00,
                               0x11,
                               0x22,
                               0x33,
                               0x44,
                               0x55,
                               0x66,
                               0x77,
                               0x88,
                               0x99,
                               0xaa,
                               0xbb,
                               0xcc,
                               0xdd,
                               0xee,
                               0xff};
  const XXH3Key expected(key_bytes);
  EmbeddingOutput output = make_embedding();
  output.hash_key = "00112233445566778899AABBCCDDEEFF";  // gitleaks:allow
  MMInputItem input;
  input.raw_data = "previous-image";
  input.hash_key = hash_string(input.raw_data);
  ASSERT_TRUE(load_embedding(output, input));
  ASSERT_TRUE(input.hash_key.has_value());
  EXPECT_EQ(*input.hash_key, expected);
  EXPECT_TRUE(input.raw_data.empty());
  input.clear();
  EXPECT_FALSE(input.is_embedding());
  EXPECT_TRUE(input.embedding.metadata.empty());
  EXPECT_TRUE(input.embedding.hash_key.empty());
  EXPECT_FALSE(input.hash_key.has_value());
}

TEST_F(MMEmbeddingRoundtripTest, RejectsMalformedKeys) {
  EmbeddingOutput output = make_embedding();
  const std::vector<std::string> invalid_keys = {
      std::string(31, '0'), std::string(33, '0'), std::string(32, 'g')};
  for (const std::string& key : invalid_keys) {
    output.hash_key = key;
    MMInputItem input;
    EXPECT_FALSE(load_embedding(output, input));
  }
}

TEST_F(MMEmbeddingRoundtripTest, KeyGenerationFailureRejectsOnlyKeylessInputs) {
  std::unique_ptr<OSSL_LIB_CTX, decltype(&OSSL_LIB_CTX_free)> context(
      OSSL_LIB_CTX_new(), OSSL_LIB_CTX_free);
  ASSERT_NE(context, nullptr);
  std::unique_ptr<OSSL_PROVIDER, decltype(&OSSL_PROVIDER_unload)> null_provider(
      OSSL_PROVIDER_load(context.get(), "null"), OSSL_PROVIDER_unload);
  ASSERT_NE(null_provider, nullptr);
  OSSL_LIB_CTX* previous_context = OSSL_LIB_CTX_set0_default(context.get());
  auto restore_context = absl::MakeCleanup([previous_context] {
    OSSL_LIB_CTX_set0_default(previous_context);
    ERR_clear_error();
  });

  // Disable random algorithms only in this test thread, without a production
  // injection hook or a process-wide replacement of OpenSSL's random method.
  unsigned char random_byte = 0;
  ASSERT_NE(RAND_bytes(&random_byte, /*num=*/1), 1);
  ERR_clear_error();

  EmbeddingOutput output = make_embedding();
  MMInputItem input;
  EXPECT_FALSE(load_embedding(output, input));
  EXPECT_FALSE(input.hash_key.has_value());
  EXPECT_FALSE(input.is_embedding());

  output.hash_key = "00112233445566778899aabbccddeeff";  // gitleaks:allow
  EXPECT_TRUE(load_embedding(output, input));
  EXPECT_TRUE(input.hash_key.has_value());
  EXPECT_EQ(input.embedding.hash_key, output.hash_key);
}

TEST_F(MMEmbeddingRoundtripTest,
       KeylessInputsGetDistinctKeysPreservedInScheduling) {
  const EmbeddingOutput original = make_embedding();
  MMInputItem original_input;
  MMInputItem binary_input;
  ASSERT_TRUE(load_embedding(original, original_input));
  ASSERT_TRUE(load_embedding(original, binary_input, /*binary=*/true));
  ASSERT_TRUE(original_input.hash_key.has_value());
  ASSERT_TRUE(binary_input.hash_key.has_value());
  EXPECT_FALSE(*original_input.hash_key == *binary_input.hash_key);

  MMInput inputs;
  inputs.insert(std::vector<MMInputItem>{original_input, binary_input});
  ModelArgs args;
  args.vision_start_token_id(kVisionStart)
      .vision_end_token_id(kVisionEnd)
      .image_token_id(kImageToken);
  MultimodalProcessor<Qwen3VLPromptProcessor> processor(
      args, nullptr, TokenizerArgs{});
  MMData data;
  ASSERT_TRUE(processor.process_multimodal(inputs, data));
  ASSERT_EQ(data.size(), 2u);
  const auto& items = data.items<MMItemVec>();
  EXPECT_EQ(items[0].state().schedule_data().key, *original_input.hash_key);
  EXPECT_EQ(items[1].state().schedule_data().key, *binary_input.hash_key);

  const std::vector<int32_t> tokens = make_tokens(items.size());
  Qwen3VLPromptProcessor prompt_processor(args);
  prompt_processor.find_mm_spans(tokens, data);
  std::vector<XXH3Key> first_hashes;
  std::vector<XXH3Key> repeated_hashes;
  extend_prefix_hashes(BlockHasherType::MM,
                       data,
                       tokens,
                       /*block_size=*/4,
                       tokens.size() / 4,
                       first_hashes);
  extend_prefix_hashes(BlockHasherType::MM,
                       data,
                       tokens,
                       /*block_size=*/4,
                       tokens.size() / 4,
                       repeated_hashes);
  EXPECT_EQ(first_hashes.size(), tokens.size() / 4);
  EXPECT_EQ(first_hashes, repeated_hashes);
}

}  // namespace
}  // namespace xllm
