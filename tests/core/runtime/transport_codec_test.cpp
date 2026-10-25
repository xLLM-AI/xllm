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
#include <string>
#include <vector>

#include "core/distributed_runtime/worker_model_codec.h"
#include "core/framework/config/execution_config.h"
#include "core/kv_cache/transfer/kv_transport_codec.h"
#include "core/kv_cache/transfer/prefetch_result.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/forward_shared_memory_manager.h"
#include "worker.pb.h"

namespace xllm {
namespace {

class ScopedRawOutputTransport final {
 public:
  ScopedRawOutputTransport()
      : previous_(
            ExecutionConfig::get_instance().use_contiguous_input_buffer()) {
    // RAW_OUTPUT serialization uses Host tensors and needs no Device stream.
    ExecutionConfig::get_instance().use_contiguous_input_buffer(false);
  }
  ~ScopedRawOutputTransport() {
    ExecutionConfig::get_instance().use_contiguous_input_buffer(previous_);
  }

 private:
  bool previous_;
};

BlockTransferInfo make_block(int32_t src_id = 7,
                             TransferType transfer_type = TransferType::G2H,
                             BlockType block_type = BlockType::KV) {
  std::array<uint8_t, XXH3_128BITS_HASH_VALUE_LEN> hash;
  hash.fill(0x5a);
  return BlockTransferInfo(src_id,
                           /*dst_id=*/11,
                           hash.data(),
                           transfer_type,
                           block_type);
}

StoragePrefetchRequest make_prefetch_request() {
  StoragePrefetchRequest request;
  request.transfer_infos = {make_block()};
  request.unit_end_offsets = {1};
  request.batch_end_unit_offsets = {1};
  return request;
}

bool write_shared_tokens(
    ForwardSharedMemoryManager& writer,
    const torch::Tensor& tokens,
    const std::vector<SpeculativeTokenStats>& stats,
    int64_t prepared_token = -1,
    const std::vector<JsonObjectOutputError>& errors = {}) {
  ForwardOutput output;
  output.sample_output.next_tokens = tokens;
  output.prepared_token = prepared_token;
  output.json_object_errors = errors;
  return writer.llm_output_write(output, stats);
}

void expect_original_output(const RawForwardOutput& output) {
  EXPECT_EQ(output.prepared_token, 81);
  ASSERT_EQ(output.outputs.size(), 1);
  ASSERT_EQ(output.outputs[0].tokens.size(), 1);
  EXPECT_EQ(output.outputs[0].tokens[0].id, 19);
  EXPECT_EQ(output.dit_forward_output.text_output,
            (std::vector<std::string>{"original"}));
}

RawForwardOutput original_output() {
  RawForwardOutput output;
  output.prepared_token = 81;
  output.outputs.resize(1);
  output.outputs[0].tokens.resize(1);
  output.outputs[0].tokens[0].id = 19;
  output.dit_forward_output.text_output = {"original"};
  return output;
}

TEST(TransportCodecTest, BlockTransferRoundTripPreservesBatchAndHash) {
  const std::vector<BlockTransferInfo> blocks{
      make_block(/*src_id=*/7, TransferType::D2H2G, BlockType::LINEAR)};
  proto::BlockTransferInfos encoded;
  ASSERT_TRUE(block_transfer_info_to_proto(/*batch_id=*/42, blocks, &encoded));

  uint64_t batch_id = 0;
  std::vector<BlockTransferInfo> decoded;
  ASSERT_TRUE(proto_to_block_transfer_info(encoded, decoded, &batch_id));
  EXPECT_EQ(batch_id, 42);
  ASSERT_EQ(decoded.size(), 1);
  EXPECT_EQ(decoded[0].src_block_id, 7);
  EXPECT_EQ(decoded[0].dst_block_id, 11);
  EXPECT_EQ(decoded[0].transfer_type, TransferType::D2H2G);
  EXPECT_EQ(decoded[0].block_type, BlockType::LINEAR);
  EXPECT_EQ(
      std::memcmp(
          decoded[0].hash_key, blocks[0].hash_key, XXH3_128BITS_HASH_VALUE_LEN),
      0);
}

TEST(TransportCodecTest, InvalidBlockTransferDecodePreservesDestination) {
  proto::BlockTransferInfos valid;
  ASSERT_TRUE(block_transfer_info_to_proto(
      /*batch_id=*/42, {make_block()}, &valid));

  std::vector<proto::BlockTransferInfos> invalid_inputs(5, valid);
  invalid_inputs[0].clear_transfer_infos();
  invalid_inputs[1].mutable_transfer_infos(0)->set_hash_key("short");
  invalid_inputs[2].mutable_transfer_infos(0)->set_hash_key(
      std::string(XXH3_128BITS_HASH_VALUE_LEN + 1, 'a'));
  invalid_inputs[3].set_transfer_type(static_cast<proto::TransferType>(99));
  invalid_inputs[4].mutable_transfer_infos(0)->set_block_type(
      static_cast<proto::BlockType>(99));

  for (const proto::BlockTransferInfos& invalid : invalid_inputs) {
    uint64_t batch_id = 81;
    std::vector<BlockTransferInfo> destination{make_block(/*src_id=*/19)};
    EXPECT_FALSE(proto_to_block_transfer_info(invalid, destination, &batch_id));
    EXPECT_EQ(batch_id, 81);
    ASSERT_EQ(destination.size(), 1);
    EXPECT_EQ(destination[0].src_block_id, 19);
  }
}

TEST(TransportCodecTest, InvalidBlockTransferEncodePreservesDestination) {
  proto::BlockTransferInfos destination;
  ASSERT_TRUE(block_transfer_info_to_proto(
      /*batch_id=*/81, {make_block(/*src_id=*/19)}, &destination));
  const std::string original = destination.SerializeAsString();
  const std::vector<std::vector<BlockTransferInfo>> invalid_inputs{
      {},
      {make_block(), make_block(/*src_id=*/8, TransferType::H2D)},
      {make_block(/*src_id=*/7, static_cast<TransferType>(99))},
      {make_block(
          /*src_id=*/7, TransferType::G2H, static_cast<BlockType>(99))},
      {make_block(/*src_id=*/-1)}};

  for (const auto& invalid : invalid_inputs) {
    EXPECT_FALSE(block_transfer_info_to_proto(
        /*batch_id=*/42, invalid, &destination));
    EXPECT_EQ(destination.SerializeAsString(), original);
  }
}

TEST(TransportCodecTest, PrefetchRoundTripAcceptsUnusedSourceBlock) {
  for (int32_t source_id : {-1, 7}) {
    StoragePrefetchRequest request = make_prefetch_request();
    request.transfer_infos[0].src_block_id = source_id;
    proto::PrefetchRequest encoded;
    ASSERT_TRUE(request.to_proto(&encoded));
    StoragePrefetchRequest decoded;
    ASSERT_TRUE(decoded.from_proto(encoded));
    ASSERT_EQ(decoded.transfer_infos.size(), 1U);
    EXPECT_EQ(decoded.transfer_infos[0].src_block_id, source_id);
    EXPECT_EQ(decoded.transfer_infos[0].dst_block_id, 11);
    EXPECT_EQ(decoded.transfer_infos[0].transfer_type, TransferType::G2H);
    EXPECT_EQ(std::memcmp(decoded.transfer_infos[0].hash_key,
                          request.transfer_infos[0].hash_key,
                          XXH3_128BITS_HASH_VALUE_LEN),
              0);
    EXPECT_EQ(decoded.unit_end_offsets, request.unit_end_offsets);
    EXPECT_EQ(decoded.batch_end_unit_offsets, request.batch_end_unit_offsets);
  }
}

TEST(TransportCodecTest, InvalidPrefetchDecodePreservesDestination) {
  proto::PrefetchRequest valid;
  ASSERT_TRUE(make_prefetch_request().to_proto(&valid));
  std::vector<proto::PrefetchRequest> invalid_inputs(7, valid);
  invalid_inputs[0].clear_transfer_infos();
  invalid_inputs[1].mutable_transfer_infos(0)->set_hash_key("short");
  invalid_inputs[2].mutable_transfer_infos(0)->set_block_type(
      static_cast<proto::BlockType>(99));
  invalid_inputs[3].set_unit_end_offsets(0, 2);
  invalid_inputs[4].set_batch_end_unit_offsets(0, 0);
  invalid_inputs[5].mutable_transfer_infos(0)->set_src_block_id(-2);
  invalid_inputs[6].mutable_transfer_infos(0)->set_dst_block_id(-1);

  for (const proto::PrefetchRequest& invalid : invalid_inputs) {
    StoragePrefetchRequest destination = make_prefetch_request();
    destination.transfer_infos[0].src_block_id = 19;
    EXPECT_FALSE(destination.from_proto(invalid));
    ASSERT_EQ(destination.transfer_infos.size(), 1);
    EXPECT_EQ(destination.transfer_infos[0].src_block_id, 19);
    EXPECT_EQ(destination.unit_end_offsets, (std::vector<uint32_t>{1}));
    EXPECT_EQ(destination.batch_end_unit_offsets, (std::vector<uint32_t>{1}));
  }
}

TEST(TransportCodecTest, InvalidPrefetchEncodePreservesDestination) {
  proto::PrefetchRequest destination;
  ASSERT_TRUE(make_prefetch_request().to_proto(&destination));
  const std::string original = destination.SerializeAsString();
  std::vector<StoragePrefetchRequest> invalid_inputs(4,
                                                     make_prefetch_request());
  invalid_inputs[0].unit_end_offsets = {2};
  invalid_inputs[1].transfer_infos[0].block_type = static_cast<BlockType>(99);
  invalid_inputs[2].transfer_infos[0].src_block_id = -2;
  invalid_inputs[3].transfer_infos[0].dst_block_id = -1;
  for (const auto& invalid : invalid_inputs) {
    EXPECT_FALSE(invalid.to_proto(&destination));
    EXPECT_EQ(destination.SerializeAsString(), original);
  }
}

TEST(TransportCodecTest, PrefetchReuseReplacesTransfersAndOffsets) {
  StoragePrefetchRequest first = make_prefetch_request();
  first.transfer_infos.emplace_back(make_block(/*src_id=*/8));
  first.unit_end_offsets = {1, 2};
  first.batch_end_unit_offsets = {1, 2};
  proto::PrefetchRequest encoded;
  ASSERT_TRUE(first.to_proto(&encoded));
  StoragePrefetchRequest decoded;
  ASSERT_TRUE(decoded.from_proto(encoded));

  ASSERT_TRUE(make_prefetch_request().to_proto(&encoded));
  ASSERT_TRUE(decoded.from_proto(encoded));
  EXPECT_EQ(encoded.transfer_infos_size(), 1);
  ASSERT_EQ(decoded.transfer_infos.size(), 1);
  EXPECT_EQ(decoded.transfer_infos[0].src_block_id, 7);
  EXPECT_EQ(decoded.unit_end_offsets, (std::vector<uint32_t>{1}));
  EXPECT_EQ(decoded.batch_end_unit_offsets, (std::vector<uint32_t>{1}));
}

TEST(TransportCodecTest, InvalidMultimodalOutputEncodePreservesDestination) {
  proto::VLMForwardResponse destination;
  destination.add_outputs()->add_tokens()->set_id(19);
  const std::string original = destination.SerializeAsString();
  ForwardOutput source;
  source.sample_output.next_tokens = torch::tensor({7}, torch::kInt32);
  source.sample_output.mm_embeddings = {{torch::Tensor()}};
  EXPECT_FALSE(encode_vlm_output(source, /*stats=*/{}, &destination));
  EXPECT_EQ(destination.SerializeAsString(), original);
}

TEST(TransportCodecTest, InvalidDiTOutputEncodePreservesDestination) {
  proto::DITForwardResponse destination;
  destination.mutable_dit_forward_output()->add_text_output("original");
  const std::string original = destination.SerializeAsString();
  ForwardOutput source;
  source.dit_forward_output.tensors = {torch::Tensor()};
  EXPECT_FALSE(encode_dit_output(source, &destination));
  EXPECT_EQ(destination.SerializeAsString(), original);
}

TEST(TransportCodecTest, InvalidOutputTensorLayoutsPreserveDestination) {
  proto::LLMForwardResponse destination;
  destination.add_outputs()->add_tokens()->set_id(19);
  const std::string original = destination.SerializeAsString();
  ForwardOutput valid;
  valid.sample_output.next_tokens = torch::tensor({7, 8}, torch::kInt32);
  std::vector<ForwardOutput> invalid_inputs(10, valid);
  invalid_inputs[0].sample_output.next_tokens = torch::tensor({7.0f, 8.0f});
  invalid_inputs[1].sample_output.logprobs = torch::tensor({-0.5f});
  invalid_inputs[2].sample_output.top_tokens =
      torch::tensor({{7}, {8}}, torch::kInt64);
  invalid_inputs[3].sample_output.top_tokens =
      torch::tensor({{7}, {8}}, torch::kInt64);
  invalid_inputs[3].sample_output.top_logprobs =
      torch::tensor({{-0.5f}, {-1.0f}});
  invalid_inputs[4].sample_output.embeddings = torch::tensor({1.0f, 2.0f});
  invalid_inputs[5].sample_output.embeddings = torch::tensor({{1.0f}});
  invalid_inputs[6].sample_output.next_tokens =
      torch::tensor({{7, 8}, {9, 10}}, torch::kInt32).transpose(0, 1);
  invalid_inputs[7].beam_search_output.src_seq_idxes =
      torch::tensor({1, 0}, torch::kInt64);
  invalid_inputs[8].sample_output.mm_embeddings = {{torch::tensor({1.0f})},
                                                   {torch::tensor({2.0f})},
                                                   {torch::tensor({3.0f})}};
  invalid_inputs[9].sample_output.next_tokens =
      torch::empty({0}, torch::kInt32);
  invalid_inputs[9].sample_output.embeddings = torch::tensor({{1.0f}, {2.0f}});
  for (const ForwardOutput& invalid : invalid_inputs) {
    EXPECT_FALSE(encode_llm_output(invalid, /*stats=*/{}, &destination));
    EXPECT_EQ(destination.SerializeAsString(), original);
  }
  EXPECT_FALSE(encode_llm_output(valid, {{1, 2}}, &destination));
  EXPECT_EQ(destination.SerializeAsString(), original);
}

TEST(TransportCodecTest, InvalidMultimodalOutputDecodePreservesDestination) {
  proto::VLMForwardResponse source;
  proto::SquenceOutput* sequence = source.add_outputs();
  sequence->add_tokens()->set_id(7);
  sequence->mutable_mm_embeddings()->add_tensors();
  RawForwardOutput destination = original_output();
  EXPECT_FALSE(decode_vlm_output(source, destination));
  expect_original_output(destination);
}

TEST(TransportCodecTest, InvalidBeamOutputDecodePreservesDestination) {
  proto::RECForwardResponse source;
  source.add_src_seq_idxes(0);
  source.add_out_tokens(7);
  RawForwardOutput destination = original_output();
  EXPECT_FALSE(decode_rec_output(source, destination));
  expect_original_output(destination);
}

TEST(TransportCodecTest, InvalidDiTOutputDecodePreservesDestination) {
  proto::DITForwardResponse source;
  source.mutable_dit_forward_output()->mutable_tensors()->add_tensors();
  RawForwardOutput destination = original_output();
  EXPECT_FALSE(decode_dit_output(source, destination));
  expect_original_output(destination);
}

TEST(TransportCodecTest, OutputRoundTripPreservesTokenAndTensorPayloads) {
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
  proto::VLMForwardResponse encoded;
  constexpr int64_t kLargeRoundCount = (int64_t{1} << 33) + 5;
  ASSERT_TRUE(encode_vlm_output(
      source, {{1, 2, kLargeRoundCount}, {2, 3, 1}}, &encoded));
  RawForwardOutput decoded;
  ASSERT_TRUE(decode_vlm_output(encoded, decoded));

  ASSERT_EQ(decoded.outputs.size(), 2);
  ASSERT_EQ(decoded.outputs[0].tokens.size(), 1);
  const RawToken& token = decoded.outputs[0].tokens[0];
  EXPECT_EQ(token.id, 7);
  ASSERT_TRUE(token.logprob.has_value());
  EXPECT_FLOAT_EQ(*token.logprob, -0.5f);
  EXPECT_EQ(token.top_tokens, (std::vector<int64_t>{7, 9}));
  EXPECT_EQ(token.top_logprobs, (std::vector<float>{-0.5f, -1.5f}));
  EXPECT_EQ(token.embeddings, (std::vector<float>{1.0f, 2.0f}));
  EXPECT_EQ(decoded.outputs[0].speculative_token_stats.accepted_tokens, 1);
  EXPECT_EQ(decoded.outputs[1].speculative_token_stats.proposed_tokens, 3);
  EXPECT_EQ(decoded.outputs[0].speculative_token_stats.verified_rounds,
            kLargeRoundCount);
  EXPECT_EQ(decoded.outputs[1].speculative_token_stats.verified_rounds, 1);
  ASSERT_EQ(decoded.outputs[0].mm_embeddings.size(), 1);
  EXPECT_TRUE(torch::equal(decoded.outputs[0].mm_embeddings[0],
                           source.sample_output.mm_embeddings[0][0]));
  EXPECT_EQ(decoded.src_seq_idxes, (std::vector<int32_t>{1, 0}));
  EXPECT_EQ(decoded.out_tokens, (std::vector<int32_t>{8, 7}));
  EXPECT_EQ(decoded.out_logprobs, (std::vector<float>{-1.0f, -0.5f}));
  ASSERT_EQ(decoded.json_object_errors.size(), 1);
  EXPECT_EQ(decoded.json_object_errors[0].sample_sequence_id, "sequence-0");
  EXPECT_EQ(decoded.json_object_errors[0].message, "invalid JSON");
}

TEST(TransportCodecTest, DiTOutputRoundTripPreservesTensorAndTextPayloads) {
  ForwardOutput source;
  source.dit_forward_output.tensors = {torch::tensor({5.0f, 6.0f})};
  source.dit_forward_output.text_output = {"generated"};
  proto::DITForwardResponse encoded;
  ASSERT_TRUE(encode_dit_output(source, &encoded));
  RawForwardOutput decoded;
  ASSERT_TRUE(decode_dit_output(encoded, decoded));

  ASSERT_EQ(decoded.dit_forward_output.tensors.size(), 1);
  EXPECT_TRUE(torch::equal(decoded.dit_forward_output.tensors[0],
                           source.dit_forward_output.tensors[0]));
  EXPECT_EQ(decoded.dit_forward_output.text_output,
            (std::vector<std::string>{"generated"}));
  EXPECT_TRUE(decoded.outputs.empty());
}

TEST(TransportCodecTest, SpeculativeOutputStopsAtPaddingAndPreservesPayloads) {
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
  proto::LLMForwardResponse encoded;
  ASSERT_TRUE(encode_llm_output(source, {{1, 2}, {2, 2}}, &encoded));
  RawForwardOutput decoded;
  ASSERT_TRUE(decode_llm_output(encoded, decoded));

  ASSERT_EQ(decoded.outputs.size(), 2);
  ASSERT_EQ(decoded.outputs[0].tokens.size(), 1);
  EXPECT_EQ(decoded.outputs[0].tokens[0].id, 7);
  EXPECT_EQ(decoded.outputs[0].tokens[0].top_tokens, (std::vector<int64_t>{7}));
  EXPECT_EQ(decoded.outputs[0].tokens[0].embeddings,
            (std::vector<float>{1.0f}));
  ASSERT_EQ(decoded.outputs[1].tokens.size(), 2);
  const RawToken& token = decoded.outputs[1].tokens[1];
  EXPECT_EQ(token.id, 9);
  ASSERT_TRUE(token.logprob.has_value());
  EXPECT_FLOAT_EQ(*token.logprob, -1.5f);
  EXPECT_EQ(token.top_tokens, (std::vector<int64_t>{9}));
  EXPECT_EQ(token.top_logprobs, (std::vector<float>{-1.5f}));
  EXPECT_EQ(token.embeddings, (std::vector<float>{5.0f}));
  EXPECT_EQ(decoded.outputs[1].speculative_token_stats.accepted_tokens, 2);
}

TEST(TransportCodecTest, MissingLegacyVerifiedRoundsDecodeAsZeroOnReuse) {
  proto::LLMForwardResponse encoded;
  auto* row = encoded.add_outputs();
  row->add_tokens()->set_id(7);
  auto* stats = row->mutable_speculative_token_stats();
  stats->set_accepted_tokens(2);
  stats->set_proposed_tokens(7);
  stats->set_verified_rounds(9);
  RawForwardOutput decoded;
  ASSERT_TRUE(decode_llm_output(encoded, decoded));
  ASSERT_EQ(decoded.outputs.size(), 1U);
  EXPECT_EQ(decoded.outputs[0].speculative_token_stats.verified_rounds, 9);

  // A legacy wire message has only fields 1 and 2 in its stats submessage.
  stats->clear_verified_rounds();
  std::string wire;
  ASSERT_TRUE(encoded.SerializeToString(&wire));
  proto::LLMForwardResponse legacy;
  ASSERT_TRUE(legacy.ParseFromString(wire));
  ASSERT_TRUE(decode_llm_output(legacy, decoded));
  ASSERT_EQ(decoded.outputs.size(), 1U);
  EXPECT_EQ(decoded.outputs[0].speculative_token_stats.accepted_tokens, 2);
  EXPECT_EQ(decoded.outputs[0].speculative_token_stats.proposed_tokens, 7);
  EXPECT_EQ(decoded.outputs[0].speculative_token_stats.verified_rounds, 0);
}

TEST(TransportCodecTest,
     SharedOutputPreservesVerifiedRoundsAndFollowingFields) {
  const ScopedRawOutputTransport host_only;
  const std::string name = ForwardSharedMemoryManager::create_unique_name(
      "verified_rounds_" + std::to_string(static_cast<int64_t>(getpid())),
      /*dp_group=*/0,
      ForwardType::RAW_OUTPUT,
      /*rank=*/0);
  bool writer_is_creator = false;
  ForwardSharedMemoryManager writer(
      name, /*size=*/4096, writer_is_creator, ForwardType::RAW_OUTPUT);
  bool reader_is_creator = false;
  ForwardSharedMemoryManager reader(
      name, /*size=*/4096, reader_is_creator, ForwardType::RAW_OUTPUT);
  constexpr int64_t kLargeRoundCount = (int64_t{1} << 33) + 7;
  const std::vector<SpeculativeTokenStats> stats{
      {1, 3, 1}, {0, 0, 1}, {2, 7, kLargeRoundCount}};
  ASSERT_TRUE(write_shared_tokens(
      writer,
      torch::tensor({{30, 31, -1}, {40, -1, -1}, {50, 51, 52}}, torch::kInt64),
      stats,
      /*prepared_token=*/37,
      {{"sequence-2", "error after round stats"}}));
  RawForwardOutput decoded;
  ASSERT_TRUE(reader.llm_output_read(decoded));
  ASSERT_EQ(decoded.outputs.size(), stats.size());
  for (size_t row = 0; row < stats.size(); ++row) {
    EXPECT_EQ(decoded.outputs[row].speculative_token_stats.accepted_tokens,
              stats[row].accepted_tokens);
    EXPECT_EQ(decoded.outputs[row].speculative_token_stats.proposed_tokens,
              stats[row].proposed_tokens);
    EXPECT_EQ(decoded.outputs[row].speculative_token_stats.verified_rounds,
              stats[row].verified_rounds);
  }
  ASSERT_EQ(decoded.outputs[1].tokens.size(), 1U);
  EXPECT_EQ(decoded.outputs[1].tokens[0].id, 40);
  EXPECT_EQ(decoded.prepared_token, 37);
  ASSERT_EQ(decoded.json_object_errors.size(), 1U);
  EXPECT_EQ(decoded.json_object_errors[0].message, "error after round stats");
}

TEST(TransportCodecTest, SharedOutputSizeIncludesVerifiedRoundWord) {
  const ScopedRawOutputTransport host_only;
  // One token, twelve container lengths (one is INT32), five INT64 values
  // (id, three stats, prepared token), and two optional-payload flags.
  constexpr size_t kRequiredBytes =
      sizeof(ControlMetadata) + 11 * sizeof(uint64_t) + sizeof(int32_t) +
      5 * sizeof(int64_t) + 2 * sizeof(bool) + 2 * sizeof(uint64_t);
  const auto tokens = torch::tensor({7}, torch::kInt64);
  for (const size_t bytes : {kRequiredBytes - 1, kRequiredBytes}) {
    const std::string name = ForwardSharedMemoryManager::create_unique_name(
        "round_size_" + std::to_string(static_cast<int64_t>(getpid())) + "_" +
            std::to_string(bytes),
        /*dp_group=*/0,
        ForwardType::RAW_OUTPUT,
        /*rank=*/0);
    bool is_creator = false;
    ForwardSharedMemoryManager writer(
        name, bytes, is_creator, ForwardType::RAW_OUTPUT);
    EXPECT_EQ(write_shared_tokens(writer, tokens, {{0, 0, 1}}),
              bytes == kRequiredBytes);
  }
}

TEST(TransportCodecTest, EmbeddingOnlyOutputUsesPlaceholderToken) {
  ForwardOutput source;
  source.sample_output.embeddings = torch::tensor({{1.0f, 2.0f}});
  proto::LLMForwardResponse encoded;
  ASSERT_TRUE(encode_llm_output(source, /*stats=*/{}, &encoded));
  RawForwardOutput decoded;
  ASSERT_TRUE(decode_llm_output(encoded, decoded));
  ASSERT_EQ(decoded.outputs.size(), 1);
  ASSERT_EQ(decoded.outputs[0].tokens.size(), 1);
  EXPECT_EQ(decoded.outputs[0].tokens[0].id, -1);
  EXPECT_FALSE(decoded.outputs[0].tokens[0].logprob.has_value());
  EXPECT_EQ(decoded.outputs[0].tokens[0].embeddings,
            (std::vector<float>{1.0f, 2.0f}));
}

TEST(TransportCodecTest, OutputReuseClearsFieldsAbsentFromTextOnlyResult) {
  ForwardOutput first;
  first.sample_output.next_tokens = torch::tensor({7}, torch::kInt32);
  first.sample_output.mm_embeddings = {{torch::tensor({1.0f})}};
  first.json_object_errors = {{"sequence-0", "invalid JSON"}};
  first.dit_forward_output.tensors = {torch::tensor({2.0f})};
  proto::DITForwardResponse encoded;
  ASSERT_TRUE(encode_dit_output(first, &encoded));
  RawForwardOutput decoded = original_output();
  ASSERT_TRUE(decode_dit_output(encoded, decoded));

  ForwardOutput second;
  second.dit_forward_output.text_output = {"text only"};
  ASSERT_TRUE(encode_dit_output(second, &encoded));
  ASSERT_TRUE(decode_dit_output(encoded, decoded));
  EXPECT_TRUE(decoded.outputs.empty());
  EXPECT_TRUE(decoded.json_object_errors.empty());
  EXPECT_TRUE(decoded.dit_forward_output.tensors.empty());
  EXPECT_EQ(decoded.dit_forward_output.text_output,
            (std::vector<std::string>{"text only"}));
}

TEST(TransportCodecTest, GenerationParamsEncodeReuseClearsUnsetOptionals) {
  DiTGenerationParams source;
  source.seed = 42;
  source.seed_is_set = true;
  source.max_new_tokens = 16;
  source.diffusion_steps = 4;
  proto::DiTGenerationParams encoded;
  ASSERT_TRUE(source.to_proto(&encoded));
  ASSERT_TRUE(encoded.has_seed());
  ASSERT_TRUE(encoded.has_max_new_tokens());
  ASSERT_TRUE(encoded.has_diffusion_steps());

  source.seed_is_set = false;
  source.max_new_tokens = 0;
  source.diffusion_steps = 0;
  ASSERT_TRUE(source.to_proto(&encoded));
  EXPECT_FALSE(encoded.has_seed());
  EXPECT_FALSE(encoded.seed_is_set());
  EXPECT_FALSE(encoded.has_max_new_tokens());
  EXPECT_FALSE(encoded.has_diffusion_steps());
}

TEST(TransportCodecTest, GenerationParamsDecodeReuseRestoresOptionalDefaults) {
  proto::DiTGenerationParams source;
  source.set_seed(42);
  source.set_seed_is_set(true);
  source.set_max_new_tokens(16);
  source.set_diffusion_steps(4);
  source.set_temperature(0.25f);
  source.set_top_k(7);
  source.set_top_p(0.5f);
  source.set_repetition_penalty(1.5f);
  source.set_audio_duration_frames(8);
  source.set_audio_steps(6);
  source.set_audio_guidance_method("test");
  source.set_audio_sampling_rate(1234);
  DiTGenerationParams decoded;
  ASSERT_TRUE(decoded.from_proto(source));
  ASSERT_EQ(decoded.seed, 42);
  ASSERT_EQ(decoded.audio_steps, 6);

  source.Clear();
  ASSERT_TRUE(decoded.from_proto(source));
  const DiTGenerationParams defaults;
  EXPECT_FALSE(decoded.seed_is_set);
  EXPECT_EQ(decoded.seed, defaults.seed);
  EXPECT_EQ(decoded.max_new_tokens, defaults.max_new_tokens);
  EXPECT_EQ(decoded.diffusion_steps, defaults.diffusion_steps);
  EXPECT_FLOAT_EQ(decoded.temperature, defaults.temperature);
  EXPECT_EQ(decoded.top_k, defaults.top_k);
  EXPECT_FLOAT_EQ(decoded.top_p, defaults.top_p);
  EXPECT_FLOAT_EQ(decoded.repetition_penalty, defaults.repetition_penalty);
  EXPECT_EQ(decoded.audio_duration_frames, defaults.audio_duration_frames);
  EXPECT_EQ(decoded.audio_steps, defaults.audio_steps);
  EXPECT_EQ(decoded.audio_guidance_method, defaults.audio_guidance_method);
  EXPECT_EQ(decoded.audio_sampling_rate, defaults.audio_sampling_rate);
}

TEST(TransportCodecTest, GenerationParamsAcceptLegacyAndExplicitZeroSeed) {
  proto::DiTGenerationParams legacy;
  legacy.set_seed(42);
  ASSERT_FALSE(legacy.seed_is_set());
  DiTGenerationParams decoded;
  ASSERT_TRUE(decoded.from_proto(legacy));
  EXPECT_TRUE(decoded.seed_is_set);
  EXPECT_EQ(decoded.seed, 42);

  DiTGenerationParams explicit_zero;
  explicit_zero.seed_is_set = true;
  explicit_zero.seed = 0;
  proto::DiTGenerationParams encoded;
  ASSERT_TRUE(explicit_zero.to_proto(&encoded));
  ASSERT_TRUE(encoded.has_seed());
  EXPECT_EQ(encoded.seed(), 0);
  ASSERT_TRUE(decoded.from_proto(encoded));
  EXPECT_TRUE(decoded.seed_is_set);
  EXPECT_EQ(decoded.seed, 0);
}

}  // namespace
}  // namespace xllm
