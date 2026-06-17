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

#include "sample_slot.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "core/framework/config/rec_config.h"
#include "framework/block/block_manager_impl.h"
#include "platform/device.h"
#include "request.h"
#include "request_state.h"

namespace xllm {
namespace {

class CharTokenizer final : public Tokenizer {
 public:
  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool add_special_tokens = true) const override {
    if (ids == nullptr) {
      return false;
    }
    ids->clear();
    if (add_special_tokens) {
      ids->push_back(kBosTokenId);
    }
    size_t pos = 0;
    while (pos < text.size()) {
      if (text.size() - pos >= kEmbTokenLen &&
          std::memcmp(text.data() + pos, kEmbToken, kEmbTokenLen) == 0) {
        ids->push_back(kEmbTokenId);
        pos += kEmbTokenLen;
        continue;
      }
      ids->push_back(static_cast<unsigned char>(text[pos]));
      ++pos;
    }
    return true;
  }

  std::string decode(const Slice<int32_t>& ids,
                     bool skip_special_tokens) const override {
    std::string text;
    for (const auto token_id : ids) {
      if (skip_special_tokens && token_id == kBosTokenId) {
        continue;
      }
      if (token_id == kEmbTokenId) {
        text.append(kEmbToken, kEmbTokenLen);
        continue;
      }
      text.push_back(static_cast<char>(token_id));
    }
    return text;
  }

  std::optional<int32_t> token_to_id(
      const std::string_view& token) const override {
    if (token == std::string_view(kEmbToken, kEmbTokenLen)) {
      return kEmbTokenId;
    }
    return std::nullopt;
  }

  std::string id_to_token(int32_t id) const override {
    if (id == kBosTokenId) {
      return "<bos>";
    }
    if (id == kEmbTokenId) {
      return std::string(kEmbToken, kEmbTokenLen);
    }
    return std::string(1, static_cast<char>(id));
  }

  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<CharTokenizer>();
  }

 private:
  static constexpr int32_t kBosTokenId = 1;
  static constexpr int32_t kEmbTokenId = 100000;
  static constexpr char kEmbToken[] = "<emb_0>";
  static constexpr size_t kEmbTokenLen = sizeof(kEmbToken) - 1;
};

class ScopedBoolFlag final {
 public:
  ScopedBoolFlag(bool* flag, bool value) : flag_(flag), old_value_(*flag) {
    *flag_ = value;
  }

  ~ScopedBoolFlag() { *flag_ = old_value_; }

 private:
  bool* flag_;
  bool old_value_;
};

TEST(SampleSlotTest, BuildSampleSlotsKeepsMatchOrderAndSampleIds) {
  CharTokenizer tokenizer;
  std::vector<SampleSlot> sample_slots;

  ASSERT_TRUE(build_sample_slots(
      "sample-req", "A<emb_0>B<emb_0>C", "<emb_0>", tokenizer, &sample_slots));

  ASSERT_EQ(sample_slots.size(), 2);

  EXPECT_EQ(sample_slots[0].request_id, "sample-req");
  EXPECT_EQ(sample_slots[0].sample_id, 0);
  EXPECT_EQ(sample_slots[0].token_position, 1);

  EXPECT_EQ(sample_slots[1].sample_id, 1);
  EXPECT_EQ(sample_slots[1].token_position, 3);
}

class UnstableLiteralTokenizer final : public Tokenizer {
 public:
  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool add_special_tokens = true) const override {
    if (ids == nullptr) {
      return false;
    }
    ids->clear();
    if (add_special_tokens) {
      ids->push_back(1);
    }
    for (char ch : text) {
      ids->push_back(static_cast<unsigned char>(ch));
    }
    return true;
  }
};

TEST(SampleSlotTest, BuildSampleSlotsRejectsUnstableLiteralToken) {
  UnstableLiteralTokenizer tokenizer;
  std::vector<SampleSlot> sample_slots;

  EXPECT_FALSE(build_sample_slots(
      "sample-req", "A<emb_0>B<emb_0>C", "<emb_0>", tokenizer, &sample_slots));
  EXPECT_TRUE(sample_slots.empty());
}

TEST(SampleSlotTest, RequestPropagatesSampleSlotsToSequenceRuntime) {
  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  RequestState request_state(
      "abc",
      std::vector<int32_t>{10, 11, 12},
      sampling_param,
      SchedulerParam{},
      stopping_checker,
      /*seq_capacity=*/8,
      /*n=*/1,
      /*best_of=*/1,
      /*logprobs=*/false,
      /*stream=*/false,
      /*echo=*/false,
      /*skip_special_tokens=*/true,
      /*enable_schedule_overlap=*/false,
      [](const RequestOutput&) { return true; },
      OutputsFunc{});

  SampleSlot first_slot;
  first_slot.request_id = "sample-req";
  first_slot.sample_id = 0;
  first_slot.token_position = 2;

  SampleSlot second_slot;
  second_slot.request_id = "sample-req";
  second_slot.sample_id = 1;
  second_slot.token_position = 4;

  request_state.sample_slots = {first_slot, second_slot};

  Request request("sample-req", "", "", request_state);

  ASSERT_EQ(request.sequences().size(), 1);
  const auto& runtime_sample_slots = request.sequences()[0]->sample_slots();
  ASSERT_EQ(runtime_sample_slots.size(), 2);
  EXPECT_EQ(runtime_sample_slots[0].sample_id, 0);
  EXPECT_EQ(runtime_sample_slots[0].token_position, 2);
  EXPECT_EQ(runtime_sample_slots[1].sample_id, 1);
}

TEST(SampleSlotTest, RequestOutputSplitsSampleResultsBySampleId) {
  torch::Device device(Device::type_torch(), 0);
  BlockManager::Options options;
  options.num_blocks(4).block_size(4);
  BlockManagerImpl manager(options);

  CharTokenizer tokenizer;
  RequestSamplingParam sampling_param;
  sampling_param.logprobs = true;
  sampling_param.top_logprobs = 2;

  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(1);

  RequestState request_state(
      "abc",
      std::vector<int32_t>{1, 'a', 'b', 'c'},
      sampling_param,
      SchedulerParam{},
      stopping_checker,
      /*seq_capacity=*/8,
      /*n=*/1,
      /*best_of=*/1,
      /*logprobs=*/true,
      /*stream=*/false,
      /*echo=*/false,
      /*skip_special_tokens=*/true,
      /*enable_schedule_overlap=*/false,
      [](const RequestOutput&) { return true; },
      OutputsFunc{});

  SampleSlot first_slot;
  first_slot.request_id = "sample-req";
  first_slot.sample_id = 0;
  first_slot.token_position = 1;

  SampleSlot second_slot = first_slot;
  second_slot.sample_id = 1;
  second_slot.token_position = 2;

  request_state.sample_slots = {first_slot, second_slot};

  Request request("sample-req", "", "", request_state);
  auto* seq = request.sequences()[0].get();
  seq->add_kv_blocks(manager.allocate(1));
  seq->kv_state().set_kv_cache_tokens_num(seq->num_prompt_tokens());

  std::vector<int64_t> top_tokens = {'X', 'Y'};
  std::vector<float> top_logprobs = {-0.10f, -1.20f};
  Token first_token('X');
  first_token.logprob = -0.10f;
  first_token.top_tokens = top_tokens;
  first_token.top_logprobs = top_logprobs;
  seq->append_token(first_token);

  Token missing_logprob_token('Z');
  seq->append_token(missing_logprob_token);

  RequestOutput output = request.generate_output(tokenizer);

  ASSERT_TRUE(output.status.has_value());
  EXPECT_TRUE(output.status->ok());
  ASSERT_TRUE(output.usage.has_value());
  EXPECT_EQ(output.usage->num_generated_tokens, 2);
  ASSERT_EQ(output.outputs.size(), 2);

  EXPECT_EQ(output.outputs[0].index, 0U);
  EXPECT_EQ(output.outputs[0].text, "X");
  ASSERT_TRUE(output.outputs[0].logprobs.has_value());
  ASSERT_EQ(output.outputs[0].logprobs->size(), 1);
  EXPECT_EQ(output.outputs[0].logprobs->front().token, "X");
  ASSERT_TRUE(output.outputs[0].logprobs->front().top_logprobs.has_value());
  ASSERT_EQ(output.outputs[0].logprobs->front().top_logprobs->size(), 2);
  EXPECT_EQ(output.outputs[0].logprobs->front().top_logprobs->at(0).token, "X");

  EXPECT_EQ(output.outputs[1].index, 1U);
  EXPECT_TRUE(output.outputs[1].text.empty());
  EXPECT_FALSE(output.outputs[1].logprobs.has_value());
  ASSERT_TRUE(output.outputs[1].finish_reason.has_value());
  EXPECT_EQ(output.outputs[1].finish_reason.value(), "empty_logprobs");
}

TEST(SampleSlotTest, RequestOutputStableSortsOutOfOrderSampleIds) {
  torch::Device device(Device::type_torch(), 0);
  BlockManager::Options options;
  options.num_blocks(4).block_size(4);
  BlockManagerImpl manager(options);

  CharTokenizer tokenizer;
  RequestSamplingParam sampling_param;
  sampling_param.logprobs = true;

  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(1);

  RequestState request_state(
      "abc",
      std::vector<int32_t>{1, 'a', 'b', 'c'},
      sampling_param,
      SchedulerParam{},
      stopping_checker,
      /*seq_capacity=*/8,
      /*n=*/1,
      /*best_of=*/1,
      /*logprobs=*/true,
      /*stream=*/false,
      /*echo=*/false,
      /*skip_special_tokens=*/true,
      /*enable_schedule_overlap=*/false,
      [](const RequestOutput&) { return true; },
      OutputsFunc{});

  SampleSlot slot2;
  slot2.request_id = "sample-req";
  slot2.sample_id = 2;
  slot2.token_position = 1;

  SampleSlot slot0 = slot2;
  slot0.sample_id = 0;
  slot0.token_position = 2;

  SampleSlot slot1 = slot2;
  slot1.sample_id = 1;
  slot1.token_position = 3;

  request_state.sample_slots = {slot2, slot0, slot1};

  Request request("sample-req", "", "", request_state);
  auto* seq = request.sequences()[0].get();
  seq->add_kv_blocks(manager.allocate(1));
  seq->kv_state().set_kv_cache_tokens_num(seq->num_prompt_tokens());

  Token slot2_token('C');
  slot2_token.logprob = -0.30f;
  seq->append_token(slot2_token);

  Token slot0_token('A');
  slot0_token.logprob = -0.10f;
  seq->append_token(slot0_token);

  Token slot1_token('B');
  slot1_token.logprob = -0.20f;
  seq->append_token(slot1_token);

  RequestOutput output = request.generate_output(tokenizer);

  ASSERT_EQ(output.outputs.size(), 3);
  EXPECT_EQ(output.outputs[0].index, 0U);
  EXPECT_EQ(output.outputs[0].text, "A");
  EXPECT_EQ(output.outputs[1].index, 1U);
  EXPECT_EQ(output.outputs[1].text, "B");
  EXPECT_EQ(output.outputs[2].index, 2U);
  EXPECT_EQ(output.outputs[2].text, "C");
}

TEST(SampleSlotTest, OneRecOutputCarriesTokenLogprobsWhenEnabled) {
  ScopedBoolFlag enable_output_sku_logprobs(
      &RecConfig::get_instance().enable_output_sku_logprobs(), true);
  ScopedBoolFlag enable_convert_tokens_to_item(
      &RecConfig::get_instance().enable_convert_tokens_to_item(), false);

  CharTokenizer tokenizer;
  RequestSamplingParam sampling_param;
  sampling_param.logprobs = true;

  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(3);

  RequestState request_state(
      /*prompt=*/"",
      std::vector<int32_t>{11, 12},
      sampling_param,
      SchedulerParam{},
      stopping_checker,
      /*seq_capacity=*/8,
      /*n=*/1,
      /*best_of=*/1,
      /*logprobs=*/true,
      /*stream=*/false,
      /*echo=*/false,
      /*skip_special_tokens=*/true,
      /*enable_schedule_overlap=*/false,
      [](const RequestOutput&) { return true; },
      OutputsFunc{});
  request_state.rec_type = RecType::kOneRec;

  Request request("onerec-score",
                  /*x_request_id=*/"",
                  /*x_request_time=*/"",
                  request_state);
  auto* seq = request.sequences()[0].get();

  Token first_token(101);
  first_token.logprob = -0.10f;
  seq->append_token(first_token);

  Token second_token(102);
  second_token.logprob = -0.20f;
  seq->append_token(second_token);

  Token third_token(103);
  third_token.logprob = -0.30f;
  seq->append_token(third_token);

  RequestOutput output = request.generate_output(tokenizer);

  ASSERT_EQ(output.outputs.size(), 1);
  ASSERT_EQ(output.outputs[0].token_ids.size(), 3U);
  ASSERT_EQ(output.outputs[0].token_ids_logprobs.size(), 3U);
  ASSERT_TRUE(output.outputs[0].token_ids_logprobs[0].has_value());
  ASSERT_TRUE(output.outputs[0].token_ids_logprobs[1].has_value());
  ASSERT_TRUE(output.outputs[0].token_ids_logprobs[2].has_value());
  EXPECT_FLOAT_EQ(output.outputs[0].token_ids_logprobs[0].value(), -0.10f);
  EXPECT_FLOAT_EQ(output.outputs[0].token_ids_logprobs[1].value(), -0.20f);
  EXPECT_FLOAT_EQ(output.outputs[0].token_ids_logprobs[2].value(), -0.30f);
}

}  // namespace
}  // namespace xllm
