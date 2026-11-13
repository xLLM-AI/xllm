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

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/common/macros.h"
#include "core/framework/model/model_args.h"

namespace xllm {
namespace {

class NestedEmbeddingArgs final {
 public:
  REFLECT_PROPERTIES(NestedEmbeddingArgs);

  PROPERTY(int64_t, vocab_size) = -1;
  PROPERTY(bool, tie_word_embeddings) = false;
};

class NestedModelArgs final {
 public:
  REFLECT_PROPERTIES(NestedModelArgs);

  PROPERTY(NestedEmbeddingArgs, token_embedding_args);
};

class ReflectionOwnerArgs final {
 public:
  REFLECT_PROPERTIES(ReflectionOwnerArgs);

  PROPERTY(NestedModelArgs, model_args);
  PROPERTY(int64_t, owner_width) = 0;
};

class RecordingPropertyVisitor final : public PropertyVisitor {
 public:
  void visit(const std::string& name, bool value) override {
    record_visit(name);
    bool_values_.emplace(name, value);
  }
  void visit(const std::string& name, int32_t value) override {
    record_visit(name);
    int32_values_.emplace(name, value);
  }
  void visit(const std::string& name, int64_t value) override {
    record_visit(name);
    int64_values_.emplace(name, value);
  }
  void visit(const std::string& name, float value) override {
    record_visit(name);
    float_values_.emplace(name, value);
  }
  void visit(const std::string& name, double) override { record_visit(name); }
  void visit(const std::string& name, const std::string& value) override {
    record_visit(name);
    string_values_.emplace(name, value);
  }
  void visit(const std::string& name,
             const std::vector<int32_t>& value) override {
    record_visit(name);
    int32_vector_values_.emplace(name, value);
  }
  void visit(const std::string& name,
             const std::vector<int64_t>& value) override {
    record_visit(name);
    int64_vector_values_.emplace(name, value);
  }
  void visit(const std::string& name, const std::vector<float>&) override {
    record_visit(name);
  }
  void visit(const std::string& name, const std::vector<double>&) override {
    record_visit(name);
  }
  void visit(const std::string& name, const std::vector<bool>&) override {
    record_visit(name);
  }
  void visit(const std::string& name,
             const std::vector<std::string>& value) override {
    record_visit(name);
    string_vector_values_.emplace(name, value);
  }
  void visit(const std::string& name,
             const std::unordered_set<int32_t>& value) override {
    record_visit(name);
    int32_set_values_.emplace(name, value);
  }
  void visit_absent(const std::string& name) override {
    record_visit(name);
    absent_names_.emplace(name);
  }

  std::unordered_map<std::string, int32_t> visit_counts_;
  std::unordered_set<std::string> absent_names_;

  std::unordered_map<std::string, bool> bool_values_;
  std::unordered_map<std::string, int32_t> int32_values_;
  std::unordered_map<std::string, int64_t> int64_values_;
  std::unordered_map<std::string, float> float_values_;
  std::unordered_map<std::string, std::string> string_values_;
  std::unordered_map<std::string, std::vector<int32_t>> int32_vector_values_;
  std::unordered_map<std::string, std::vector<int64_t>> int64_vector_values_;
  std::unordered_map<std::string, std::vector<std::string>>
      string_vector_values_;
  std::unordered_map<std::string, std::unordered_set<int32_t>>
      int32_set_values_;

 private:
  void record_visit(const std::string& name) { ++visit_counts_[name]; }
};

TEST(PropertyReflectTest, RecursivelyFlattensMultipleLevelsBesideFlatFields) {
  ReflectionOwnerArgs args;
  args.model_args().token_embedding_args().vocab_size(/*value=*/32000);
  args.model_args().token_embedding_args().tie_word_embeddings(/*value=*/true);
  args.owner_width(/*value=*/4096);

  RecordingPropertyVisitor visitor;
  visit_properties(args, visitor);

  EXPECT_EQ(visitor.int64_values_.at("vocab_size"), 32000);
  EXPECT_TRUE(visitor.bool_values_.at("tie_word_embeddings"));
  EXPECT_EQ(visitor.int64_values_.at("owner_width"), 4096);
  EXPECT_EQ(visitor.visit_counts_.at("vocab_size"), 1);
  EXPECT_EQ(visitor.visit_counts_.at("tie_word_embeddings"), 1);
  EXPECT_EQ(visitor.visit_counts_.at("owner_width"), 1);
  EXPECT_EQ(visitor.int64_values_.count("model_args.vocab_size"), 0);
  EXPECT_EQ(
      visitor.int64_values_.count("model_args.token_embedding_args.vocab_size"),
      0);
  EXPECT_EQ(visitor.bool_values_.count("model_args.tie_word_embeddings"), 0);
  EXPECT_EQ(visitor.bool_values_.count(
                "model_args.token_embedding_args.tie_word_embeddings"),
            0);
}

TEST(PropertyReflectTest,
     KeepsModelArgsEmbeddingAccessorsAndFlatReflectionKeys) {
  ModelArgs args{};
  args.token_embedding_args().vocab_size(/*value=*/32000);
  args.token_embedding_args().tie_word_embeddings(/*value=*/true);

  EXPECT_EQ(args.vocab_size(), 32000);
  EXPECT_TRUE(args.tie_word_embeddings());

  args.vocab_size(/*value=*/32128);
  args.tie_word_embeddings(/*value=*/false);
  EXPECT_EQ(args.token_embedding_args().vocab_size(), 32128);
  EXPECT_FALSE(args.token_embedding_args().tie_word_embeddings());

  RecordingPropertyVisitor visitor;
  visit_properties(args, visitor);

  EXPECT_EQ(visitor.int64_values_.at("vocab_size"), 32128);
  EXPECT_FALSE(visitor.bool_values_.at("tie_word_embeddings"));
  EXPECT_EQ(visitor.visit_counts_.at("vocab_size"), 1);
  EXPECT_EQ(visitor.visit_counts_.at("tie_word_embeddings"), 1);
  EXPECT_EQ(visitor.int64_values_.count("token_embedding_args.vocab_size"), 0);
  EXPECT_EQ(
      visitor.bool_values_.count("token_embedding_args.tie_word_embeddings"),
      0);
}

TEST(PropertyReflectTest, SharesSpecialTokenStorageAndKeepsFlatReflectionKeys) {
  static_assert(std::is_same_v<
                decltype(std::declval<const ModelArgs&>().eos_token_id_vec()),
                const std::vector<int32_t>&>);
  static_assert(
      std::is_same_v<decltype(std::declval<ModelArgs&>().eos_token_id_vec()),
                     std::vector<int32_t>&>);
  static_assert(
      std::is_same_v<decltype(std::declval<ModelArgs&&>().eos_token_id_vec()),
                     std::vector<int32_t>&&>);
  static_assert(std::is_same_v<
                decltype(std::declval<const ModelArgs&>().stop_token_ids()),
                const std::unordered_set<int32_t>&>);
  static_assert(
      std::is_same_v<decltype(std::declval<ModelArgs&>().stop_token_ids()),
                     std::unordered_set<int32_t>&>);
  static_assert(
      std::is_same_v<decltype(std::declval<ModelArgs&&>().stop_token_ids()),
                     std::unordered_set<int32_t>&&>);
  static_assert(noexcept(std::declval<ModelArgs&>().eos_token_id_vec()));
  static_assert(noexcept(std::declval<ModelArgs&>().stop_token_ids()));

  ModelArgs args{};
  args.special_token_args()
      .bos_token_id(/*value=*/1001)
      .eos_token_id(/*value=*/1002)
      .eos_token_id_vec(/*value=*/{1002, 1003})
      .pad_token_id(/*value=*/1000)
      .stop_token_ids(/*value=*/{1003, 1004});
  EXPECT_EQ(args.bos_token_id(), 1001);
  EXPECT_EQ(args.eos_token_id(), 1002);
  EXPECT_EQ(args.eos_token_id_vec(), std::vector<int32_t>({1002, 1003}));
  EXPECT_EQ(args.pad_token_id(), 1000);
  EXPECT_EQ(args.stop_token_ids(), std::unordered_set<int32_t>({1003, 1004}));
  EXPECT_EQ(&args.bos_token_id(), &args.special_token_args().bos_token_id());
  EXPECT_EQ(&args.eos_token_id(), &args.special_token_args().eos_token_id());
  EXPECT_EQ(&args.eos_token_id_vec(),
            &args.special_token_args().eos_token_id_vec());
  EXPECT_EQ(&args.pad_token_id(), &args.special_token_args().pad_token_id());
  EXPECT_EQ(&args.stop_token_ids(),
            &args.special_token_args().stop_token_ids());

  const std::vector<int32_t> eos_token_ids = {2003, 2002, 2003};
  std::unordered_set<int32_t> stop_token_ids = {2002, 2004};
  ModelArgs& chained = args.bos_token_id(/*value=*/2001)
                           .eos_token_id(/*value=*/2002)
                           .eos_token_id_vec(eos_token_ids)
                           .pad_token_id(/*value=*/2000)
                           .stop_token_ids(std::move(stop_token_ids));
  EXPECT_EQ(&chained, &args);
  EXPECT_EQ(args.special_token_args().bos_token_id(), 2001);
  EXPECT_EQ(args.special_token_args().eos_token_id(), 2002);
  EXPECT_EQ(args.special_token_args().eos_token_id_vec(), eos_token_ids);
  EXPECT_EQ(args.special_token_args().pad_token_id(), 2000);
  EXPECT_EQ(args.special_token_args().stop_token_ids(),
            std::unordered_set<int32_t>({2002, 2004}));

  RecordingPropertyVisitor visitor;
  visit_properties(args, visitor);
  EXPECT_EQ(visitor.int32_values_.at("bos_token_id"), 2001);
  EXPECT_EQ(visitor.int32_values_.at("eos_token_id"), 2002);
  EXPECT_EQ(visitor.int32_vector_values_.at("eos_token_id_vec"), eos_token_ids);
  EXPECT_EQ(visitor.int32_values_.at("pad_token_id"), 2000);
  EXPECT_EQ(visitor.int32_set_values_.at("stop_token_ids"),
            std::unordered_set<int32_t>({2002, 2004}));
  for (const std::string name : {"bos_token_id",
                                 "eos_token_id",
                                 "eos_token_id_vec",
                                 "pad_token_id",
                                 "stop_token_ids"}) {
    EXPECT_EQ(visitor.visit_counts_.at(name), 1) << name;
    EXPECT_EQ(visitor.visit_counts_.count("special_token_args." + name), 0)
        << name;
  }
  EXPECT_EQ(visitor.visit_counts_.count("special_token_args"), 0);
}

TEST(PropertyReflectTest, OwnsIndependentSpecialTokensAfterCopy) {
  ModelArgs original{};
  original.eos_token_id_vec(/*value=*/{151329, 151336});
  original.stop_token_ids(/*value=*/{151329, 151338});
  // Some loaders populate only the EOS list and keep the scalar sentinel.
  EXPECT_EQ(original.eos_token_id(), -1);

  ModelArgs copied = original;
  copied.special_token_args()
      .bos_token_id(/*value=*/1)
      .eos_token_id(/*value=*/151336)
      .pad_token_id(
          /*value=*/2);
  copied.eos_token_id_vec().front() = 151337;
  copied.special_token_args().stop_token_ids().erase(151329);
  copied.stop_token_ids().insert(151337);

  EXPECT_EQ(original.bos_token_id(), 0);
  EXPECT_EQ(original.eos_token_id(), -1);
  EXPECT_EQ(original.eos_token_id_vec(),
            std::vector<int32_t>({151329, 151336}));
  EXPECT_EQ(original.pad_token_id(), 0);
  EXPECT_EQ(original.stop_token_ids(),
            std::unordered_set<int32_t>({151329, 151338}));

  RecordingPropertyVisitor visitor;
  visit_properties(copied, visitor);
  EXPECT_EQ(visitor.int32_values_.at("bos_token_id"), 1);
  EXPECT_EQ(visitor.int32_values_.at("eos_token_id"), 151336);
  EXPECT_EQ(visitor.int32_vector_values_.at("eos_token_id_vec"),
            std::vector<int32_t>({151337, 151336}));
  EXPECT_EQ(visitor.int32_values_.at("pad_token_id"), 2);
  EXPECT_EQ(visitor.int32_set_values_.at("stop_token_ids"),
            std::unordered_set<int32_t>({151337, 151338}));
}

TEST(PropertyReflectTest, SharesMoERoutingStorageAndKeepsFlatReflectionKeys) {
  static_assert(
      std::is_same_v<decltype(std::declval<const ModelArgs&>().scoring_func()),
                     const std::string&>);
  static_assert(
      std::is_same_v<decltype(std::declval<ModelArgs&>().scoring_func()),
                     std::string&>);
  static_assert(
      std::is_same_v<decltype(std::declval<ModelArgs&&>().scoring_func()),
                     std::string&&>);
  static_assert(noexcept(std::declval<const ModelArgs&>().scoring_func()));
  static_assert(noexcept(std::declval<ModelArgs&>().scoring_func()));
  static_assert(noexcept(std::declval<ModelArgs&&>().scoring_func()));

  ModelArgs args{};
  args.moe_args()
      .n_routed_experts(/*value=*/64)
      .n_shared_experts(/*value=*/2)
      .num_experts_per_tok(/*value=*/6)
      .moe_intermediate_size(/*value=*/2048)
      .routed_scaling_factor(/*value=*/2.5f)
      .norm_topk_prob(/*value=*/true)
      .n_group(/*value=*/8)
      .topk_group(/*value=*/4)
      .topk_method(/*value=*/"group_limited_greedy")
      .scoring_func(/*value=*/"softmax");

  EXPECT_EQ(args.n_routed_experts(), 64);
  EXPECT_EQ(args.n_shared_experts(), 2);
  EXPECT_EQ(args.num_experts_per_tok(), 6);
  EXPECT_EQ(args.moe_intermediate_size(), 2048);
  EXPECT_FLOAT_EQ(args.routed_scaling_factor(), 2.5f);
  EXPECT_TRUE(args.norm_topk_prob());
  EXPECT_EQ(args.n_group(), 8);
  EXPECT_EQ(args.topk_group(), 4);
  EXPECT_EQ(args.topk_method(), "group_limited_greedy");
  EXPECT_EQ(args.scoring_func(), "softmax");
  EXPECT_EQ(&args.n_routed_experts(), &args.moe_args().n_routed_experts());
  EXPECT_EQ(&args.scoring_func(), &args.moe_args().scoring_func());

  const std::string topk_method = "noaux_tc";
  std::string scoring_func = "sigmoid";
  ModelArgs& chained = args.n_routed_experts(/*value=*/128)
                           .n_shared_experts(/*value=*/1)
                           .num_experts_per_tok(/*value=*/8)
                           .moe_intermediate_size(/*value=*/4096)
                           .routed_scaling_factor(/*value=*/1.5f)
                           .norm_topk_prob(/*value=*/false)
                           .n_group(/*value=*/16)
                           .topk_group(/*value=*/2)
                           .topk_method(topk_method)
                           .scoring_func(std::move(scoring_func));
  EXPECT_EQ(&chained, &args);
  EXPECT_EQ(args.moe_args().n_routed_experts(), 128);
  EXPECT_EQ(args.moe_args().n_shared_experts(), 1);
  EXPECT_EQ(args.moe_args().num_experts_per_tok(), 8);
  EXPECT_EQ(args.moe_args().moe_intermediate_size(), 4096);
  EXPECT_FLOAT_EQ(args.moe_args().routed_scaling_factor(), 1.5f);
  EXPECT_FALSE(args.moe_args().norm_topk_prob());
  EXPECT_EQ(args.moe_args().n_group(), 16);
  EXPECT_EQ(args.moe_args().topk_group(), 2);
  EXPECT_EQ(args.moe_args().topk_method(), topk_method);
  EXPECT_EQ(args.moe_args().scoring_func(), "sigmoid");

  RecordingPropertyVisitor visitor;
  visit_properties(args, visitor);
  EXPECT_EQ(visitor.int32_values_.at("n_routed_experts"), 128);
  EXPECT_EQ(visitor.int32_values_.at("n_shared_experts"), 1);
  EXPECT_EQ(visitor.int32_values_.at("num_experts_per_tok"), 8);
  EXPECT_EQ(visitor.int32_values_.at("moe_intermediate_size"), 4096);
  EXPECT_FLOAT_EQ(visitor.float_values_.at("routed_scaling_factor"), 1.5f);
  EXPECT_FALSE(visitor.bool_values_.at("norm_topk_prob"));
  EXPECT_EQ(visitor.int32_values_.at("n_group"), 16);
  EXPECT_EQ(visitor.int32_values_.at("topk_group"), 2);
  EXPECT_EQ(visitor.string_values_.at("topk_method"), topk_method);
  EXPECT_EQ(visitor.string_values_.at("scoring_func"), "sigmoid");
  for (const std::string name : {"n_routed_experts",
                                 "n_shared_experts",
                                 "num_experts_per_tok",
                                 "moe_intermediate_size",
                                 "routed_scaling_factor",
                                 "norm_topk_prob",
                                 "n_group",
                                 "topk_group",
                                 "topk_method",
                                 "scoring_func"}) {
    EXPECT_EQ(visitor.visit_counts_.at(name), 1) << name;
    EXPECT_EQ(visitor.visit_counts_.count("moe_args." + name), 0) << name;
  }
  EXPECT_EQ(visitor.visit_counts_.count("moe_args"), 0);
}

TEST(PropertyReflectTest, SharesMtpModelStorageAndKeepsFlatReflectionKeys) {
  static_assert(
      std::is_same_v<decltype(std::declval<const ModelArgs&>().mtp_mlp_type()),
                     const std::string&>);
  static_assert(
      std::is_same_v<decltype(std::declval<ModelArgs&>().mtp_mlp_type()),
                     std::string&>);
  static_assert(
      std::is_same_v<decltype(std::declval<ModelArgs&&>().mtp_mlp_type()),
                     std::string&&>);
  static_assert(noexcept(std::declval<const ModelArgs&>().mtp_mlp_type()));
  static_assert(noexcept(std::declval<ModelArgs&>().mtp_mlp_type()));
  static_assert(noexcept(std::declval<ModelArgs&&>().mtp_mlp_type()));

  ModelArgs args{};
  args.mtp_args()
      .num_nextn_predict_layers(/*value=*/2)
      .mtp_start_layer_idx(/*value=*/48)
      .mtp_mlp_type(/*value=*/"dense");
  EXPECT_EQ(args.num_nextn_predict_layers(), 2);
  EXPECT_EQ(args.mtp_start_layer_idx(), 48);
  EXPECT_EQ(args.mtp_mlp_type(), "dense");
  EXPECT_EQ(&args.num_nextn_predict_layers(),
            &args.mtp_args().num_nextn_predict_layers());
  EXPECT_EQ(&args.mtp_start_layer_idx(),
            &args.mtp_args().mtp_start_layer_idx());
  EXPECT_EQ(&args.mtp_mlp_type(), &args.mtp_args().mtp_mlp_type());

  const std::string mlp_type = "moe";
  ModelArgs& chained = args.num_nextn_predict_layers(/*value=*/1)
                           .mtp_start_layer_idx(/*value=*/61)
                           .mtp_mlp_type(mlp_type);
  EXPECT_EQ(&chained, &args);
  args.mtp_mlp_type() = "dense";
  EXPECT_EQ(args.mtp_args().num_nextn_predict_layers(), 1);
  EXPECT_EQ(args.mtp_args().mtp_start_layer_idx(), 61);
  EXPECT_EQ(args.mtp_args().mtp_mlp_type(), "dense");

  RecordingPropertyVisitor visitor;
  visit_properties(args, visitor);
  EXPECT_EQ(visitor.int32_values_.at("num_nextn_predict_layers"), 1);
  EXPECT_EQ(visitor.int64_values_.at("mtp_start_layer_idx"), 61);
  EXPECT_EQ(visitor.string_values_.at("mtp_mlp_type"), "dense");
  for (const std::string name :
       {"num_nextn_predict_layers", "mtp_start_layer_idx", "mtp_mlp_type"}) {
    EXPECT_EQ(visitor.visit_counts_.at(name), 1) << name;
    EXPECT_EQ(visitor.visit_counts_.count("mtp_args." + name), 0) << name;
  }
  EXPECT_EQ(visitor.visit_counts_.count("mtp_args"), 0);
}

TEST(PropertyReflectTest, OwnsIndependentMtpModelParametersAfterCopy) {
  ModelArgs original{};
  original.num_nextn_predict_layers(/*value=*/1)
      .mtp_start_layer_idx(/*value=*/61)
      .mtp_mlp_type(
          /*value=*/"dense");

  ModelArgs copied = original;
  copied.mtp_args()
      .num_nextn_predict_layers(/*value=*/2)
      .mtp_start_layer_idx(/*value=*/62)
      .mtp_mlp_type(/*value=*/"moe");
  EXPECT_EQ(original.num_nextn_predict_layers(), 1);
  EXPECT_EQ(original.mtp_start_layer_idx(), 61);
  EXPECT_EQ(original.mtp_mlp_type(), "dense");
  EXPECT_EQ(copied.num_nextn_predict_layers(), 2);
  EXPECT_EQ(copied.mtp_start_layer_idx(), 62);
  EXPECT_EQ(copied.mtp_mlp_type(), "moe");
}

TEST(PropertyReflectTest,
     SharesAttentionHeadStorageAndKeepsFlatReflectionKeys) {
  static_assert(
      std::is_same_v<decltype(std::declval<const ModelArgs&>().n_kv_heads()),
                     const std::optional<int64_t>&>);
  static_assert(
      std::is_same_v<decltype(std::declval<ModelArgs&>().n_kv_heads()),
                     std::optional<int64_t>&>);
  static_assert(
      std::is_same_v<decltype(std::declval<ModelArgs&&>().n_kv_heads()),
                     std::optional<int64_t>&&>);
  static_assert(noexcept(std::declval<const ModelArgs&>().n_kv_heads()));
  static_assert(noexcept(std::declval<ModelArgs&>().n_kv_heads()));
  static_assert(noexcept(std::declval<ModelArgs&&>().n_kv_heads()));

  ModelArgs args{};
  args.attention_head_args()
      .head_dim(/*value=*/128)
      .decoder_head_dim(/*value=*/64)
      .n_heads(/*value=*/32)
      .decoder_n_heads(/*value=*/16)
      .actual_n_heads(/*value=*/28)
      .n_kv_heads(/*value=*/8)
      .decoder_n_kv_heads(/*value=*/4);
  EXPECT_EQ(args.head_dim(), 128);
  EXPECT_EQ(args.decoder_head_dim(), 64);
  EXPECT_EQ(args.n_heads(), 32);
  EXPECT_EQ(args.decoder_n_heads(), 16);
  EXPECT_EQ(args.actual_n_heads(), 28);
  EXPECT_EQ(args.n_kv_heads(), 8);
  EXPECT_EQ(args.decoder_n_kv_heads(), 4);
  EXPECT_EQ(&args.n_heads(), &args.attention_head_args().n_heads());
  EXPECT_EQ(&args.n_kv_heads(), &args.attention_head_args().n_kv_heads());

  const std::optional<int64_t> kv_heads = 16;
  std::optional<int64_t> decoder_kv_heads = 8;
  ModelArgs& chained = args.head_dim(/*value=*/256)
                           .decoder_head_dim(/*value=*/128)
                           .n_heads(/*value=*/64)
                           .decoder_n_heads(/*value=*/32)
                           .actual_n_heads(/*value=*/60)
                           .n_kv_heads(kv_heads)
                           .decoder_n_kv_heads(std::move(decoder_kv_heads));
  EXPECT_EQ(&chained, &args);
  args.n_heads() = 72;
  EXPECT_EQ(args.attention_head_args().head_dim(), 256);
  EXPECT_EQ(args.attention_head_args().decoder_head_dim(), 128);
  EXPECT_EQ(args.attention_head_args().n_heads(), 72);
  EXPECT_EQ(args.attention_head_args().decoder_n_heads(), 32);
  EXPECT_EQ(args.attention_head_args().actual_n_heads(), 60);
  EXPECT_EQ(args.attention_head_args().n_kv_heads(), 16);
  EXPECT_EQ(args.attention_head_args().decoder_n_kv_heads(), 8);

  RecordingPropertyVisitor visitor;
  visit_properties(args, visitor);
  EXPECT_EQ(visitor.int64_values_.at("head_dim"), 256);
  EXPECT_EQ(visitor.int64_values_.at("decoder_head_dim"), 128);
  EXPECT_EQ(visitor.int64_values_.at("n_heads"), 72);
  EXPECT_EQ(visitor.int64_values_.at("decoder_n_heads"), 32);
  EXPECT_EQ(visitor.int64_values_.at("actual_n_heads"), 60);
  EXPECT_EQ(visitor.int64_values_.at("n_kv_heads"), 16);
  EXPECT_EQ(visitor.int64_values_.at("decoder_n_kv_heads"), 8);
  for (const std::string name : {"head_dim",
                                 "decoder_head_dim",
                                 "n_heads",
                                 "decoder_n_heads",
                                 "actual_n_heads",
                                 "n_kv_heads",
                                 "decoder_n_kv_heads"}) {
    EXPECT_EQ(visitor.visit_counts_.at(name), 1) << name;
    EXPECT_EQ(visitor.visit_counts_.count("attention_head_args." + name), 0)
        << name;
    EXPECT_EQ(visitor.absent_names_.count(name), 0) << name;
  }
  EXPECT_EQ(visitor.visit_counts_.count("attention_head_args"), 0);
}

TEST(PropertyReflectTest, SharesMlaGeometryStorageAndKeepsFlatReflectionKeys) {
  static_assert(
      std::is_same_v<decltype(std::declval<const ModelArgs&>().q_lora_rank()),
                     const int32_t&>);
  static_assert(
      std::is_same_v<decltype(std::declval<ModelArgs&>().q_lora_rank()),
                     int32_t&>);
  static_assert(
      std::is_same_v<decltype(std::declval<ModelArgs&&>().q_lora_rank()),
                     int32_t&&>);
  static_assert(noexcept(std::declval<const ModelArgs&>().q_lora_rank()));
  static_assert(noexcept(std::declval<ModelArgs&>().q_lora_rank()));
  static_assert(noexcept(std::declval<ModelArgs&&>().q_lora_rank()));

  ModelArgs args{};
  args.mla_args()
      .qk_nope_head_dim(/*value=*/128)
      .qk_rope_head_dim(/*value=*/64)
      .v_head_dim(/*value=*/128)
      .q_lora_rank(/*value=*/1536)
      .kv_lora_rank(/*value=*/512);
  EXPECT_EQ(args.qk_nope_head_dim(), 128);
  EXPECT_EQ(args.qk_rope_head_dim(), 64);
  EXPECT_EQ(args.v_head_dim(), 128);
  EXPECT_EQ(args.q_lora_rank(), 1536);
  EXPECT_EQ(args.kv_lora_rank(), 512);
  EXPECT_EQ(&args.qk_nope_head_dim(), &args.mla_args().qk_nope_head_dim());
  EXPECT_EQ(&args.qk_rope_head_dim(), &args.mla_args().qk_rope_head_dim());
  EXPECT_EQ(&args.v_head_dim(), &args.mla_args().v_head_dim());
  EXPECT_EQ(&args.q_lora_rank(), &args.mla_args().q_lora_rank());
  EXPECT_EQ(&args.kv_lora_rank(), &args.mla_args().kv_lora_rank());

  ModelArgs& chained = args.qk_nope_head_dim(/*value=*/192)
                           .qk_rope_head_dim(/*value=*/32)
                           .v_head_dim(/*value=*/256)
                           .q_lora_rank(/*value=*/2048)
                           .kv_lora_rank(/*value=*/256);
  EXPECT_EQ(&chained, &args);
  EXPECT_EQ(args.mla_args().qk_nope_head_dim(), 192);
  EXPECT_EQ(args.mla_args().qk_rope_head_dim(), 32);
  EXPECT_EQ(args.mla_args().v_head_dim(), 256);
  EXPECT_EQ(args.mla_args().q_lora_rank(), 2048);
  EXPECT_EQ(args.mla_args().kv_lora_rank(), 256);

  RecordingPropertyVisitor visitor;
  visit_properties(args, visitor);
  EXPECT_EQ(visitor.int32_values_.at("qk_nope_head_dim"), 192);
  EXPECT_EQ(visitor.int32_values_.at("qk_rope_head_dim"), 32);
  EXPECT_EQ(visitor.int32_values_.at("v_head_dim"), 256);
  EXPECT_EQ(visitor.int32_values_.at("q_lora_rank"), 2048);
  EXPECT_EQ(visitor.int32_values_.at("kv_lora_rank"), 256);
  for (const std::string name : {"qk_nope_head_dim",
                                 "qk_rope_head_dim",
                                 "v_head_dim",
                                 "q_lora_rank",
                                 "kv_lora_rank"}) {
    EXPECT_EQ(visitor.visit_counts_.at(name), 1) << name;
    EXPECT_EQ(visitor.visit_counts_.count("mla_args." + name), 0) << name;
  }
  EXPECT_EQ(visitor.visit_counts_.count("mla_args"), 0);
}

TEST(PropertyReflectTest, KeepsMlaGeometryIndependentFromExecutionSelection) {
  ModelArgs args{};
  EXPECT_FALSE(args.attention_args().enable_mla());
  EXPECT_EQ(&args.enable_mla(), &args.attention_args().enable_mla());
  args.mla_args()
      .qk_nope_head_dim(/*value=*/128)
      .qk_rope_head_dim(/*value=*/64)
      .v_head_dim(/*value=*/128)
      .q_lora_rank(/*value=*/0)
      .kv_lora_rank(/*value=*/512);
  // Some models pad the logical head width instead of using the QK sum.
  args.head_dim(/*value=*/256).rotary_dim(/*value=*/64);
  EXPECT_FALSE(args.enable_mla());
  EXPECT_EQ(args.head_dim(), 256);
  EXPECT_EQ(args.rotary_dim(), 64);

  args.attention_args().enable_mla(/*value=*/true);
  EXPECT_TRUE(args.attention_args().enable_mla());
  EXPECT_TRUE(args.enable_mla());
  args.enable_mla(/*value=*/true);
  EXPECT_EQ(args.qk_nope_head_dim(), 128);
  EXPECT_EQ(args.qk_rope_head_dim(), 64);
  EXPECT_EQ(args.v_head_dim(), 128);
  EXPECT_EQ(args.q_lora_rank(), 0);
  EXPECT_EQ(args.kv_lora_rank(), 512);

  args.mla_args().qk_rope_head_dim(/*value=*/32);
  args.q_lora_rank() = 1536;
  EXPECT_TRUE(args.enable_mla());
  EXPECT_EQ(args.head_dim(), 256);
  EXPECT_EQ(args.rotary_dim(), 64);

  RecordingPropertyVisitor visitor;
  visit_properties(args, visitor);
  EXPECT_TRUE(visitor.bool_values_.at("enable_mla"));
  EXPECT_EQ(visitor.visit_counts_.at("enable_mla"), 1);
  EXPECT_EQ(visitor.visit_counts_.count("attention_args"), 0);
  EXPECT_EQ(visitor.visit_counts_.count("attention_args.enable_mla"), 0);
  EXPECT_EQ(visitor.int32_values_.at("qk_rope_head_dim"), 32);
  EXPECT_EQ(visitor.int32_values_.at("q_lora_rank"), 1536);
  EXPECT_EQ(visitor.int64_values_.at("head_dim"), 256);
  EXPECT_EQ(visitor.int64_values_.at("rotary_dim"), 64);
}

TEST(PropertyReflectTest, OwnsAttentionTypeSelectionAfterCopy) {
  ModelArgs original{};
  original.attention_args().enable_mla(/*value=*/true);

  ModelArgs copied = original;
  copied.enable_mla(/*value=*/false);

  EXPECT_TRUE(original.enable_mla());
  EXPECT_FALSE(copied.enable_mla());
  EXPECT_EQ(&original.enable_mla(), &original.attention_args().enable_mla());
  EXPECT_EQ(&copied.enable_mla(), &copied.attention_args().enable_mla());
  EXPECT_NE(&original.enable_mla(), &copied.enable_mla());
}

TEST(PropertyReflectTest,
     SharesLinearAttentionStorageAndKeepsFlatReflectionKeys) {
  static_assert(std::is_same_v<decltype(std::declval<const ModelArgs&>()
                                            .linear_num_key_heads()),
                               const int64_t&>);
  static_assert(std::is_same_v<
                decltype(std::declval<ModelArgs&>().linear_num_key_heads()),
                int64_t&>);
  static_assert(std::is_same_v<
                decltype(std::declval<ModelArgs&&>().linear_num_key_heads()),
                int64_t&&>);
  static_assert(noexcept(std::declval<ModelArgs&>().linear_num_key_heads()));

  ModelArgs args{};
  args.linear_attention_args()
      .linear_conv_kernel_dim(/*value=*/4)
      .linear_key_head_dim(/*value=*/128)
      .linear_value_head_dim(/*value=*/96)
      .linear_num_key_heads(/*value=*/16)
      .linear_num_value_heads(/*value=*/32)
      .linear_lower_bound(/*value=*/-4.0f)
      .mamba_ssm_dtype(/*value=*/"float32");
  EXPECT_EQ(args.linear_conv_kernel_dim(), 4);
  EXPECT_EQ(args.linear_key_head_dim(), 128);
  EXPECT_EQ(args.linear_value_head_dim(), 96);
  EXPECT_EQ(args.linear_num_key_heads(), 16);
  EXPECT_EQ(args.linear_num_value_heads(), 32);
  EXPECT_FLOAT_EQ(args.linear_lower_bound(), -4.0f);
  EXPECT_EQ(args.mamba_ssm_dtype(), "float32");
  EXPECT_EQ(&args.linear_conv_kernel_dim(),
            &args.linear_attention_args().linear_conv_kernel_dim());
  EXPECT_EQ(&args.linear_key_head_dim(),
            &args.linear_attention_args().linear_key_head_dim());
  EXPECT_EQ(&args.linear_value_head_dim(),
            &args.linear_attention_args().linear_value_head_dim());
  EXPECT_EQ(&args.linear_num_key_heads(),
            &args.linear_attention_args().linear_num_key_heads());
  EXPECT_EQ(&args.linear_num_value_heads(),
            &args.linear_attention_args().linear_num_value_heads());
  EXPECT_EQ(&args.linear_lower_bound(),
            &args.linear_attention_args().linear_lower_bound());
  EXPECT_EQ(&args.mamba_ssm_dtype(),
            &args.linear_attention_args().mamba_ssm_dtype());

  constexpr int64_t kLargeHeadCount = int64_t{1} << 33;
  ModelArgs& chained = args.linear_conv_kernel_dim(/*value=*/2)
                           .linear_key_head_dim(/*value=*/64)
                           .linear_value_head_dim(/*value=*/128)
                           .linear_num_key_heads(kLargeHeadCount)
                           .linear_num_value_heads(/*value=*/64)
                           .linear_lower_bound(/*value=*/-3.0f)
                           .mamba_ssm_dtype(/*value=*/"bfloat16");
  EXPECT_EQ(&chained, &args);
  EXPECT_EQ(args.linear_attention_args().linear_conv_kernel_dim(), 2);
  EXPECT_EQ(args.linear_attention_args().linear_key_head_dim(), 64);
  EXPECT_EQ(args.linear_attention_args().linear_value_head_dim(), 128);
  EXPECT_EQ(args.linear_attention_args().linear_num_key_heads(),
            kLargeHeadCount);
  EXPECT_EQ(args.linear_attention_args().linear_num_value_heads(), 64);
  EXPECT_FLOAT_EQ(args.linear_attention_args().linear_lower_bound(), -3.0f);
  EXPECT_EQ(args.linear_attention_args().mamba_ssm_dtype(), "bfloat16");

  RecordingPropertyVisitor visitor;
  visit_properties(args, visitor);
  EXPECT_EQ(visitor.int32_values_.at("linear_conv_kernel_dim"), 2);
  EXPECT_EQ(visitor.int32_values_.at("linear_key_head_dim"), 64);
  EXPECT_EQ(visitor.int32_values_.at("linear_value_head_dim"), 128);
  EXPECT_EQ(visitor.int64_values_.at("linear_num_key_heads"), kLargeHeadCount);
  EXPECT_EQ(visitor.int32_values_.at("linear_num_value_heads"), 64);
  EXPECT_FLOAT_EQ(visitor.float_values_.at("linear_lower_bound"), -3.0f);
  EXPECT_EQ(visitor.string_values_.at("mamba_ssm_dtype"), "bfloat16");
  for (const std::string name : {"linear_conv_kernel_dim",
                                 "linear_key_head_dim",
                                 "linear_value_head_dim",
                                 "linear_num_key_heads",
                                 "linear_num_value_heads",
                                 "linear_lower_bound",
                                 "mamba_ssm_dtype"}) {
    EXPECT_EQ(visitor.visit_counts_.at(name), 1) << name;
    EXPECT_EQ(visitor.visit_counts_.count("linear_attention_args." + name), 0)
        << name;
  }
  EXPECT_EQ(visitor.visit_counts_.count("linear_attention_args"), 0);
}

TEST(PropertyReflectTest, OwnsIndependentLinearStateParametersAfterCopy) {
  ModelArgs original{};
  original.linear_conv_kernel_dim(/*value=*/4)
      .linear_key_head_dim(/*value=*/128)
      .linear_value_head_dim(/*value=*/64)
      .linear_num_key_heads(/*value=*/16)
      .linear_num_value_heads(/*value=*/32)
      .linear_lower_bound(/*value=*/-5.0f)
      .mamba_ssm_dtype(/*value=*/"float32");
  original.dtype(/*value=*/"bfloat16").full_attention_interval(/*value=*/4);

  ModelArgs copied = original;
  copied.linear_attention_args()
      .linear_conv_kernel_dim(/*value=*/2)
      .linear_key_head_dim(/*value=*/64)
      .linear_value_head_dim(/*value=*/128)
      .linear_num_key_heads(/*value=*/8)
      .linear_num_value_heads(/*value=*/16)
      .linear_lower_bound(/*value=*/-2.0f)
      .mamba_ssm_dtype(/*value=*/"bfloat16");
  EXPECT_EQ(original.linear_conv_kernel_dim(), 4);
  EXPECT_EQ(original.linear_key_head_dim(), 128);
  EXPECT_EQ(original.linear_value_head_dim(), 64);
  EXPECT_EQ(original.linear_num_key_heads(), 16);
  EXPECT_EQ(original.linear_num_value_heads(), 32);
  EXPECT_FLOAT_EQ(original.linear_lower_bound(), -5.0f);
  EXPECT_EQ(original.mamba_ssm_dtype(), "float32");
  EXPECT_EQ(copied.mamba_ssm_dtype(), "bfloat16");
  EXPECT_EQ(copied.dtype(), "bfloat16");
  EXPECT_EQ(copied.full_attention_interval(), 4);
  EXPECT_TRUE(has_linear_attention_layers(copied));
  EXPECT_FALSE(is_full_attention_layer(copied, 0));
  EXPECT_TRUE(is_full_attention_layer(copied, 3));
}

TEST(PropertyReflectTest, PreservesHybridAttentionDefaults) {
  const ModelArgs args{};
  EXPECT_FALSE(has_linear_attention_layers(args));
  EXPECT_TRUE(is_full_attention_layer(args, 0));
  EXPECT_TRUE(is_full_attention_layer(args, 3));
}

TEST(PropertyReflectTest,
     SharesHybridAttentionStorageAndKeepsFlatReflectionKeys) {
  static_assert(
      std::is_same_v<decltype(std::declval<const ModelArgs&>().layer_types()),
                     const std::vector<std::string>&>);
  static_assert(
      std::is_same_v<decltype(std::declval<ModelArgs&>().full_attn_layers()),
                     std::vector<int32_t>&>);
  static_assert(
      std::is_same_v<decltype(std::declval<ModelArgs&&>().layer_types()),
                     std::vector<std::string>&&>);
  static_assert(noexcept(std::declval<ModelArgs&>().layer_types()));

  ModelArgs args{};
  args.hybrid_attention_args()
      .full_attention_interval(/*value=*/4)
      .full_attn_layers(/*value=*/{1})
      .layer_types(/*value=*/{"linear_attention", "deepseek_sparse_attention"});
  EXPECT_EQ(args.full_attention_interval(), 4);
  EXPECT_EQ(args.full_attn_layers(), (std::vector<int32_t>{1}));
  EXPECT_EQ(args.layer_types(),
            (std::vector<std::string>{"linear_attention",
                                      "deepseek_sparse_attention"}));
  EXPECT_EQ(&args.full_attention_interval(),
            &args.hybrid_attention_args().full_attention_interval());
  EXPECT_EQ(&args.full_attn_layers(),
            &args.hybrid_attention_args().full_attn_layers());
  EXPECT_EQ(&args.layer_types(), &args.hybrid_attention_args().layer_types());

  ModelArgs& chained =
      args.full_attention_interval(/*value=*/2)
          .full_attn_layers(/*value=*/{0, 2})
          .layer_types(
              /*value=*/{"full_attention", "linear_attention", "attention"});
  EXPECT_EQ(&chained, &args);
  EXPECT_EQ(args.hybrid_attention_args().full_attention_interval(), 2);
  EXPECT_EQ(args.hybrid_attention_args().full_attn_layers(),
            (std::vector<int32_t>{0, 2}));
  EXPECT_EQ(args.hybrid_attention_args().layer_types(),
            (std::vector<std::string>{
                "full_attention", "linear_attention", "attention"}));

  RecordingPropertyVisitor visitor;
  visit_properties(args, visitor);
  EXPECT_EQ(visitor.int32_values_.at("full_attention_interval"), 2);
  EXPECT_EQ(visitor.int32_vector_values_.at("full_attn_layers"),
            (std::vector<int32_t>{0, 2}));
  EXPECT_EQ(visitor.string_vector_values_.at("layer_types"),
            (std::vector<std::string>{
                "full_attention", "linear_attention", "attention"}));
  for (const std::string name :
       {"full_attention_interval", "full_attn_layers", "layer_types"}) {
    EXPECT_EQ(visitor.visit_counts_.at(name), 1) << name;
    EXPECT_EQ(visitor.visit_counts_.count("hybrid_attention_args." + name), 0)
        << name;
  }
  EXPECT_EQ(visitor.visit_counts_.count("hybrid_attention_args"), 0);
}

TEST(PropertyReflectTest, PreservesHybridLayerClassificationAfterCopy) {
  ModelArgs original{};
  original.full_attention_interval(/*value=*/4).full_attn_layers(/*value=*/{3});
  EXPECT_TRUE(has_linear_attention_layers(original));
  EXPECT_FALSE(is_full_attention_layer(original, 0));
  EXPECT_FALSE(is_full_attention_layer(original, 2));
  EXPECT_TRUE(is_full_attention_layer(original, 3));
  EXPECT_TRUE(is_full_attention_layer(original, 7));

  original.layer_types(/*value=*/{"full_attention",
                                  "attention",
                                  "deepseek_sparse_attention",
                                  "linear_attention"});
  EXPECT_TRUE(is_full_attention_layer(original, 0));
  EXPECT_TRUE(is_full_attention_layer(original, 1));
  EXPECT_TRUE(is_full_attention_layer(original, 2));
  EXPECT_FALSE(is_full_attention_layer(original, 3));
  EXPECT_TRUE(is_full_attention_layer(original, 7));
  EXPECT_TRUE(has_linear_attention_layers(original));

  ModelArgs copied = original;
  copied.hybrid_attention_args().full_attn_layers(/*value=*/{0, 1, 2, 3});
  copied.hybrid_attention_args().layer_types()[3] = "full_attention";
  EXPECT_EQ(original.full_attn_layers(), (std::vector<int32_t>{3}));
  EXPECT_EQ(original.layer_types().at(3), "linear_attention");
  EXPECT_EQ(copied.full_attn_layers(), (std::vector<int32_t>{0, 1, 2, 3}));
  EXPECT_TRUE(is_full_attention_layer(copied, 3));
  EXPECT_FALSE(has_linear_attention_layers(copied));
}

TEST(PropertyReflectTest, SharesRopeStorageAndKeepsFlatReflectionKeys) {
  static_assert(
      std::is_same_v<decltype(std::declval<const ModelArgs&>().rotary_dim()),
                     const int64_t&>);
  static_assert(std::is_same_v<decltype(std::declval<ModelArgs&>()
                                            .rope_scaling_mrope_section()),
                               std::vector<int64_t>&>);
  static_assert(std::is_same_v<decltype(std::declval<ModelArgs&&>()
                                            .rope_scaling_mrope_section()),
                               std::vector<int64_t>&&>);
  static_assert(
      noexcept(std::declval<ModelArgs&>().rope_scaling_mrope_section()));

  ModelArgs args{};
  args.rope_args()
      .rotary_dim(/*value=*/64)
      .rope_theta(/*value=*/500000.0f)
      .rope_scaling_rope_type(/*value=*/"yarn")
      .rope_scaling_factor(/*value=*/8.0f)
      .rope_scaling_low_freq_factor(/*value=*/1.0f)
      .rope_scaling_high_freq_factor(/*value=*/4.0f)
      .rope_extrapolation_factor(/*value=*/2.0f)
      .rope_scaling_original_max_position_embeddings(/*value=*/8192)
      .rope_scaling_beta_fast(/*value=*/32)
      .rope_scaling_beta_slow(/*value=*/1)
      .rope_scaling_attn_factor(/*value=*/1.5f)
      .rope_scaling_mscale(/*value=*/1.25f)
      .rope_scaling_mscale_all_dim(/*value=*/0.75f)
      .rope_scaling_mrope_section(/*value=*/{16, 24, 24})
      .rope_scaling_mrope_interleaved(/*value=*/true)
      .rope_scaling(/*value=*/2)
      .partial_rotary_factor(/*value=*/0.5f);
  EXPECT_EQ(args.rotary_dim(), 64);
  EXPECT_EQ(&args.rotary_dim(), &args.rope_args().rotary_dim());
  EXPECT_EQ(&args.rope_theta(), &args.rope_args().rope_theta());
  EXPECT_EQ(&args.rope_scaling_rope_type(),
            &args.rope_args().rope_scaling_rope_type());
  EXPECT_EQ(&args.rope_scaling_mrope_section(),
            &args.rope_args().rope_scaling_mrope_section());

  ModelArgs& chained = args.rotary_dim(/*value=*/32)
                           .rope_theta(/*value=*/1000000.0f)
                           .rope_scaling_rope_type(/*value=*/"mrope")
                           .rope_scaling_mrope_section(/*value=*/{8, 12, 12})
                           .partial_rotary_factor(/*value=*/0.25f);
  EXPECT_EQ(&chained, &args);
  EXPECT_EQ(args.rope_args().rotary_dim(), 32);
  EXPECT_FLOAT_EQ(args.rope_args().rope_theta(), 1000000.0f);
  EXPECT_EQ(args.rope_args().rope_scaling_rope_type(), "mrope");
  EXPECT_EQ(args.rope_args().rope_scaling_mrope_section(),
            (std::vector<int64_t>{8, 12, 12}));
  EXPECT_FLOAT_EQ(args.rope_args().partial_rotary_factor(), 0.25f);

  RecordingPropertyVisitor visitor;
  visit_properties(args, visitor);
  EXPECT_EQ(visitor.int64_values_.at("rotary_dim"), 32);
  EXPECT_FLOAT_EQ(visitor.float_values_.at("rope_theta"), 1000000.0f);
  EXPECT_EQ(visitor.string_values_.at("rope_scaling_rope_type"), "mrope");
  EXPECT_FLOAT_EQ(visitor.float_values_.at("rope_scaling_factor"), 8.0f);
  EXPECT_FLOAT_EQ(visitor.float_values_.at("rope_scaling_low_freq_factor"),
                  1.0f);
  EXPECT_FLOAT_EQ(visitor.float_values_.at("rope_scaling_high_freq_factor"),
                  4.0f);
  EXPECT_FLOAT_EQ(visitor.float_values_.at("rope_extrapolation_factor"), 2.0f);
  EXPECT_EQ(
      visitor.int64_values_.at("rope_scaling_original_max_position_embeddings"),
      8192);
  EXPECT_EQ(visitor.int64_values_.at("rope_scaling_beta_fast"), 32);
  EXPECT_EQ(visitor.int64_values_.at("rope_scaling_beta_slow"), 1);
  EXPECT_FLOAT_EQ(visitor.float_values_.at("rope_scaling_attn_factor"), 1.5f);
  EXPECT_FLOAT_EQ(visitor.float_values_.at("rope_scaling_mscale"), 1.25f);
  EXPECT_FLOAT_EQ(visitor.float_values_.at("rope_scaling_mscale_all_dim"),
                  0.75f);
  EXPECT_EQ(visitor.int64_vector_values_.at("rope_scaling_mrope_section"),
            (std::vector<int64_t>{8, 12, 12}));
  EXPECT_TRUE(visitor.bool_values_.at("rope_scaling_mrope_interleaved"));
  EXPECT_EQ(visitor.int32_values_.at("rope_scaling"), 2);
  EXPECT_FLOAT_EQ(visitor.float_values_.at("partial_rotary_factor"), 0.25f);
  for (const std::string name :
       {"rotary_dim",
        "rope_theta",
        "rope_scaling_rope_type",
        "rope_scaling_factor",
        "rope_scaling_low_freq_factor",
        "rope_scaling_high_freq_factor",
        "rope_extrapolation_factor",
        "rope_scaling_original_max_position_embeddings",
        "rope_scaling_beta_fast",
        "rope_scaling_beta_slow",
        "rope_scaling_attn_factor",
        "rope_scaling_mscale",
        "rope_scaling_mscale_all_dim",
        "rope_scaling_mrope_section",
        "rope_scaling_mrope_interleaved",
        "rope_scaling",
        "partial_rotary_factor"}) {
    EXPECT_EQ(visitor.visit_counts_.at(name), 1) << name;
    EXPECT_EQ(visitor.visit_counts_.count("rope_args." + name), 0) << name;
  }
  EXPECT_EQ(visitor.visit_counts_.count("rope_args"), 0);
}

TEST(PropertyReflectTest, OwnsIndependentRopeScalingWithoutChangingContext) {
  ModelArgs original{};
  original.rope_scaling_rope_type(/*value=*/"mrope")
      .rope_scaling_mrope_section(/*value=*/{16, 24, 24})
      .rotary_dim(/*value=*/64)
      .head_dim(/*value=*/256)
      .max_position_embeddings(/*value=*/131072)
      .use_absolute_position_embedding(/*value=*/false);

  ModelArgs copied = original;
  copied.rope_args()
      .rope_scaling_rope_type(/*value=*/"yarn")
      .rotary_dim(/*value=*/32);
  copied.rope_scaling_mrope_section()[0] = 8;
  copied.rope_args().rope_scaling_mrope_interleaved(/*value=*/true);
  EXPECT_EQ(original.rope_scaling_rope_type(), "mrope");
  EXPECT_EQ(original.rope_scaling_mrope_section(),
            (std::vector<int64_t>{16, 24, 24}));
  EXPECT_EQ(original.rotary_dim(), 64);
  EXPECT_FALSE(original.rope_scaling_mrope_interleaved());
  EXPECT_EQ(copied.rope_scaling_rope_type(), "yarn");
  EXPECT_EQ(copied.rope_scaling_mrope_section(),
            (std::vector<int64_t>{8, 24, 24}));
  EXPECT_TRUE(copied.rope_scaling_mrope_interleaved());
  EXPECT_EQ(copied.rotary_dim(), 32);
  EXPECT_EQ(copied.head_dim(), 256);
  EXPECT_EQ(copied.max_position_embeddings(), 131072);
  EXPECT_FALSE(copied.use_absolute_position_embedding());
}

TEST(PropertyReflectTest,
     SharesMultimodalTokenStorageAndKeepsFlatReflectionKeys) {
  static_assert(std::is_same_v<
                decltype(std::declval<const ModelArgs&>().image_token_id()),
                const int32_t&>);
  static_assert(
      std::is_same_v<decltype(std::declval<ModelArgs&>().image_token_id()),
                     int32_t&>);
  static_assert(noexcept(std::declval<const ModelArgs&>().image_token_id()));
  static_assert(noexcept(std::declval<ModelArgs&>().image_token_id()));

  ModelArgs args{};
  ModelArgs& chained = args.vision_start_token_id(/*value=*/101)
                           .vision_end_token_id(/*value=*/102)
                           .vision_token_id(/*value=*/103)
                           .image_token_id(/*value=*/104)
                           .video_token_id(/*value=*/105)
                           .audio_token_id(/*value=*/106)
                           .image_start_token_id(/*value=*/107)
                           .image_end_token_id(/*value=*/108)
                           .video_start_token_id(/*value=*/109)
                           .video_end_token_id(/*value=*/110);
  EXPECT_EQ(&chained, &args);

  const std::array<std::pair<const char*, int32_t>, 10> fields = {{
      {"vision_start_token_id", 101},
      {"vision_end_token_id", 102},
      {"vision_token_id", 103},
      {"image_token_id", 104},
      {"video_token_id", 105},
      {"audio_token_id", 106},
      {"image_start_token_id", 107},
      {"image_end_token_id", 108},
      {"video_start_token_id", 109},
      {"video_end_token_id", 110},
  }};
  const std::array<std::pair<const int32_t*, const int32_t*>, 10> storage = {{
      {&args.vision_start_token_id(),
       &args.multimodal_token_args().vision_start_token_id()},
      {&args.vision_end_token_id(),
       &args.multimodal_token_args().vision_end_token_id()},
      {&args.vision_token_id(),
       &args.multimodal_token_args().vision_token_id()},
      {&args.image_token_id(), &args.multimodal_token_args().image_token_id()},
      {&args.video_token_id(), &args.multimodal_token_args().video_token_id()},
      {&args.audio_token_id(), &args.multimodal_token_args().audio_token_id()},
      {&args.image_start_token_id(),
       &args.multimodal_token_args().image_start_token_id()},
      {&args.image_end_token_id(),
       &args.multimodal_token_args().image_end_token_id()},
      {&args.video_start_token_id(),
       &args.multimodal_token_args().video_start_token_id()},
      {&args.video_end_token_id(),
       &args.multimodal_token_args().video_end_token_id()},
  }};
  for (std::size_t index = 0; index < fields.size(); ++index) {
    EXPECT_EQ(*storage[index].first, fields[index].second)
        << fields[index].first;
    EXPECT_EQ(storage[index].first, storage[index].second)
        << fields[index].first;
  }

  RecordingPropertyVisitor visitor;
  visit_properties(args, visitor);
  for (const auto& [name, value] : fields) {
    EXPECT_EQ(visitor.int32_values_.at(name), value) << name;
    EXPECT_EQ(visitor.visit_counts_.at(name), 1) << name;
    EXPECT_EQ(visitor.visit_counts_.count("multimodal_token_args." +
                                          std::string(name)),
              0)
        << name;
  }
  EXPECT_EQ(visitor.visit_counts_.count("multimodal_token_args"), 0);
}

TEST(PropertyReflectTest, OwnsIndependentMultimodalTokensAfterCopy) {
  ModelArgs original{};
  original.vision_start_token_id(/*value=*/101)
      .image_token_id(/*value=*/104)
      .audio_token_id(/*value=*/106);

  ModelArgs copied = original;
  copied.multimodal_token_args()
      .vision_start_token_id(/*value=*/201)
      .image_token_id(/*value=*/204)
      .audio_token_id(/*value=*/206);

  EXPECT_EQ(original.vision_start_token_id(), 101);
  EXPECT_EQ(original.image_token_id(), 104);
  EXPECT_EQ(original.audio_token_id(), 106);
  EXPECT_EQ(copied.vision_start_token_id(), 201);
  EXPECT_EQ(copied.image_token_id(), 204);
  EXPECT_EQ(copied.audio_token_id(), 206);
}

TEST(PropertyReflectTest,
     SharesSlidingWindowStorageAndKeepsFlatReflectionKeys) {
  static_assert(std::is_same_v<
                decltype(std::declval<const ModelArgs&>().sliding_window()),
                const int32_t&>);
  static_assert(
      std::is_same_v<decltype(std::declval<ModelArgs&>().sliding_window()),
                     int32_t&>);
  static_assert(
      std::is_same_v<decltype(std::declval<ModelArgs&&>().sliding_window()),
                     int32_t&&>);
  static_assert(noexcept(std::declval<const ModelArgs&>().sliding_window()));
  static_assert(noexcept(std::declval<ModelArgs&>().sliding_window()));
  static_assert(noexcept(std::declval<ModelArgs&&>().sliding_window()));

  ModelArgs args{};
  args.sliding_window_args()
      .use_sliding_window(/*value=*/true)
      .sliding_window(/*value=*/4096)
      .max_window_layers(/*value=*/24);
  EXPECT_TRUE(args.use_sliding_window());
  EXPECT_EQ(args.sliding_window(), 4096);
  EXPECT_EQ(args.max_window_layers(), 24);
  EXPECT_EQ(&args.use_sliding_window(),
            &args.sliding_window_args().use_sliding_window());
  EXPECT_EQ(&args.sliding_window(),
            &args.sliding_window_args().sliding_window());
  EXPECT_EQ(&args.max_window_layers(),
            &args.sliding_window_args().max_window_layers());

  ModelArgs& chained = args.use_sliding_window(/*value=*/false)
                           .sliding_window(/*value=*/2048)
                           .max_window_layers(/*value=*/12);
  EXPECT_EQ(&chained, &args);

  RecordingPropertyVisitor visitor;
  visit_properties(args, visitor);
  EXPECT_FALSE(visitor.bool_values_.at("use_sliding_window"));
  EXPECT_EQ(visitor.int32_values_.at("sliding_window"), 2048);
  EXPECT_EQ(visitor.int32_values_.at("max_window_layers"), 12);
  for (const std::string name :
       {"use_sliding_window", "sliding_window", "max_window_layers"}) {
    EXPECT_EQ(visitor.visit_counts_.at(name), 1) << name;
    EXPECT_EQ(visitor.visit_counts_.count("sliding_window_args." + name), 0)
        << name;
  }
  EXPECT_EQ(visitor.visit_counts_.count("sliding_window_args"), 0);
}

TEST(PropertyReflectTest,
     SharesConfidenceHeadStorageAndKeepsFlatReflectionKeys) {
  static_assert(std::is_same_v<decltype(std::declval<const ModelArgs&>()
                                            .enable_confidence_head()),
                               const bool&>);
  static_assert(std::is_same_v<
                decltype(std::declval<ModelArgs&>().enable_confidence_head()),
                bool&>);
  static_assert(std::is_same_v<
                decltype(std::declval<ModelArgs&&>().enable_confidence_head()),
                bool&&>);
  static_assert(
      noexcept(std::declval<const ModelArgs&>().enable_confidence_head()));
  static_assert(noexcept(std::declval<ModelArgs&>().enable_confidence_head()));
  static_assert(noexcept(std::declval<ModelArgs&&>().enable_confidence_head()));

  ModelArgs args{};
  args.confidence_head_args()
      .enable_confidence_head(/*value=*/true)
      .confidence_head_with_markov(/*value=*/false);
  EXPECT_TRUE(args.enable_confidence_head());
  EXPECT_FALSE(args.confidence_head_with_markov());
  EXPECT_EQ(&args.enable_confidence_head(),
            &args.confidence_head_args().enable_confidence_head());
  EXPECT_EQ(&args.confidence_head_with_markov(),
            &args.confidence_head_args().confidence_head_with_markov());

  ModelArgs& chained = args.enable_confidence_head(/*value=*/false)
                           .confidence_head_with_markov(/*value=*/true);
  EXPECT_EQ(&chained, &args);
  EXPECT_FALSE(args.confidence_head_args().enable_confidence_head());
  EXPECT_TRUE(args.confidence_head_args().confidence_head_with_markov());

  RecordingPropertyVisitor visitor;
  visit_properties(args, visitor);
  EXPECT_FALSE(visitor.bool_values_.at("enable_confidence_head"));
  EXPECT_TRUE(visitor.bool_values_.at("confidence_head_with_markov"));
  for (const std::string name :
       {"enable_confidence_head", "confidence_head_with_markov"}) {
    EXPECT_EQ(visitor.visit_counts_.at(name), 1) << name;
    EXPECT_EQ(visitor.visit_counts_.count("confidence_head_args." + name), 0)
        << name;
  }
  EXPECT_EQ(visitor.visit_counts_.count("confidence_head_args"), 0);
}

TEST(PropertyReflectTest,
     SharesImagePreprocessStorageAndPreservesFlatReflectionKeys) {
  ModelArgs args{};
  args.image_preprocess_args().mm_image_do_center_crop() = true;
  args.image_preprocess_args().mm_image_crop_height_size() = 224;
  args.image_preprocess_args().mm_image_crop_width_size() = 256;
  args.image_preprocess_args().mm_image_do_resize() = true;
  args.image_preprocess_args().mm_image_resize_shortest_edge() = 384;
  args.image_preprocess_args().mm_image_resample() = 3;
  args.image_preprocess_args().mm_image_do_rescale() = true;
  args.image_preprocess_args().mm_image_rescale_factor() = 0.5;
  args.image_preprocess_args().mm_image_do_normalize() = true;
  args.image_preprocess_args().mm_image_normalize_mean() = {0.1, 0.2, 0.3};
  args.image_preprocess_args().mm_image_normalize_std() = {0.9, 0.8, 0.7};

  EXPECT_TRUE(args.mm_image_do_center_crop());
  EXPECT_EQ(args.mm_image_crop_height_size(), 224);
  EXPECT_EQ(args.mm_image_crop_width_size(), 256);
  EXPECT_TRUE(args.mm_image_do_resize());
  EXPECT_EQ(args.mm_image_resize_shortest_edge(), 384);
  EXPECT_EQ(args.mm_image_resample(), 3);
  EXPECT_TRUE(args.mm_image_do_rescale());
  EXPECT_DOUBLE_EQ(args.mm_image_rescale_factor(), 0.5);
  EXPECT_TRUE(args.mm_image_do_normalize());
  EXPECT_EQ(args.mm_image_normalize_mean(),
            (std::vector<double>{0.1, 0.2, 0.3}));
  EXPECT_EQ(args.mm_image_normalize_std(),
            (std::vector<double>{0.9, 0.8, 0.7}));

  RecordingPropertyVisitor visitor;
  visit_properties(args, visitor);
  const std::array<const char*, 11> keys = {
      "mm_image_do_center_crop",
      "mm_image_crop_height_size",
      "mm_image_crop_width_size",
      "mm_image_do_resize",
      "mm_image_resize_shortest_edge",
      "mm_image_resample",
      "mm_image_do_rescale",
      "mm_image_rescale_factor",
      "mm_image_do_normalize",
      "mm_image_normalize_mean",
      "mm_image_normalize_std",
  };
  for (const char* key : keys) {
    const std::string key_name(key);
    EXPECT_EQ(visitor.visit_counts_.at(key_name), 1) << key_name;
    EXPECT_EQ(visitor.visit_counts_.count("image_preprocess_args." + key_name),
              0)
        << key_name;
  }
}

TEST(PropertyReflectTest, OwnsIndependentImagePreprocessVectorsAfterCopy) {
  ModelArgs original{};
  original.mm_image_normalize_mean() = {0.1, 0.2, 0.3};
  original.mm_image_normalize_std() = {0.9, 0.8, 0.7};

  ModelArgs copy = original;
  copy.image_preprocess_args().mm_image_normalize_mean()[0] = 1.0;
  copy.mm_image_normalize_std()[1] = 2.0;

  EXPECT_DOUBLE_EQ(original.mm_image_normalize_mean()[0], 0.1);
  EXPECT_DOUBLE_EQ(original.image_preprocess_args().mm_image_normalize_std()[1],
                   0.8);
  EXPECT_DOUBLE_EQ(copy.mm_image_normalize_mean()[0], 1.0);
  EXPECT_DOUBLE_EQ(copy.image_preprocess_args().mm_image_normalize_std()[1],
                   2.0);
}

TEST(PropertyReflectTest, PreservesOptionalAttentionHeadFallbacksAfterCopy) {
  ModelArgs args{};
  args.n_heads(/*value=*/32).decoder_n_heads(/*value=*/16);
  EXPECT_EQ(args.n_kv_heads().value_or(args.n_heads()), 32);
  EXPECT_EQ(args.decoder_n_kv_heads().value_or(args.decoder_n_heads()), 16);

  args.n_kv_heads(/*value=*/0);
  args.attention_head_args().decoder_n_kv_heads(/*value=*/8);
  EXPECT_EQ(args.n_kv_heads().value_or(args.n_heads()), 0);
  EXPECT_EQ(args.decoder_n_kv_heads().value_or(args.decoder_n_heads()), 8);

  ModelArgs copied = args;
  copied.n_kv_heads().reset();
  copied.attention_head_args().decoder_n_kv_heads().reset();
  EXPECT_EQ(copied.n_kv_heads().value_or(copied.n_heads()), 32);
  EXPECT_EQ(copied.decoder_n_kv_heads().value_or(copied.decoder_n_heads()), 16);
  EXPECT_EQ(args.n_kv_heads(), 0);
  EXPECT_EQ(args.decoder_n_kv_heads(), 8);

  RecordingPropertyVisitor original_visitor;
  visit_properties(args, original_visitor);
  EXPECT_EQ(original_visitor.int64_values_.at("n_kv_heads"), 0);
  EXPECT_EQ(original_visitor.int64_values_.at("decoder_n_kv_heads"), 8);

  RecordingPropertyVisitor copied_visitor;
  visit_properties(copied, copied_visitor);
  for (const std::string name : {"n_kv_heads", "decoder_n_kv_heads"}) {
    EXPECT_EQ(original_visitor.absent_names_.count(name), 0) << name;
    EXPECT_EQ(original_visitor.visit_counts_.at(name), 1) << name;
    EXPECT_EQ(copied_visitor.absent_names_.count(name), 1) << name;
    EXPECT_EQ(copied_visitor.int64_values_.count(name), 0) << name;
    EXPECT_EQ(copied_visitor.visit_counts_.at(name), 1) << name;
  }
}

TEST(PropertyReflectTest,
     PreservesRemainingModelArgumentGroupsAndFlatReflectionKeys) {
  const ModelArgs args{};
  EXPECT_EQ(args.model_type(), "");
  EXPECT_EQ(args.dtype(), "");
  EXPECT_EQ(args.hidden_size(), 0);
  EXPECT_EQ(args.kv_cache_storage_head_dim(), 0);
  EXPECT_EQ(args.max_batch_size(), 0);
  EXPECT_EQ(args.max_seq_len(), 0);
  EXPECT_FALSE(args.use_qk_norm());
  EXPECT_FALSE(args.attn_scalar().has_value());
  EXPECT_EQ(args.draft_vocab_size(), 0);
  EXPECT_EQ(args.index_kpool(), 1);
  EXPECT_EQ(args.index_topk_freq(), 1);
  EXPECT_FLOAT_EQ(args.hc_eps(), 1e-6f);

  RecordingPropertyVisitor visitor;
  visit_properties(args, visitor);

  const std::vector<std::string> model_architecture_names = {
      "model_type",
      "dtype",
      "hidden_size",
      "hidden_act",
      "intermediate_size",
      "n_layers",
      "n_encoder_layers",
      "max_position_embeddings",
      "use_absolute_position_embedding",
      "initializer_range",
      "rms_norm_eps",
      "layer_norm_eps",
      "mlp_layer_types",
      "in_channels",
      "out_channels"};
  const std::vector<std::string> token_embedding_names = {
      "vocab_size",
      "tie_word_embeddings",
      "encoder_embedding_mode",
      "embedding_mode"};
  const std::vector<std::string> attention_head_names = {"head_dim",
                                                         "decoder_head_dim",
                                                         "n_heads",
                                                         "decoder_n_heads",
                                                         "actual_n_heads",
                                                         "num_attention_heads",
                                                         "n_kv_heads",
                                                         "decoder_n_kv_heads"};
  const std::vector<std::string> kv_cache_names = {
      "kv_cache_storage_head_dim", "max_batch_size", "max_seq_len"};
  const std::vector<std::string> attention_config_names = {"use_qk_norm",
                                                           "attn_scalar",
                                                           "no_bias",
                                                           "qkv_bias",
                                                           "attention_bias",
                                                           "attention_dropout",
                                                           "attn_output_gate"};
  const std::vector<std::string> speculative_names = {
      "draft_vocab_size",
      "max_concurrent_prefills_per_dp",
      "layers_to_capture",
      "norm_before_fc",
      "norm_before_residual",
      "norm_output",
      "fc_norm"};
  const std::vector<std::string> indexer_names = {
      "index_head_dim",
      "index_n_heads",
      "index_topk",
      "index_kpool",
      "index_kpool_compress",
      "index_kpool_always_select_tail",
      "indexer_rope_interleave",
      "index_topk_freq",
      "index_topk_pattern",
      "index_skip_topk_offset",
      "index_share_for_mtp_iteration",
      "indexer_types"};
  const std::vector<std::string> deepseek_v4_names = {
      "rope_head_dim",
      "o_lora_rank",
      "o_groups",
      "compress_ratios",
      "compress_rope_theta",
      "window_size",
      "n_activated_experts",
      "n_hash_layers",
      "factor",
      "beta_fast",
      "beta_slow",
      "scale_fmt",
  };
  const std::vector<std::string> hyperconnection_names = {
      "hc_mult", "hc_sinkhorn_iters", "hc_eps"};
  const std::vector<std::string> vision_names = {
      "query_num",
      "vision_custom_adapter",
      "vision_max_slice_nums",
      "mm_dropout",
      "mm_hidden_act",
      "mm_hidden_size",
      "mm_image_size",
      "mm_intermediate_size",
      "mm_num_channels",
      "mm_initializer_range",
      "mm_layer_norm_eps",
      "mm_num_attention_heads",
      "mm_num_hidden_layers",
      "mm_output_attentions",
      "mm_output_hidden_states",
      "mm_output_scores",
      "mm_patch_size",
      "mm_projection_dim",
      "mm_projector_hidden_size",
      "mm_spatial_merge_size",
      "mm_spatial_patch_size",
      "mm_remove_invalid_values",
      "mm_repetition_penalty",
      "mm_return_dict",
      "mm_return_dict_in_generate",
      "mm_temperature",
      "mm_tie_encoder_decoder",
      "mm_tie_word_embeddings",
      "mm_top_k",
      "mm_top_p",
      "mm_torchscript",
      "mm_use_bfloat16",
      "mm_head_dim",
      "mm_vocab_size",
      "mm_window_size",
      "mm_fullatt_block_indexes",
      "mm_deepstack_visual_indexes",
      "mm_tokens_per_second",
      "mm_temporal_patch_size",
      "mm_projector_type",
      "mm_num_position_embeddings",
      "mm_projector_hidden_act",
      "mm_projector_n_layers",
      "mm_vision_feature_layer",
      "mm_vision_feature_select_strategy",
      "mm_init_pos_emb_width",
      "mm_init_pos_emb_height",
      "mm_init_pos_emb_time"};
  const std::vector<std::string> moe_names = {
      "first_k_dense_replace",
      "moe_layer_freq",
      "use_moe",
      "moe_score_func",
      "moe_route_scale",
      "moe_use_shared_experts",
      "swiglu_limit",
      "decoder_sparse_step",
      "mlp_only_layers",
      "num_experts",
      "output_router_logits",
      "router_aux_loss_coef",
      "shared_expert_intermediate_size"};

  for (const auto& names : {model_architecture_names,
                            token_embedding_names,
                            attention_head_names,
                            kv_cache_names,
                            attention_config_names,
                            speculative_names,
                            indexer_names,
                            deepseek_v4_names,
                            hyperconnection_names,
                            vision_names,
                            moe_names}) {
    for (const std::string& name : names) {
      EXPECT_EQ(visitor.visit_counts_.at(name), 1) << name;
    }
  }
  for (const std::string& group : {"model_architecture_args",
                                   "token_embedding_args",
                                   "attention_head_args",
                                   "kv_cache_args",
                                   "attention_config_args",
                                   "speculative_args",
                                   "indexer_args",
                                   "deepseek_v4_args",
                                   "hyperconnection_args",
                                   "vision_args",
                                   "moe_args"}) {
    EXPECT_EQ(visitor.visit_counts_.count(group), 0) << group;
  }
  EXPECT_EQ(visitor.visit_counts_.count("model_architecture_args.model_type"),
            0);
  EXPECT_EQ(visitor.visit_counts_.count("moe_args.first_k_dense_replace"), 0);
}

TEST(PropertyReflectTest, KeepsMovedFieldsInIndependentArgumentGroups) {
  ModelArgs args{};
  args.token_embedding_args()
      .encoder_embedding_mode(/*value=*/true)
      .embedding_mode(/*value=*/true);
  args.attention_head_args().num_attention_heads(/*value=*/96);
  args.model_architecture_args()
      .initializer_range(/*value=*/0.01f)
      .rms_norm_eps(/*value=*/1e-5f)
      .layer_norm_eps(/*value=*/1e-6f)
      .mlp_layer_types(/*value=*/std::vector<std::string>{"dense", "sparse"})
      .in_channels(/*value=*/16)
      .out_channels(/*value=*/32);
  args.vision_args().query_num(/*value=*/64);
  args.mm_init_pos_emb_width(/*value=*/192);
  args.mm_init_pos_emb_height(/*value=*/128);
  args.mm_init_pos_emb_time(/*value=*/6);
  args.hyperconnection_args()
      .hc_mult(/*value=*/4)
      .hc_sinkhorn_iters(/*value=*/8)
      .hc_eps(/*value=*/1e-4f);

  EXPECT_TRUE(args.encoder_embedding_mode());
  EXPECT_TRUE(args.embedding_mode());
  EXPECT_EQ(args.num_attention_heads(), 96);
  EXPECT_FLOAT_EQ(args.initializer_range(), 0.01f);
  EXPECT_FLOAT_EQ(args.rms_norm_eps(), 1e-5f);
  EXPECT_FLOAT_EQ(args.layer_norm_eps(), 1e-6f);
  EXPECT_EQ(args.mlp_layer_types(),
            (std::vector<std::string>{"dense", "sparse"}));
  EXPECT_EQ(args.in_channels(), 16);
  EXPECT_EQ(args.out_channels(), 32);
  EXPECT_EQ(args.query_num(), 64);
  EXPECT_EQ(args.mm_init_pos_emb_width(), 192);
  EXPECT_EQ(args.mm_init_pos_emb_height(), 128);
  EXPECT_EQ(args.mm_init_pos_emb_time(), 6);
  EXPECT_EQ(args.hc_mult(), 4);
  EXPECT_EQ(args.hc_sinkhorn_iters(), 8);
  EXPECT_FLOAT_EQ(args.hc_eps(), 1e-4f);

  // Replacing one functional group must not reset the fields owned by the
  // other groups.
  args.moe_args(/*value=*/MoEArgs{});
  args.attention_config_args(/*value=*/AttentionConfigArgs{});
  args.deepseek_v4_args(/*value=*/DeepseekV4Args{});
  args.vision_args(/*value=*/VisionArgs{});
  EXPECT_EQ(args.num_attention_heads(), 96);
  EXPECT_FLOAT_EQ(args.rms_norm_eps(), 1e-5f);
  EXPECT_EQ(args.query_num(), 0);
  EXPECT_EQ(args.mm_init_pos_emb_width(), 64);
  EXPECT_EQ(args.mm_init_pos_emb_height(), 64);
  EXPECT_EQ(args.mm_init_pos_emb_time(), 4);
  EXPECT_EQ(args.hc_mult(), 4);
  EXPECT_EQ(args.hc_sinkhorn_iters(), 8);
  EXPECT_FLOAT_EQ(args.hc_eps(), 1e-4f);

  args.mm_init_pos_emb_width(/*value=*/192);
  args.mm_init_pos_emb_height(/*value=*/128);
  args.mm_init_pos_emb_time(/*value=*/6);
  args.image_video_args(/*value=*/ImageVideoArgs{});
  EXPECT_EQ(args.mm_init_pos_emb_width(), 192);
  EXPECT_EQ(args.mm_init_pos_emb_height(), 128);
  EXPECT_EQ(args.mm_init_pos_emb_time(), 6);

  RecordingPropertyVisitor visitor;
  visit_properties(args, visitor);
  for (const std::string& name : {"encoder_embedding_mode",
                                  "embedding_mode",
                                  "num_attention_heads",
                                  "initializer_range",
                                  "rms_norm_eps",
                                  "layer_norm_eps",
                                  "mlp_layer_types",
                                  "in_channels",
                                  "out_channels",
                                  "query_num",
                                  "mm_init_pos_emb_width",
                                  "mm_init_pos_emb_height",
                                  "mm_init_pos_emb_time",
                                  "hc_mult",
                                  "hc_sinkhorn_iters",
                                  "hc_eps"}) {
    EXPECT_EQ(visitor.visit_counts_.at(name), 1) << name;
    EXPECT_EQ(visitor.visit_counts_.count("model_args." + name), 0) << name;
  }
}

TEST(PropertyReflectTest,
     SharesRemainingModelArgumentGroupStorageAndFlatAccessors) {
  ModelArgs args{};
  args.model_architecture_args()
      .model_type(/*value=*/"qwen3")
      .hidden_size(/*value=*/4096);
  args.kv_cache_args().max_seq_len(/*value=*/32768);
  args.attention_config_args().qkv_bias(/*value=*/true);
  args.indexer_args().index_topk(/*value=*/64);
  args.deepseek_v4_args().rope_head_dim(/*value=*/64);
  args.moe_args().first_k_dense_replace(/*value=*/3);

  EXPECT_EQ(args.model_type(), "qwen3");
  EXPECT_EQ(args.hidden_size(), 4096);
  EXPECT_EQ(args.max_seq_len(), 32768);
  EXPECT_TRUE(args.qkv_bias());
  EXPECT_EQ(args.index_topk(), 64);
  EXPECT_EQ(args.rope_head_dim(), 64);
  EXPECT_EQ(args.first_k_dense_replace(), 3);
  EXPECT_EQ(&args.model_type(), &args.model_architecture_args().model_type());
  EXPECT_EQ(&args.max_seq_len(), &args.kv_cache_args().max_seq_len());
  EXPECT_EQ(&args.qkv_bias(), &args.attention_config_args().qkv_bias());
  EXPECT_EQ(&args.index_topk(), &args.indexer_args().index_topk());
  EXPECT_EQ(&args.rope_head_dim(), &args.deepseek_v4_args().rope_head_dim());
  EXPECT_EQ(&args.first_k_dense_replace(),
            &args.moe_args().first_k_dense_replace());

  RecordingPropertyVisitor visitor;
  visit_properties(args, visitor);
  EXPECT_EQ(visitor.string_values_.at("model_type"), "qwen3");
  EXPECT_EQ(visitor.int64_values_.at("hidden_size"), 4096);
  EXPECT_EQ(visitor.int64_values_.at("max_seq_len"), 32768);
  EXPECT_TRUE(visitor.bool_values_.at("qkv_bias"));
  EXPECT_EQ(visitor.int32_values_.at("index_topk"), 64);
  EXPECT_EQ(visitor.int32_values_.at("rope_head_dim"), 64);
  EXPECT_EQ(visitor.int32_values_.at("first_k_dense_replace"), 3);
}

TEST(PropertyReflectTest, OwnsRemainingModelArgumentGroupsAfterCopy) {
  ModelArgs original{};
  original.speculative_args().layers_to_capture(/*value=*/{1, 4});
  original.indexer_args().indexer_types(/*value=*/{"dense", "sparse"});
  original.deepseek_v4_args().compress_ratios(/*value=*/{2, 4});
  original.moe_args().mlp_only_layers(/*value=*/{0, 2});

  ModelArgs copied = original;
  copied.speculative_args().layers_to_capture(/*value=*/{3});
  copied.indexer_args().indexer_types(/*value=*/{"shared"});
  copied.deepseek_v4_args().compress_ratios(/*value=*/{8});
  copied.moe_args().mlp_only_layers(/*value=*/{1});

  EXPECT_EQ(original.layers_to_capture(), std::vector<int32_t>({1, 4}));
  EXPECT_EQ(original.indexer_types(),
            std::vector<std::string>({"dense", "sparse"}));
  EXPECT_EQ(original.compress_ratios(), std::vector<int32_t>({2, 4}));
  EXPECT_EQ(original.mlp_only_layers(), std::vector<int32_t>({0, 2}));
  EXPECT_EQ(copied.layers_to_capture(), std::vector<int32_t>({3}));
  EXPECT_EQ(copied.indexer_types(), std::vector<std::string>({"shared"}));
  EXPECT_EQ(copied.compress_ratios(), std::vector<int32_t>({8}));
  EXPECT_EQ(copied.mlp_only_layers(), std::vector<int32_t>({1}));
}

}  // namespace
}  // namespace xllm
