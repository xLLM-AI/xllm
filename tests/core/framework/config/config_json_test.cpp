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

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>

#include "core/common/global_flags.h"
#include "core/framework/config/config_utils.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"

namespace xllm {
namespace {

inline constexpr std::string_view kInlineConfig = R"json({
  "block_size": 16,
  "max_memory_utilization": 0.5,
  "enable_prefix_cache": false,
  "max_tokens_per_batch": 8192,
  "max_seqs_per_batch": 64,
  "model_impl": "py",
  "python_model_path": "/tmp/xllm-python-model",
  "python_graph_backend": "cudagraphs"
})json";

inline constexpr std::string_view kUpdatedConfig = R"json({
  "block_size": 32,
  "max_tokens_per_batch": 4096
})json";

inline constexpr std::string_view kMalformedConfig = R"json({
  "block_size":
})json";

class ConfigJsonFileFlagGuard final {
 public:
  explicit ConfigJsonFileFlagGuard(const std::string& config_json_file)
      : old_config_json_file_(FLAGS_config_json_file) {
    FLAGS_config_json_file = config_json_file;
  }

  ~ConfigJsonFileFlagGuard() { FLAGS_config_json_file = old_config_json_file_; }

 private:
  std::string old_config_json_file_;
};

class DumpConfigJsonFlagGuard final {
 public:
  explicit DumpConfigJsonFlagGuard(const std::string& dump_config_json_file)
      : old_enable_dump_config_json_(FLAGS_enable_dump_config_json),
        old_dump_config_json_file_(FLAGS_dump_config_json_file) {
    FLAGS_dump_config_json_file = dump_config_json_file;
  }

  ~DumpConfigJsonFlagGuard() {
    FLAGS_enable_dump_config_json = old_enable_dump_config_json_;
    FLAGS_dump_config_json_file = old_dump_config_json_file_;
  }

 private:
  bool old_enable_dump_config_json_;
  std::string old_dump_config_json_file_;
};

class CpSizeFlagGuard final {
 public:
  CpSizeFlagGuard() : old_cp_size_(FLAGS_cp_size) {}
  ~CpSizeFlagGuard() { FLAGS_cp_size = old_cp_size_; }

 private:
  int32_t old_cp_size_;
};

class ConfigFlagGuard final {
 public:
  ConfigFlagGuard()
      : old_block_size_(FLAGS_block_size),
        old_max_memory_utilization_(FLAGS_max_memory_utilization),
        old_enable_prefix_cache_(FLAGS_enable_prefix_cache),
        old_max_tokens_per_batch_(FLAGS_max_tokens_per_batch),
        old_max_seqs_per_batch_(FLAGS_max_seqs_per_batch),
        old_model_impl_(FLAGS_model_impl),
        old_python_model_path_(FLAGS_python_model_path),
        old_python_graph_backend_(FLAGS_python_graph_backend) {}

  ~ConfigFlagGuard() {
    FLAGS_block_size = old_block_size_;
    FLAGS_max_memory_utilization = old_max_memory_utilization_;
    FLAGS_enable_prefix_cache = old_enable_prefix_cache_;
    FLAGS_max_tokens_per_batch = old_max_tokens_per_batch_;
    FLAGS_max_seqs_per_batch = old_max_seqs_per_batch_;
    FLAGS_model_impl = old_model_impl_;
    FLAGS_python_model_path = old_python_model_path_;
    FLAGS_python_graph_backend = old_python_graph_backend_;
  }

 private:
  int32_t old_block_size_;
  double old_max_memory_utilization_;
  bool old_enable_prefix_cache_;
  int32_t old_max_tokens_per_batch_;
  int32_t old_max_seqs_per_batch_;
  std::string old_model_impl_;
  std::string old_python_model_path_;
  std::string old_python_graph_backend_;
};

class StartupConfigGuard final {
 public:
  StartupConfigGuard()
      : model_config_(ModelConfig::get_instance()),
        execution_config_(ExecutionConfig::get_instance()),
        kv_cache_config_(KVCacheConfig::get_instance()),
        scheduler_config_(SchedulerConfig::get_instance()),
        old_model_impl_(model_config_.model_impl()),
        old_python_model_path_(model_config_.python_model_path()),
        old_python_graph_backend_(execution_config_.python_graph_backend()),
        old_block_size_(kv_cache_config_.block_size()),
        old_enable_prefix_cache_(kv_cache_config_.enable_prefix_cache()),
        old_max_tokens_per_batch_(scheduler_config_.max_tokens_per_batch()),
        old_max_seqs_per_batch_(scheduler_config_.max_seqs_per_batch()),
        old_enable_chunked_prefill_(
            scheduler_config_.enable_chunked_prefill()) {}

  ~StartupConfigGuard() {
    model_config_.model_impl(old_model_impl_)
        .python_model_path(old_python_model_path_);
    execution_config_.python_graph_backend(old_python_graph_backend_);
    kv_cache_config_.block_size(old_block_size_)
        .enable_prefix_cache(old_enable_prefix_cache_);
    scheduler_config_.max_tokens_per_batch(old_max_tokens_per_batch_)
        .max_seqs_per_batch(old_max_seqs_per_batch_)
        .enable_chunked_prefill(old_enable_chunked_prefill_);
  }

 private:
  ModelConfig& model_config_;
  ExecutionConfig& execution_config_;
  KVCacheConfig& kv_cache_config_;
  SchedulerConfig& scheduler_config_;
  std::string old_model_impl_;
  std::string old_python_model_path_;
  std::string old_python_graph_backend_;
  int32_t old_block_size_;
  bool old_enable_prefix_cache_;
  int32_t old_max_tokens_per_batch_;
  int32_t old_max_seqs_per_batch_;
  bool old_enable_chunked_prefill_;
};

void write_config_file(const std::filesystem::path& config_path,
                       std::string_view config_json) {
  std::ofstream config_file(config_path);
  config_file << config_json;
}

nlohmann::ordered_json read_json_file(const std::filesystem::path& file_path) {
  std::ifstream input_file(file_path);
  EXPECT_TRUE(input_file.is_open()) << file_path;
  return nlohmann::ordered_json::parse(input_file);
}

std::filesystem::path config_test_file_path() {
  const std::filesystem::path source_config_path =
      std::filesystem::path(__FILE__).parent_path() / "config_test.json";
  if (std::filesystem::exists(source_config_path)) {
    return source_config_path;
  }

  const std::filesystem::path copied_config_path = "config_test.json";
  if (std::filesystem::exists(copied_config_path)) {
    return copied_config_path;
  }

  return std::filesystem::path("tests/core/framework/config/config_test.json");
}

TEST(ConfigJsonTest, FromJsonUsesParsedOverrides) {
  const JsonReader json = config::parse_json_string(kInlineConfig);
  ConfigFlagGuard flag_guard;

  ModelConfig model_config;
  model_config.from_json(json);

  ExecutionConfig execution_config;
  execution_config.from_json(json);

  KVCacheConfig kv_cache_config;
  kv_cache_config.from_flags();
  kv_cache_config.from_json(json);

  SchedulerConfig scheduler_config;
  scheduler_config.from_flags();
  scheduler_config.from_json(json);

  EXPECT_EQ(kv_cache_config.block_size(), 16);
  EXPECT_DOUBLE_EQ(kv_cache_config.max_memory_utilization(), 0.5);
  EXPECT_FALSE(kv_cache_config.enable_prefix_cache());
  EXPECT_EQ(scheduler_config.max_tokens_per_batch(), 8192);
  EXPECT_EQ(scheduler_config.max_seqs_per_batch(), 64);
  // model_impl is no longer canonicalized: the raw "py" alias is preserved and
  // recognized via is_python_model_impl(). from_json still mirrors it into the
  // matching gflag.
  EXPECT_EQ(model_config.model_impl(), "py");
  EXPECT_TRUE(ModelConfig::is_python_model_impl(model_config.model_impl()));
  EXPECT_EQ(model_config.python_model_path(), "/tmp/xllm-python-model");
  EXPECT_EQ(execution_config.python_graph_backend(), "cudagraphs");

  EXPECT_EQ(FLAGS_model_impl, "py");
  EXPECT_EQ(FLAGS_python_model_path, "/tmp/xllm-python-model");
  EXPECT_EQ(FLAGS_python_graph_backend, "cudagraphs");

  EXPECT_EQ(kv_cache_config.kv_cache_dtype(), "auto");
  EXPECT_EQ(kv_cache_config.indexer_cache_dtype(), "auto");
  EXPECT_EQ(scheduler_config.max_decode_token_per_sequence(), 256);
}

TEST(KVCacheConfigValidationTest, AcceptsSupportedIndexerCacheDtypes) {
  KVCacheConfig config;

  config.indexer_cache_dtype("auto");
  config.validate();

  config.indexer_cache_dtype("int8");
  config.validate();
}

TEST(KVCacheConfigValidationTest, RejectsUnsupportedIndexerCacheDtypes) {
  EXPECT_DEATH(
      {
        KVCacheConfig config;
        config.indexer_cache_dtype("fp8");
        config.validate();
      },
      "indexer_cache_dtype.*auto.*int8");
  EXPECT_DEATH(
      {
        KVCacheConfig config;
        config.indexer_cache_dtype("INT8");
        config.validate();
      },
      "indexer_cache_dtype.*auto.*int8");
  EXPECT_DEATH(
      {
        KVCacheConfig config;
        config.indexer_cache_dtype(" int8 ");
        config.validate();
      },
      "indexer_cache_dtype.*auto.*int8");
}

TEST(ConfigJsonTest, ParallelConfigReadsContextParallelSize) {
  CpSizeFlagGuard flag_guard;
  const JsonReader json =
      config::parse_json_string(R"json({"cp_size": 4})json");
  ParallelConfig parallel_config;
  parallel_config.from_json(json);

  EXPECT_EQ(parallel_config.cp_size(), 4);
}

TEST(ConfigJsonTest, RegistersOnlyContextParallelCommandLineOption) {
  google::CommandLineFlagInfo flag_info;
  EXPECT_TRUE(google::GetCommandLineFlagInfo("cp_size", &flag_info));
  EXPECT_EQ(flag_info.default_value, "1");

  const std::string removed_flag = std::string("enable_") + "prefill_sp";
  EXPECT_FALSE(
      google::GetCommandLineFlagInfo(removed_flag.c_str(), &flag_info));
}

TEST(ConfigJsonTest, LoadJsonFileReadsConfigFixture) {
  ConfigFlagGuard flag_guard;
  const std::filesystem::path config_path = config_test_file_path();
  ASSERT_TRUE(std::filesystem::exists(config_path)) << config_path;

  const JsonReader json = config::load_json_file(config_path.string());

  KVCacheConfig kv_cache_config;
  kv_cache_config.from_flags();
  kv_cache_config.from_json(json);

  SchedulerConfig scheduler_config;
  scheduler_config.from_flags();
  scheduler_config.from_json(json);

  EXPECT_EQ(kv_cache_config.block_size(), 24);
  EXPECT_EQ(kv_cache_config.max_cache_size(), 1048576);
  EXPECT_DOUBLE_EQ(kv_cache_config.max_memory_utilization(), 0.65);
  EXPECT_EQ(kv_cache_config.kv_cache_dtype(), "int8");
  EXPECT_EQ(kv_cache_config.indexer_cache_dtype(), "int8");
  EXPECT_FALSE(kv_cache_config.enable_prefix_cache());
  EXPECT_EQ(kv_cache_config.xxh3_128bits_seed(), 2048);
  EXPECT_TRUE(kv_cache_config.enable_xtensor());
  EXPECT_EQ(kv_cache_config.phy_page_granularity_size(), 4096);

  EXPECT_EQ(scheduler_config.max_tokens_per_batch(), 2048);
  EXPECT_EQ(scheduler_config.max_seqs_per_batch(), 32);
  EXPECT_TRUE(scheduler_config.enable_schedule_overlap());
  EXPECT_DOUBLE_EQ(scheduler_config.prefill_scheduling_memory_usage_threshold(),
                   0.75);
  EXPECT_FALSE(scheduler_config.enable_chunked_prefill());
  EXPECT_EQ(scheduler_config.max_tokens_per_chunk_for_prefill(), 512);
  EXPECT_EQ(scheduler_config.chunked_match_frequency(), 3);
  EXPECT_TRUE(scheduler_config.use_zero_evict());
  EXPECT_EQ(scheduler_config.max_decode_token_per_sequence(), 128);
  EXPECT_EQ(scheduler_config.priority_strategy(), "priority");
  EXPECT_FALSE(scheduler_config.enable_online_preempt_offline());
  EXPECT_DOUBLE_EQ(scheduler_config.aggressive_coeff(), 1.5);
  EXPECT_DOUBLE_EQ(scheduler_config.starve_threshold(), 2.0);
  EXPECT_FALSE(scheduler_config.enable_starve_prevent());
}

TEST(ConfigJsonTest, InitializeLoadsConfigJsonFileFromFlag) {
  ConfigFlagGuard config_flag_guard;
  const std::filesystem::path config_path =
      std::filesystem::temp_directory_path() / "xllm_config_json_test.json";
  write_config_file(config_path, kInlineConfig);

  ConfigJsonFileFlagGuard flag_guard(config_path.string());

  KVCacheConfig kv_cache_config;
  kv_cache_config.initialize();

  SchedulerConfig scheduler_config;
  scheduler_config.initialize();

  EXPECT_EQ(kv_cache_config.block_size(), 16);
  EXPECT_EQ(scheduler_config.max_tokens_per_batch(), 8192);

  std::filesystem::remove(config_path);
}

TEST(ConfigJsonTest, InitializeReusesCachedConfigJsonForSameFile) {
  ConfigFlagGuard config_flag_guard;
  const std::filesystem::path config_path =
      std::filesystem::temp_directory_path() /
      "xllm_config_json_test_cached.json";
  write_config_file(config_path, kInlineConfig);

  ConfigJsonFileFlagGuard flag_guard(config_path.string());

  KVCacheConfig kv_cache_config;
  kv_cache_config.initialize();

  write_config_file(config_path, kUpdatedConfig);

  SchedulerConfig scheduler_config;
  scheduler_config.initialize();

  EXPECT_EQ(kv_cache_config.block_size(), 16);
  EXPECT_EQ(scheduler_config.max_tokens_per_batch(), 8192);

  std::filesystem::remove(config_path);
}

TEST(ConfigJsonTest, MalformedJsonFileKeepsFlagDefaults) {
  const std::filesystem::path config_path =
      std::filesystem::temp_directory_path() /
      "xllm_config_json_test_malformed.json";
  write_config_file(config_path, kMalformedConfig);

  ConfigJsonFileFlagGuard flag_guard(config_path.string());

  KVCacheConfig kv_cache_config;
  kv_cache_config.initialize();

  SchedulerConfig scheduler_config;
  scheduler_config.initialize();

  EXPECT_EQ(kv_cache_config.block_size(), 128);
  EXPECT_EQ(scheduler_config.max_tokens_per_batch(), 10240);

  std::filesystem::remove(config_path);
}

TEST(ConfigJsonTest, MissingJsonFileKeepsFlagDefaults) {
  const std::filesystem::path config_path =
      std::filesystem::temp_directory_path() /
      "xllm_config_json_test_missing.json";
  std::filesystem::remove(config_path);

  ConfigJsonFileFlagGuard flag_guard(config_path.string());

  KVCacheConfig kv_cache_config;
  kv_cache_config.initialize();

  SchedulerConfig scheduler_config;
  scheduler_config.initialize();

  EXPECT_EQ(kv_cache_config.block_size(), 128);
  EXPECT_DOUBLE_EQ(kv_cache_config.max_memory_utilization(), 0.8);
  EXPECT_EQ(scheduler_config.max_tokens_per_batch(), 10240);
  EXPECT_EQ(scheduler_config.max_seqs_per_batch(), 1024);
}

TEST(ConfigJsonTest, DumpStartupConfigSkipsWhenDisabled) {
  const std::filesystem::path dump_path =
      std::filesystem::temp_directory_path() /
      "xllm_dump_config_json_test_disabled.json";
  std::filesystem::remove(dump_path);
  DumpConfigJsonFlagGuard flag_guard(dump_path.string());
  FLAGS_enable_dump_config_json = false;

  config::dump_startup_config();

  EXPECT_FALSE(std::filesystem::exists(dump_path));
}

TEST(ConfigJsonTest, DumpStartupConfigWritesNonDefaultValuesOnly) {
  const std::filesystem::path dump_path =
      std::filesystem::temp_directory_path() /
      "xllm_dump_config_json_test_non_default.json";
  std::filesystem::remove(dump_path);
  DumpConfigJsonFlagGuard flag_guard(dump_path.string());
  StartupConfigGuard startup_config_guard;

  ModelConfig::get_instance().model_impl("python").python_model_path(
      "/tmp/xllm-python-model");
  ExecutionConfig::get_instance().python_graph_backend("cudagraphs");
  KVCacheConfig::get_instance().block_size(256).enable_prefix_cache(false);
  SchedulerConfig::get_instance()
      .max_tokens_per_batch(2048)
      .max_seqs_per_batch(128)
      .enable_chunked_prefill(false);
  FLAGS_enable_dump_config_json = true;

  config::dump_startup_config();

  ASSERT_TRUE(std::filesystem::exists(dump_path));
  const nlohmann::ordered_json config_json = read_json_file(dump_path);
  EXPECT_EQ(config_json.at("model_impl").get<std::string>(), "python");
  EXPECT_EQ(config_json.at("python_model_path").get<std::string>(),
            "/tmp/xllm-python-model");
  EXPECT_EQ(config_json.at("python_graph_backend").get<std::string>(),
            "cudagraphs");
  EXPECT_EQ(config_json.at("block_size").get<int32_t>(), 256);
  EXPECT_FALSE(config_json.at("enable_prefix_cache").get<bool>());
  EXPECT_EQ(config_json.at("max_tokens_per_batch").get<int32_t>(), 2048);
  EXPECT_EQ(config_json.at("max_seqs_per_batch").get<int32_t>(), 128);
  EXPECT_FALSE(config_json.at("enable_chunked_prefill").get<bool>());

  EXPECT_FALSE(config_json.contains("max_cache_size"));
  EXPECT_FALSE(config_json.contains("kv_cache_dtype"));
  EXPECT_FALSE(config_json.contains("indexer_cache_dtype"));
  EXPECT_FALSE(config_json.contains("priority_strategy"));

  std::filesystem::remove(dump_path);
}

}  // namespace
}  // namespace xllm
