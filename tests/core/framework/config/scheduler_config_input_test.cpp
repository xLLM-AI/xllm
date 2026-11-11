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

#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "core/common/global_flags.h"
#include "core/framework/config/config_input.h"
#include "core/framework/config/scheduler_config.h"
#include "core/util/json_reader.h"

namespace xllm {
namespace {

using config::ConfigInputSource;
using config::FlagInputSnapshot;

void parse_flags(std::vector<std::string> arguments) {
  arguments.insert(arguments.begin(), "scheduler_config_input_test");
  std::vector<char*> pointers;
  pointers.reserve(arguments.size() + 1);
  for (std::string& argument : arguments) {
    pointers.emplace_back(argument.data());
  }
  pointers.emplace_back(nullptr);
  int argc = static_cast<int>(arguments.size());
  char** argv = pointers.data();
  google::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);
}

JsonReader parse_json(std::string_view text) {
  JsonReader reader;
  EXPECT_TRUE(reader.parse_text(std::string(text)));
  return reader;
}

class ScopedEnvironment final {
 public:
  explicit ScopedEnvironment(std::string name) : name_(std::move(name)) {
    if (const char* value = std::getenv(name_.c_str())) {
      previous_value_ = value;
    }
  }

  ~ScopedEnvironment() {
    if (previous_value_.has_value()) {
      setenv(name_.c_str(), previous_value_->c_str(), /*overwrite=*/1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  bool set(const char* value) const {
    return setenv(name_.c_str(), value, /*overwrite=*/1) == 0;
  }

  bool unset() const { return unsetenv(name_.c_str()) == 0; }

 private:
  std::string name_;
  std::optional<std::string> previous_value_;
};

class ScopedFlagFile final {
 public:
  explicit ScopedFlagFile(std::string_view contents) {
    std::string path_template =
        (std::filesystem::temp_directory_path() / "xllm_scheduler_flags_XXXXXX")
            .string();
    const int fd = mkstemp(path_template.data());
    EXPECT_GE(fd, 0);
    if (fd < 0) {
      return;
    }
    close(fd);
    path_ = std::move(path_template);
    std::ofstream output(path_);
    output << contents;
    output.close();
    EXPECT_TRUE(output.good());
  }

  ~ScopedFlagFile() {
    if (!path_.empty()) {
      std::error_code error;
      std::filesystem::remove(path_, error);
    }
  }

  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

class SchedulerConfigInputTest : public testing::Test {
 private:
  google::FlagSaver flag_saver_;
};

TEST_F(SchedulerConfigInputTest, ExplicitDefaultOverridesConflictingJson) {
  parse_flags({"--max_tokens_per_batch=10240"});
  const FlagInputSnapshot flags = FlagInputSnapshot::capture();
  SchedulerConfigInput input = scheduler_config_input_from_flags(flags);

  merge_scheduler_config_json(input,
                              parse_json(R"({"max_tokens_per_batch":4096})"));
  const SchedulerConfig effective = resolve_scheduler_config(input);

  EXPECT_TRUE(flags.is_specified("max_tokens_per_batch"));
  EXPECT_EQ(input.max_tokens_per_batch.value, 10240);
  EXPECT_EQ(input.max_tokens_per_batch.source, ConfigInputSource::GFLAGS);
  EXPECT_EQ(effective.max_tokens_per_batch(), 10240);
}

TEST_F(SchedulerConfigInputTest, NegatedBooleanOverridesJson) {
  parse_flags({"--noenable_chunked_prefill"});
  SchedulerConfigInput input =
      scheduler_config_input_from_flags(FlagInputSnapshot::capture());

  merge_scheduler_config_json(input,
                              parse_json(R"({"enable_chunked_prefill":true})"));

  EXPECT_EQ(input.enable_chunked_prefill.value, false);
  EXPECT_EQ(input.enable_chunked_prefill.source, ConfigInputSource::GFLAGS);
  EXPECT_FALSE(resolve_scheduler_config(input).enable_chunked_prefill());
}

TEST_F(SchedulerConfigInputTest, FlagFilePreservesExplicitDefault) {
  const ScopedFlagFile file("--max_seqs_per_batch=200\n");
  ASSERT_FALSE(file.path().empty());
  parse_flags({"--flagfile=" + file.path()});
  SchedulerConfigInput input =
      scheduler_config_input_from_flags(FlagInputSnapshot::capture());

  merge_scheduler_config_json(input,
                              parse_json(R"({"max_seqs_per_batch":64})"));

  EXPECT_EQ(input.max_seqs_per_batch.source, ConfigInputSource::GFLAGS);
  EXPECT_EQ(resolve_scheduler_config(input).max_seqs_per_batch(), 200);
}

TEST_F(SchedulerConfigInputTest, FromEnvPreservesExplicitDefault) {
  const ScopedEnvironment environment("FLAGS_priority_strategy");
  ASSERT_TRUE(environment.set("fcfs"));
  parse_flags({"--fromenv=priority_strategy"});
  SchedulerConfigInput input =
      scheduler_config_input_from_flags(FlagInputSnapshot::capture());

  merge_scheduler_config_json(
      input, parse_json(R"({"priority_strategy":"deadline"})"));

  EXPECT_EQ(input.priority_strategy.source, ConfigInputSource::GFLAGS);
  EXPECT_EQ(resolve_scheduler_config(input).priority_strategy(), "fcfs");
}

TEST_F(SchedulerConfigInputTest, TryFromEnvPreservesExplicitZero) {
  const ScopedEnvironment environment("FLAGS_max_tokens_per_chunk_for_prefill");
  ASSERT_TRUE(environment.set("0"));
  parse_flags({"--tryfromenv=max_tokens_per_chunk_for_prefill"});
  SchedulerConfigInput input =
      scheduler_config_input_from_flags(FlagInputSnapshot::capture());

  merge_scheduler_config_json(
      input, parse_json(R"({"max_tokens_per_chunk_for_prefill":512})"));

  EXPECT_EQ(input.max_tokens_per_chunk_for_prefill.source,
            ConfigInputSource::GFLAGS);
  EXPECT_EQ(resolve_scheduler_config(input).max_tokens_per_chunk_for_prefill(),
            0);
}

TEST_F(SchedulerConfigInputTest, MissingTryFromEnvLeavesInputUnspecified) {
  const ScopedEnvironment environment("FLAGS_aggressive_coeff");
  ASSERT_TRUE(environment.unset());
  parse_flags({"--tryfromenv=aggressive_coeff"});
  const FlagInputSnapshot flags = FlagInputSnapshot::capture();
  SchedulerConfigInput input = scheduler_config_input_from_flags(flags);

  EXPECT_FALSE(flags.is_specified("aggressive_coeff"));
  EXPECT_FALSE(input.aggressive_coeff.value.has_value());
  EXPECT_EQ(input.aggressive_coeff.source, ConfigInputSource::UNSPECIFIED);
  merge_scheduler_config_json(input,
                              parse_json(R"({"aggressive_coeff":1.25})"));

  EXPECT_EQ(input.aggressive_coeff.source, ConfigInputSource::JSON);
  EXPECT_DOUBLE_EQ(resolve_scheduler_config(input).aggressive_coeff(), 1.25);
}

TEST_F(SchedulerConfigInputTest, EnvironmentRequiresGflagsOptIn) {
  const ScopedEnvironment environment("FLAGS_max_tokens_per_batch");
  ASSERT_TRUE(environment.set("2048"));
  parse_flags({});
  const SchedulerConfigInput input =
      scheduler_config_input_from_flags(FlagInputSnapshot::capture());

  EXPECT_FALSE(input.max_tokens_per_batch.value.has_value());
  EXPECT_EQ(resolve_scheduler_config(input).max_tokens_per_batch(), 10240);
}

TEST_F(SchedulerConfigInputTest, JsonWritebackDoesNotChangeCapturedInput) {
  parse_flags({});
  const FlagInputSnapshot flags = FlagInputSnapshot::capture();
  const SchedulerConfigInput original =
      scheduler_config_input_from_flags(flags);
  SchedulerConfigInput input = original;
  merge_scheduler_config_json(input,
                              parse_json(R"({"max_tokens_per_batch":8192})"));
  SchedulerConfig effective;
  effective.initialize(input);
  ASSERT_EQ(FLAGS_max_tokens_per_batch, 8192);

  google::CommandLineFlagInfo flag_info;
  ASSERT_TRUE(
      google::GetCommandLineFlagInfo("max_tokens_per_batch", &flag_info));
  EXPECT_FALSE(flag_info.is_default);
  FLAGS_max_tokens_per_batch = 17;

  EXPECT_FALSE(flags.is_specified("max_tokens_per_batch"));
  EXPECT_FALSE(original.max_tokens_per_batch.value.has_value());
  EXPECT_EQ(original.max_tokens_per_batch.source,
            ConfigInputSource::UNSPECIFIED);
  EXPECT_EQ(input.max_tokens_per_batch.value, 8192);
  EXPECT_EQ(input.max_tokens_per_batch.source, ConfigInputSource::JSON);
  EXPECT_EQ(resolve_scheduler_config(input).max_tokens_per_batch(), 8192);
}

TEST_F(SchedulerConfigInputTest, TypedCaptureOutlivesGflagsChanges) {
  parse_flags({"--max_tokens_per_batch=2048", "--priority_strategy=priority"});
  const SchedulerConfigInput captured =
      scheduler_config_input_from_flags(FlagInputSnapshot::capture());
  SchedulerConfigInput replacement;
  merge_scheduler_config_json(replacement,
                              parse_json(R"({"max_tokens_per_batch":4096,
                     "priority_strategy":"deadline"})"));
  SchedulerConfig effective;
  effective.initialize(replacement);

  const SchedulerConfig original = resolve_scheduler_config(captured);
  EXPECT_EQ(original.max_tokens_per_batch(), 2048);
  EXPECT_EQ(original.priority_strategy(), "priority");
  EXPECT_EQ(captured.priority_strategy.source, ConfigInputSource::GFLAGS);
  EXPECT_EQ(FLAGS_max_tokens_per_batch, 4096);
  EXPECT_EQ(FLAGS_priority_strategy, "deadline");
}

TEST_F(SchedulerConfigInputTest, JsonPreservesFalseZeroAndEmptyString) {
  SchedulerConfigInput input;
  merge_scheduler_config_json(input,
                              parse_json(R"({"enable_chunked_prefill":false,
                     "max_tokens_per_chunk_for_prefill":0,
                     "prefill_scheduling_memory_usage_threshold":0,
                     "priority_strategy":""})"));
  const SchedulerConfig effective = resolve_scheduler_config(input);

  EXPECT_FALSE(effective.enable_chunked_prefill());
  EXPECT_EQ(effective.max_tokens_per_chunk_for_prefill(), 0);
  EXPECT_DOUBLE_EQ(effective.prefill_scheduling_memory_usage_threshold(), 0.0);
  EXPECT_TRUE(effective.priority_strategy().empty());
  EXPECT_EQ(input.enable_chunked_prefill.source, ConfigInputSource::JSON);
  EXPECT_EQ(input.priority_strategy.source, ConfigInputSource::JSON);
  nlohmann::ordered_json dump = nlohmann::ordered_json::object();
  effective.append_config_json(dump);
  EXPECT_EQ(dump.at("enable_chunked_prefill"), false);
  EXPECT_EQ(dump.at("max_tokens_per_chunk_for_prefill"), 0);
  EXPECT_EQ(dump.at("priority_strategy"), "");
}

TEST_F(SchedulerConfigInputTest, JsonMissingNullAndObjectRemainUnspecified) {
  SchedulerConfigInput input;
  merge_scheduler_config_json(
      input,
      parse_json(R"({"max_tokens_per_batch":null,"priority_strategy":{}})"));

  EXPECT_FALSE(input.max_tokens_per_batch.value.has_value());
  EXPECT_FALSE(input.priority_strategy.value.has_value());
  EXPECT_FALSE(input.enable_chunked_prefill.value.has_value());
  EXPECT_EQ(input.max_tokens_per_batch.source, ConfigInputSource::UNSPECIFIED);
  EXPECT_EQ(input.priority_strategy.source, ConfigInputSource::UNSPECIFIED);
  const SchedulerConfig effective = resolve_scheduler_config(input);
  EXPECT_EQ(effective.max_tokens_per_batch(), 10240);
  EXPECT_EQ(effective.priority_strategy(), "fcfs");
  EXPECT_TRUE(effective.enable_chunked_prefill());
}

TEST_F(SchedulerConfigInputTest, JsonOnlyFillsInputsWithoutValues) {
  parse_flags({"--max_seqs_per_batch=64"});
  SchedulerConfigInput input =
      scheduler_config_input_from_flags(FlagInputSnapshot::capture());
  input.max_tokens_per_batch = {2048, ConfigInputSource::API};
  merge_scheduler_config_json(input, parse_json(R"({"max_tokens_per_batch":8192,
                     "max_seqs_per_batch":128,
                     "priority_strategy":"deadline"})"));

  const SchedulerConfig effective = resolve_scheduler_config(input);
  EXPECT_EQ(effective.max_tokens_per_batch(), 2048);
  EXPECT_EQ(effective.max_seqs_per_batch(), 64);
  EXPECT_EQ(effective.priority_strategy(), "deadline");
  EXPECT_EQ(input.max_tokens_per_batch.source, ConfigInputSource::API);
  EXPECT_EQ(input.max_seqs_per_batch.source, ConfigInputSource::GFLAGS);
  EXPECT_EQ(input.priority_strategy.source, ConfigInputSource::JSON);
}

TEST_F(SchedulerConfigInputTest, JsonTypeErrorsPropagateAtInputBoundary) {
  for (std::string_view json : {R"({"max_tokens_per_batch":"many"})",
                                R"({"enable_chunked_prefill":"false"})",
                                R"({"priority_strategy":17})"}) {
    SCOPED_TRACE(json);
    SchedulerConfigInput input;
    EXPECT_THROW(merge_scheduler_config_json(input, parse_json(json)),
                 nlohmann::json::type_error);
  }
}

TEST_F(SchedulerConfigInputTest, ExplicitFlagSkipsLowerPriorityJsonTypeError) {
  parse_flags({"--max_tokens_per_batch=10240"});
  SchedulerConfigInput input =
      scheduler_config_input_from_flags(FlagInputSnapshot::capture());

  EXPECT_NO_THROW(merge_scheduler_config_json(
      input, parse_json(R"({"max_tokens_per_batch":"ignored"})")));
  EXPECT_EQ(resolve_scheduler_config(input).max_tokens_per_batch(), 10240);
}

TEST_F(SchedulerConfigInputTest, ResolverDoesNotReadOrWriteGflags) {
  parse_flags({"--max_tokens_per_batch=8192",
               "--max_seqs_per_batch=32",
               "--priority_strategy=deadline"});
  SchedulerConfigInput input;
  input.max_tokens_per_batch = {2048, ConfigInputSource::API};
  merge_scheduler_config_json(
      input, parse_json(R"({"enable_chunked_prefill":false})"));

  const SchedulerConfig effective = resolve_scheduler_config(input);

  EXPECT_EQ(effective.max_tokens_per_batch(), 2048);
  EXPECT_EQ(effective.max_seqs_per_batch(), 200);
  EXPECT_EQ(effective.priority_strategy(), "fcfs");
  EXPECT_FALSE(effective.enable_chunked_prefill());
  EXPECT_EQ(FLAGS_max_tokens_per_batch, 8192);
  EXPECT_EQ(FLAGS_max_seqs_per_batch, 32);
  EXPECT_EQ(FLAGS_priority_strategy, "deadline");
  EXPECT_TRUE(FLAGS_enable_chunked_prefill);
  EXPECT_EQ(input.max_tokens_per_batch.source, ConfigInputSource::API);
  EXPECT_FALSE(input.max_seqs_per_batch.value.has_value());
}

TEST_F(SchedulerConfigInputTest, UnspecifiedInputKeepsLegacyDefaultsAndDump) {
  parse_flags({});
  const SchedulerConfigInput input =
      scheduler_config_input_from_flags(FlagInputSnapshot::capture());
  const SchedulerConfig effective = resolve_scheduler_config(input);

  EXPECT_FALSE(input.max_tokens_per_batch.value.has_value());
  EXPECT_FALSE(input.enable_schedule_overlap.value.has_value());
  EXPECT_EQ(effective.max_tokens_per_batch(), 10240);
  EXPECT_EQ(effective.max_seqs_per_batch(), 200);
  EXPECT_FALSE(effective.enable_schedule_overlap());
  EXPECT_TRUE(effective.enable_chunked_prefill());
  EXPECT_EQ(effective.max_tokens_per_chunk_for_prefill(), -1);
  EXPECT_EQ(effective.request_queue_size(), 100000);
  nlohmann::ordered_json dump = nlohmann::ordered_json::object();
  effective.append_config_json(dump);
  EXPECT_TRUE(dump.empty());
}

TEST_F(SchedulerConfigInputTest,
       InitializationAppliesResolvedInputAndWriteback) {
  SchedulerConfigInput input;
  input.max_tokens_per_batch = {2048, ConfigInputSource::API};
  merge_scheduler_config_json(input,
                              parse_json(R"({"enable_chunked_prefill":false,
                     "priority_strategy":"priority"})"));
  SchedulerConfig effective;

  effective.initialize(input);

  EXPECT_EQ(effective.max_tokens_per_batch(), 2048);
  EXPECT_FALSE(effective.enable_chunked_prefill());
  EXPECT_EQ(effective.priority_strategy(), "priority");
  EXPECT_EQ(FLAGS_max_tokens_per_batch, 2048);
  EXPECT_FALSE(FLAGS_enable_chunked_prefill);
  EXPECT_EQ(FLAGS_priority_strategy, "priority");
  EXPECT_EQ(input.max_tokens_per_batch.source, ConfigInputSource::API);
  nlohmann::ordered_json dump = nlohmann::ordered_json::object();
  effective.append_config_json(dump);
  const nlohmann::ordered_json expected = {{"max_tokens_per_batch", 2048},
                                           {"enable_chunked_prefill", false},
                                           {"priority_strategy", "priority"}};
  EXPECT_EQ(dump, expected);
}

}  // namespace
}  // namespace xllm
