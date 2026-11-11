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

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "core/common/global_flags.h"
#include "core/framework/config/config_input.h"
#include "core/framework/config/eplb_config.h"
#include "core/util/json_reader.h"

namespace xllm {
namespace {

using config::ConfigInputSource;
using config::FlagInputSnapshot;

void parse_flags(std::vector<std::string> arguments) {
  arguments.insert(arguments.begin(), "eplb_config_input_test");
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

void expect_defaults(const EPLBConfig& config) {
  EXPECT_FALSE(config.enable_eplb());
  EXPECT_EQ(config.redundant_experts_num(), 1);
  EXPECT_EQ(config.eplb_update_interval(), 1000);
  EXPECT_DOUBLE_EQ(config.eplb_min_peak_load_improvement(), 0.05);
  EXPECT_EQ(config.eplb_policy_kind(), "balanced");
  EXPECT_FALSE(config.eplb_use_decode_only_load());
  EXPECT_EQ(config.eplb_prepare_timeout_seconds(), 30);
  EXPECT_EQ(config.expert_parallel_degree(), 0);
  EXPECT_TRUE(config.rank_tablefile().empty());
}

void expect_sources(const EplbConfigInput& input, ConfigInputSource source) {
  EXPECT_EQ(input.enable_eplb.source, source);
  EXPECT_EQ(input.redundant_experts_num.source, source);
  EXPECT_EQ(input.eplb_update_interval.source, source);
  EXPECT_EQ(input.eplb_min_peak_load_improvement.source, source);
  EXPECT_EQ(input.eplb_policy_kind.source, source);
  EXPECT_EQ(input.eplb_use_decode_only_load.source, source);
  EXPECT_EQ(input.eplb_prepare_timeout_seconds.source, source);
  EXPECT_EQ(input.expert_parallel_degree.source, source);
  EXPECT_EQ(input.rank_tablefile.source, source);
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
        (std::filesystem::temp_directory_path() / "xllm_eplb_flags_XXXXXX")
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

class EplbConfigInputTest : public testing::Test {
 private:
  google::FlagSaver flag_saver_;
};

TEST_F(EplbConfigInputTest, ExplicitDefaultsOverrideConflictingJson) {
  parse_flags({"--enable_eplb=false",
               "--redundant_experts_num=1",
               "--eplb_update_interval=1000",
               "--eplb_min_peak_load_improvement=0.05",
               "--eplb_policy_kind=balanced",
               "--eplb_use_decode_only_load=false",
               "--eplb_prepare_timeout_seconds=30",
               "--expert_parallel_degree=0",
               "--rank_tablefile="});
  EplbConfigInput input =
      eplb_config_input_from_flags(FlagInputSnapshot::capture());
  merge_eplb_config_json(input, parse_json(R"({"enable_eplb":true,
      "redundant_experts_num":3,"eplb_update_interval":7,
      "eplb_min_peak_load_improvement":0.2,"eplb_policy_kind":"greedy",
      "eplb_use_decode_only_load":true,"eplb_prepare_timeout_seconds":45,
      "expert_parallel_degree":4,"rank_tablefile":"rank.json"})"));

  expect_sources(input, ConfigInputSource::GFLAGS);
  expect_defaults(resolve_eplb_config(input));
  EXPECT_EQ(input.enable_eplb.value, false);
  EXPECT_EQ(input.expert_parallel_degree.value, 0);
  EXPECT_EQ(input.rank_tablefile.value, "");
}

TEST_F(EplbConfigInputTest, NegatedDefaultBooleansOverrideJson) {
  parse_flags({"--noenable_eplb", "--noeplb_use_decode_only_load"});
  EplbConfigInput input =
      eplb_config_input_from_flags(FlagInputSnapshot::capture());
  merge_eplb_config_json(input, parse_json(R"({"enable_eplb":true,
      "eplb_use_decode_only_load":true})"));

  EXPECT_EQ(input.enable_eplb.source, ConfigInputSource::GFLAGS);
  EXPECT_EQ(input.eplb_use_decode_only_load.source, ConfigInputSource::GFLAGS);
  EXPECT_FALSE(resolve_eplb_config(input).enable_eplb());
  EXPECT_FALSE(resolve_eplb_config(input).eplb_use_decode_only_load());
}

TEST_F(EplbConfigInputTest, FlagFilePreservesFalseZeroAndEmptyString) {
  const ScopedFlagFile file(
      "--enable_eplb=false\n--expert_parallel_degree=0\n"
      "--rank_tablefile=\n");
  ASSERT_FALSE(file.path().empty());
  parse_flags({"--flagfile=" + file.path()});
  EplbConfigInput input =
      eplb_config_input_from_flags(FlagInputSnapshot::capture());
  merge_eplb_config_json(input, parse_json(R"({"enable_eplb":true,
      "expert_parallel_degree":4,"rank_tablefile":"rank.json"})"));

  EXPECT_EQ(input.enable_eplb.source, ConfigInputSource::GFLAGS);
  EXPECT_EQ(input.expert_parallel_degree.source, ConfigInputSource::GFLAGS);
  EXPECT_EQ(input.rank_tablefile.source, ConfigInputSource::GFLAGS);
  expect_defaults(resolve_eplb_config(input));
}

TEST_F(EplbConfigInputTest, FromEnvPreservesExplicitDefault) {
  const ScopedEnvironment environment("FLAGS_eplb_policy_kind");
  ASSERT_TRUE(environment.set("balanced"));
  parse_flags({"--fromenv=eplb_policy_kind"});
  EplbConfigInput input =
      eplb_config_input_from_flags(FlagInputSnapshot::capture());
  merge_eplb_config_json(input, parse_json(R"({"eplb_policy_kind":"greedy"})"));

  EXPECT_EQ(input.eplb_policy_kind.source, ConfigInputSource::GFLAGS);
  EXPECT_EQ(resolve_eplb_config(input).eplb_policy_kind(), "balanced");
}

TEST_F(EplbConfigInputTest, TryFromEnvPreservesExplicitZero) {
  const ScopedEnvironment environment("FLAGS_redundant_experts_num");
  ASSERT_TRUE(environment.set("0"));
  parse_flags({"--tryfromenv=redundant_experts_num"});
  EplbConfigInput input =
      eplb_config_input_from_flags(FlagInputSnapshot::capture());
  merge_eplb_config_json(input, parse_json(R"({"redundant_experts_num":3})"));

  EXPECT_EQ(input.redundant_experts_num.source, ConfigInputSource::GFLAGS);
  EXPECT_EQ(resolve_eplb_config(input).redundant_experts_num(), 0);
}

TEST_F(EplbConfigInputTest, MissingTryFromEnvLeavesInputUnspecified) {
  const ScopedEnvironment environment("FLAGS_rank_tablefile");
  ASSERT_TRUE(environment.unset());
  parse_flags({"--tryfromenv=rank_tablefile"});
  const FlagInputSnapshot flags = FlagInputSnapshot::capture();
  EplbConfigInput input = eplb_config_input_from_flags(flags);

  EXPECT_FALSE(flags.is_specified("rank_tablefile"));
  EXPECT_FALSE(input.rank_tablefile.value.has_value());
  EXPECT_EQ(input.rank_tablefile.source, ConfigInputSource::UNSPECIFIED);
  merge_eplb_config_json(input,
                         parse_json(R"({"rank_tablefile":"rank.json"})"));
  EXPECT_EQ(input.rank_tablefile.source, ConfigInputSource::JSON);
  EXPECT_EQ(resolve_eplb_config(input).rank_tablefile(), "rank.json");
}

TEST_F(EplbConfigInputTest, EnvironmentRequiresGflagsOptIn) {
  const ScopedEnvironment environment("FLAGS_enable_eplb");
  ASSERT_TRUE(environment.set("true"));
  parse_flags({});
  const EplbConfigInput input =
      eplb_config_input_from_flags(FlagInputSnapshot::capture());

  EXPECT_FALSE(input.enable_eplb.value.has_value());
  EXPECT_EQ(input.enable_eplb.source, ConfigInputSource::UNSPECIFIED);
  EXPECT_FALSE(resolve_eplb_config(input).enable_eplb());
}

TEST_F(EplbConfigInputTest, JsonWritebackDoesNotChangeCapturedInput) {
  parse_flags({});
  const FlagInputSnapshot flags = FlagInputSnapshot::capture();
  const EplbConfigInput original = eplb_config_input_from_flags(flags);
  EplbConfigInput input = original;
  merge_eplb_config_json(input, parse_json(R"({"enable_eplb":true})"));
  EPLBConfig effective;
  effective.initialize(input);
  ASSERT_TRUE(FLAGS_enable_eplb);
  google::CommandLineFlagInfo flag_info;
  ASSERT_TRUE(google::GetCommandLineFlagInfo("enable_eplb", &flag_info));
  EXPECT_FALSE(flag_info.is_default);

  EXPECT_FALSE(flags.is_specified("enable_eplb"));
  EXPECT_FALSE(original.enable_eplb.value.has_value());
  EXPECT_EQ(original.enable_eplb.source, ConfigInputSource::UNSPECIFIED);
  EXPECT_EQ(input.enable_eplb.value, true);
  EXPECT_EQ(input.enable_eplb.source, ConfigInputSource::JSON);
}

TEST_F(EplbConfigInputTest, TypedSnapshotPreservesEveryFieldAfterWriteback) {
  parse_flags({"--enable_eplb=true",
               "--redundant_experts_num=3",
               "--eplb_update_interval=6000000000",
               "--eplb_min_peak_load_improvement=0.125",
               "--eplb_policy_kind=DeepSeek_Hier",
               "--eplb_use_decode_only_load=true",
               "--eplb_prepare_timeout_seconds=45",
               "--expert_parallel_degree=4",
               "--rank_tablefile=captured.json"});
  const EplbConfigInput captured =
      eplb_config_input_from_flags(FlagInputSnapshot::capture());
  EplbConfigInput replacement;
  merge_eplb_config_json(replacement, parse_json(R"({"enable_eplb":false,
      "redundant_experts_num":7,"eplb_update_interval":13,
      "eplb_min_peak_load_improvement":0.3,"eplb_policy_kind":"greedy",
      "eplb_use_decode_only_load":false,"eplb_prepare_timeout_seconds":60,
      "expert_parallel_degree":8,"rank_tablefile":"replacement.json"})"));
  EPLBConfig effective;
  effective.initialize(replacement);
  const EPLBConfig original = resolve_eplb_config(captured);

  expect_sources(captured, ConfigInputSource::GFLAGS);
  EXPECT_TRUE(original.enable_eplb());
  EXPECT_EQ(original.redundant_experts_num(), 3);
  EXPECT_EQ(original.eplb_update_interval(), 6000000000LL);
  EXPECT_DOUBLE_EQ(original.eplb_min_peak_load_improvement(), 0.125);
  EXPECT_EQ(original.eplb_policy_kind(), "DeepSeek_Hier");
  EXPECT_TRUE(original.eplb_use_decode_only_load());
  EXPECT_EQ(original.eplb_prepare_timeout_seconds(), 45);
  EXPECT_EQ(original.expert_parallel_degree(), 4);
  EXPECT_EQ(original.rank_tablefile(), "captured.json");
  EXPECT_FALSE(FLAGS_enable_eplb);
  EXPECT_EQ(FLAGS_redundant_experts_num, 7);
  EXPECT_EQ(FLAGS_eplb_update_interval, 13);
  EXPECT_DOUBLE_EQ(FLAGS_eplb_min_peak_load_improvement, 0.3);
  EXPECT_EQ(FLAGS_eplb_policy_kind, "greedy");
  EXPECT_FALSE(FLAGS_eplb_use_decode_only_load);
  EXPECT_EQ(google::GetCommandLineFlagInfoOrDie("eplb_prepare_timeout_seconds")
                .current_value,
            "60");
  EXPECT_EQ(FLAGS_expert_parallel_degree, 8);
  EXPECT_EQ(FLAGS_rank_tablefile, "replacement.json");
}

TEST_F(EplbConfigInputTest, JsonPreservesFalseZeroAndEmptyString) {
  EplbConfigInput input;
  merge_eplb_config_json(input, parse_json(R"({"enable_eplb":false,
      "redundant_experts_num":0,"eplb_update_interval":0,
      "eplb_min_peak_load_improvement":0,"eplb_policy_kind":"",
      "eplb_use_decode_only_load":false,"eplb_prepare_timeout_seconds":0,
      "expert_parallel_degree":0,"rank_tablefile":""})"));
  const EPLBConfig effective = resolve_eplb_config(input);

  expect_sources(input, ConfigInputSource::JSON);
  EXPECT_FALSE(effective.enable_eplb());
  EXPECT_EQ(effective.redundant_experts_num(), 0);
  EXPECT_EQ(effective.eplb_update_interval(), 0);
  EXPECT_DOUBLE_EQ(effective.eplb_min_peak_load_improvement(), 0.0);
  EXPECT_TRUE(effective.eplb_policy_kind().empty());
  EXPECT_FALSE(effective.eplb_use_decode_only_load());
  EXPECT_EQ(effective.eplb_prepare_timeout_seconds(), 0);
  EXPECT_EQ(effective.expert_parallel_degree(), 0);
  EXPECT_TRUE(effective.rank_tablefile().empty());
}

TEST_F(EplbConfigInputTest, JsonMissingNullAndObjectRemainUnspecified) {
  EplbConfigInput input;
  merge_eplb_config_json(input, parse_json(R"({"enable_eplb":null,
      "redundant_experts_num":{},"eplb_update_interval":null,
      "eplb_min_peak_load_improvement":{},"eplb_policy_kind":null,
      "eplb_use_decode_only_load":{},"eplb_prepare_timeout_seconds":null,
      "expert_parallel_degree":{},"rank_tablefile":null})"));

  expect_sources(input, ConfigInputSource::UNSPECIFIED);
  EXPECT_FALSE(input.enable_eplb.value.has_value());
  EXPECT_FALSE(input.redundant_experts_num.value.has_value());
  EXPECT_FALSE(input.eplb_update_interval.value.has_value());
  EXPECT_FALSE(input.eplb_min_peak_load_improvement.value.has_value());
  EXPECT_FALSE(input.eplb_policy_kind.value.has_value());
  EXPECT_FALSE(input.eplb_use_decode_only_load.value.has_value());
  EXPECT_FALSE(input.eplb_prepare_timeout_seconds.value.has_value());
  EXPECT_FALSE(input.expert_parallel_degree.value.has_value());
  EXPECT_FALSE(input.rank_tablefile.value.has_value());
  expect_defaults(resolve_eplb_config(input));
}

TEST_F(EplbConfigInputTest, JsonOnlyFillsInputsWithoutValues) {
  parse_flags({"--eplb_update_interval=0"});
  EplbConfigInput input =
      eplb_config_input_from_flags(FlagInputSnapshot::capture());
  input.enable_eplb = {false, ConfigInputSource::API};
  input.eplb_policy_kind = {"", ConfigInputSource::API};
  merge_eplb_config_json(input, parse_json(R"({"enable_eplb":true,
      "eplb_update_interval":10,"eplb_policy_kind":"balanced",
      "rank_tablefile":"rank.json"})"));
  const EPLBConfig effective = resolve_eplb_config(input);

  EXPECT_FALSE(effective.enable_eplb());
  EXPECT_EQ(effective.eplb_update_interval(), 0);
  EXPECT_TRUE(effective.eplb_policy_kind().empty());
  EXPECT_EQ(effective.rank_tablefile(), "rank.json");
  EXPECT_EQ(input.enable_eplb.source, ConfigInputSource::API);
  EXPECT_EQ(input.eplb_policy_kind.source, ConfigInputSource::API);
  EXPECT_EQ(input.eplb_update_interval.source, ConfigInputSource::GFLAGS);
  EXPECT_EQ(input.rank_tablefile.source, ConfigInputSource::JSON);
  EXPECT_EQ(input.redundant_experts_num.source, ConfigInputSource::UNSPECIFIED);
}

TEST_F(EplbConfigInputTest, JsonTypeErrorsPropagateAtInputBoundary) {
  for (std::string_view json : {R"({"enable_eplb":"false"})",
                                R"({"redundant_experts_num":"one"})",
                                R"({"eplb_update_interval":"often"})",
                                R"({"eplb_min_peak_load_improvement":"small"})",
                                R"({"eplb_policy_kind":17})",
                                R"({"eplb_use_decode_only_load":"true"})",
                                R"({"eplb_prepare_timeout_seconds":"later"})",
                                R"({"expert_parallel_degree":"all"})",
                                R"({"rank_tablefile":false})"}) {
    SCOPED_TRACE(json);
    EplbConfigInput input;
    EXPECT_THROW(merge_eplb_config_json(input, parse_json(json)),
                 nlohmann::json::type_error);
  }
}

TEST_F(EplbConfigInputTest, ExplicitFlagsSkipLowerPriorityJsonTypeErrors) {
  parse_flags({"--enable_eplb=false",
               "--redundant_experts_num=1",
               "--eplb_update_interval=1000",
               "--eplb_min_peak_load_improvement=0.05",
               "--eplb_policy_kind=balanced",
               "--eplb_use_decode_only_load=false",
               "--eplb_prepare_timeout_seconds=30",
               "--expert_parallel_degree=0",
               "--rank_tablefile="});
  EplbConfigInput input =
      eplb_config_input_from_flags(FlagInputSnapshot::capture());

  EXPECT_NO_THROW(merge_eplb_config_json(input, parse_json(R"({
      "enable_eplb":"ignored","redundant_experts_num":"ignored",
      "eplb_update_interval":"ignored","eplb_min_peak_load_improvement":[],
      "eplb_policy_kind":17,"eplb_use_decode_only_load":"ignored",
      "eplb_prepare_timeout_seconds":"ignored","expert_parallel_degree":[],
      "rank_tablefile":false})")));
  expect_defaults(resolve_eplb_config(input));
}

TEST_F(EplbConfigInputTest, ResolverDoesNotReadOrWriteGflags) {
  parse_flags({"--enable_eplb=true",
               "--redundant_experts_num=3",
               "--eplb_update_interval=7",
               "--eplb_min_peak_load_improvement=0.2",
               "--eplb_policy_kind=greedy",
               "--eplb_use_decode_only_load=true",
               "--eplb_prepare_timeout_seconds=45",
               "--expert_parallel_degree=4",
               "--rank_tablefile=rank.json"});
  EplbConfigInput input;
  input.redundant_experts_num = {0, ConfigInputSource::API};
  const EPLBConfig effective = resolve_eplb_config(input);

  EXPECT_FALSE(effective.enable_eplb());
  EXPECT_EQ(effective.redundant_experts_num(), 0);
  EXPECT_EQ(effective.eplb_update_interval(), 1000);
  EXPECT_DOUBLE_EQ(effective.eplb_min_peak_load_improvement(), 0.05);
  EXPECT_EQ(effective.eplb_policy_kind(), "balanced");
  EXPECT_FALSE(effective.eplb_use_decode_only_load());
  EXPECT_EQ(effective.eplb_prepare_timeout_seconds(), 30);
  EXPECT_EQ(effective.expert_parallel_degree(), 0);
  EXPECT_TRUE(effective.rank_tablefile().empty());
  EXPECT_TRUE(FLAGS_enable_eplb);
  EXPECT_EQ(FLAGS_redundant_experts_num, 3);
  EXPECT_EQ(FLAGS_eplb_update_interval, 7);
  EXPECT_DOUBLE_EQ(FLAGS_eplb_min_peak_load_improvement, 0.2);
  EXPECT_EQ(FLAGS_eplb_policy_kind, "greedy");
  EXPECT_TRUE(FLAGS_eplb_use_decode_only_load);
  EXPECT_EQ(google::GetCommandLineFlagInfoOrDie("eplb_prepare_timeout_seconds")
                .current_value,
            "45");
  EXPECT_EQ(FLAGS_expert_parallel_degree, 4);
  EXPECT_EQ(FLAGS_rank_tablefile, "rank.json");
  EXPECT_EQ(input.redundant_experts_num.source, ConfigInputSource::API);
  EXPECT_FALSE(input.enable_eplb.value.has_value());
}

TEST_F(EplbConfigInputTest, DisabledInputDoesNotEagerlyValidatePolicyValues) {
  EplbConfigInput input;
  input.redundant_experts_num = {-1, ConfigInputSource::API};
  input.eplb_update_interval = {-1, ConfigInputSource::API};
  input.eplb_min_peak_load_improvement = {
      std::numeric_limits<double>::quiet_NaN(), ConfigInputSource::API};
  input.eplb_policy_kind = {"unknown-alias", ConfigInputSource::API};
  input.eplb_prepare_timeout_seconds = {0, ConfigInputSource::API};
  EPLBConfig effective;

  effective.initialize(input);

  EXPECT_FALSE(effective.enable_eplb());
  EXPECT_EQ(effective.redundant_experts_num(), -1);
  EXPECT_EQ(effective.eplb_update_interval(), -1);
  EXPECT_TRUE(std::isnan(effective.eplb_min_peak_load_improvement()));
  EXPECT_EQ(effective.eplb_policy_kind(), "unknown-alias");
  EXPECT_EQ(effective.eplb_prepare_timeout_seconds(), 0);
}

TEST_F(EplbConfigInputTest, UnspecifiedInputKeepsLegacyDefaultsAndEmptyDump) {
  parse_flags({});
  const EplbConfigInput input =
      eplb_config_input_from_flags(FlagInputSnapshot::capture());
  const EPLBConfig effective = resolve_eplb_config(input);

  expect_sources(input, ConfigInputSource::UNSPECIFIED);
  expect_defaults(effective);
  nlohmann::ordered_json dump = nlohmann::ordered_json::object();
  effective.append_config_json(dump);
  EXPECT_TRUE(dump.empty());
}

TEST_F(EplbConfigInputTest, InitializationWritesAllFieldsInLegacyDumpOrder) {
  EplbConfigInput input;
  merge_eplb_config_json(input, parse_json(R"({"enable_eplb":true,
      "redundant_experts_num":3,"eplb_update_interval":6000000000,
      "eplb_min_peak_load_improvement":0.2,"eplb_policy_kind":"greedy",
      "eplb_use_decode_only_load":true,"eplb_prepare_timeout_seconds":45,
      "expert_parallel_degree":4,"rank_tablefile":"rank.json"})"));
  EPLBConfig effective;

  effective.initialize(input);

  expect_sources(input, ConfigInputSource::JSON);
  EXPECT_TRUE(FLAGS_enable_eplb);
  EXPECT_EQ(FLAGS_redundant_experts_num, 3);
  EXPECT_EQ(FLAGS_eplb_update_interval, 6000000000LL);
  EXPECT_DOUBLE_EQ(FLAGS_eplb_min_peak_load_improvement, 0.2);
  EXPECT_EQ(FLAGS_eplb_policy_kind, "greedy");
  EXPECT_TRUE(FLAGS_eplb_use_decode_only_load);
  EXPECT_EQ(google::GetCommandLineFlagInfoOrDie("eplb_prepare_timeout_seconds")
                .current_value,
            "45");
  EXPECT_EQ(FLAGS_expert_parallel_degree, 4);
  EXPECT_EQ(FLAGS_rank_tablefile, "rank.json");
  nlohmann::ordered_json dump = nlohmann::ordered_json::object();
  effective.append_config_json(dump);
  const nlohmann::ordered_json expected = {
      {"enable_eplb", true},
      {"redundant_experts_num", 3},
      {"eplb_update_interval", 6000000000LL},
      {"eplb_min_peak_load_improvement", 0.2},
      {"eplb_policy_kind", "greedy"},
      {"eplb_use_decode_only_load", true},
      {"eplb_prepare_timeout_seconds", 45},
      {"expert_parallel_degree", 4},
      {"rank_tablefile", "rank.json"}};
  EXPECT_EQ(dump.dump(), expected.dump());
}

TEST_F(EplbConfigInputTest, LegacyAdaptersKeepExplicitValuesAndJsonWriteback) {
  parse_flags({"--enable_eplb=false", "--rank_tablefile="});
  EPLBConfig effective;
  effective.from_flags();
  effective.from_json(parse_json(R"({"enable_eplb":true,
      "redundant_experts_num":3,"eplb_update_interval":7,
      "eplb_min_peak_load_improvement":0.2,"eplb_policy_kind":"greedy",
      "eplb_use_decode_only_load":true,"eplb_prepare_timeout_seconds":45,
      "expert_parallel_degree":4,"rank_tablefile":"rank.json"})"));

  EXPECT_FALSE(effective.enable_eplb());
  EXPECT_EQ(effective.redundant_experts_num(), 3);
  EXPECT_EQ(effective.eplb_update_interval(), 7);
  EXPECT_DOUBLE_EQ(effective.eplb_min_peak_load_improvement(), 0.2);
  EXPECT_EQ(effective.eplb_policy_kind(), "greedy");
  EXPECT_TRUE(effective.eplb_use_decode_only_load());
  EXPECT_EQ(effective.eplb_prepare_timeout_seconds(), 45);
  EXPECT_EQ(effective.expert_parallel_degree(), 4);
  EXPECT_TRUE(effective.rank_tablefile().empty());
  EXPECT_EQ(FLAGS_redundant_experts_num, 3);
  EXPECT_EQ(FLAGS_eplb_policy_kind, "greedy");
  EXPECT_TRUE(FLAGS_rank_tablefile.empty());
}

TEST_F(EplbConfigInputTest, LegacyInitializeResolvesCurrentInputs) {
  parse_flags({"--enable_eplb=true", "--eplb_update_interval=7"});
  EPLBConfig effective;

  effective.initialize();

  EXPECT_TRUE(effective.enable_eplb());
  EXPECT_EQ(effective.eplb_update_interval(), 7);
  EXPECT_EQ(effective.redundant_experts_num(), 1);
  EXPECT_EQ(effective.eplb_policy_kind(), "balanced");
  EXPECT_EQ(FLAGS_eplb_update_interval, 7);
}

}  // namespace
}  // namespace xllm
