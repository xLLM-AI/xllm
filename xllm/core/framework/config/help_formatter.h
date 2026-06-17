/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#pragma once

#include <gflags/gflags.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "core/framework/config/beam_search_config.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/distributed_config.h"
#include "core/framework/config/dit_config.h"
#include "core/framework/config/eplb_config.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/kv_cache_store_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/option_category.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/profile_config.h"
#include "core/framework/config/rec_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/service_config.h"
#include "core/framework/config/speculative_config.h"

namespace xllm {

class HelpFormatter {
 public:
  [[nodiscard]] static const std::vector<OptionCategory>& option_categories() {
    static const OptionCategory kConfigOptionCategory = {
        "CONFIG OPTIONS",
        {"config_json_file",
         "enable_dump_config_json",
         "dump_config_json_file"}};
    static const std::vector<OptionCategory> kOptionCategories = {
        kConfigOptionCategory,
        ServiceConfig::option_category(),
        ModelConfig::option_category(),
        LoadConfig::option_category(),
        KVCacheConfig::option_category(),
        KVCacheStoreConfig::option_category(),
        BeamSearchConfig::option_category(),
        SchedulerConfig::option_category(),
        ParallelConfig::option_category(),
        EPLBConfig::option_category(),
        DistributedConfig::option_category(),
        DisaggPDConfig::option_category(),
        SpeculativeConfig::option_category(),
        ProfileConfig::option_category(),
        ExecutionConfig::option_category(),
        KernelConfig::option_category(),
        DiTConfig::option_category(),
        RecConfig::option_category()};
    return kOptionCategories;
  }

  static std::string generate_help() {
    std::ostringstream oss;

    oss << "USAGE: xllm --model <PATH> [OPTIONS]\n\n";

    oss << "REQUIRED OPTIONS:\n";
    oss << "  --model <PATH>: Path to the model directory. This is "
           "the only required flag.\n\n";

    oss << "HELP OPTIONS:\n";
    oss << "  -h, --help: Display this help message and exit.\n\n";

    // Print flags(options) by category
    for (const OptionCategory& option_category : option_categories()) {
      std::ostringstream category_oss;

      for (const std::string& option_name : option_category.option_names) {
        google::CommandLineFlagInfo option_info;
        if (google::GetCommandLineFlagInfo(option_name.c_str(), &option_info)) {
          category_oss << "  --" << option_info.name;
          if (!option_info.description.empty()) {
            category_oss << ": " << option_info.description;
          }
          category_oss << "\n";
        }
      }

      std::string category_help = category_oss.str();
      if (!category_help.empty()) {
        oss << option_category.category_name << ":\n";
        oss << category_help << "\n";
      }
    }

    oss << "For more information and all available options, visit:\n";
    oss << "  https://github.com/jd-opensource/xllm/blob/main/xllm/core/"
           "framework/config/\n";
    oss << "Documentation: "
           "https://docs.xllm-ai.com/en/cli_reference/\n";

    return oss.str();
  }

  static void print_help() { std::cout << generate_help(); }

  static void print_usage() {
    std::cout << "USAGE: xllm --model <PATH> [OPTIONS]\n";
    std::cout << "Try 'xllm --help' for more information.\n";
  }

  static void print_error(const std::string& error_msg) {
    std::cerr << "Error: " << error_msg << "\n\n";
    print_usage();
  }
};

}  // namespace xllm
