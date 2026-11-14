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

#include "models/dit/utils/minimax_h3_backend_selection.h"

#include <gtest/gtest.h>

#include "core/framework/config/dit_config.h"
#include "core/framework/config/load_config.h"

namespace xllm {
namespace {

TEST(H3BackendSelection, DefaultAndExplicitNativePreserveRollingChoice) {
  DiTConfig dit;
  LoadConfig load;
  EXPECT_EQ(select_h3_production_construction(dit.dit_h3_execution_profile(),
                                              load.enable_rolling_load()),
            H3BackendConstruction::NATIVE);
  load.enable_rolling_load(true);
  EXPECT_EQ(select_h3_production_construction(dit.dit_h3_execution_profile(),
                                              load.enable_rolling_load()),
            H3BackendConstruction::NATIVE);
  EXPECT_TRUE(load.enable_rolling_load());
}

TEST(H3BackendSelection, QualifiedRequiresResidentBeforeFactoryConstruction) {
  DiTConfig dit;
  LoadConfig load;
  dit.dit_h3_execution_profile(std::string(kMiniMaxH3ExecutionProfile));
  load.enable_rolling_load(false);
  EXPECT_EQ(select_h3_production_construction(dit.dit_h3_execution_profile(),
                                              load.enable_rolling_load()),
            H3BackendConstruction::A3_16CARD);
  load.enable_rolling_load(true);
  EXPECT_THROW(select_h3_production_construction(dit.dit_h3_execution_profile(),
                                                 load.enable_rolling_load()),
               std::invalid_argument);
}

TEST(H3BackendSelection, UnknownProfileCannotConstructEitherBackend) {
  EXPECT_THROW(select_h3_production_construction("unknown", false),
               std::invalid_argument);
}

}  // namespace
}  // namespace xllm
