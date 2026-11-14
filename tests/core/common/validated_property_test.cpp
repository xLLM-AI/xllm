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

#include <algorithm>
#include <limits>
#include <type_traits>
#include <utility>

#include "core/common/macros.h"

namespace xllm {
namespace {

class ValidatedOptions final {
 public:
  REFLECT_PROPERTIES(ValidatedOptions);

  VALIDATED_PROPERTY(int32_t, token_count, [](int32_t value) {
    return std::max(value, int32_t{0});
  }) = 0;
};

static_assert(
    std::is_same_v<decltype(std::declval<ValidatedOptions&>().token_count()),
                   int32_t>);
static_assert(
    std::is_same_v<decltype(std::declval<ValidatedOptions&>().token_count(1)),
                   ValidatedOptions&>);

TEST(ValidatedPropertyTest, PreservesValidValuesAndCorrectsInvalidValues) {
  ValidatedOptions options;
  EXPECT_EQ(options.token_count(), 0);
  EXPECT_EQ(&options.token_count(/*value=*/4), &options);
  EXPECT_EQ(options.token_count(), 4);
  options.token_count(/*value=*/-1);
  EXPECT_EQ(options.token_count(), 0);
  options.token_count(std::numeric_limits<int32_t>::min());
  EXPECT_EQ(options.token_count(), 0);
  options.token_count(std::numeric_limits<int32_t>::max());
  EXPECT_EQ(options.token_count(), std::numeric_limits<int32_t>::max());
}

TEST(ValidatedPropertyTest, RegistersFieldForReflection) {
  const auto& fields = reflect::PropertyTable<ValidatedOptions>::fields();
  ASSERT_EQ(fields.size(), 1);
  EXPECT_STREQ(fields.front().name, "token_count");
  EXPECT_NE(fields.front().emit, nullptr);
}

}  // namespace
}  // namespace xllm
