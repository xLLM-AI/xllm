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

#include "framework/request/sequence_kv_state.h"

#include <gtest/gtest.h>

namespace xllm {

TEST(KVCacheStateTest, TransferCursorTracksAndResets) {
  KVCacheState state;
  EXPECT_EQ(state.next_transfer_block_idx(), 0u);

  state.set_next_transfer_block_idx(2);
  EXPECT_EQ(state.next_transfer_block_idx(), 2u);

  state.advance_transfer_block_idx(5);
  EXPECT_EQ(state.next_transfer_block_idx(), 5u);

  state.advance_transfer_block_idx(3);
  EXPECT_EQ(state.next_transfer_block_idx(), 5u);

  state.reset();
  EXPECT_EQ(state.next_transfer_block_idx(), 0u);
}

}  // namespace xllm
