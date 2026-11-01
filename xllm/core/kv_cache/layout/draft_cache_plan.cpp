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

#include "core/kv_cache/layout/draft_cache_plan.h"

#include <glog/logging.h>

#include <algorithm>

#include "core/framework/config/speculative_config.h"
#include "core/framework/model/model_args.h"
#include "core/kv_cache/block/block_utils.h"
#include "core/runtime/options.h"
#include "util/utils.h"

namespace xllm {

bool should_use_draft_swa_cache(const ModelArgs& draft_args,
                                const runtime::Options& options) {
  const bool block_draft =
      SpeculativeConfig::is_dspark_algorithm(options.speculative_algorithm()) ||
      SpeculativeConfig::is_dflash2_algorithm(options.speculative_algorithm());
  const bool ordinary_draft =
      draft_args.model_type() == "qwen3" ||
      draft_args.model_type() == "DSparkDraftModel" ||
      is_dflash2_draft_model_type(draft_args.model_type());
  // DSV4 retains its grouped SWA/compressed structure; GLM MLA drafts have
  // their own cache geometry. The model types above use ordinary paged KV.
  return options.enable_draft_swa_manage() && block_draft && ordinary_draft &&
         draft_args.use_sliding_window() && draft_args.sliding_window() > 0;
}

DraftSwaCachePlan resolve_draft_swa_cache_plan(
    const ModelArgs& draft_args,
    const runtime::Options& options) {
  if (!should_use_draft_swa_cache(draft_args, options)) {
    return {};
  }
  CHECK_GT(options.block_size(), 0);
  const int64_t speculative = options.num_speculative_tokens();
  const int64_t window = draft_args.sliding_window();
  // Preserve the block anchor, the asynchronous speculative upper bound and
  // one additional alignment block; this is not DSV4's semantic window alone.
  const int64_t retained = get_swa_blocks_per_seq(window + 2 * speculative + 1,
                                                  options.block_size()) +
                           1;
  const int64_t burst = util::ceil_div<int64_t>(
      std::max<int64_t>(options.max_tokens_per_batch(), 1),
      options.block_size());
  return {
      window,
      swa_pool_block_count(
          retained, std::max<int64_t>(options.max_seqs_per_batch(), 1), burst)};
}

}  // namespace xllm
