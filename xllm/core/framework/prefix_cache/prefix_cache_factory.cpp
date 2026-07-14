#include "prefix_cache_factory.h"

#include <absl/strings/numbers.h>
#include <absl/strings/str_split.h>

#include "linear_state_prefix_cache.h"

namespace xllm {

std::unique_ptr<PrefixCache> create_prefix_cache(PrefixCache::Options options) {
  int32_t block_size = options.block_size();
  BlockHasherType hasher_type = options.hasher_type();
  if (options.block_type() == BlockType::LINEAR) {
    // block_size here is the linear checkpoint stride (one prefill chunk),
    // routed into the cache's single block-boundary slot by BlockManagerImpl.
    return std::make_unique<LinearStatePrefixCache>(block_size, hasher_type);
  }
  return std::make_unique<PrefixCache>(block_size, hasher_type);
}

}  // namespace xllm
