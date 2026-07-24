#include "prefix_cache_factory.h"

#include <absl/strings/numbers.h>
#include <absl/strings/str_split.h>

#include "linear_state_prefix_cache.h"

namespace xllm {

std::unique_ptr<PrefixCache> create_prefix_cache(PrefixCache::Options options) {
  int32_t block_size = options.block_size();
  BlockHasherType hasher_type = options.hasher_type();
  // Two block types need the gap-tolerant probe:
  //   LINEAR -- block_size here is the linear checkpoint stride (one prefill
  //             chunk), routed into the cache's single block-boundary slot
  //             by BlockManagerImpl.
  //   SWA    -- block_size here is the SWA base block. Middle misses are
  //             harmless (attention only reads the tail; the composite
  //             enforces the tail-continuity check).
  // Both share the same subclass; the type name is retained to keep the
  // SWA-adoption diff narrow.
  if (options.block_type() == BlockType::LINEAR ||
      options.block_type() == BlockType::SWA) {
    return std::make_unique<LinearStatePrefixCache>(block_size, hasher_type);
  }
  return std::make_unique<PrefixCache>(block_size, hasher_type);
}

}  // namespace xllm
