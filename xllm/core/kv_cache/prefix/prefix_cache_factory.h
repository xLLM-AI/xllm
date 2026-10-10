#pragma once
#include <string>

#include "core/kv_cache/prefix/prefix_cache.h"

namespace xllm {

std::unique_ptr<PrefixCache> create_prefix_cache(PrefixCache::Options options);

}  // namespace xllm
