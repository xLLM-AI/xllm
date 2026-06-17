#pragma once
#include <string>

#include "prefix_cache.h"

namespace xllm {

std::unique_ptr<PrefixCache> create_prefix_cache(PrefixCache::Options options);

}  // namespace xllm
