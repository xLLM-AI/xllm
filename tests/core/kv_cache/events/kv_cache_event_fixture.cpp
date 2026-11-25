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

// CPU-only protocol fixture using the production cache and publisher. This
// exercises metadata/residency, not a model forward or a physical NPU copy.
#include <glog/logging.h>

#include <array>
#include <iostream>
#include <memory>
#include <string>

#include "core/framework/block/block_manager_pool.h"
#include "core/framework/block/composite_block_manager.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/kv_cache/block/block_manager_impl.h"
#include "core/kv_cache/events/kv_cache_event_publisher.h"

namespace {

class FixtureBlockManagerPool final : public xllm::BlockManagerPool {
 public:
  explicit FixtureBlockManagerPool(const Options& options)
      : BlockManagerPool(options, /*dp_size=*/2) {}

  xllm::BlockManager* device(int32_t rank) {
    auto* composite = block_managers_[rank].get();
    return composite->leaf_entries().at(xllm::BlockType::KV).leaf.get();
  }

  bool publisher_healthy() const {
    return event_publisher_ && event_publisher_->healthy();
  }

  void stop_publisher() { event_publisher_.reset(); }
};

}  // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  CHECK(argc == 2 || (argc == 3 && std::string(argv[2]) == "--pool"))
      << "Expected publisher configuration JSON and optional --pool";
  auto journal = std::make_shared<xllm::KVCacheEventJournal>();
  std::unique_ptr<xllm::KVCacheEventPublisher> publisher;
  std::unique_ptr<FixtureBlockManagerPool> target_pool;
  std::unique_ptr<FixtureBlockManagerPool> draft_pool;
  if (argc == 3) {
    // Keep global reporting enabled while the draft receives no event config.
    xllm::KVCacheConfig::get_instance().kv_events_config(argv[1]);
    xllm::BlockManagerPool::Options options;
    options.num_blocks(64).block_size(2).model_id("fixture").kv_events_config(
        argv[1]);
    target_pool = std::make_unique<FixtureBlockManagerPool>(options);
    journal = target_pool->options().event_journal();
  } else {
    publisher = std::make_unique<xllm::KVCacheEventPublisher>(
        journal, nlohmann::json::parse(argv[1]), "fixture", /*dp_size=*/2);
  }
  std::array<std::unique_ptr<xllm::BlockManagerImpl>, 2> devices;
  std::array<std::unique_ptr<xllm::BlockManagerImpl>, 2> hosts;
  for (int32_t rank = 0; rank < 2; ++rank) {
    xllm::BlockManager::Options options;
    options.num_blocks(64)
        .block_size(2)
        .enable_prefix_cache(true)
        .event_journal(journal)
        .event_scope({"fixture", rank, "gpu"});
    if (!target_pool) {
      devices[rank] = std::make_unique<xllm::BlockManagerImpl>(options);
    }
    options.event_scope({"fixture", rank, "cpu"});
    hosts[rank] = std::make_unique<xllm::BlockManagerImpl>(options);
  }
  std::string line;
  std::cout << "ready" << std::endl;
  while (std::getline(std::cin, line)) {
    const nlohmann::json command = nlohmann::json::parse(line);
    const std::string operation = command.at("op").get<std::string>();
    if (operation == "stop") {
      if (target_pool) {
        target_pool->stop_publisher();
      } else {
        publisher.reset();
      }
      std::cout << "stopped" << std::endl;
      break;
    }
    if (operation == "state") {
      std::cout << journal->read("", 0).dump() << std::endl;
      continue;
    }
    if (operation == "status") {
      const bool healthy =
          target_pool ? target_pool->publisher_healthy() : publisher->healthy();
      std::cout << nlohmann::json({{"publisher_healthy", healthy},
                                   {"journal_enabled", journal->enabled()}})
                       .dump()
                << std::endl;
      continue;
    }
    if (operation == "create_draft") {
      CHECK(target_pool != nullptr);
      xllm::BlockManagerPool::Options options;
      options.num_blocks(64)
          .block_size(2)
          .model_id("draft")
          .enable_prefix_cache(command.value("prefix_cache", true));
      draft_pool = std::make_unique<FixtureBlockManagerPool>(options);
      xllm::BlockManager* draft = draft_pool->device(0);
      std::vector<xllm::Block> blocks = draft->allocate(1);
      CHECK_EQ(blocks.size(), 1);
      draft->cache(std::vector<int32_t>{700, 701}, blocks);
      draft->deallocate(blocks);
      std::cout << nlohmann::json(
                       {{"draft_journal_attached",
                         draft_pool->options().event_journal() != nullptr},
                        {"draft_publisher_healthy",
                         draft_pool->publisher_healthy()}})
                       .dump()
                << std::endl;
      continue;
    }
    if (operation == "disable_events") {
      journal->disable();
      std::cout << "ok" << std::endl;
      continue;
    }
    const int32_t rank = command.at("rank").get<int32_t>();
    CHECK_GE(rank, 0);
    CHECK_LT(rank, 2);
    xllm::BlockManager* device =
        target_pool ? target_pool->device(rank) : devices[rank].get();
    if (operation == "store") {
      const auto tokens = command.at("tokens").get<std::vector<int32_t>>();
      std::vector<xllm::Block> blocks = device->allocate(tokens.size() / 2);
      CHECK_EQ(blocks.size(), tokens.size() / 2);
      device->cache(tokens, blocks);
      device->deallocate(blocks);
    } else if (operation == "copy") {
      const auto tokens = command.at("tokens").get<std::vector<int32_t>>();
      std::vector<xllm::Block> source = device->allocate_shared(tokens);
      CHECK_EQ(source.size(), tokens.size() / 2);
      std::vector<xllm::Block> blocks = hosts[rank]->allocate(source.size());
      CHECK_EQ(blocks.size(), source.size());
      for (size_t index = 0; index < blocks.size(); ++index) {
        blocks[index].set_hash_value(source[index].get_immutable_hash_value());
      }
      hosts[rank]->cache(blocks);
      hosts[rank]->deallocate(blocks);
      device->deallocate(source);
    } else if (operation == "clear") {
      const std::string medium = command.at("medium").get<std::string>();
      CHECK(medium == "gpu" || medium == "cpu");
      xllm::BlockManager* manager =
          medium == "gpu" ? device : hosts[rank].get();
      const size_t capacity = manager->num_total_blocks();
      std::vector<xllm::Block> blocks = manager->allocate(capacity);
      CHECK_EQ(blocks.size(), capacity);
      manager->deallocate(blocks);
    } else {
      LOG(FATAL) << "Unknown fixture operation: " << operation;
    }
    std::cout << "ok" << std::endl;
  }
  return 0;
}
