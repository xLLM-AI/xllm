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

#include "core/distributed_runtime/model_residency_coordinator.h"

#include <folly/futures/Future.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/distributed_runtime/distributed_memory_coordinator.h"
#include "core/distributed_runtime/distributed_worker_manager.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/model_loader/model_loader.h"
#include "core/kv_cache/storage/paged_kv_cache_page_allocator.h"

namespace xllm {
namespace {

class RecordingWorkerClient final : public WorkerClient {
 public:
  explicit RecordingWorkerClient(std::string model_id)
      : model_id_(std::move(model_id)) {}

  folly::SemiFuture<bool> sleep_async(MasterStatus master_status) override {
    ++sleep_calls_;
    sleep_status_ = master_status;
    pages_sleeping_at_sleep_ = DistributedMemoryCoordinator::get_instance()
                                   .kv_cache_page_allocator()
                                   .is_suspended(model_id_);
    return folly::makeSemiFuture(sleep_result_);
  }

  folly::SemiFuture<bool> wakeup_async(const WakeupOptions& options) override {
    ++wakeup_calls_;
    wakeup_options_ = options;
    pages_awake_at_wakeup_ = !DistributedMemoryCoordinator::get_instance()
                                  .kv_cache_page_allocator()
                                  .is_suspended(model_id_);
    return folly::makeSemiFuture(wakeup_result_);
  }

  bool sleep_result_ = true;
  bool wakeup_result_ = true;
  int32_t sleep_calls_ = 0;
  int32_t wakeup_calls_ = 0;
  MasterStatus sleep_status_ = MasterStatus::WAKEUP;
  WakeupOptions wakeup_options_;
  bool pages_sleeping_at_sleep_ = false;
  bool pages_awake_at_wakeup_ = false;

 private:
  const std::string model_id_;
};

class MetadataModelLoader final : public ModelLoader {
 public:
  std::unique_ptr<Tokenizer> tokenizer() const override { return nullptr; }

  std::vector<std::unique_ptr<StateDict>>& get_state_dicts() override {
    return state_dicts_;
  }

  std::string model_weights_path() const override { return {}; }

  int64_t get_total_weight_size() const override { return total_weight_size_; }

  int64_t get_non_decoder_weight_size() const override {
    return non_decoder_weight_size_;
  }

  int64_t get_max_decoder_layer_weight_size() const override {
    return max_decoder_layer_weight_size_;
  }

  int64_t total_weight_size_ = 0;
  int64_t non_decoder_weight_size_ = 0;
  int64_t max_decoder_layer_weight_size_ = -1;

 private:
  std::vector<std::unique_ptr<StateDict>> state_dicts_;
};

void expect_weight_segments(const std::vector<WeightSegment>& actual,
                            const std::vector<WeightSegment>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t index = 0; index < expected.size(); ++index) {
    EXPECT_EQ(actual[index].offset, expected[index].offset);
    EXPECT_EQ(actual[index].size, expected[index].size);
  }
}

void expect_wakeup_options(const WakeupOptions& actual,
                           const WakeupOptions& expected) {
  EXPECT_EQ(actual.master_status, expected.master_status);
  EXPECT_EQ(actual.remote_addrs, expected.remote_addrs);
  ASSERT_EQ(actual.src_weight_segments.size(),
            expected.src_weight_segments.size());
  for (size_t index = 0; index < expected.src_weight_segments.size(); ++index) {
    expect_weight_segments(actual.src_weight_segments[index],
                           expected.src_weight_segments[index]);
  }
}

}  // namespace

class ModelResidencyCoordinatorTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    // Empty models exercise lifecycle transitions without physical pages,
    // device initialization, distributed mapping, or background preallocation.
    DistributedMemoryCoordinator::get_instance().init_page_budget(
        /*total_pages=*/8,
        /*dp_size=*/1,
        /*worker_count=*/2,
        KVCacheConfig::get_instance().phy_page_granularity_size(),
        /*enable_prealloc=*/false);
  }

  void SetUp() override {
    static uint64_t next_model_id = 0;
    model_id_ =
        "model_residency_coordinator_test_" + std::to_string(next_model_id++);
    ASSERT_TRUE(DistributedMemoryCoordinator::get_instance()
                    .kv_cache_page_allocator()
                    .register_model(model_id_, /*num_layers=*/2));

    auto& load_config = LoadConfig::get_instance();
    saved_enable_rolling_load_ = load_config.enable_rolling_load();
    saved_cached_layers_ = load_config.rolling_load_num_cached_layers();
    load_config.enable_rolling_load(false);
  }

  void TearDown() override {
    LoadConfig::get_instance()
        .enable_rolling_load(saved_enable_rolling_load_)
        .rolling_load_num_cached_layers(saved_cached_layers_);
  }

  std::shared_ptr<DistributedWorkerManager> make_manager(
      std::vector<std::shared_ptr<RecordingWorkerClient>> clients) {
    std::vector<std::shared_ptr<WorkerClient>> worker_clients;
    worker_clients.reserve(clients.size());
    for (auto& client : clients) {
      worker_clients.emplace_back(std::move(client));
    }
    return std::shared_ptr<DistributedWorkerManager>(
        new DistributedWorkerManager(std::move(worker_clients)));
  }

  ModelResidencyCoordinator make_coordinator(
      bool enabled,
      std::shared_ptr<DistributedWorkerManager> manager) {
    return make_coordinator(enabled, std::move(manager), model_id_);
  }

  ModelResidencyCoordinator make_coordinator(
      bool enabled,
      std::shared_ptr<DistributedWorkerManager> manager,
      std::string model_id) {
    ModelResidencyCoordinator::Options options;
    options.enabled = enabled;
    options.model_id = std::move(model_id);
    return ModelResidencyCoordinator(std::move(options), std::move(manager));
  }

  void suspend_model_pages() {
    auto& page_allocator =
        DistributedMemoryCoordinator::get_instance().kv_cache_page_allocator();
    const auto plan = page_allocator.begin_suspend(model_id_);
    ASSERT_TRUE(plan.has_value());
    page_allocator.finish_suspend(model_id_);
  }

  std::string model_id_;

 private:
  bool saved_enable_rolling_load_ = false;
  int32_t saved_cached_layers_ = 0;
};

TEST_F(ModelResidencyCoordinatorTest,
       DisabledOperationsPreserveOutputsAndModelState) {
  auto worker = std::make_shared<RecordingWorkerClient>(model_id_);
  auto coordinator =
      make_coordinator(/*enabled=*/false, make_manager({worker}));
  const MetadataModelLoader model_loader{};
  EXPECT_TRUE(coordinator.initialize_model(model_loader,
                                           /*num_layers=*/0,
                                           /*dp_size=*/0,
                                           /*tp_size=*/0,
                                           MasterStatus::DEEP_SLEEP));
  EXPECT_TRUE(coordinator.finish_initialization(MasterStatus::DEEP_SLEEP));
  EXPECT_FALSE(coordinator.sleep(MasterStatus::DEEP_SLEEP));
  EXPECT_FALSE(coordinator.wakeup(WakeupOptions{}));
  EXPECT_FALSE(DistributedMemoryCoordinator::get_instance()
                   .kv_cache_page_allocator()
                   .is_suspended(model_id_));
  EXPECT_EQ(worker->sleep_calls_, 0);
  EXPECT_EQ(worker->wakeup_calls_, 0);

  std::vector<size_t> free_pages = {3, 5};
  std::unordered_map<std::string, std::vector<WeightSegment>> weight_segments =
      {{"existing", {{7, 11}}}};
  coordinator.get_virtual_memory_info(free_pages, weight_segments);
  EXPECT_EQ(free_pages, std::vector<size_t>({3, 5}));
  ASSERT_EQ(weight_segments.size(), 1);
  ASSERT_EQ(weight_segments.count("existing"), 1);
  expect_weight_segments(weight_segments.at("existing"), {{7, 11}});
}

TEST_F(ModelResidencyCoordinatorTest,
       RecordsSleepingWeightPagesWithoutAllocation) {
  const int64_t page_size =
      KVCacheConfig::get_instance().phy_page_granularity_size();
  MetadataModelLoader model_loader{};
  model_loader.total_weight_size_ = 5 * page_size + 1;
  const std::vector<MasterStatus> statuses = {MasterStatus::LIGHT_SLEEP,
                                              MasterStatus::DEEP_SLEEP};

  for (size_t index = 0; index < statuses.size(); ++index) {
    const std::string model_id =
        model_id_ + "_bootstrap_" + std::to_string(index);
    auto first = std::make_shared<RecordingWorkerClient>(model_id);
    auto second = std::make_shared<RecordingWorkerClient>(model_id);
    auto coordinator = make_coordinator(
        /*enabled=*/true, make_manager({first, second}), model_id);

    ASSERT_TRUE(coordinator.initialize_model(model_loader,
                                             /*num_layers=*/2,
                                             /*dp_size=*/1,
                                             /*tp_size=*/2,
                                             statuses[index]));

    auto& memory_coordinator = DistributedMemoryCoordinator::get_instance();
    EXPECT_EQ(memory_coordinator.get_model_parallel_strategy(model_id),
              std::make_pair(/*dp_size=*/1, /*tp_size=*/2));
    // 5 pages + 1 byte across two TP ranks needs 3 pages per worker, plus
    // the existing 20-page weight safety margin.
    EXPECT_EQ(memory_coordinator.get_weight_page_count(model_id), 23);
    EXPECT_EQ(memory_coordinator.get_worker_free_page_counts(),
              std::vector<size_t>({8, 8}));
    EXPECT_FALSE(
        memory_coordinator.kv_cache_page_allocator().is_suspended(model_id));

    EXPECT_TRUE(coordinator.finish_initialization(statuses[index]));
    EXPECT_TRUE(
        memory_coordinator.kv_cache_page_allocator().is_suspended(model_id));
    EXPECT_EQ(memory_coordinator.get_weight_page_count(model_id), 23);
    EXPECT_EQ(memory_coordinator.get_worker_free_page_counts(),
              std::vector<size_t>({8, 8}));
    EXPECT_EQ(first->sleep_calls_, 0);
    EXPECT_EQ(second->sleep_calls_, 0);
  }
}

TEST_F(ModelResidencyCoordinatorTest,
       RollingLoadBudgetsOnlyCachedDecoderLayers) {
  LoadConfig::get_instance()
      .enable_rolling_load(true)
      .rolling_load_num_cached_layers(2);
  const int64_t page_size =
      KVCacheConfig::get_instance().phy_page_granularity_size();
  MetadataModelLoader model_loader{};
  model_loader.total_weight_size_ = 100 * page_size;
  model_loader.non_decoder_weight_size_ = page_size + 1;
  model_loader.max_decoder_layer_weight_size_ = 3 * page_size + 1;
  const std::string model_id = model_id_ + "_rolling";
  auto first = std::make_shared<RecordingWorkerClient>(model_id);
  auto second = std::make_shared<RecordingWorkerClient>(model_id);
  auto coordinator = make_coordinator(
      /*enabled=*/true, make_manager({first, second}), model_id);

  ASSERT_TRUE(coordinator.initialize_model(model_loader,
                                           /*num_layers=*/10,
                                           /*dp_size=*/1,
                                           /*tp_size=*/2,
                                           MasterStatus::LIGHT_SLEEP));

  auto& memory_coordinator = DistributedMemoryCoordinator::get_instance();
  // Non-decoder weights + two cached layers total 7 pages + 3 bytes.
  // Two TP ranks need 4 pages each, plus the 20-page weight safety margin.
  EXPECT_EQ(memory_coordinator.get_weight_page_count(model_id), 24);
  EXPECT_EQ(memory_coordinator.get_worker_free_page_counts(),
            std::vector<size_t>({8, 8}));
  EXPECT_TRUE(coordinator.finish_initialization(MasterStatus::LIGHT_SLEEP));
  EXPECT_TRUE(
      memory_coordinator.kv_cache_page_allocator().is_suspended(model_id));
  EXPECT_EQ(memory_coordinator.get_weight_page_count(model_id), 24);
}

TEST_F(ModelResidencyCoordinatorTest,
       InvalidWeightMetadataDoesNotAllocatePages) {
  LoadConfig::get_instance().enable_rolling_load(true);
  struct WeightMetadata {
    int64_t total_size;
    int64_t non_decoder_size;
    int64_t max_layer_size;
  };
  const std::vector<WeightMetadata> invalid_metadata = {
      {0, 16, 32}, {128, 0, 32}, {128, 129, 32}, {128, 16, 0}, {128, 16, -1}};

  for (size_t index = 0; index < invalid_metadata.size(); ++index) {
    MetadataModelLoader model_loader{};
    model_loader.total_weight_size_ = invalid_metadata[index].total_size;
    model_loader.non_decoder_weight_size_ =
        invalid_metadata[index].non_decoder_size;
    model_loader.max_decoder_layer_weight_size_ =
        invalid_metadata[index].max_layer_size;
    const std::string model_id =
        model_id_ + "_invalid_" + std::to_string(index);
    auto first = std::make_shared<RecordingWorkerClient>(model_id);
    auto second = std::make_shared<RecordingWorkerClient>(model_id);
    auto coordinator = make_coordinator(
        /*enabled=*/true, make_manager({first, second}), model_id);

    EXPECT_FALSE(coordinator.initialize_model(model_loader,
                                              /*num_layers=*/2,
                                              /*dp_size=*/1,
                                              /*tp_size=*/2,
                                              MasterStatus::DEEP_SLEEP));
    EXPECT_EQ(DistributedMemoryCoordinator::get_instance()
                  .get_worker_free_page_counts(),
              std::vector<size_t>({8, 8}));
  }
}

TEST_F(ModelResidencyCoordinatorTest, AwakeInitializationLeavesModelAwake) {
  auto coordinator = make_coordinator(/*enabled=*/true, make_manager({}));

  EXPECT_TRUE(coordinator.finish_initialization(MasterStatus::WAKEUP));
  EXPECT_FALSE(DistributedMemoryCoordinator::get_instance()
                   .kv_cache_page_allocator()
                   .is_suspended(model_id_));
}

TEST_F(ModelResidencyCoordinatorTest, MissingWorkerManagerPreservesModelState) {
  auto coordinator = make_coordinator(/*enabled=*/true, nullptr);
  MetadataModelLoader model_loader{};
  model_loader.total_weight_size_ = 1;
  EXPECT_FALSE(coordinator.initialize_model(model_loader,
                                            /*num_layers=*/2,
                                            /*dp_size=*/1,
                                            /*tp_size=*/1,
                                            MasterStatus::DEEP_SLEEP));
  EXPECT_EQ(
      DistributedMemoryCoordinator::get_instance().get_model_parallel_strategy(
          model_id_),
      std::make_pair(/*dp_size=*/1, /*tp_size=*/2));
  EXPECT_FALSE(coordinator.sleep(MasterStatus::DEEP_SLEEP));
  EXPECT_FALSE(DistributedMemoryCoordinator::get_instance()
                   .kv_cache_page_allocator()
                   .is_suspended(model_id_));

  suspend_model_pages();
  EXPECT_FALSE(coordinator.wakeup(WakeupOptions{}));
  EXPECT_TRUE(DistributedMemoryCoordinator::get_instance()
                  .kv_cache_page_allocator()
                  .is_suspended(model_id_));
}

TEST_F(ModelResidencyCoordinatorTest, EmptyWorkerManagerPreservesModelState) {
  auto coordinator = make_coordinator(/*enabled=*/true, make_manager({}));
  MetadataModelLoader model_loader{};
  model_loader.total_weight_size_ = 1;
  EXPECT_FALSE(coordinator.initialize_model(model_loader,
                                            /*num_layers=*/2,
                                            /*dp_size=*/1,
                                            /*tp_size=*/1,
                                            MasterStatus::DEEP_SLEEP));
  EXPECT_EQ(
      DistributedMemoryCoordinator::get_instance().get_model_parallel_strategy(
          model_id_),
      std::make_pair(/*dp_size=*/1, /*tp_size=*/2));
  EXPECT_FALSE(coordinator.sleep(MasterStatus::DEEP_SLEEP));
  EXPECT_FALSE(DistributedMemoryCoordinator::get_instance()
                   .kv_cache_page_allocator()
                   .is_suspended(model_id_));

  suspend_model_pages();
  EXPECT_FALSE(coordinator.wakeup(WakeupOptions{}));
  EXPECT_TRUE(DistributedMemoryCoordinator::get_instance()
                  .kv_cache_page_allocator()
                  .is_suspended(model_id_));
}

TEST_F(ModelResidencyCoordinatorTest, ReportsSharedPageCountsWithoutWorkerRpc) {
  auto coordinator = make_coordinator(/*enabled=*/true, nullptr);
  std::vector<size_t> free_pages = {3, 5};
  std::unordered_map<std::string, std::vector<WeightSegment>> weight_segments =
      {{"stale", {{7, 11}}}};

  coordinator.get_virtual_memory_info(free_pages, weight_segments);

  EXPECT_EQ(free_pages, std::vector<size_t>({8, 8}));
  EXPECT_TRUE(weight_segments.empty());
}

TEST_F(ModelResidencyCoordinatorTest, BroadcastsLifecycleAfterPageTransitions) {
  auto first = std::make_shared<RecordingWorkerClient>(model_id_);
  auto second = std::make_shared<RecordingWorkerClient>(model_id_);
  auto coordinator =
      make_coordinator(/*enabled=*/true, make_manager({first, second}));

  // An awake model cannot wake up again and must not dispatch to workers.
  EXPECT_FALSE(coordinator.wakeup(WakeupOptions{}));
  EXPECT_TRUE(coordinator.sleep(MasterStatus::DEEP_SLEEP));
  EXPECT_FALSE(coordinator.sleep(MasterStatus::DEEP_SLEEP));
  WakeupOptions options;
  options.master_status = MasterStatus::DEEP_SLEEP;
  EXPECT_TRUE(coordinator.wakeup(options));

  for (const auto& worker : {first, second}) {
    EXPECT_EQ(worker->sleep_calls_, 1);
    EXPECT_EQ(worker->sleep_status_, MasterStatus::DEEP_SLEEP);
    EXPECT_TRUE(worker->pages_sleeping_at_sleep_);
    EXPECT_EQ(worker->wakeup_calls_, 1);
    EXPECT_TRUE(worker->pages_awake_at_wakeup_);
    expect_wakeup_options(worker->wakeup_options_, options);
  }
}

TEST_F(ModelResidencyCoordinatorTest,
       ReportsSleepFailureAfterCallingEveryWorker) {
  auto first = std::make_shared<RecordingWorkerClient>(model_id_);
  auto second = std::make_shared<RecordingWorkerClient>(model_id_);
  first->sleep_result_ = false;
  auto coordinator =
      make_coordinator(/*enabled=*/true, make_manager({first, second}));

  EXPECT_FALSE(coordinator.sleep(MasterStatus::LIGHT_SLEEP));
  EXPECT_TRUE(DistributedMemoryCoordinator::get_instance()
                  .kv_cache_page_allocator()
                  .is_suspended(model_id_));
  EXPECT_EQ(first->sleep_calls_, 1);
  EXPECT_EQ(second->sleep_calls_, 1);
}

TEST_F(ModelResidencyCoordinatorTest,
       AssignsP2pSourcesAndSegmentsByWorkerRank) {
  auto first = std::make_shared<RecordingWorkerClient>(model_id_);
  auto second = std::make_shared<RecordingWorkerClient>(model_id_);
  auto coordinator =
      make_coordinator(/*enabled=*/true, make_manager({first, second}));
  suspend_model_pages();

  WakeupOptions options;
  options.master_status = MasterStatus::DEEP_SLEEP;
  options.remote_addrs = {"source-0", "source-1"};
  options.src_weight_segments = {{{32, 8}, {64, 16}}, {{128, 32}}};
  EXPECT_TRUE(coordinator.wakeup(options));

  WakeupOptions first_options;
  first_options.master_status = MasterStatus::DEEP_SLEEP;
  first_options.remote_addrs = {"source-0"};
  first_options.src_weight_segments = {{{32, 8}, {64, 16}}};
  expect_wakeup_options(first->wakeup_options_, first_options);
  WakeupOptions second_options;
  second_options.master_status = MasterStatus::DEEP_SLEEP;
  second_options.remote_addrs = {"source-1"};
  second_options.src_weight_segments = {{{128, 32}}};
  expect_wakeup_options(second->wakeup_options_, second_options);
  EXPECT_EQ(first->wakeup_calls_, 1);
  EXPECT_EQ(second->wakeup_calls_, 1);
}

TEST_F(ModelResidencyCoordinatorTest,
       MissingP2pSegmentsStayEmptyForTheCorrespondingRank) {
  auto first = std::make_shared<RecordingWorkerClient>(model_id_);
  auto second = std::make_shared<RecordingWorkerClient>(model_id_);
  auto coordinator =
      make_coordinator(/*enabled=*/true, make_manager({first, second}));
  suspend_model_pages();

  WakeupOptions options;
  options.master_status = MasterStatus::LIGHT_SLEEP;
  options.remote_addrs = {"source-0", "source-1"};
  options.src_weight_segments = {{{32, 8}}};
  EXPECT_TRUE(coordinator.wakeup(options));

  WakeupOptions second_options;
  second_options.master_status = MasterStatus::LIGHT_SLEEP;
  second_options.remote_addrs = {"source-1"};
  expect_wakeup_options(second->wakeup_options_, second_options);
  ASSERT_EQ(first->wakeup_options_.src_weight_segments.size(), 1);
  expect_weight_segments(first->wakeup_options_.src_weight_segments.front(),
                         {{32, 8}});
}

TEST_F(ModelResidencyCoordinatorTest,
       BroadcastsSharedSourcesWhenRankCountsDiffer) {
  auto first = std::make_shared<RecordingWorkerClient>(model_id_);
  auto second = std::make_shared<RecordingWorkerClient>(model_id_);
  auto coordinator =
      make_coordinator(/*enabled=*/true, make_manager({first, second}));
  suspend_model_pages();

  WakeupOptions options;
  options.master_status = MasterStatus::DEEP_SLEEP;
  options.remote_addrs = {"shared-source"};
  options.src_weight_segments = {{{32, 8}, {64, 16}}};
  EXPECT_TRUE(coordinator.wakeup(options));

  expect_wakeup_options(first->wakeup_options_, options);
  expect_wakeup_options(second->wakeup_options_, options);
}

TEST_F(ModelResidencyCoordinatorTest,
       ReportsWakeupFailureAfterCallingEveryWorker) {
  auto first = std::make_shared<RecordingWorkerClient>(model_id_);
  auto second = std::make_shared<RecordingWorkerClient>(model_id_);
  first->wakeup_result_ = false;
  auto coordinator =
      make_coordinator(/*enabled=*/true, make_manager({first, second}));
  suspend_model_pages();

  EXPECT_FALSE(coordinator.wakeup(WakeupOptions{}));
  EXPECT_EQ(first->wakeup_calls_, 1);
  EXPECT_EQ(second->wakeup_calls_, 1);
}

}  // namespace xllm
