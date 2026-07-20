/* Copyright 2025-2026 The xLLM Authors.

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

#include "framework/kv_cache_transfer/hierarchy_kv_cache_transfer.h"

#include <folly/futures/Future.h>
#include <sys/mman.h>

#include <cerrno>
#include <cstring>
#include <memory>

#include "framework/kv_cache_transfer/kv_cache_store.h"
namespace xllm {

constexpr uint64_t MBUF_SIZE = 128 * 1024 * 1024;
constexpr uint32_t BATCH_COPY_MAX_SIZE = 4096;
constexpr uint32_t TIMEOUT_S = 60;      // second
constexpr uint32_t TIMEOUT_MS = 60000;  // millisecond

HierarchyKVCacheTransfer::HierarchyKVCacheTransfer(
    const Options& options,
    const torch::Device& device,
    std::vector<xllm::KVCache>* kv_caches_ptr)
    : options_(options), device_(device), kv_caches_ptr_(kv_caches_ptr) {
  device_.set_device();
  device_.init_device_context();
  h2d_threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/2,
      /*init_func=*/[this]() mutable { device_.set_device(); },
      /*cpu_binding=*/false,
      /*pool_name=*/"HierarchyKVCacheTransfer.h2d");
  d2h_threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/5,
      /*init_func=*/[this]() mutable { device_.set_device(); },
      /*cpu_binding=*/false,
      /*pool_name=*/"HierarchyKVCacheTransfer.d2h");
  for (int i = 0; i < h2d_threadpool_->size() + d2h_threadpool_->size(); i++) {
    copy_stream_.enqueue(device_.get_stream_from_pool(TIMEOUT_MS));
  }

  if (options_.host_blocks_factor() > 1) {
    create_page_aligned_host_cache();
  }

  if (options_.enable_kvcache_store()) {
    KVCacheStoreInitConfig config;
    config.localhost_name = options_.store_local_hostname();
    config.protocol = options_.store_protocol();
    config.metadata_server = options_.store_metadata_server();
    config.master_server_address = options_.store_master_server_address();
    config.tp_rank = options_.tp_rank();
    config.total_size = page_aligned_data_size_;
    config.tensor_data = page_aligned_data_;

    if (!KVCacheStore::get_instance().init(config, &host_kv_caches_)) {
      LOG(FATAL) << "Init KVCacheStore fail!";
    }
  }
}

HierarchyKVCacheTransfer::~HierarchyKVCacheTransfer() {
  if (page_aligned_data_ != nullptr) {
#if defined(USE_NPU)
    aclrtHostUnregister(page_aligned_data_);
#endif
    munlock(page_aligned_data_, page_aligned_data_size_);
    munmap(page_aligned_data_, page_aligned_data_size_);
  }
}

uint32_t HierarchyKVCacheTransfer::transfer_kv_blocks(
    const uint64_t batch_id,
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  CHECK(!block_transfer_info.empty());

  switch (block_transfer_info[0].transfer_type) {
    case TransferType::H2D: {
      h2d_threadpool_->schedule(
          [this,
           batch_id = batch_id,
           block_transfer_info = std::move(block_transfer_info)]() mutable {
            Slice<BlockTransferInfo> info_slice{block_transfer_info};
            h2d_batch_copy(batch_id, info_slice);
          });
      break;
    }
    case TransferType::D2G:
      return offload_kv_blocks(std::move(block_transfer_info));
    case TransferType::G2D: {
      // TODO load_kv_blocks async
      LOG(ERROR) << "Unsupport copy type G2D.";
      break;
    }
    default:
      LOG(ERROR) << "Unsupport copy type: "
                 << uint32_t(block_transfer_info[0].transfer_type);
      break;
  }
  return 0;
}

uint32_t HierarchyKVCacheTransfer::transfer_kv_blocks(
    const uint64_t batch_id,
    Slice<BlockTransferInfo>& block_transfer_info) {
  CHECK(!block_transfer_info.empty());

  switch (block_transfer_info[0].transfer_type) {
    case TransferType::G2H:
      return load_from_store(block_transfer_info);
    default:
      LOG(ERROR) << "Unsupport copy type: "
                 << uint32_t(block_transfer_info[0].transfer_type);
      break;
  }
  return 0;
}

void HierarchyKVCacheTransfer::set_layer_synchronizer(
    ModelInputParams& params) {
#if defined(USE_NPU)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (layer_wise_load_synchronizer_.count(params.meta.batch_id) != 0) {
      params.parallel.layer_wise_load_synchronizer =
          layer_wise_load_synchronizer_[params.meta.batch_id];
      layer_wise_load_synchronizer_.erase(params.meta.batch_id);
      uint32_t event_cnt =
          params.parallel.layer_wise_load_synchronizer->get_event_size();
      params.parallel.layers_per_bacth_copy =
          (options_.layers() + event_cnt - 1) / event_cnt;
    }
  }
#endif
}

uint32_t HierarchyKVCacheTransfer::offload_kv_blocks(
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  if (block_transfer_info.empty()) {
    return 0;
  }

  const int64_t num_layers = options_.layers();
  uint32_t max_blocks_per_batch =
      BATCH_COPY_MAX_SIZE / (cache_tensor_cnt_ * num_layers);
  uint32_t total_slice =
      (block_transfer_info.size() + max_blocks_per_batch - 1) /
      max_blocks_per_batch;

  Slice transfer_info_slice(block_transfer_info);
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(total_slice);

  for (size_t i = 0; i < block_transfer_info.size();
       i += max_blocks_per_batch) {
    folly::Promise<bool> promise;
    auto future = promise.getSemiFuture();
    auto slice = transfer_info_slice.slice(
        i, std::min(i + max_blocks_per_batch, block_transfer_info.size()));

    d2h_threadpool_->schedule([this,
                               promise = std::move(promise),
                               slice = std::move(slice)]() mutable {
      bool ret = d2h_batch_copy(slice);
      auto success_cnt = offload_to_store(slice);
      if (success_cnt != slice.size()) {
        LOG(WARNING) << "KVCacheStore not all put success: " << success_cnt
                     << "/" << slice.size();
      }
      promise.setValue(ret);
    });

    futures.emplace_back(std::move(future));
  }

  if (!futures.empty()) {
    try {
      // TODO(kangmeng): add timeout
      auto all_results = folly::collect(futures).get();
      if (!std::all_of(all_results.begin(), all_results.end(), [](bool result) {
            return result;
          })) {
        LOG(FATAL) << "Not all D2H copy returned true";
      }
    } catch (const std::exception& e) {
      LOG(FATAL) << "Future execution failed: " << e.what();
    }
  }

  return block_transfer_info.size();
}

bool HierarchyKVCacheTransfer::d2h_batch_copy(
    Slice<BlockTransferInfo>& block_transfer_info) {
#if defined(USE_NPU)
  const int64_t num_layers = options_.layers();
  uint32_t num_batches =
      block_transfer_info.size() * num_layers * cache_tensor_cnt_;
  void** srcs = new void*[num_batches];
  void** dsts = new void*[num_batches];
  size_t* copy_size = new size_t[num_batches];
  aclrtMemcpyBatchAttr attrs[1] = {d2h_attrs_};
  size_t attrs_indexes[1] = {0};
  size_t fail_index;
  uint32_t curr_index = 0;

  for (const auto& info : block_transfer_info) {
    auto dst_k_cache = host_kv_caches_.at(info.dst_block_id).get_k_cache();
    auto dst_v_cache = host_kv_caches_.at(info.dst_block_id).get_v_cache();
    auto dst_index_cache =
        host_kv_caches_.at(info.dst_block_id).get_index_cache();
    auto dst_index_cache_scale =
        host_kv_caches_.at(info.dst_block_id).get_indexer_cache_scale();

    for (int layer_id = 0; layer_id < num_layers; layer_id++) {
      auto src_k_cache = kv_caches_ptr_->at(layer_id).get_k_cache();
      srcs[curr_index] = src_k_cache[info.src_block_id].data_ptr();
      dsts[curr_index] = dst_k_cache[layer_id].data_ptr();
      copy_size[curr_index] = cache_size_per_layer_[0];
      curr_index++;

      if (cache_size_per_layer_[1] != 0) {
        auto src_v_cache = kv_caches_ptr_->at(layer_id).get_v_cache();
        srcs[curr_index] = src_v_cache[info.src_block_id].data_ptr();
        dsts[curr_index] = dst_v_cache[layer_id].data_ptr();
        copy_size[curr_index] = cache_size_per_layer_[1];
        curr_index++;
      }

      if (cache_size_per_layer_[2] != 0) {
        auto src_index_cache = kv_caches_ptr_->at(layer_id).get_index_cache();
        srcs[curr_index] = src_index_cache[info.src_block_id].data_ptr();
        dsts[curr_index] = dst_index_cache[layer_id].data_ptr();
        copy_size[curr_index] = cache_size_per_layer_[2];
        curr_index++;
      }

      if (cache_size_per_layer_[3] != 0) {
        auto src_index_cache_scale =
            kv_caches_ptr_->at(layer_id).get_indexer_cache_scale();
        CHECK(src_index_cache_scale.has_value())
            << "index cache scale is required for D2H copy.";
        CHECK(dst_index_cache_scale.has_value())
            << "host index cache scale is required for D2H copy.";
        srcs[curr_index] =
            src_index_cache_scale.value()[info.src_block_id].data_ptr();
        dsts[curr_index] = dst_index_cache_scale.value()[layer_id].data_ptr();
        copy_size[curr_index] = cache_size_per_layer_[3];
        curr_index++;
      }
    }
  }

  std::unique_ptr<Stream> stream;
  copy_stream_.wait_dequeue(stream);
  c10::StreamGuard streamGuard = stream->set_stream_guard();

  // TODO(kangmeng): change to async API
  aclError ret = aclrtMemcpyBatch(dsts,
                                  copy_size,
                                  srcs,
                                  copy_size,
                                  num_batches,
                                  attrs,
                                  attrs_indexes,
                                  1,
                                  &fail_index);
  if (ret != 0 || fail_index != SIZE_MAX) {
    LOG(ERROR) << "aclrtMemcpyBatch error: " << ret
               << ", fail_index:" << fail_index;
    copy_stream_.enqueue(std::move(stream));
    return false;
  }

  if (stream->synchronize() != 0) {
    LOG(ERROR) << "d2h_batch_copy timeout!";
    copy_stream_.enqueue(std::move(stream));
    return false;
  }

  copy_stream_.enqueue(std::move(stream));

  delete[] dsts;
  delete[] srcs;
  delete[] copy_size;
#endif
  return true;
}

bool HierarchyKVCacheTransfer::h2d_batch_copy(
    const uint64_t batch_id,
    Slice<BlockTransferInfo>& block_transfer_info) {
#if defined(USE_NPU)
  CHECK(block_transfer_info.size() < BATCH_COPY_MAX_SIZE / cache_tensor_cnt_)
      << "h2d_batch_copy support copy blocks less than "
      << BATCH_COPY_MAX_SIZE / cache_tensor_cnt_ << ", but got "
      << block_transfer_info.size();

  if (block_transfer_info.empty()) {
    return true;
  }

  const int64_t num_layers = options_.layers();
  uint32_t layers_per_bacth_copy =
      num_layers / options_.layers_wise_copy_batchs();
  uint32_t num_batches = block_transfer_info.size() * cache_tensor_cnt_;
  while (num_batches * layers_per_bacth_copy > BATCH_COPY_MAX_SIZE) {
    layers_per_bacth_copy--;
  }

  uint32_t copy_cnt =
      (num_layers + layers_per_bacth_copy - 1) / layers_per_bacth_copy;
  auto synchronizer = std::make_shared<NPULayerSynchronizerImpl>(copy_cnt);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (layer_wise_load_synchronizer_.count(batch_id) != 0) {
      LOG(FATAL) << "Batch id already exists!";
    }
    layer_wise_load_synchronizer_[batch_id] = synchronizer;
  }

  aclrtMemcpyBatchAttr attrs[1] = {h2d_attrs_};
  size_t attrs_indexes[1] = {0};

  std::unique_ptr<Stream> stream;
  copy_stream_.wait_dequeue(stream);
  c10::StreamGuard streamGuard = stream->set_stream_guard();
  aclError ret = 0;

  void** srcs = new void*[num_batches * layers_per_bacth_copy];
  void** dsts = new void*[num_batches * layers_per_bacth_copy];
  size_t* copy_size = new size_t[num_batches * layers_per_bacth_copy];

  for (int index = 0; index < copy_cnt; index++) {
    int layer_id = index * layers_per_bacth_copy;
    size_t fail_index = 0;
    uint32_t curr_index = 0;
    uint32_t layer_cnt = 0;

    while (layer_id < (index + 1) * layers_per_bacth_copy &&
           layer_id < num_layers) {
      auto dst_k_cache = kv_caches_ptr_->at(layer_id).get_k_cache();
      auto dst_v_cache = kv_caches_ptr_->at(layer_id).get_v_cache();
      auto dst_index_cache = kv_caches_ptr_->at(layer_id).get_index_cache();
      auto dst_index_cache_scale =
          kv_caches_ptr_->at(layer_id).get_indexer_cache_scale();

      for (const auto& info : block_transfer_info) {
        auto src_k_cache = host_kv_caches_.at(info.src_block_id).get_k_cache();
        srcs[curr_index] = src_k_cache[layer_id].data_ptr();
        dsts[curr_index] = dst_k_cache[info.dst_block_id].data_ptr();
        copy_size[curr_index] = cache_size_per_layer_[0];
        curr_index++;

        if (cache_size_per_layer_[1] != 0) {
          auto src_v_cache =
              host_kv_caches_.at(info.src_block_id).get_v_cache();
          srcs[curr_index] = src_v_cache[layer_id].data_ptr();
          dsts[curr_index] = dst_v_cache[info.dst_block_id].data_ptr();
          copy_size[curr_index] = cache_size_per_layer_[1];
          curr_index++;
        }

        if (cache_size_per_layer_[2] != 0) {
          auto src_index_cache =
              host_kv_caches_.at(info.src_block_id).get_index_cache();
          srcs[curr_index] = src_index_cache[layer_id].data_ptr();
          dsts[curr_index] = dst_index_cache[info.dst_block_id].data_ptr();
          copy_size[curr_index] = cache_size_per_layer_[2];
          curr_index++;
        }

        if (cache_size_per_layer_[3] != 0) {
          auto src_index_cache_scale =
              host_kv_caches_.at(info.src_block_id).get_indexer_cache_scale();
          CHECK(src_index_cache_scale.has_value())
              << "host index cache scale is required for H2D copy.";
          CHECK(dst_index_cache_scale.has_value())
              << "index cache scale is required for H2D copy.";
          srcs[curr_index] = src_index_cache_scale.value()[layer_id].data_ptr();
          dsts[curr_index] =
              dst_index_cache_scale.value()[info.dst_block_id].data_ptr();
          copy_size[curr_index] = cache_size_per_layer_[3];
          curr_index++;
        }
      }
      layer_id++;
      layer_cnt++;
    }

    ret = aclrtMemcpyBatch(dsts,
                           copy_size,
                           srcs,
                           copy_size,
                           num_batches * layer_cnt,
                           attrs,
                           attrs_indexes,
                           1,
                           &fail_index);

    if (ret != 0 || fail_index != SIZE_MAX) {
      LOG(ERROR) << "aclrtMemcpyBatch error: " << ret
                 << ", fail_index:" << fail_index;
    } else {
      auto* event = synchronizer->get_event(index);
      ret = aclrtRecordEvent(*event, stream->get_stream()->stream());
      if (ret != 0) {
        LOG(ERROR) << "aclrtRecordEvent error: " << ret;
      }
    }
    auto* event_flag = synchronizer->get_event_flag(index);
    event_flag->store(true, std::memory_order_release);
    if (ret != 0) break;
  }

  if (stream->synchronize() != 0) {
    LOG(ERROR) << "h2d_batch_copy timeout!";
    copy_stream_.enqueue(std::move(stream));
    return false;
  }

  copy_stream_.enqueue(std::move(stream));

  delete[] dsts;
  delete[] srcs;
  delete[] copy_size;
#endif
  return true;
}

uint32_t HierarchyKVCacheTransfer::offload_to_store(
    Slice<BlockTransferInfo>& block_transfer_info) {
  if (!options_.enable_kvcache_store()) {
    return block_transfer_info.size();
  }

  return KVCacheStore::get_instance().batch_put(block_transfer_info);
}

uint32_t HierarchyKVCacheTransfer::load_from_store(
    Slice<BlockTransferInfo>& block_transfer_info) {
  if (!options_.enable_kvcache_store()) {
    return 0;
  }
  return KVCacheStore::get_instance().batch_get(block_transfer_info);
}

void HierarchyKVCacheTransfer::create_page_aligned_host_cache() {
  CHECK(kv_caches_ptr_->size() > 0) << "hbm kv cache size should > 0.";
  CHECK(options_.host_blocks_factor() > 1) << "host_blocks_factor should > 1.";

  std::vector<std::vector<int64_t>> tensor_shapes =
      kv_caches_ptr_->at(0).get_shapes();
  tensor_shapes.resize(4);

  CHECK(!tensor_shapes[0].empty()) << "k cache should not be empty!";

#if defined(USE_NPU)
  int32_t device_id = device_.index();
  h2d_attrs_.dstLoc.id = device_id;
  h2d_attrs_.dstLoc.type = aclrtMemLocationType::ACL_MEM_LOCATION_TYPE_DEVICE;
  h2d_attrs_.srcLoc.id = device_id;
  h2d_attrs_.srcLoc.type = aclrtMemLocationType::ACL_MEM_LOCATION_TYPE_HOST;
  memset(h2d_attrs_.rsv, 0, 16);

  d2h_attrs_.dstLoc.id = device_id;
  d2h_attrs_.dstLoc.type = aclrtMemLocationType::ACL_MEM_LOCATION_TYPE_HOST;
  d2h_attrs_.srcLoc.id = device_id;
  d2h_attrs_.srcLoc.type = aclrtMemLocationType::ACL_MEM_LOCATION_TYPE_DEVICE;
  memset(d2h_attrs_.rsv, 0, 16);
#endif

  cache_size_per_layer_.resize(4, 0);
  cache_size_per_layer_[0] =
      kv_caches_ptr_->at(0).get_k_cache()[0].numel() *
      kv_caches_ptr_->at(0).get_k_cache()[0].element_size();

  if (!tensor_shapes[1].empty()) {
    cache_size_per_layer_[1] =
        kv_caches_ptr_->at(0).get_v_cache()[0].numel() *
        kv_caches_ptr_->at(0).get_v_cache()[0].element_size();
    cache_tensor_cnt_++;
  }

  if (!tensor_shapes[2].empty()) {
    cache_size_per_layer_[2] =
        kv_caches_ptr_->at(0).get_index_cache()[0].numel() *
        kv_caches_ptr_->at(0).get_index_cache()[0].element_size();
    cache_tensor_cnt_++;
  }

  if (!tensor_shapes[3].empty()) {
    auto index_cache_scale = kv_caches_ptr_->at(0).get_indexer_cache_scale();
    CHECK(index_cache_scale.has_value())
        << "index cache scale shape exists but tensor is missing.";
    cache_size_per_layer_[3] = index_cache_scale.value()[0].numel() *
                               index_cache_scale.value()[0].element_size();
    cache_tensor_cnt_++;
  }

  uint64_t num_blocks = tensor_shapes[0][0] * options_.host_blocks_factor();
  std::vector<uint64_t> cache_size_per_tensor(4, 0);

  for (size_t i = 0; i < tensor_shapes.size(); i++) {
    if (!tensor_shapes[i].empty()) {
      tensor_shapes[i][0] = options_.layers();
      cache_size_per_tensor[i] = cache_size_per_layer_[i] * options_.layers();
      page_aligned_data_size_ += num_blocks * cache_size_per_tensor[i];
    }
  }

  size_t page_size = sysconf(_SC_PAGESIZE);
  page_aligned_data_size_ =
      ((page_aligned_data_size_ + page_size - 1) / page_size) * page_size;

  page_aligned_data_ = mmap(nullptr,
                            page_aligned_data_size_,
                            PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS,
                            -1,
                            0);

  if (page_aligned_data_ == MAP_FAILED) {
    LOG(FATAL) << "Failed to allocate aligned memory pool!";
  }

  if (mlock(page_aligned_data_, page_aligned_data_size_) != 0) {
    int err = errno;
    munmap(page_aligned_data_, page_aligned_data_size_);
    uint64_t limit_kb = (page_aligned_data_size_ + 1023) / 1024;
    LOG(FATAL) << "Failed to lock memory pool! mlock errno=" << err << " ("
               << strerror(err) << "), size=" << page_aligned_data_size_
               << " bytes (~" << (page_aligned_data_size_ >> 20) << " MiB). "
               << "Run: ulimit -l " << limit_kb << " (or ulimit -l unlimited) "
               << "then restart the process.";
  }

#if defined(USE_NPU)
  auto ret = aclrtHostRegister(page_aligned_data_,
                               page_aligned_data_size_,
                               aclrtHostRegisterType::ACL_HOST_REGISTER_MAPPED,
                               &mapped_ptr_);
  if (ret != 0) {
    LOG(FATAL) << "aclrtHostRegister fail: " << ret;
  }
#endif

  size_t current_offset = 0;
  auto k_options = torch::TensorOptions()
                       .dtype(kv_caches_ptr_->at(0).get_k_cache().scalar_type())
                       .device(torch::kCPU);
  torch::ScalarType v_dtype =
      kv_caches_ptr_->at(0).get_v_cache().defined()
          ? kv_caches_ptr_->at(0).get_v_cache().scalar_type()
          : kv_caches_ptr_->at(0).get_k_cache().scalar_type();
  torch::ScalarType index_dtype =
      kv_caches_ptr_->at(0).get_index_cache().defined()
          ? kv_caches_ptr_->at(0).get_index_cache().scalar_type()
          : kv_caches_ptr_->at(0).get_k_cache().scalar_type();
  auto v_options = torch::TensorOptions().dtype(v_dtype).device(torch::kCPU);
  auto index_options =
      torch::TensorOptions().dtype(index_dtype).device(torch::kCPU);
  std::optional<torch::Tensor> first_index_cache_scale =
      kv_caches_ptr_->at(0).get_indexer_cache_scale();
  auto index_scale_options =
      torch::TensorOptions()
          .dtype(first_index_cache_scale.has_value()
                     ? first_index_cache_scale->scalar_type()
                     : kv_caches_ptr_->at(0).get_k_cache().scalar_type())
          .device(torch::kCPU);
  host_kv_caches_.reserve(num_blocks);

  for (size_t i = 0; i < num_blocks; ++i) {
    torch::Tensor key_cache, value_cache, index_cache;
    std::optional<torch::Tensor> index_cache_scale;
    void* k_tensor_ptr =
        static_cast<char*>(page_aligned_data_) + current_offset;
    key_cache = torch::from_blob(k_tensor_ptr, tensor_shapes[0], k_options);
    current_offset += cache_size_per_tensor[0];

    if (!tensor_shapes[1].empty()) {
      void* v_tensor_ptr =
          static_cast<char*>(page_aligned_data_) + current_offset;
      value_cache = torch::from_blob(v_tensor_ptr, tensor_shapes[1], v_options);
      current_offset += cache_size_per_tensor[1];
    }

    if (!tensor_shapes[2].empty()) {
      void* index_tensor_ptr =
          static_cast<char*>(page_aligned_data_) + current_offset;
      index_cache =
          torch::from_blob(index_tensor_ptr, tensor_shapes[2], index_options);
      current_offset += cache_size_per_tensor[2];
    }

    if (!tensor_shapes[3].empty()) {
      void* index_scale_tensor_ptr =
          static_cast<char*>(page_aligned_data_) + current_offset;
      index_cache_scale = torch::from_blob(
          index_scale_tensor_ptr, tensor_shapes[3], index_scale_options);
      current_offset += cache_size_per_tensor[3];
    }

    if (index_cache.defined()) {
      host_kv_caches_.emplace_back(
          IndexedKVCacheTensors{KVCacheTensors{key_cache, value_cache},
                                index_cache,
                                index_cache_scale});
      continue;
    }

    host_kv_caches_.emplace_back(KVCacheTensors{key_cache, value_cache});
  }

  LOG(INFO) << "host k cache shape: "
            << host_kv_caches_[0].get_k_cache().sizes();
  LOG(INFO) << "host v cache shape: "
            << host_kv_caches_[0].get_v_cache().sizes();
  LOG(INFO) << "host index cache shape: "
            << host_kv_caches_[0].get_index_cache().sizes();
  if (host_kv_caches_[0].get_indexer_cache_scale().has_value()) {
    LOG(INFO) << "host index cache scale shape: "
              << host_kv_caches_[0].get_indexer_cache_scale().value().sizes();
  }

  LOG(INFO) << "Host block init finish, total size: " << num_blocks;
}

}  // namespace xllm
