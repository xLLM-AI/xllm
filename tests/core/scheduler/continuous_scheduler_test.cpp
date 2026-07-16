#include "continuous_scheduler.h"

#include <absl/time/clock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <optional>

#include "chunked_prefill_scheduler.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/platform/platform.h"
#include "distributed_runtime/engine.h"
#include "prefill_only_scheduler.h"
#include "scheduler_factory.h"
#include "util/utils.h"

namespace xllm {

namespace {
class FakeTokenizer : public Tokenizer {
 public:
  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool add_special_tokens = true) const {
    NOT_IMPLEMENTED();
  }
  std::string decode(const Slice<int32_t>& ids,
                     bool skip_special_tokens) const {
    return "";
  }
  std::optional<int32_t> token_to_id(const std::string_view& token) const {
    NOT_IMPLEMENTED();
  }
  std::string id_to_token(int32_t id) const { NOT_IMPLEMENTED(); }
  size_t vocab_size() const { NOT_IMPLEMENTED(); }
  std::unique_ptr<Tokenizer> clone() const {
    return std::make_unique<FakeTokenizer>();
  }
};

class FakeEngine : public Engine {
 public:
  FakeEngine(int32_t num_blocks,
             int32_t block_size,
             bool enable_prefix_cache = false) {
    BlockManagerPool::Options opt;
    opt.num_blocks_ = num_blocks;
    opt.block_size_ = block_size;
    opt.enable_prefix_cache_ = enable_prefix_cache;
    fake_tokenizer_ = std::make_unique<FakeTokenizer>();
    fake_block_manager_ = std::make_unique<BlockManagerPool>(opt, 1);
  }
  ForwardOutput step(std::vector<Batch>& batch) { return {}; }
  void update_last_step_result(std::vector<Batch>& batch) { NOT_IMPLEMENTED(); }
  const Tokenizer* tokenizer() const { return fake_tokenizer_.get(); }
  BlockManagerPool* block_manager_pool() const {
    return fake_block_manager_.get();
  }
  const ModelArgs& model_args() const { NOT_IMPLEMENTED(); }
  const TokenizerArgs& tokenizer_args() const { NOT_IMPLEMENTED(); }
  std::vector<int64_t> get_active_activation_memory() const { return {0}; }
  bool init() override { return true; }

 private:
  std::unique_ptr<Tokenizer> fake_tokenizer_;
  std::unique_ptr<BlockManagerPool> fake_block_manager_;
};

class TestContinuousScheduler final : public ContinuousScheduler {
 public:
  TestContinuousScheduler(Engine* engine, const Options& options)
      : ContinuousScheduler(engine, options) {}

  void reject_stream(const std::shared_ptr<Request>& request) {
    response_processor_->process_stream_request(request);
    response_processor_->wait_completion();
  }

  void reject_streams(const std::vector<std::shared_ptr<Request>>& requests) {
    std::vector<std::shared_ptr<Request>> request_copy = requests;
    response_processor_->batch_process_stream_requests(request_copy);
    response_processor_->wait_completion();
  }
};

template <typename T>
class ScopedConfigValue final {
 public:
  ScopedConfigValue(T& value, T new_value) : value_(value), old_(value) {
    value_ = new_value;
  }

  ~ScopedConfigValue() { value_ = old_; }

 private:
  T& value_;
  T old_;
};

ContinuousScheduler::Options create_scheduler_options(
    int32_t max_tokens_per_batch,
    int32_t max_seqs_per_batch,
    int32_t num_speculative_tokens,
    int32_t max_tokens_per_chunk_for_prefill,
    int32_t dp_size,
    const std::string& priority_strategy = "fcfs",
    bool enable_profile_kv_blocks = true,
    bool enable_latency_aware_schedule = false,
    int32_t max_global_ttft_ms = std::numeric_limits<int32_t>::max(),
    int32_t max_global_tpot_ms = std::numeric_limits<int32_t>::max()) {
  ContinuousScheduler::Options opt;
  opt.num_speculative_tokens_ = num_speculative_tokens;
  opt.max_tokens_per_chunk_for_prefill_ = max_tokens_per_chunk_for_prefill;
  opt.max_tokens_per_batch_ = max_tokens_per_batch;
  opt.max_seqs_per_batch_ = max_seqs_per_batch;
  opt.dp_size_ = dp_size;
  opt.priority_strategy_ = priority_strategy;
  opt.enable_profile_kv_blocks_ = enable_profile_kv_blocks;
  opt.enable_latency_aware_schedule_ = enable_latency_aware_schedule;
  opt.max_global_ttft_ms_ = max_global_ttft_ms;
  opt.max_global_tpot_ms_ = max_global_tpot_ms;
  return opt;
}

std::vector<std::shared_ptr<Request>> generate_request(
    const std::vector<int32_t>& prompt_lens,
    const std::vector<int32_t>& max_tokens,
    std::optional<std::vector<bool>> offlines,
    std::optional<std::vector<int32_t>> priorities,
    std::optional<std::vector<int32_t>> ns,
    std::optional<std::vector<int32_t>> beam_widths,
    int32_t max_context_len) {
  std::vector<std::shared_ptr<Request>> requests;
  EXPECT_TRUE(prompt_lens.size() == max_tokens.size());

  size_t batch_size = prompt_lens.size();
  std::vector<bool> offline_vec;
  std::vector<int32_t> priority_vec;
  if (offlines.has_value()) {
    offline_vec = *offlines;
  } else {
    offline_vec = std::vector<bool>(batch_size, false);
  }

  if (priorities.has_value()) {
    priority_vec = priorities.value();
  } else {
    priority_vec = std::vector<int32_t>(
        batch_size, static_cast<int32_t>(RequestPriority::NORMAL));
  }

  std::vector<int32_t> n_vec;
  std::vector<int32_t> beam_width_vec;
  if (ns.has_value()) {
    n_vec = *ns;
  } else {
    n_vec = std::vector<int32_t>(batch_size, 1);
  }
  if (beam_widths.has_value()) {
    beam_width_vec = *beam_widths;
  } else {
    beam_width_vec = std::vector<int32_t>(batch_size, 0);
  }

  for (size_t i = 0; i < batch_size; ++i) {
    std::vector<int32_t> prompt_token_ids;
    prompt_token_ids.resize(prompt_lens[i]);
    RequestSamplingParam sampling_param;
    sampling_param.beam_width = beam_width_vec[i];
    SchedulerParam scheduler_param;
    scheduler_param.offline = offline_vec[i];
    scheduler_param.priority = static_cast<RequestPriority>(priority_vec[i]);

    StoppingChecker stopping_checker;
    stopping_checker.set_max_generated_tokens(max_tokens[i]);
    stopping_checker.set_max_context_len(max_context_len);
    stopping_checker.set_ignore_eos(true);

    RequestState req_state("x",
                           prompt_token_ids,
                           sampling_param,
                           scheduler_param,
                           stopping_checker,
                           prompt_lens[i] + 30000,
                           n_vec[i],
                           1,
                           false,
                           false,
                           false,
                           false,
                           false,
                           nullptr,
                           nullptr);

    auto request =
        std::make_shared<Request>("1", "1", "1", std::move(req_state), "1");
    requests.emplace_back(request);
  }

  return requests;
}

std::shared_ptr<Request> generate_request_with_prompt_tokens(
    const std::vector<int32_t>& prompt_token_ids,
    int32_t max_tokens,
    int32_t max_context_len) {
  RequestSamplingParam sampling_param;
  SchedulerParam scheduler_param;

  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(max_tokens);
  stopping_checker.set_max_context_len(max_context_len);
  stopping_checker.set_ignore_eos(true);

  RequestState req_state("x",
                         prompt_token_ids,
                         sampling_param,
                         scheduler_param,
                         stopping_checker,
                         prompt_token_ids.size() + 30000,
                         1,
                         1,
                         false,
                         false,
                         false,
                         false,
                         false,
                         nullptr,
                         nullptr);

  return std::make_shared<Request>("1", "1", "1", std::move(req_state), "1");
}

std::shared_ptr<Request> generate_request_with_best_of(
    const std::vector<int32_t>& prompt_token_ids,
    int32_t max_tokens,
    int32_t max_context_len,
    size_t n,
    size_t best_of) {
  RequestSamplingParam sampling_param;
  SchedulerParam scheduler_param;

  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(max_tokens);
  stopping_checker.set_max_context_len(max_context_len);
  stopping_checker.set_ignore_eos(true);

  RequestState req_state("x",
                         prompt_token_ids,
                         sampling_param,
                         scheduler_param,
                         stopping_checker,
                         prompt_token_ids.size() + 30000,
                         n,
                         best_of,
                         false,
                         false,
                         false,
                         false,
                         false,
                         nullptr,
                         nullptr);

  return std::make_shared<Request>("1", "1", "1", std::move(req_state), "1");
}

// dont not consider speculative decoding.
void update_requests(std::vector<std::shared_ptr<Request>> requests) {
  for (auto req : requests) {
    for (auto& seq : req->sequences()) {
      if (seq->kv_state().kv_cache_tokens_num() == 0) {
        seq->kv_state().incr_kv_cache_tokens_num(seq->num_prompt_tokens());
      } else {
        seq->kv_state().incr_kv_cache_tokens_num(1);
      }
      Token token(1);
      seq->append_token(token);
    }
  }
}

void make_request_decode_ready(const std::shared_ptr<Request>& request) {
  for (auto& seq : request->sequences()) {
    seq->kv_state().set_kv_cache_tokens_num(seq->num_prompt_tokens());
    Token token(1);
    seq->append_token(token);
  }
}

void set_chunk_kv(const std::shared_ptr<Request>& request, size_t kv_tokens) {
  for (auto& seq : request->sequences()) {
    seq->kv_state().set_kv_cache_tokens_num(kv_tokens);
  }
}

}  // namespace

TEST(ContinuousSchedulerFactoryTest,
     ChunkedPrefillWithoutSPUsesChunkedScheduler) {
  ContinuousScheduler::Options opt =
      create_scheduler_options(10000, 256, 0, 1024, 1);
  opt.enable_chunked_prefill() = true;

  auto engine = std::make_unique<FakeEngine>(32, 32);
  auto scheduler = create_continuous_scheduler(engine.get(), opt);

  EXPECT_NE(dynamic_cast<ChunkedPrefillScheduler*>(scheduler.get()), nullptr);
  EXPECT_EQ(dynamic_cast<PrefillOnlyScheduler*>(scheduler.get()), nullptr);
}

TEST(ContinuousSchedulerFactoryTest,
     ChunkedPrefillWithCpUsesBackendPartitionScheduler) {
  ContinuousScheduler::Options opt =
      create_scheduler_options(10000, 256, 0, 1024, 1);
  opt.enable_chunked_prefill() = true;
  opt.cp_size() = 2;

  auto engine = std::make_unique<FakeEngine>(32, 32);
  auto scheduler = create_continuous_scheduler(engine.get(), opt);

  if (Platform::uses_model_cp_partition()) {
    EXPECT_NE(dynamic_cast<PrefillOnlyScheduler*>(scheduler.get()), nullptr);
    EXPECT_EQ(dynamic_cast<ChunkedPrefillScheduler*>(scheduler.get()), nullptr);
  } else {
    EXPECT_NE(dynamic_cast<ChunkedPrefillScheduler*>(scheduler.get()), nullptr);
    EXPECT_EQ(dynamic_cast<PrefillOnlyScheduler*>(scheduler.get()), nullptr);
  }
}

TEST(ContinuousSchedulerFactoryTest,
     ChunkedPrefillWithCpAndSpeculativeUsesPrefillOnlyScheduler) {
  ContinuousScheduler::Options opt =
      create_scheduler_options(10000, 256, 4, 1024, 1);
  opt.enable_chunked_prefill() = true;
  opt.cp_size() = 2;

  auto engine = std::make_unique<FakeEngine>(32, 32);
  auto scheduler = create_continuous_scheduler(engine.get(), opt);

  EXPECT_NE(dynamic_cast<PrefillOnlyScheduler*>(scheduler.get()), nullptr);
  EXPECT_EQ(dynamic_cast<ChunkedPrefillScheduler*>(scheduler.get()), nullptr);
}

TEST(ContinuousSchedulerFactoryTest, ModelPartitionCpDoesNotBuildMixedBatch) {
  if (!Platform::uses_model_cp_partition()) {
    GTEST_SKIP() << "model-side CP partition is not enabled on this backend";
  }
  ContinuousScheduler::Options opt = create_scheduler_options(8, 8, 0, 4, 1);
  opt.enable_chunked_prefill() = true;
  opt.cp_size() = 2;

  auto engine = std::make_unique<FakeEngine>(32, 32);
  auto scheduler = create_continuous_scheduler(engine.get(), opt);
  auto* prefill_only = dynamic_cast<PrefillOnlyScheduler*>(scheduler.get());
  ASSERT_NE(prefill_only, nullptr);

  auto requests = generate_request({2, 10},
                                   {8, 8},
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   30000);
  for (auto& req : requests) {
    prefill_only->add_request(req);
  }

  auto batches = prefill_only->prepare_batch_test();
  ASSERT_EQ(batches.size(), 1);
  ASSERT_EQ(batches[0].size(), 2);
  const auto& allowed_max_tokens = batches[0].get_allowed_max_tokens();
  ASSERT_EQ(allowed_max_tokens.size(), 2);

  make_request_decode_ready(requests[0]);
  set_chunk_kv(requests[1], allowed_max_tokens[1]);

  batches = prefill_only->prepare_batch_test();
  ASSERT_EQ(batches.size(), 1);
  ASSERT_EQ(batches[0].size(), 1);

  const auto forward_input =
      batches[0].prepare_forward_input(1, 0, ModelArgs());
  EXPECT_TRUE(
      forward_input.input_params.meta.batch_forward_type.is_chunked_prefill());
  EXPECT_FALSE(forward_input.input_params.meta.batch_forward_type.is_mixed());
  EXPECT_EQ(forward_input.input_params.meta.num_sequences, 1);
  EXPECT_EQ(batches[0].get_allowed_max_tokens()[0],
            opt.max_tokens_per_chunk_for_prefill());
}

TEST(SchedulerFactoryTest, DisaggPDChunkedPrefillTakesPriorityOverCp) {
  ScopedConfigValue<bool> use_mix_scheduler(
      SchedulerConfig::get_instance().use_mix_scheduler(), false);
  ContinuousScheduler::Options opt =
      create_scheduler_options(10000, 256, 2, 1024, 1);
  opt.enable_disagg_pd() = true;
  opt.enable_pd_ooc() = false;
  opt.enable_chunked_prefill() = true;
  opt.cp_size() = 2;

  EXPECT_EQ(select_scheduler_kind(opt),
            SchedulerKind::DISAGG_PD_CHUNKED_PREFILL);
}

TEST(SchedulerFactoryTest, DisaggPDTakesPriorityOverCp) {
  ScopedConfigValue<bool> use_mix_scheduler(
      SchedulerConfig::get_instance().use_mix_scheduler(), false);
  ContinuousScheduler::Options opt =
      create_scheduler_options(10000, 256, 0, 1024, 1);
  opt.enable_disagg_pd() = true;
  opt.enable_pd_ooc() = false;
  opt.enable_chunked_prefill() = false;
  opt.cp_size() = 2;

  EXPECT_EQ(select_scheduler_kind(opt), SchedulerKind::DISAGG_PD);
}

TEST(SchedulerFactoryTest, DisaggPDOOCKeepsPDOOCKind) {
  ScopedConfigValue<bool> use_mix_scheduler(
      SchedulerConfig::get_instance().use_mix_scheduler(), false);
  ContinuousScheduler::Options opt =
      create_scheduler_options(10000, 256, 0, 1024, 1);
  opt.enable_disagg_pd() = true;
  opt.enable_pd_ooc() = true;
  opt.enable_chunked_prefill() = true;

  EXPECT_EQ(select_scheduler_kind(opt), SchedulerKind::PD_OOC);
}

// TEST-1:
// test preempt
TEST(ContinuousSchedulerTest, OnDecodePreemptOffDecode) {
  // set max free blocks: 9, support 9*32=288 tokens
  // actually only 8 free blocks , because default 1 block is for padding
  int block_num = 9;
  int block_size = 32;
  int max_tokens_per_chunk_for_prefill = 1024;
  // set chunked max_tokens budgets 10000 per step
  ContinuousScheduler::Options opt = create_scheduler_options(
      10000, 256, 0, max_tokens_per_chunk_for_prefill, 1);
  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
  BlockManagerPool* block_manager_pool = engine->block_manager_pool();
  EXPECT_TRUE(scheduler != nullptr);

  std::vector<std::shared_ptr<Request>> running_requests;

  // 1. schedule two new online prefill requests
  auto requests = generate_request({127, 127},
                                   {10, 10},
                                   std::vector<bool>{true, false},
                                   std::vector<int32_t>{2, 2},
                                   std::nullopt,
                                   std::nullopt,
                                   30000);
  running_requests = requests;
  for (auto req : requests) {
    scheduler->add_request(req);
  }
  auto batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 2);
  update_requests(running_requests);

  batch = scheduler->prepare_batch_test();

  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 2);
  update_requests(running_requests);

  int free_blocks_before_preempt =
      util::max(block_manager_pool->num_free_blocks());
  batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 1);
  int free_blocks_after_preempt =
      util::max(block_manager_pool->num_free_blocks());
  EXPECT_TRUE(free_blocks_after_preempt > free_blocks_before_preempt);

  // check the running request is online request
  EXPECT_TRUE(scheduler->get_running_requests().size() == 1);
  EXPECT_TRUE(scheduler->get_running_requests()[0]->offline() == false);
  EXPECT_TRUE(scheduler->get_waiting_requests_num() == 1);
}

// TEST-2:
// test preempt
TEST(ContinuousSchedulerTest, OnPrefillPreemptOffDecode) {
  // set max free blocks: 9, support 9*32=288 tokens
  // actually only 8 free blocks , because default 1 block is for padding
  int block_num = 9;
  int block_size = 32;
  int max_tokens_per_chunk_for_prefill = 1024;
  // set chunked max_tokens budgets 10000 per step
  ContinuousScheduler::Options opt = create_scheduler_options(
      10000, 256, 0, max_tokens_per_chunk_for_prefill, 1);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      2.0);

  {
    // 1. two offline decode requests then one online prefill request
    // preempt them
    auto engine = std::make_unique<FakeEngine>(block_num, block_size);
    auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
    BlockManagerPool* block_manager_pool = engine->block_manager_pool();
    EXPECT_TRUE(scheduler != nullptr);

    std::vector<std::shared_ptr<Request>> running_requests;

    auto requests = generate_request({100, 100},
                                     {10, 10},
                                     std::vector<bool>{true, true},
                                     std::vector<int32_t>{2, 2},
                                     std::nullopt,
                                     std::nullopt,
                                     30000);
    running_requests = requests;
    for (auto req : requests) {
      scheduler->add_request(req);
    }
    auto batch = scheduler->prepare_batch_test();
    EXPECT_TRUE(batch.size() == 1);
    EXPECT_TRUE(batch[0].size() == 2);
    EXPECT_TRUE(util::max(block_manager_pool->num_free_blocks()) == 0);
    update_requests(running_requests);

    batch = scheduler->prepare_batch_test();
    EXPECT_TRUE(batch.size() == 1);
    EXPECT_TRUE(batch[0].size() == 2);
    EXPECT_TRUE(util::max(block_manager_pool->num_free_blocks()) == 0);
    update_requests(running_requests);

    auto new_requests = generate_request({80},
                                         {10},
                                         std::vector<bool>{false},
                                         std::vector<int32_t>{2},
                                         std::nullopt,
                                         std::nullopt,
                                         30000);  // use 3 blocks
    scheduler->add_request(new_requests[0]);
    batch = scheduler->prepare_batch_test();
    EXPECT_TRUE(batch.size() == 1);
    EXPECT_TRUE(batch[0].size() == 1);

    // online prefill request preempt offline decode request
    EXPECT_TRUE(scheduler->get_running_requests().size() == 1);
    EXPECT_TRUE(scheduler->get_running_requests()[0]->offline() == false);
    EXPECT_TRUE(scheduler->get_waiting_requests_num() == 1);

    // offline is evicted
    EXPECT_TRUE(util::max(block_manager_pool->num_free_blocks()) == 1);
  }

  // 2. another case: longer online prefill request arrives, but can not
  // evict offline because evicting offline is not enough
  {
    auto engine = std::make_unique<FakeEngine>(block_num, block_size);
    auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
    BlockManagerPool* block_manager_pool = engine->block_manager_pool();
    EXPECT_TRUE(scheduler != nullptr);

    std::vector<std::shared_ptr<Request>> running_requests;
    // one online, one offline
    auto requests = generate_request({100, 100},
                                     {10, 10},
                                     std::vector<bool>{true, false},
                                     std::vector<int32_t>{2, 2},
                                     std::nullopt,
                                     std::nullopt,
                                     30000);
    running_requests = requests;
    for (auto req : requests) {
      scheduler->add_request(req);
    }
    auto batch = scheduler->prepare_batch_test();
    EXPECT_TRUE(batch.size() == 1);
    EXPECT_TRUE(batch[0].size() == 2);
    EXPECT_TRUE(util::max(block_manager_pool->num_free_blocks()) == 0);
    update_requests(running_requests);

    auto new_requests = generate_request({200},
                                         {10},
                                         std::vector<bool>{false},
                                         std::vector<int32_t>{2},
                                         std::nullopt,
                                         std::nullopt,
                                         30000);
    scheduler->add_request(new_requests[0]);
    batch = scheduler->prepare_batch_test();
    // online is still waiting
    EXPECT_TRUE(batch.size() == 1);
    EXPECT_TRUE(batch[0].size() == 2);
    EXPECT_TRUE(scheduler->get_waiting_requests().size() == 1);
    EXPECT_TRUE(scheduler->get_waiting_requests()[0].get() ==
                new_requests[0].get());
  }
}

// TEST-3:
// test priority schedule
TEST(ContinuousSchedulerTest, PrioritySchedule) {
  // set max free blocks: 12
  // actually only 11 free blocks , because default 1 block is for padding
  int block_num = 12;
  int block_size = 32;
  int max_tokens_per_chunk_for_prefill = 1024;
  // set chunked max_tokens budgets 10000 per step
  ContinuousScheduler::Options opt = create_scheduler_options(
      10000, 256, 0, max_tokens_per_chunk_for_prefill, 1, "priority");
  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
  EXPECT_TRUE(scheduler != nullptr);

  std::vector<std::shared_ptr<Request>> running_requests;

  // 1: HIGH, 2: NORMAL, 3: LOW
  auto requests = generate_request({128, 128, 128},
                                   {10, 10, 10},
                                   std::vector<bool>{false, false, false},
                                   std::vector<int32_t>{3, 3, 2},
                                   std::nullopt,
                                   std::nullopt,
                                   30000);
  for (auto req : requests) {
    scheduler->add_request(req);
  }
  auto batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 2);
  EXPECT_TRUE(scheduler->get_running_requests().size() == 2);
  EXPECT_TRUE(scheduler->get_running_requests()[0]->priority() ==
              RequestPriority::NORMAL /*NORMAL*/);
  EXPECT_TRUE(scheduler->get_running_requests()[1]->priority() ==
              RequestPriority::LOW /*LOW*/);
  running_requests = scheduler->get_running_requests();
  update_requests(running_requests);

  // new HIGH priority request arrives, its prefill starts
  auto new_requests = generate_request({32},
                                       {10},
                                       std::vector<bool>{false},
                                       std::vector<int32_t>{1},
                                       std::nullopt,
                                       std::nullopt,
                                       30000);  // use 1 blocks
  scheduler->add_request(new_requests[0]);
  batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 1);
  EXPECT_TRUE(scheduler->get_running_requests().size() == 1);
  update_requests(new_requests);

  // only HIGH and NORMAL requests decode
  batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 2);
  EXPECT_TRUE(scheduler->get_running_requests().size() == 2);
  EXPECT_TRUE(scheduler->get_running_requests()[0]->priority() ==
              RequestPriority::HIGH /*HIGH*/);
  EXPECT_TRUE(scheduler->get_running_requests()[1]->priority() ==
              RequestPriority::NORMAL /*NORMAL*/);
}

// TEST-4:
// beam strict mode should not partially schedule one request.
TEST(ContinuousSchedulerTest, BeamStrictNoPartialScheduling) {
  ContinuousScheduler::Options opt =
      create_scheduler_options(2, 8, 0, 1024, 1, "fcfs");
  auto engine = std::make_unique<FakeEngine>(64, 32);
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
  EXPECT_TRUE(scheduler != nullptr);

  auto req1 = generate_request({64},
                               {10},
                               std::nullopt,
                               std::nullopt,
                               std::vector<int32_t>{1},
                               std::vector<int32_t>{0},
                               30000)[0];
  make_request_decode_ready(req1);

  auto beam_req = generate_request({64},
                                   {10},
                                   std::nullopt,
                                   std::nullopt,
                                   std::vector<int32_t>{1},
                                   std::vector<int32_t>{2},
                                   30000)[0];
  // Manually duplicate beam sequence to simulate a decoded beam group.
  beam_req->sequences().emplace_back(
      std::make_unique<Sequence>(*beam_req->sequences()[0]));
  ASSERT_EQ(beam_req->sequences().size(), 2u);
  make_request_decode_ready(beam_req);

  scheduler->add_request(req1);
  scheduler->add_request(beam_req);
  auto batch = scheduler->prepare_batch_test();
  ASSERT_EQ(batch.size(), 1u);
  EXPECT_EQ(batch[0].size(), 1u);
  EXPECT_EQ(batch[0][0], req1->sequences()[0].get());
  EXPECT_EQ(scheduler->get_running_requests().size(), 1u);
  EXPECT_EQ(scheduler->get_waiting_requests_num(), 0u);
  EXPECT_NE(batch[0][0], beam_req->sequences()[0].get());
  EXPECT_NE(batch[0][0], beam_req->sequences()[1].get());
}

// TEST-5:
// test latency budget
TEST(ContinuousSchedulerTest, LatencySchedule) {
  // block is enough
  int block_num = 12;
  int block_size = 32;
  int max_tokens_per_chunk_for_prefill = 1024;
  // set chunked max_tokens budgets 10000 per step
  ContinuousScheduler::Options opt =
      create_scheduler_options(10000,
                               256,
                               0,
                               max_tokens_per_chunk_for_prefill,
                               1,
                               "fcfs",
                               false,
                               true,
                               350,
                               25);
  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
  EXPECT_TRUE(scheduler != nullptr);

  // mannuly created profile data for y=0.5x^2+10x
  std::vector<std::pair<int32_t, double>> created_profile_data = {
      {2, 22}, {4, 48}, {6, 78}, {8, 112}};
  auto profile_manager = scheduler->get_profile_manager();
  // fit y=0.5x^2+10x
  profile_manager->train_prefill_time_predictor(created_profile_data);

  auto requests = generate_request({10, 10, 10},
                                   {10, 10, 10},
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   30000);
  // check if time equation fits well
  EXPECT_TRUE(
      static_cast<int32_t>(std::round(profile_manager->predict_step_time(
          requests[0]->sequences()[0].get(), true, true))) == 150);
  EXPECT_TRUE(static_cast<int32_t>(std::round(
                  profile_manager->predict_step_time(2, 0, true, true))) == 22);

  std::vector<std::shared_ptr<Request>> running_requests;

  // 1. two requests enter prefill
  for (auto req : requests) {
    scheduler->add_request(req);
  }
  auto batch = scheduler->prepare_batch_test();

  EXPECT_EQ(batch.size(), 1);
  // 2*150 < ttft_slo=350 < 3 * 150, only two requests enter prefill
  EXPECT_EQ(batch[0].size(), 2);
  EXPECT_EQ(scheduler->get_running_requests().size(), 2);
  running_requests = scheduler->get_running_requests();
  update_requests(running_requests);

  // 2. one request enter prefill
  batch = scheduler->prepare_batch_test();
  EXPECT_EQ(batch.size(), 1);
  EXPECT_EQ(batch[0].size(), 1);
  EXPECT_EQ(scheduler->get_running_requests().size(), 1);
  running_requests = scheduler->get_running_requests();
  update_requests(running_requests);

  // 3. two requests start decode
  // batch = scheduler->prepare_batch_test();
  // EXPECT_TRUE(batch.size() == 1);
  // // 2*10 < tpot_slo=25 < 3 * 10, only two requests enter decode
  // EXPECT_TRUE(batch[0].size() == 2);
  // EXPECT_TRUE(scheduler->get_running_requests().size() == 2);
}

TEST(BlockManagerPoolTest, AllocateFailureRollsBackSharedPrefixBlocks) {
  auto engine = std::make_unique<FakeEngine>(3, 4, true);
  BlockManagerPool* block_manager_pool = engine->block_manager_pool();

  auto cached_request =
      generate_request_with_prompt_tokens({1, 2, 3, 4, 5, 6, 7, 8}, 1, 30000);
  auto failed_request = generate_request_with_prompt_tokens(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 1, 30000);
  auto later_request =
      generate_request_with_prompt_tokens({20, 21, 22, 23}, 1, 30000);

  auto* cached_sequence = cached_request->sequences()[0].get();
  ASSERT_TRUE(block_manager_pool->allocate(cached_sequence,
                                           cached_sequence->num_tokens()));
  cached_sequence->kv_state().set_kv_cache_tokens_num(
      cached_sequence->num_tokens());
  block_manager_pool->deallocate(cached_sequence);

  const size_t free_blocks_before_failure =
      util::max(block_manager_pool->num_free_blocks());
  const size_t used_blocks_before_failure =
      util::min(block_manager_pool->num_used_blocks());
  EXPECT_EQ(free_blocks_before_failure, 0);

  auto* failed_sequence = failed_request->sequences()[0].get();
  EXPECT_FALSE(block_manager_pool->allocate(failed_sequence,
                                            failed_sequence->num_tokens()));
  EXPECT_EQ(failed_sequence->kv_state().num_blocks(BlockType::KV), 0);
  EXPECT_EQ(failed_sequence->kv_state().shared_blocks_num(BlockType::KV), 0);
  EXPECT_EQ(util::max(block_manager_pool->num_free_blocks()),
            free_blocks_before_failure);
  EXPECT_EQ(util::min(block_manager_pool->num_used_blocks()),
            used_blocks_before_failure);

  auto* later_sequence = later_request->sequences()[0].get();
  EXPECT_TRUE(block_manager_pool->allocate(later_sequence,
                                           later_sequence->num_tokens()));
  EXPECT_EQ(later_sequence->kv_state().num_blocks(BlockType::KV), 1);

  (void)engine.release();
}

TEST(ContinuousSchedulerTest,
     PDDecodeBestOfNExpandsAndSharesPromptViaPrefixCache) {
  // Disagg PD decode instance flow:
  //   1. request arrives from the prefill instance with kv_cache_tokens_num
  //      already advanced (try_allocate + append_token(first_token)).
  //   2. ContinuousScheduler::prepare_batch must short-circuit the waiting
  //      queue and push directly into running_requests_.
  //   3. handle_running_requests must trigger expand_sequences(true) and
  //      cache(seq[0]) so the expanded seq[1..best_of-1] can hit the
  //      shared prompt blocks via prefix cache.
  ContinuousScheduler::Options opt =
      create_scheduler_options(1024, 16, 0, 1024, 1);
  auto engine =
      std::make_unique<FakeEngine>(32, 4, /*enable_prefix_cache=*/true);
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
  BlockManagerPool* block_manager_pool = engine->block_manager_pool();

  constexpr size_t kBestOf = 4;
  constexpr size_t kN = 2;
  auto request = generate_request_with_best_of({1, 2, 3, 4, 5, 6, 7, 8},
                                               /*max_tokens=*/4,
                                               /*max_context_len=*/30000,
                                               kN,
                                               kBestOf);

  Sequence* seq0 = request->sequences()[0].get();
  ASSERT_TRUE(block_manager_pool->try_allocate(seq0));
  ASSERT_EQ(seq0->kv_state().kv_cache_tokens_num(), seq0->num_prompt_tokens());
  Token first(42);
  seq0->append_token(first);

  scheduler->add_request(request);

  auto batch = scheduler->prepare_batch_test();
  EXPECT_EQ(batch.size(), 1);
  EXPECT_EQ(request->sequences().size(), kBestOf);
  EXPECT_EQ(batch[0].size(), kBestOf);

  for (size_t i = 1; i < kBestOf; ++i) {
    EXPECT_GT(
        request->sequences()[i]->kv_state().shared_blocks_num(BlockType::KV),
        0u)
        << "expanded seq " << i
        << " should reuse seq[0] prompt blocks via prefix cache";
  }

  scheduler.reset();
  (void)engine.release();
}

TEST(ContinuousSchedulerTest, PDDecodeBestOfOneSkipsExpansionAndShares) {
  // Sanity check: best_of==n==1 should not trigger any expansion, and the
  // request should still flow through the PD-decode short-circuit.
  auto request = generate_request_with_best_of({1, 2, 3, 4, 5, 6, 7, 8},
                                               /*max_tokens=*/4,
                                               /*max_context_len=*/30000,
                                               /*n=*/1,
                                               /*best_of=*/1);
  Sequence* seq0 = request->sequences()[0].get();

  ContinuousScheduler::Options opt =
      create_scheduler_options(1024, 16, 0, 1024, 1);
  // Prefix cache is not under test here; disabling it avoids teardown putting
  // blocks into the prefix-cache table instead of the free list.
  auto engine =
      std::make_unique<FakeEngine>(32, 4, /*enable_prefix_cache=*/false);
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
  BlockManagerPool* block_manager_pool = engine->block_manager_pool();

  ASSERT_TRUE(block_manager_pool->try_allocate(seq0));
  ASSERT_EQ(seq0->kv_state().kv_cache_tokens_num(), seq0->num_prompt_tokens());
  Token first(42);
  seq0->append_token(first);

  scheduler->add_request(request);

  auto batch = scheduler->prepare_batch_test();
  EXPECT_EQ(batch.size(), 1);
  EXPECT_EQ(request->sequences().size(), 1u);
  EXPECT_EQ(batch[0].size(), 1u);

  // prepare_batch may allocate extra decode blocks; release them before engine
  // teardown (BlockManagerImpl checks all blocks are on the free list).
  block_manager_pool->deallocate_without_cache(seq0);
  (void)engine.release();
}

TEST(ContinuousSchedulerTest, RejectedStreamCancelsAtSchedulingBoundary) {
  ContinuousScheduler::Options opt =
      create_scheduler_options(1024, 16, 0, 1024, 1);
  opt.enable_schedule_overlap() = false;
  auto engine = std::make_unique<FakeEngine>(32, 4);
  auto scheduler = std::make_unique<TestContinuousScheduler>(engine.get(), opt);
  BlockManagerPool* block_manager_pool = engine->block_manager_pool();
  const size_t initial_free_blocks =
      util::max(block_manager_pool->num_free_blocks());

  auto request = generate_request_with_prompt_tokens({1, 2, 3, 4}, 4, 30000);
  request->state().stream = true;
  request->state().output_func = [](const RequestOutput&) { return false; };
  make_request_decode_ready(request);
  scheduler->add_request(request);

  std::vector<Batch> batch = scheduler->prepare_batch_test();
  ASSERT_EQ(batch.size(), 1u);
  ASSERT_EQ(batch[0].size(), 1u);
  EXPECT_LT(util::max(block_manager_pool->num_free_blocks()),
            initial_free_blocks);

  scheduler->reject_stream(request);

  EXPECT_FALSE(request->cancelled());

  scheduler->step(absl::ZeroDuration());

  EXPECT_TRUE(request->cancelled());
  EXPECT_TRUE(scheduler->get_running_requests().empty());
  EXPECT_EQ(util::max(block_manager_pool->num_free_blocks()),
            initial_free_blocks);

  scheduler->reject_stream(request);
  scheduler->step(absl::ZeroDuration());

  EXPECT_TRUE(request->cancelled());
  EXPECT_TRUE(scheduler->get_running_requests().empty());
  EXPECT_EQ(util::max(block_manager_pool->num_free_blocks()),
            initial_free_blocks);
}

TEST(ContinuousSchedulerTest, BatchRejectedStreamsCancelAtSchedulingBoundary) {
  ContinuousScheduler::Options opt =
      create_scheduler_options(1024, 16, 0, 1024, 1);
  opt.enable_schedule_overlap() = false;
  auto engine = std::make_unique<FakeEngine>(32, 4);
  auto scheduler = std::make_unique<TestContinuousScheduler>(engine.get(), opt);
  BlockManagerPool* block_manager_pool = engine->block_manager_pool();

  std::vector<std::shared_ptr<Request>> requests =
      generate_request({4, 4},
                       {4, 4},
                       std::nullopt,
                       std::nullopt,
                       std::nullopt,
                       std::nullopt,
                       30000);
  for (std::shared_ptr<Request>& request : requests) {
    request->state().stream = true;
    request->state().output_func = [](const RequestOutput&) { return true; };
    make_request_decode_ready(request);
    scheduler->add_request(request);
  }
  requests[0]->state().outputs_func = [](const std::vector<RequestOutput>&) {
    return std::vector<bool>{false, true};
  };

  std::vector<Batch> batch = scheduler->prepare_batch_test();
  ASSERT_EQ(batch.size(), 1u);
  ASSERT_EQ(batch[0].size(), 2u);
  const size_t free_blocks_before_cancel =
      util::max(block_manager_pool->num_free_blocks());

  scheduler->reject_streams(requests);
  scheduler->reject_streams(requests);

  EXPECT_FALSE(requests[0]->cancelled());
  EXPECT_FALSE(requests[1]->cancelled());

  scheduler->step(absl::ZeroDuration());

  EXPECT_TRUE(requests[0]->cancelled());
  EXPECT_FALSE(requests[1]->cancelled());
  ASSERT_EQ(scheduler->get_running_requests().size(), 1u);
  EXPECT_EQ(scheduler->get_running_requests()[0], requests[1]);
  EXPECT_GT(util::max(block_manager_pool->num_free_blocks()),
            free_blocks_before_cancel);

  requests[1]->state().outputs_func = [](const std::vector<RequestOutput>&) {
    return std::vector<bool>{false};
  };
  scheduler->reject_streams({requests[1]});
  EXPECT_FALSE(requests[1]->cancelled());
  scheduler->step(absl::ZeroDuration());
  EXPECT_TRUE(requests[1]->cancelled());
  EXPECT_TRUE(scheduler->get_running_requests().empty());
}

TEST(ContinuousSchedulerTest,
     BatchStreamMarksPrefillFinishedWhenFirstTokenDecodesEmpty) {
  ContinuousScheduler::Options opt =
      create_scheduler_options(1024, 16, 0, 1024, 1);
  auto engine = std::make_unique<FakeEngine>(32, 4);
  auto scheduler = std::make_unique<TestContinuousScheduler>(engine.get(), opt);

  auto request = generate_request_with_prompt_tokens({1, 2, 3, 4}, 4, 30000);
  request->state().stream = true;
  make_request_decode_ready(request);

  bool callback_called = false;
  bool outputs_empty = false;
  bool finished_on_prefill_instance = false;
  request->state().outputs_func =
      [&](const std::vector<RequestOutput>& outputs) {
        callback_called = true;
        outputs_empty = outputs.size() == 1 && outputs[0].outputs.empty();
        finished_on_prefill_instance =
            outputs.size() == 1 && outputs[0].finished_on_prefill_instance;
        return std::vector<bool>{true};
      };

  scheduler->reject_streams({request});

  EXPECT_TRUE(callback_called);
  EXPECT_TRUE(outputs_empty);
  EXPECT_TRUE(finished_on_prefill_instance);
}
// ============== Async RL training: Pause/Resume tests ==============

// TEST: pause()/resume() state transitions are correct and idempotent.
TEST(ContinuousSchedulerTest, PauseResumeStateTransition) {
  int block_num = 9;
  int block_size = 32;
  ContinuousScheduler::Options opt =
      create_scheduler_options(10000, 256, 0, 1024, 1);
  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);

  EXPECT_FALSE(scheduler->is_paused());
  scheduler->pause();
  EXPECT_TRUE(scheduler->is_paused());
  scheduler->pause();  // idempotent
  EXPECT_TRUE(scheduler->is_paused());
  scheduler->resume();
  EXPECT_FALSE(scheduler->is_paused());
  scheduler->resume();  // idempotent
  EXPECT_FALSE(scheduler->is_paused());

  (void)engine.release();
}

// TEST: pause preempts all running requests, frees KV cache, moves them back
// to the waiting queue (vLLM-compatible semantics for RL).
TEST(ContinuousSchedulerTest, PausePreemptsRunningAndFreesKVCache) {
  int block_num = 33;
  int block_size = 32;
  ContinuousScheduler::Options opt =
      create_scheduler_options(10000, 256, 0, 1024, 1);
  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
  BlockManagerPool* block_manager_pool = engine->block_manager_pool();
  ASSERT_TRUE(scheduler != nullptr);

  auto requests = generate_request({127, 127},
                                   {10, 10},
                                   std::vector<bool>{false, false},
                                   std::vector<int32_t>{2, 2},
                                   std::nullopt,
                                   std::nullopt,
                                   30000);
  std::vector<std::shared_ptr<Request>> running_requests = requests;
  for (auto req : requests) {
    scheduler->add_request(req);
  }
  auto batch = scheduler->prepare_batch_test();
  EXPECT_EQ(batch.size(), 1);
  EXPECT_EQ(batch[0].size(), 2);
  update_requests(running_requests);

  EXPECT_EQ(scheduler->get_running_requests().size(), 2);
  int free_blocks_before = util::max(block_manager_pool->num_free_blocks());

  scheduler->pause();
  scheduler->preempt_all_running_requests_test();

  int free_blocks_after = util::max(block_manager_pool->num_free_blocks());
  EXPECT_GT(free_blocks_after, free_blocks_before);
  EXPECT_EQ(scheduler->get_running_requests().size(), 0);
  EXPECT_EQ(scheduler->get_waiting_requests_num(), 2);

  (void)engine.release();
}

// TEST: after resume, preempted requests in the waiting queue get re-scheduled.
TEST(ContinuousSchedulerTest, ResumeReschedulesPreemptedRequests) {
  int block_num = 33;
  int block_size = 32;
  ContinuousScheduler::Options opt =
      create_scheduler_options(10000, 256, 0, 1024, 1);
  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
  ASSERT_TRUE(scheduler != nullptr);

  auto requests = generate_request({127, 127},
                                   {10, 10},
                                   std::vector<bool>{false, false},
                                   std::vector<int32_t>{2, 2},
                                   std::nullopt,
                                   std::nullopt,
                                   30000);
  std::vector<std::shared_ptr<Request>> running_requests = requests;
  for (auto req : requests) {
    scheduler->add_request(req);
  }
  (void)scheduler->prepare_batch_test();
  update_requests(running_requests);
  EXPECT_EQ(scheduler->get_running_requests().size(), 2);

  scheduler->pause();
  scheduler->preempt_all_running_requests_test();
  EXPECT_EQ(scheduler->get_running_requests().size(), 0);
  EXPECT_EQ(scheduler->get_waiting_requests_num(), 2);

  scheduler->resume();
  EXPECT_FALSE(scheduler->is_paused());

  auto batch = scheduler->prepare_batch_test();
  EXPECT_EQ(batch.size(), 1);
  EXPECT_EQ(batch[0].size(), 2);
  EXPECT_EQ(scheduler->get_running_requests().size(), 2);
  EXPECT_EQ(scheduler->get_waiting_requests_num(), 0);

  (void)engine.release();
}

// TEST: ABORT mode cancels running requests, frees KV cache, and does NOT
// push them back to the waiting queue (clients must retry).
TEST(ContinuousSchedulerTest, PauseAbortCancelsRunningRequests) {
  int block_num = 33;
  int block_size = 32;
  ContinuousScheduler::Options opt =
      create_scheduler_options(10000, 256, 0, 1024, 1);
  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
  BlockManagerPool* block_manager_pool = engine->block_manager_pool();
  ASSERT_TRUE(scheduler != nullptr);

  auto requests = generate_request({127, 127},
                                   {10, 10},
                                   std::vector<bool>{false, false},
                                   std::vector<int32_t>{2, 2},
                                   std::nullopt,
                                   std::nullopt,
                                   30000);
  std::vector<std::shared_ptr<Request>> running_requests = requests;
  for (auto req : requests) {
    // process_failed_request invokes the requests output callback; give it a
    // no-op so ABORT can notify completion without a real client.
    req->state().output_func = [](const RequestOutput&) { return true; };
    scheduler->add_request(req);
  }
  auto batch = scheduler->prepare_batch_test();
  EXPECT_EQ(batch.size(), 1);
  EXPECT_EQ(batch[0].size(), 2);
  update_requests(running_requests);

  EXPECT_EQ(scheduler->get_running_requests().size(), 2);
  int free_blocks_before = util::max(block_manager_pool->num_free_blocks());

  scheduler->abort_all_running_requests_test();

  int free_blocks_after = util::max(block_manager_pool->num_free_blocks());
  EXPECT_GT(free_blocks_after, free_blocks_before);        // KV cache freed
  EXPECT_EQ(scheduler->get_running_requests().size(), 0);  // running cleared
  EXPECT_EQ(scheduler->get_waiting_requests_num(), 0);     // NOT requeued

  (void)engine.release();
}

// With in-batch prefix cache enabled, a request admitted earlier in the same
// scheduling step publishes its full prompt blocks, so a later request with the
// same prefix reuses them (shared_kv_blocks_num > 0). When disabled, the later
// request shares nothing within the same batch.
TEST(ContinuousSchedulerTest,
     InBatchCachePrefillBlocksIncreaseSharedBlocksForLaterRequests) {
  const auto run_with_in_batch_prefix_cache =
      [](bool enable_in_batch_prefix_cache) -> size_t {
    KVCacheConfig& kv_config = KVCacheConfig::get_instance();
    const bool saved_prefix_cache = kv_config.enable_prefix_cache();
    const bool saved_in_batch = kv_config.enable_in_batch_prefix_cache();
    kv_config.enable_prefix_cache(true);
    kv_config.enable_in_batch_prefix_cache(enable_in_batch_prefix_cache);

    ContinuousScheduler::Options opt =
        create_scheduler_options(1024, 16, 0, 1024, 1);
    auto engine =
        std::make_unique<FakeEngine>(32, 4, /*enable_prefix_cache=*/true);
    auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);

    auto first_request =
        generate_request_with_prompt_tokens({1, 2, 3, 4, 5, 6, 7, 8}, 1, 30000);
    auto second_request =
        generate_request_with_prompt_tokens({1, 2, 3, 4, 5, 6, 7, 8}, 1, 30000);
    scheduler->add_request(first_request);
    scheduler->add_request(second_request);

    auto batch = scheduler->prepare_batch_test();
    EXPECT_EQ(batch.size(), 1);
    EXPECT_EQ(batch[0].size(), 2);
    EXPECT_EQ(first_request->sequences()[0]->kv_state().shared_blocks_num(
                  BlockType::KV),
              0u);
    const size_t second_shared =
        second_request->sequences()[0]->kv_state().shared_blocks_num(
            BlockType::KV);

    scheduler.reset();
    // Leak the engine: cached blocks live in the prefix-cache table at
    // teardown.
    (void)engine.release();

    kv_config.enable_prefix_cache(saved_prefix_cache);
    kv_config.enable_in_batch_prefix_cache(saved_in_batch);
    return second_shared;
  };

  const size_t shared_when_enabled = run_with_in_batch_prefix_cache(true);
  const size_t shared_when_disabled = run_with_in_batch_prefix_cache(false);

  EXPECT_GT(shared_when_enabled, shared_when_disabled);
  EXPECT_GT(shared_when_enabled, 0u);
  EXPECT_EQ(shared_when_disabled, 0u);
}

// End-to-end check through the real scheduler path (add_request ->
// prepare_batch): two requests that share only the first 2 of 3 prompt blocks
// are admitted in the same step. With in-batch prefix cache on, the first
// request publishes its full blocks, so the second request reuses EXACTLY the 2
// shared blocks (the 3rd differs and must be a miss). The prefix-cache table
// also grows by the published blocks. With it off, nothing is shared in-batch.
TEST(ContinuousSchedulerTest, InBatchCacheReusesPartialPrefixWithinSameBatch) {
  // block_size = 4. 12 tokens => 3 full blocks per prompt.
  const std::vector<int32_t> prompt_a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  // Shares blocks [0,1] (tokens 0..7) with prompt_a, differs in block 2.
  const std::vector<int32_t> prompt_b = {
      1, 2, 3, 4, 5, 6, 7, 8, 99, 100, 101, 102};

  struct Result {
    size_t first_shared = 0;
    size_t second_shared = 0;
    size_t batch_count = 0;
    size_t batch0_size = 0;
    size_t prefix_cache_blocks = 0;
  };

  const auto run = [&](bool enable_in_batch_prefix_cache) -> Result {
    KVCacheConfig& kv_config = KVCacheConfig::get_instance();
    const bool saved_prefix_cache = kv_config.enable_prefix_cache();
    const bool saved_in_batch = kv_config.enable_in_batch_prefix_cache();
    kv_config.enable_prefix_cache(true);
    kv_config.enable_in_batch_prefix_cache(enable_in_batch_prefix_cache);

    ContinuousScheduler::Options opt =
        create_scheduler_options(1024, 16, 0, 1024, 1);
    auto engine =
        std::make_unique<FakeEngine>(64, 4, /*enable_prefix_cache=*/true);
    auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);

    auto first_request =
        generate_request_with_prompt_tokens(prompt_a, 1, 30000);
    auto second_request =
        generate_request_with_prompt_tokens(prompt_b, 1, 30000);
    scheduler->add_request(first_request);
    scheduler->add_request(second_request);

    auto batch = scheduler->prepare_batch_test();

    Result r;
    r.batch_count = batch.size();
    r.batch0_size = batch.empty() ? 0 : batch[0].size();
    r.first_shared =
        first_request->sequences()[0]->kv_state().shared_blocks_num(
            BlockType::KV);
    r.second_shared =
        second_request->sequences()[0]->kv_state().shared_blocks_num(
            BlockType::KV);
    r.prefix_cache_blocks =
        engine->block_manager_pool()->num_blocks_in_prefix_cache()[0];

    scheduler.reset();
    // Leak the engine: published blocks live in the prefix-cache table at
    // teardown, which would otherwise trip the free-list check in dtor.
    (void)engine.release();

    kv_config.enable_prefix_cache(saved_prefix_cache);
    kv_config.enable_in_batch_prefix_cache(saved_in_batch);
    return r;
  };

  const Result enabled = run(true);
  const Result disabled = run(false);

  // Both requests must land in the same batch for in-batch reuse to apply.
  EXPECT_EQ(enabled.batch_count, 1u);
  EXPECT_EQ(enabled.batch0_size, 2u);

  // The first request can never share within the same step (nothing published
  // yet when it is admitted).
  EXPECT_EQ(enabled.first_shared, 0u);

  // The second request reuses exactly the 2 identical leading blocks.
  EXPECT_EQ(enabled.second_shared, 2u);

  // Prefix cache holds first request's 3 blocks plus the second request's one
  // distinct trailing block => 4 unique blocks.
  EXPECT_EQ(enabled.prefix_cache_blocks, 4u);

  // With the feature disabled, no in-batch sharing happens.
  EXPECT_EQ(disabled.first_shared, 0u);
  EXPECT_EQ(disabled.second_shared, 0u);
  EXPECT_GT(enabled.second_shared, disabled.second_shared);
}

}  // namespace xllm
