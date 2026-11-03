#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRANSFER_DIR = ROOT / "xllm/core/framework/kv_cache_transfer"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class DataSystemSourceContractsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.factory = read(TRANSFER_DIR / "kv_cache_transfer.cpp")
        cls.adapter_h = read(TRANSFER_DIR / "yuanrong_kv_cache_transfer.h")
        cls.adapter_cpp = read(TRANSFER_DIR / "yuanrong_kv_cache_transfer.cpp")
        cls.bridge_cpp = read(TRANSFER_DIR / "yuanrong_datasystem_bridge.cpp")
        cls.cmake = read(TRANSFER_DIR / "CMakeLists.txt")
        cls.smoke = read(ROOT / "tools/datasystem/smoke_pd.sh")
        cls.rendezvous_h = read(ROOT / "xllm/core/runtime/worker_rendezvous.h")
        cls.rendezvous_cpp = read(ROOT / "xllm/core/runtime/worker_rendezvous.cpp")
        cls.worker_impl = read(ROOT / "xllm/core/runtime/worker_impl.cpp")
        cls.llm_worker_impl = read(ROOT / "xllm/core/runtime/llm_worker_impl.cpp")
        cls.worker_proto = read(ROOT / "xllm/proto/worker.proto")
        cls.scheduler = read(ROOT / "xllm/core/scheduler/disagg_pd_scheduler.cpp")
        cls.engine = read(ROOT / "xllm/core/distributed_runtime/llm_engine.cpp")
        cls.batch_builder = read(
            ROOT / "xllm/core/framework/batch/batch_input_builder.cpp"
        )

    def test_old_cpp_plugin_architecture_is_absent(self) -> None:
        self.assertNotIn("libxllm_yuanrong_kv_cache_transfer.so", self.factory)
        self.assertNotIn("dlopen", self.factory)
        self.assertNotIn("yuanrong_kv_cache_transfer_plugin.cpp", self.cmake)

    def test_adapter_is_compiled_into_main_transfer_target(self) -> None:
        self.assertIn("yuanrong_kv_cache_transfer.cpp", self.cmake)
        self.assertIn("yuanrong_kv_cache_transfer.h", self.cmake)
        self.assertIn("xllm_yuanrong_datasystem_bridge SHARED", self.cmake)

    def test_mset_d2h_batch_returns_published_count(self) -> None:
        signature = "size_t YuanRongKVCacheTransfer::mset_d2h_batch("
        self.assertIn(signature, self.adapter_cpp)
        self.assertIn("size_t mset_d2h_batch(", self.adapter_h)
        self.assertIn("return keys.size();", self.adapter_cpp)

    def test_datasystem_keys_use_cross_version_safe_characters(self) -> None:
        sanitize_begin = self.adapter_cpp.index(
            "std::string sanitize_key_component"
        )
        sanitize_end = self.adapter_cpp.index(
            "void assert_supported_shape", sanitize_begin
        )
        sanitize = self.adapter_cpp[sanitize_begin:sanitize_end]
        self.assertIn("ch != '-' && ch != '_'", sanitize)
        for unsafe in ("ch != ':'", "ch != '.'", "ch != '$'", "ch != '&'"):
            self.assertNotIn(unsafe, sanitize)

    def test_datasystem_shares_process_global_cann_runtime(self) -> None:
        self.assertIn(
            'get_int_env("XLLM_YUANRONG_BRIDGE_BASE_NAMESPACE", 1)',
            self.adapter_cpp,
        )
        self.assertIn(
            'XLLM_YUANRONG_BRIDGE_BASE_NAMESPACE:-1', self.smoke
        )
        self.assertIn("flags |= RTLD_DEEPBIND", self.adapter_cpp)
        self.assertIn("stage=bridge_dlopen_begin", self.adapter_cpp)
        self.assertIn("stage=bridge_client_create_begin", self.adapter_cpp)

    def test_datasystem_uses_receiver_driven_pull(self) -> None:
        self.assertIn(
            "YuanRong DataSystem requires --kv_cache_transfer_mode=PULL",
            self.factory,
        )
        self.assertIn("bridge_mget_", self.adapter_cpp)
        self.assertIn("YuanRong MGetH2D pull", self.adapter_cpp)
        self.assertIn("MGetH2D(key_vec", self.bridge_cpp)
        self.assertIn('<< " key0="', self.adapter_cpp)
        self.assertIn('KV_TRANSFER_MODE="${KV_TRANSFER_MODE:-PULL}"', self.smoke)

    def test_kv_only_models_ignore_scheduler_linear_id_placeholders(self) -> None:
        pull_begin = self.adapter_cpp.index(
            "bool YuanRongKVCacheTransfer::pull_kv_blocks("
        )
        pull_end = self.adapter_cpp.index(
            "#if defined(USE_NPU)", pull_begin
        )
        pull = self.adapter_cpp[pull_begin:pull_end]
        self.assertIn("ignoring scheduler linear-state id placeholders", pull)
        self.assertNotIn("does not support linear-state pull yet", pull)
        self.assertIn("assert_supported_shape(kv_cache_shape);", self.adapter_cpp)

    def test_only_source_roles_publish_blocks(self) -> None:
        self.assertIn("publish_source_blocks_", self.adapter_cpp)
        self.assertIn("instance_role == InstanceRole::PREFILL", self.factory)
        self.assertIn("instance_role == InstanceRole::MIX", self.factory)
        self.assertIn("DECODE cache is receive-only", self.adapter_cpp)

    def test_datasystem_publishes_stable_content_objects(self) -> None:
        self.assertNotIn("register_blocks(", self.adapter_cpp)
        self.assertIn("synchronize_layer(num_layers_ - 1)", self.adapter_cpp)
        self.assertIn(
            "mset_d2h_batch(keys, blob_lists, kv_info.request_id)",
            self.adapter_cpp,
        )
        self.assertIn("MSetD2H(key_vec, list_vec, set_param)", self.bridge_cpp)
        self.assertIn("WriteMode::NONE_L2_CACHE_EVICT", self.bridge_cpp)
        self.assertIn("content_addressed_datasystem_publish", self.llm_worker_impl)
        self.assertIn("wait_for_kv_transfers", self.llm_worker_impl)
        self.assertIn(
            "content_addressed_datasystem_publish && !wait_for_kv_transfers()",
            self.llm_worker_impl,
        )
        self.assertIn("content_addressed ? map_end", self.batch_builder)
        self.assertIn("seq_len >= full_info.total_kv_tokens", self.batch_builder)

    def test_datasystem_key_uses_content_hash_and_parallel_topology(self) -> None:
        self.assertIn('"xllm_ds_vllm_v1_model_"', self.adapter_cpp)
        for dimension in ('"_rank_"', '"_dp_"', '"_tp_"', '"_cp_"', '"_kvsplit_"'):
            self.assertIn(dimension, self.adapter_cpp)
        self.assertIn('"_hash_" + block_hash', self.adapter_cpp)
        self.assertNotIn('sanitize_key_component(request_id)', self.adapter_cpp)
        self.assertNotIn('std::to_string(block_id)', self.adapter_cpp)
        self.assertIn("parallel_args_);", self.worker_impl)
        self.assertIn("stable_identity_hash(addr_)", self.adapter_cpp)
        self.assertIn("compute_kv_block_hashes(", self.factory)
        self.assertIn("previous = current", self.factory)
        self.assertIn("    xxHash\n", self.cmake)

    def test_pull_protocol_carries_content_hashes_end_to_end(self) -> None:
        self.assertIn("string request_id = 9;", self.worker_proto)
        self.assertIn("repeated string block_hashes = 10;", self.worker_proto)
        self.assertIn("request.set_request_id(request_id)", read(
            ROOT / "xllm/core/distributed_runtime/comm_channel.cpp"
        ))
        self.assertIn("request.mutable_block_hashes()", read(
            ROOT / "xllm/core/distributed_runtime/comm_channel.cpp"
        ))
        self.assertIn("block_hashes = compute_kv_block_hashes", self.scheduler)
        self.assertIn("dst_linear_state_ids,\n        block_hashes", self.engine)

    def test_content_object_lifecycle_is_datasystem_managed(self) -> None:
        self.assertIn("WriteMode::NONE_L2_CACHE_EVICT", self.bridge_cpp)
        self.assertNotIn("YuanRongKVCacheTransfer::release_kv_blocks", self.adapter_cpp)
        self.assertNotIn("bridge_delete_", self.adapter_cpp)
        self.assertNotIn("xllm_yr_ds_local_delete", self.bridge_cpp)
        self.assertNotIn("DevLocalDelete", self.adapter_cpp)
        self.assertNotIn("DevDelete", self.adapter_cpp)
        self.assertNotIn("registered_keys_mutex_", self.adapter_h)
        self.assertNotIn("engine_->release_kv_blocks", self.scheduler)

    def test_mooncake_and_datasystem_are_both_buildable(self) -> None:
        self.assertIn("if(USE_MOONCAKE)", self.cmake)
        self.assertIn("USE_YUANRONG_DATASYSTEM", self.cmake)
        self.assertIn("mooncake_store", self.cmake)

    def test_worker_rendezvous_constructor_matrix(self) -> None:
        guard = "#if defined(USE_NPU) && defined(USE_MOONCAKE)"
        header_guard = self.rendezvous_h.index(
            guard, self.rendezvous_h.index("class WorkerRendezvous")
        )
        header_else = self.rendezvous_h.index("#else", header_guard)
        header_end = self.rendezvous_h.index("#endif", header_else)
        self.assertIn(
            "MooncakeWeightTransfer>& weight_transfer",
            self.rendezvous_h[header_guard:header_else],
        )
        self.assertNotIn(
            "MooncakeWeightTransfer",
            self.rendezvous_h[header_else:header_end],
        )

        impl_guard = self.rendezvous_cpp.index(
            guard, self.rendezvous_cpp.index("namespace xllm")
        )
        impl_else = self.rendezvous_cpp.index("#else", impl_guard)
        impl_end = self.rendezvous_cpp.index("#endif", impl_else)
        self.assertIn("weight_transfer_(weight_transfer)", self.rendezvous_cpp[impl_guard:impl_else])
        self.assertNotIn("weight_transfer_", self.rendezvous_cpp[impl_else:impl_end])

        self.assertIn(
            "std::make_unique<WorkerRendezvous>(kv_cache_transfer_, weight_transfer_)",
            self.worker_impl,
        )


if __name__ == "__main__":
    unittest.main()
