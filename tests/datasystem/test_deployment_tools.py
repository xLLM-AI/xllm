# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
import importlib.util
from enum import IntEnum
import json
import os
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/datasystem/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validate = _load("validate_smoke")
sdk = _load("sdk_info")
stop = _load("stop_installation")


class DeploymentToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def _logs(self, key: str = "shared_key", content: str | None = "hello", success: int = 1) -> None:
        (self.root / "response.json").write_text(json.dumps({"choices": [{"message": {"content": content}}]}))
        (self.root / "prefill.log").write_text("XLLM_KV_DIAG backend=DataSystem op=MSetD2H phase=batch key0=shared_key success=1\n")
        (self.root / "decode.log").write_text(f"XLLM_KV_DIAG backend=DataSystem op=MGetH2D phase=batch key0={key} success={success}\n")

    def test_data_system_accepts_matching_successful_transfer(self) -> None:
        self._logs()
        self.assertEqual(validate.validate(self.root, "DataSystem")["matching_key0"], ["shared_key"])

    def test_data_system_rejects_mismatched_keys(self) -> None:
        self._logs(key="different")
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate.validate(self.root, "DataSystem")

    def test_data_system_rejects_failed_mget(self) -> None:
        self._logs(success=0)
        with self.assertRaisesRegex(ValueError, "successful Decode"):
            validate.validate(self.root, "DataSystem")

    def test_empty_http_answer_is_rejected(self) -> None:
        self._logs(content="")
        with self.assertRaisesRegex(ValueError, "non-empty"):
            validate.validate(self.root, "DataSystem")

    def test_null_http_answer_is_rejected(self) -> None:
        self._logs(content=None)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            validate.validate(self.root, "DataSystem")

    def test_mooncake_requires_successful_push(self) -> None:
        self._logs()
        with self.assertRaisesRegex(ValueError, "Mooncake PUSH"):
            validate.validate(self.root, "Mooncake")
        (self.root / "prefill.log").write_text("XLLM_KV_DIAG backend=Mooncake op=push success=1")
        self.assertTrue(validate.validate(self.root, "Mooncake")["push_success"])

    def test_sdk_root_follows_selected_python_package(self) -> None:
        (self.root / "include/datasystem").mkdir(parents=True)
        (self.root / "include/datasystem/hetero_client.h").touch()
        (self.root / "lib64").mkdir()
        (self.root / "lib64/libdatasystem.so").touch()
        (self.root / "datasystem_worker").touch()
        with patch.object(sdk.metadata, "version", return_value="0.8.1"), patch.object(sdk.util, "find_spec", return_value=SimpleNamespace(origin=str(self.root / "__init__.py"))), patch.dict(os.environ, {"DATASYSTEM_VERSION": "0.8.1"}):
            info = sdk._inspect()
            self.assertEqual(info["root"], str(self.root.resolve()))
            self.assertTrue(info["library"].endswith("libdatasystem.so"))

    def test_sdk_version_mismatch_is_rejected(self) -> None:
        with patch.object(sdk.metadata, "version", return_value="0.7.0"), patch.dict(os.environ, {"DATASYSTEM_VERSION": "0.8.1"}):
            with self.assertRaisesRegex(RuntimeError, "version mismatch"):
                sdk._inspect()

    def test_stop_selects_descendants_and_ignores_reused_pid(self) -> None:
        # Windows has no SIGKILL constant; this is a mocked Linux /proc test.
        sigkill = IntEnum("PortableSignal", {"SIGKILL": 9}).SIGKILL
        signal_patch = patch.object(stop.signal, "SIGKILL", sigkill, create=True)
        signal_patch.start()
        self.addCleanup(signal_patch.stop)
        for pid in (100, 101, 200, 300):
            (self.root / str(pid)).mkdir()
        identities = {100: (1, "10"), 101: (100, "11"), 200: (1, "20"), 300: (1, "30")}
        calls = []
        reads = {}
        def identity(pid):
            reads[pid] = reads.get(pid, 0) + 1
            if pid == 300 and reads[pid] > 1:
                return (1, "99")
            return identities[pid]
        def readlink(path):
            return "/install/xllm" if path.parent.name in ("100", "300") else "/other/executable"
        with patch.object(stop, "PROC", self.root), patch.object(stop, "_identity", side_effect=identity), patch.object(stop.os, "readlink", side_effect=readlink), patch.object(stop.os, "kill", side_effect=lambda pid, sig: calls.append((pid, sig))), patch.object(stop.time, "sleep"):
            stop.stop_installation(["/install/xllm"])
        self.assertEqual({pid for pid, _ in calls}, {100, 101})
        self.assertEqual({sig for _, sig in calls}, {signal.SIGTERM, signal.SIGKILL})


if __name__ == "__main__":
    unittest.main()
