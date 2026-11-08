# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exercise private operator patch preparation with real Git repositories."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.build_support import utils
from scripts.build_support.npu_ops_source import prepare_xllm_ops_source, xllm_ops_source_identity


class NpuOpsSourceTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "third_party" / "xllm_ops"
        self.source.mkdir(parents=True)
        self._git("init", "-q")
        self._git("config", "user.name", "xLLM test")
        self._git("config", "user.email", "test@example.com")
        (self.source / "kernel.cpp").write_text("original\n", encoding="utf-8")
        self._git("add", "kernel.cpp")
        self._git("commit", "-qm", "operator fixture")
        self.revision = self._git("rev-parse", "HEAD").strip()
        self.patches = self.root / "patches" / "xllm_ops"
        self.patches.mkdir(parents=True)
        self.destination = self.root / "build" / "ops_source"

    def _git(self, *args: str) -> str:
        return subprocess.check_output(["git", "-C", str(self.source), *args], text=True, stderr=subprocess.PIPE)

    def _patch(self, content: str = "patched") -> Path:
        patch = self.patches / "0001-kernel.patch"
        patch.write_text(
            "diff --git a/kernel.cpp b/kernel.cpp\n"
            "--- a/kernel.cpp\n"
            "+++ b/kernel.cpp\n"
            "@@ -1 +1 @@\n"
            f"-original\n+{content}\n",
            encoding="utf-8",
        )
        return patch

    def test_unpatched_build_keeps_original_revision_and_source(self) -> None:
        self.assertEqual(xllm_ops_source_identity(self.root), self.revision)
        self.assertEqual(prepare_xllm_ops_source(self.root, self.destination), self.source.resolve())
        self.assertFalse(self.destination.exists())

    def test_prepares_patch_without_modifying_pinned_or_dirty_source(self) -> None:
        self._patch()
        (self.source / "kernel.cpp").write_text("uncommitted source edit\n", encoding="utf-8")
        prepared = prepare_xllm_ops_source(self.root, self.destination)
        self.assertEqual((prepared / "kernel.cpp").read_text(), "patched\n")
        self.assertEqual((self.source / "kernel.cpp").read_text(), "uncommitted source edit\n")
        self.assertFalse((prepared / ".git").exists())
        self.assertEqual(self._git("rev-parse", "HEAD").strip(), self.revision)

    def test_applies_patch_relative_to_private_copy_inside_enclosing_repository(self) -> None:
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        self._patch()
        prepared = prepare_xllm_ops_source(self.root, self.destination)
        self.assertEqual((prepared / "kernel.cpp").read_text(), "patched\n")
        self.assertEqual((self.source / "kernel.cpp").read_text(), "original\n")

    def test_reuses_prepared_source_then_refreshes_after_patch_changes(self) -> None:
        self._patch()
        identity = xllm_ops_source_identity(self.root)
        prepared = prepare_xllm_ops_source(self.root, self.destination)
        sentinel = prepared / "incremental-build-input"
        sentinel.touch()
        self.assertEqual(prepare_xllm_ops_source(self.root, self.destination), prepared)
        self.assertTrue(sentinel.exists())
        self._patch("changed")
        self.assertNotEqual(xllm_ops_source_identity(self.root), identity)
        prepare_xllm_ops_source(self.root, self.destination)
        self.assertFalse(sentinel.exists())
        self.assertEqual((prepared / "kernel.cpp").read_text(), "changed\n")

    def test_refreshes_source_prepared_with_obsolete_export_format(self) -> None:
        self._patch()
        prepared = prepare_xllm_ops_source(self.root, self.destination)
        marker = prepared / ".xllm_ops_source_identity"
        marker.write_text(xllm_ops_source_identity(self.root) + "\n", encoding="utf-8")
        (prepared / "kernel.cpp").unlink()
        prepare_xllm_ops_source(self.root, self.destination)
        self.assertEqual((prepared / "kernel.cpp").read_text(), "patched\n")

    def test_patch_names_affect_source_identity(self) -> None:
        patch = self._patch()
        identity = xllm_ops_source_identity(self.root)
        patch.rename(self.patches / "0002-kernel.patch")
        self.assertNotEqual(xllm_ops_source_identity(self.root), identity)

    def test_applies_patches_in_filename_order(self) -> None:
        later_patch = self._patch("final").rename(self.patches / "0002-kernel.patch")
        later_patch.write_text(later_patch.read_text().replace("-original\n", "-intermediate\n"), encoding="utf-8")
        self._patch("intermediate")
        prepared = prepare_xllm_ops_source(self.root, self.destination)
        self.assertEqual((prepared / "kernel.cpp").read_text(), "final\n")

    def test_failed_patch_keeps_last_prepared_source(self) -> None:
        patch = self._patch()
        prepare_xllm_ops_source(self.root, self.destination)
        patch.write_text("invalid patch\n", encoding="utf-8")
        with self.assertRaises(subprocess.CalledProcessError):
            prepare_xllm_ops_source(self.root, self.destination)
        self.assertEqual((self.destination / "kernel.cpp").read_text(), "patched\n")
        self.assertEqual((self.source / "kernel.cpp").read_text(), "original\n")

    def test_copies_fetched_dependencies_privately_without_build_artifacts(self) -> None:
        self._patch()
        dependency = self.source / "third_party" / "fetched"
        dependency.mkdir(parents=True)
        header = dependency / "header.h"
        header.write_text("dependency\n", encoding="utf-8")
        (dependency / "alias.h").symlink_to(header)
        (dependency / "build").mkdir()
        (dependency / "build" / "artifact.o").touch()
        (dependency / "output").mkdir()
        (dependency / "output" / "artifact.o").touch()
        output_headers = dependency / "include" / "nlohmann" / "detail" / "output"
        output_headers.mkdir(parents=True)
        (output_headers / "binary_writer.hpp").write_text("required header\n", encoding="utf-8")
        prepared = prepare_xllm_ops_source(self.root, self.destination)
        copied = prepared / "third_party" / "fetched"
        self.assertFalse((copied / "alias.h").is_symlink())
        self.assertFalse((copied / "build").exists())
        self.assertFalse((copied / "output").exists())
        self.assertEqual(
            (copied / "include" / "nlohmann" / "detail" / "output" / "binary_writer.hpp").read_text(),
            "required header\n",
        )
        (copied / "alias.h").write_text("changed dependency\n", encoding="utf-8")
        self.assertEqual(header.read_text(), "dependency\n")

    def test_exports_nested_dependencies_at_pinned_revision(self) -> None:
        nested = self.source / "third_party" / "nested"
        nested.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(nested)], check=True)
        (nested / "header.h").write_text("pinned dependency\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(nested), "add", "header.h"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(nested),
                "-c",
                "user.name=xLLM test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-qm",
                "nested fixture",
            ],
            check=True,
        )
        self._git("add", "third_party/nested")
        self._git("commit", "-qm", "pin dependency")
        (nested / "header.h").write_text("dirty dependency\n", encoding="utf-8")
        self._patch()
        prepared = prepare_xllm_ops_source(self.root, self.destination)
        self.assertEqual((prepared / "third_party/nested/header.h").read_text(), "pinned dependency\n")
        self.assertEqual((nested / "header.h").read_text(), "dirty dependency\n")

    def test_prebuild_reuses_matching_patch_identity_and_invalidates_stale_cache(self) -> None:
        self._patch()
        identity = xllm_ops_source_identity(self.root)
        opp = self.root / "toolkit" / "opp"
        marker = opp / "vendors/xllm/.xllm_ops_git_head"
        marker.parent.mkdir(parents=True)
        marker.write_text(identity + "\n", encoding="utf-8")
        cache = self.root / "CMakeCache.txt"
        cache.write_text("XLLM_OPS_GIT_HEAD_CACHED:INTERNAL=old\nKEEP:BOOL=ON\n", encoding="utf-8")
        environment = {"ASCEND_OPP_PATH": str(opp)}
        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch.object(utils, "set_npu_envs"),
            mock.patch.object(utils, "get_base_dir", return_value=str(self.root)),
            mock.patch.object(utils, "_get_cmake_cache_path", return_value=str(cache)),
            mock.patch.object(utils, "_get_cann_version_info", return_value=("fixture", "9.0.0")),
        ):
            utils._ensure_xllm_ops_rebuild_state("npu")
            self.assertEqual(os.environ["XLLM_OPS_GIT_HEAD_CACHED"], identity)
            self._patch("changed")
            utils._ensure_xllm_ops_rebuild_state("npu")
            self.assertNotIn("XLLM_OPS_GIT_HEAD_CACHED", os.environ)
            self.assertEqual(cache.read_text(), "KEEP:BOOL=ON\n")


if __name__ == "__main__":
    unittest.main()
