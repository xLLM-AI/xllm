# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
import unittest
from unittest import mock

from scripts.build_support import utils


class BuildDependenciesTest(unittest.TestCase):
    def test_ubuntu_paths_are_supported(self) -> None:
        with mock.patch.object(
            utils.sysconfig, "get_config_var", return_value="x86_64-linux-gnu"
        ):
            dependencies = utils._get_required_dependency_files()

        self.assertIn("/usr/include/msgpack.hpp", dependencies["msgpack-cxx"])
        self.assertIn("/usr/include/xxhash.h", dependencies["xxhash-header"])
        self.assertIn(
            "/usr/lib/x86_64-linux-gnu/libxxhash.so",
            dependencies["xxhash-library"],
        )
        self.assertIn("/usr/include/zstd.h", dependencies["zstd-header"])
        self.assertIn(
            "/usr/lib/x86_64-linux-gnu/libzstd.so",
            dependencies["zstd-library"],
        )

    def test_dependency_requires_header_and_library(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            header = os.path.join(temp_dir, "include", "xxhash.h")
            library = os.path.join(temp_dir, "lib", "libxxhash.so")
            os.makedirs(os.path.dirname(header))
            os.makedirs(os.path.dirname(library))
            open(header, "w", encoding="utf-8").close()

            dependencies = {
                "xxhash-header": [header],
                "xxhash-library": [library],
            }
            missing = utils._collect_missing_dependencies(dependencies)

        self.assertNotIn("xxhash-header", missing)
        self.assertIn("xxhash-library", missing)

    def test_yalantinglibs_prefix_can_be_overridden(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"YALANTINGLIBS_PREFIX": "/opt/xllm/yalantinglibs"},
        ):
            dependencies = utils._get_required_dependency_files()

        self.assertEqual(
            dependencies["yalantinglibs"],
            ["/opt/xllm/yalantinglibs/lib/cmake/yalantinglibs/config.cmake"],
        )


if __name__ == "__main__":
    unittest.main()
