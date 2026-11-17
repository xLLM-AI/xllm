# -*- coding: utf-8 -*-
# Copyright 2026 The xLLM Authors.
#
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
"""Pre-build the host-side C++ extension by importing the submission module."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import solution  # noqa: E402  (builds/loads the extension at import)

if solution._ext is None:
    print("C++ extension build FAILED:", solution._ext_error)
    raise SystemExit(1)
print("C++ extension built and loaded:", solution._ext)
