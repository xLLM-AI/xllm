# Copyright 2026 The xLLM Authors.
#
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

"""Extract the five frozen task attachments without rewriting their contents."""

import zipfile
from pathlib import Path


def ensure_fixtures(root: Path) -> Path:
    destination = root / "official"
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(root / "official.zip") as archive:
        for name in ("任务书.md", "case.json", "op.json", "generate_data.py", "golden.py"):
            content = archive.read(name)
            target = destination / name
            if not target.exists() or target.read_bytes() != content:
                target.write_bytes(content)
    return destination
