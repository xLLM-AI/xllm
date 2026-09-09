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

"""CPU-only tests for the auto-tuning launcher helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xllm.auto_config import utils


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        (["--model", "/models/qwen3", "--port", "8000"], "/models/qwen3"),
        (["--port", "8000", "--model=/models/qwen3"], "/models/qwen3"),
        (["--model"], None),
        (["--port", "8000"], None),
    ],
)
def test_extract_model_path_accepts_both_forwarding_forms(
    extra_args: list[str],
    expected: str | None,
) -> None:
    assert utils.extract_model_path(extra_args) == expected


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"model_type": "qwen3", "model_name": "fallback"}, "qwen3"),
        ({"model_name": "fallback"}, "fallback"),
    ],
)
def test_read_model_type_prefers_model_type_and_falls_back(
    tmp_path: Path,
    config: dict[str, str],
    expected: str,
) -> None:
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    assert utils.read_model_type(str(tmp_path)) == expected


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("[]", "must contain a JSON object"),
        ('{"model_type": ""}', "must contain a string"),
        ('{"model_type": 7}', "must contain a string"),
        ("{", "failed to parse"),
    ],
)
def test_read_model_type_reports_invalid_model_config(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    (tmp_path / "config.json").write_text(contents, encoding="utf-8")

    with pytest.raises(utils.AutoTuningError, match=message):
        utils.read_model_type(str(tmp_path))


def test_read_model_type_reports_missing_config(tmp_path: Path) -> None:
    with pytest.raises(utils.AutoTuningError, match="config.json not found"):
        utils.read_model_type(str(tmp_path))


def test_generate_tuned_config_passes_model_context_and_writes_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "tiny"}', encoding="utf-8")

    tuning_dir = tmp_path / "profiles"
    tuning_dir.mkdir()
    base_config = {"nnodes": 1, "max_tokens_per_batch": 8}
    (tuning_dir / "tiny.json").write_text(json.dumps(base_config), encoding="utf-8")
    (tuning_dir / "tiny.py").write_text(
        "def tune(base_config: dict, context: dict) -> dict:\n"
        "    base_config['model_path'] = context['model_path']\n"
        "    base_config['model_type'] = context['model_type']\n"
        "    base_config['visible_device_count'] = context['visible_device_count']\n"
        "    return base_config\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    monkeypatch.setattr(utils, "auto_tuning_config_dir", lambda: str(tuning_dir))
    monkeypatch.setattr(utils.Platform, "get_device_count", classmethod(lambda cls: 2))

    output_path = utils.generate_tuned_config(["--model", str(model_path)], str(output_dir))

    assert Path(output_path) == output_dir / "tiny.tuned.json"
    assert json.loads(Path(output_path).read_text(encoding="utf-8")) == {
        **base_config,
        "model_path": str(model_path),
        "model_type": "tiny",
        "visible_device_count": 2,
    }
    assert json.loads((tuning_dir / "tiny.json").read_text(encoding="utf-8")) == base_config


def test_generate_tuned_config_rejects_missing_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "missing"}', encoding="utf-8")
    tuning_dir = tmp_path / "profiles"
    tuning_dir.mkdir()
    monkeypatch.setattr(utils, "auto_tuning_config_dir", lambda: str(tuning_dir))

    with pytest.raises(utils.AutoTuningError, match="auto-tuning is not supported"):
        utils.generate_tuned_config(["--model", str(model_path)], str(tmp_path / "output"))


def test_generate_tuned_config_rejects_non_dict_profile_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "tiny"}', encoding="utf-8")

    tuning_dir = tmp_path / "profiles"
    tuning_dir.mkdir()
    (tuning_dir / "tiny.json").write_text("{}", encoding="utf-8")
    (tuning_dir / "tiny.py").write_text(
        "def tune(base_config: dict, context: dict) -> list:\n    return []\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(utils, "auto_tuning_config_dir", lambda: str(tuning_dir))
    monkeypatch.setattr(utils.Platform, "get_device_count", classmethod(lambda cls: 1))

    with pytest.raises(utils.AutoTuningError, match=r"tune\(\) must return a dict"):
        utils.generate_tuned_config(["--model", str(model_path)], str(tmp_path / "output"))
