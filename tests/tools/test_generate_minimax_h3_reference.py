# Copyright 2026 The xLLM Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for H3 reference export and its SGLang execution boundary."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from tools import generate_minimax_h3_reference as reference


class MiniMaxH3ReferenceCliTest(unittest.TestCase):
    def test_help_without_site_packages(self) -> None:
        result = subprocess.run(
            [sys.executable, "-S", reference.__file__, "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--devices {4,8}", result.stdout)

    def test_rejects_unsupported_sglang_before_importing_runtime(self) -> None:
        with (
            patch.object(reference.importlib.metadata, "version", return_value="0.0"),
            self.assertRaisesRegex(RuntimeError, r"requires SGLang 0\.5\.23; found 0\.0"),
        ):
            reference._capture_conditioning(Path("unused-output"))


@unittest.skipUnless(importlib.util.find_spec("sglang"), "The pinned SGLang reference environment is required")
class MiniMaxH3ReferenceExportTest(unittest.TestCase):
    def setUp(self) -> None:
        import torch
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
            MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages import (
            text_encoding,
        )

        self.torch = torch
        self.text_encoding = text_encoding
        temporary = tempfile.TemporaryDirectory(prefix="xllm h3 reference ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.hidden_states = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
        self.token_tags = torch.tensor([0, 1, 2])
        self.batch = SimpleNamespace(
            sampling_params=SimpleNamespace(prompt="test prompt"),
            extra={
                MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY: {
                    "positive": {"hidden_states": self.hidden_states, "text_token_tags": self.token_tags}
                }
            },
        )
        # Isolate model execution while exercising the real observer and torch I/O.
        forward = patch.object(text_encoding.MiniMaxH3TextEncodingStage, "forward", return_value=self.batch)
        self.forward = forward.start()
        self.addCleanup(forward.stop)

    def test_exports_actual_tp2_conditioning_only_on_rank_zero(self) -> None:
        source = reference._capture_conditioning(self.root)
        stage = SimpleNamespace(text_encoder=SimpleNamespace(_encoder_tp_group=SimpleNamespace(world_size=2)))
        with patch.object(self.torch.distributed, "get_rank", return_value=0):
            result = self.text_encoding.MiniMaxH3TextEncodingStage.forward(stage, self.batch, None)
        self.assertIs(result, self.batch)
        self.forward.assert_called_once_with(stage, self.batch, None)
        conditioning = self.torch.load(self.root / "conditioning.pt", weights_only=True)
        self.assertTrue(self.torch.equal(conditioning["prompt_embed"], self.hidden_states.unsqueeze(0)))
        self.assertEqual(conditioning["prompt_embed"].dtype, self.torch.bfloat16)
        self.assertTrue(self.torch.equal(conditioning["prompt_token_tags"], self.token_tags))
        self.assertEqual(conditioning["prompt"], self.batch.sampling_params.prompt)
        self.assertEqual(source["version"], "0.5.23")
        self.assertEqual(
            source["text_encoding_sha256"],
            hashlib.sha256(Path(self.text_encoding.__file__).read_bytes()).hexdigest(),
        )
        (self.root / "conditioning.pt").unlink()
        with patch.object(self.torch.distributed, "get_rank", return_value=1):
            result = self.text_encoding.MiniMaxH3TextEncodingStage.forward(stage, self.batch, None)
        self.assertIs(result, self.batch)
        self.assertFalse((self.root / "conditioning.pt").exists())

    def test_rejects_missing_encoder_api_and_non_tp2_encoding(self) -> None:
        reference._capture_conditioning(self.root)
        for encoder in (SimpleNamespace(), SimpleNamespace(_encoder_tp_group=SimpleNamespace(world_size=1))):
            with self.subTest(encoder=encoder), patch.object(self.torch.distributed, "get_rank", return_value=0):
                with self.assertRaisesRegex(RuntimeError, "pinned SGLang TP2 text encoder API"):
                    self.text_encoding.MiniMaxH3TextEncodingStage.forward(
                        SimpleNamespace(text_encoder=encoder), self.batch, None
                    )
                self.assertFalse((self.root / "conditioning.pt").exists())

    def test_manifest_records_actual_parameters_configs_sources_and_outputs(self) -> None:
        from sglang.multimodal_gen.runtime.entrypoints import diffusion_generator

        model = self.root / "model"
        (model / "transformer").mkdir(parents=True)
        (model / "model_index.json").write_text('{"model": "test"}\n')
        (model / "transformer/config.json").write_text('{"heads": 56}\n')
        output = self.root / "output"

        def generate(*, sampling_params_kwargs: dict[str, Any]) -> object:
            target = Path(sampling_params_kwargs["output_path"])
            self.torch.save({"prompt_embed": self.hidden_states}, target / "conditioning.pt")
            (target / sampling_params_kwargs["output_file_name"]).write_bytes(b"reference output")
            return object()

        generator = Mock()
        generator.generate.side_effect = generate
        argv = [
            reference.__file__,
            "--model",
            str(model),
            "--devices",
            "4",
            "--prompt",
            "test prompt",
            "--output",
            str(output),
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.dict(os.environ),
            patch.object(diffusion_generator.DiffGenerator, "from_pretrained", return_value=generator) as factory,
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):
            reference.main()
        report = json.loads((output / "reference.json").read_text())
        self.assertEqual(json.loads(stdout.getvalue()), report)
        self.assertEqual(report["runtime_kwargs"], factory.call_args.kwargs)
        self.assertEqual(report["sampling_params"], generator.generate.call_args.kwargs["sampling_params_kwargs"])
        self.assertEqual((report["tp"], report["sp"]), (2, 2))
        self.assertEqual(report["model"]["path"], str(model))
        for name, digest in report["model"]["config_sha256"].items():
            self.assertEqual(digest, hashlib.sha256((model / name).read_bytes()).hexdigest())
        sources = {
            "generate_minimax_h3_reference.py": reference.__file__,
            "sglang/text_encoding.py": self.text_encoding.__file__,
            "sglang/diffusion_generator.py": diffusion_generator.__file__,
        }
        for name, source in sources.items():
            self.assertEqual(report["source_sha256"][name], hashlib.sha256(Path(source).read_bytes()).hexdigest())
        for name in ("conditioning.pt", "reference.mp4"):
            self.assertEqual(report["sha256"][name], hashlib.sha256((output / name).read_bytes()).hexdigest())
        generator.shutdown.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
