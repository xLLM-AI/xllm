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
# ==============================================================================

import base64
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from xllm.generate import _build_parser, _build_request, _load_conditioning


class GenerateCommandTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="xllm generate test ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.binary = self.root / "engine"
        self.binary.write_text(
            f"#!{sys.executable}\n"
            + textwrap.dedent("""
            import json, os, signal, sys, time
            from pathlib import Path
            args = sys.argv[1:]
            rank = int(next(value.split('=', 1)[1] for value in args if value.startswith('--node_rank=')))
            root = Path(os.environ['GENERATE_TEST_ROOT'])
            offset = args.index('generate')
            modality, request_path, output_path = args[offset + 1:offset + 4]
            request = json.loads(Path(request_path).read_text())
            (root / f'rank_{rank}.json').write_text(json.dumps({'args': args, 'request': request, 'modality': modality}))
            def stop(signum, frame):
                (root / f'stopped_{rank}').touch()
                sys.exit(0)
            signal.signal(signal.SIGTERM, stop)
            if rank:
                if os.environ.get('GENERATE_WORKER_EXIT'):
                    sys.exit(0)
                while True:
                    time.sleep(0.05)
            if os.environ.get('GENERATE_WORKER_EXIT'):
                while True:
                    time.sleep(0.05)
            if '--nnodes=2' in args:
                deadline = time.monotonic() + 5
                while not (root / 'rank_1.json').exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
            code = int(os.environ.get('GENERATE_PRIMARY_EXIT', '0'))
            if not code:
                Path(output_path).write_bytes(b'encoded media')
            sys.exit(code)
        """)
        )
        self.binary.chmod(0o755)

    def _run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "xllm.launch_server",
                "generate",
                *args,
                "--binary-path",
                str(self.binary),
                "--no-log-files",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            env={**os.environ, "GENERATE_TEST_ROOT": str(self.root), **env},
        )

    def test_generate_saves_result_and_stops_other_ranks(self) -> None:
        output = self.root / "video output.mp4"
        result = self._run(
            "/model with spaces",
            "--modality",
            "video",
            "--prompt",
            "a bird",
            "--nnodes",
            "2",
            "--output-file-path",
            str(output),
            "--seed",
            "0",
            "--num-inference-steps",
            "2",
            "--tp-size",
            "2",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output.read_bytes(), b"encoded media")
        self.assertTrue((self.root / "stopped_1").is_file())
        record = json.loads((self.root / "rank_0.json").read_text())
        self.assertEqual(record["request"]["parameters"], {"seed": 0, "num_inference_steps": 2})
        self.assertIn("--tp_size", record["args"])
        self.assertIn("--model=/model with spaces", record["args"])
        self.assertEqual(record["modality"], "video")

    def test_worker_exit_is_failure_even_with_zero_exit_code(self) -> None:
        result = self._run(
            "/model", "--prompt", "a bird", "--nnodes", "2", "--output-path", str(self.root), GENERATE_WORKER_EXIT="1"
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertTrue((self.root / "stopped_0").is_file())
        self.assertFalse((self.root / "output.png").exists())

    def test_primary_failure_is_preserved_and_workers_stop(self) -> None:
        result = self._run(
            "/model", "--prompt", "a bird", "--nnodes", "2", "--output-path", str(self.root), GENERATE_PRIMARY_EXIT="17"
        )
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertTrue((self.root / "stopped_1").is_file())

    def test_unknown_option_value_is_not_consumed_as_model(self) -> None:
        result = self._run(
            "--master-node-addr",
            "127.0.0.1:12345",
            "--model-path",
            "/model",
            "--prompt",
            "a bird",
            "--output-path",
            str(self.root),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads((self.root / "rank_0.json").read_text())
        self.assertIn("127.0.0.1:12345", record["args"])
        self.assertIn("--model=/model", record["args"])

    def test_request_json_and_explicit_cli_overrides(self) -> None:
        path = self.root / "request.json"
        path.write_text(
            json.dumps(
                {
                    "input": {"prompt": "old", "media_sources": [{"name": "first_frame"}]},
                    "parameters": {"seed": 8, "num_frames": 124},
                }
            )
        )
        args = _build_parser().parse_args(
            ["--model", "/model", "--request-file", str(path), "--prompt", "new", "--seed", "0"]
        )
        request = _build_request(args)
        self.assertEqual(request["input"]["prompt"], "new")
        self.assertEqual(request["input"]["media_sources"], [{"name": "first_frame"}])
        self.assertEqual(request["parameters"], {"seed": 0, "num_frames": 124})

    def test_explicit_runtime_options_override_config(self) -> None:
        config = self.root / "runtime.json"
        config.write_text(json.dumps({"model": "/from-config", "nnodes": 2, "port": 9123}))
        result = self._run(
            "--config-json-file",
            str(config),
            "--nnodes",
            "1",
            "--port",
            "9124",
            "--prompt",
            "a bird",
            "--output-path",
            str(self.root),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads((self.root / "rank_0.json").read_text())
        self.assertIn("--nnodes=1", record["args"])
        self.assertIn("--port=9124", record["args"])
        self.assertIn("--model=/from-config", record["args"])
        self.assertFalse((self.root / "rank_1.json").exists())

    def test_rejects_duplicate_model_and_wrong_extension(self) -> None:
        for args in (
            ("/first", "--model", "/second"),
            ("/model", "--prompt", "bird", "--output-file-path", "video.mp4"),
        ):
            with self.subTest(args=args):
                result = self._run(*args)
                self.assertEqual(result.returncode, 2)
                self.assertFalse((self.root / "rank_0.json").exists())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required for conditioning")
    def test_conditioning_preserves_bf16_features_and_token_tags(self) -> None:
        import torch

        embedding = torch.arange(24, dtype=torch.bfloat16).reshape(1, 3, 8)
        tags = torch.tensor([1, 0, 1], dtype=torch.int64)
        path = self.root / "conditioning.pt"
        torch.save(
            {
                "prompt": "bird",
                "prompt_embed": embedding,
                "prompt_token_tags": tags,
                "pooled_prompt_embed": embedding[:, 0],
            },
            path,
        )
        inputs = {}
        _load_conditioning(path, inputs)
        entry = inputs["prompt_embed"]
        restored = torch.frombuffer(
            bytearray(base64.b64decode(entry["contents"]["bytes_contents"])), dtype=torch.bfloat16
        ).reshape(entry["shape"])
        torch.testing.assert_close(restored, embedding[0], rtol=0, atol=0)
        self.assertEqual(entry["datatype"], "BF16")
        self.assertEqual(entry["parameters"]["prompt_token_tags"]["int64_tensor_param"], [1, 0, 1])
        self.assertEqual(inputs["pooled_prompt_embed"]["shape"], [8])
        self.assertEqual(inputs["prompt"], "bird")
        with self.assertRaisesRegex(ValueError, "prompt differs"):
            _load_conditioning(path, {"prompt": "different"})

    def test_invalid_dimensions_do_not_start_engine(self) -> None:
        result = self._run("/model", "--prompt", "a bird", "--width", "128")
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be supplied together", result.stderr)
        self.assertFalse((self.root / "rank_0.json").exists())

    def test_help_does_not_load_native_extension(self) -> None:
        result = self._run("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--conditioning", result.stdout)
        self.assertFalse((self.root / "rank_0.json").exists())


if __name__ == "__main__":
    unittest.main()
