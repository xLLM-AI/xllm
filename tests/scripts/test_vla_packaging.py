# Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
"""Verify VLA discovery and imports from the installed package layout."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from setuptools import Distribution
from setuptools.command.build_py import build_py

from scripts.build_support.utils import load_setup_module


class VlaPackagingTest(unittest.TestCase):
    def test_build_py_includes_vla_outside_checkout(self) -> None:
        setup = load_setup_module()
        packages = setup._find_python_packages()
        self.assertIn("xllm.vla", packages)
        distribution = Distribution(
            {
                "packages": packages,
                "package_dir": {"xllm_kernel": "xllm-kernel/python/xllm_kernel"},
                "py_modules": ["xllm/__init__"],
            }
        )
        distribution.script_name = "setup.py"
        with tempfile.TemporaryDirectory() as directory:
            build_root = Path(directory) / "site-packages"
            command = build_py(distribution)
            command.build_lib = str(build_root)
            command.ensure_finalized()
            command.run()
            source = Path(setup.__file__).parent / "xllm" / "vla"
            for module in source.glob("*.py"):
                self.assertEqual((build_root / "xllm" / "vla" / module.name).read_bytes(), module.read_bytes())
            env = os.environ.copy()
            env["PYTHONPATH"] = str(build_root)
            # Discovery exercises the installed layout without requiring optional
            # model dependencies or importing the hardware extension on the host.
            script = """
import importlib.util
import sys
from pathlib import Path

root = Path(sys.argv[1])
for name in ('gr00t', 'gr00t_action', 'gr00t_attention', 'gr00t_execution', 'gr00t_processing'):
    spec = importlib.util.find_spec('xllm.vla.' + name)
    assert spec is not None, name
    assert Path(spec.origin) == root / 'xllm' / 'vla' / (name + '.py'), spec.origin
assert 'xllm_export' not in sys.modules
"""
            subprocess.run([sys.executable, "-c", script, str(build_root)], cwd=directory, env=env, check=True)


if __name__ == "__main__":
    unittest.main()
