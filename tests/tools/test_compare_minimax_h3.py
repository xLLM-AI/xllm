# Copyright 2026 The xLLM Authors.
# SPDX-License-Identifier: Apache-2.0

"""Exercise H3 media acceptance through the real FFmpeg command line."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg and FFprobe are required")
class MiniMaxH3ComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="xllm h3 media ")
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        cls.reference = cls.root / "reference.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=1344x768:r=24",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=32000:cl=stereo",
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-frames:v",
                "124",
                "-t",
                str(124 / 24),
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-threads",
                "1",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(cls.reference),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )

    def _compare(self, actual: Path, output: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parents[2] / "tools/compare_minimax_h3.py"),
                "--actual",
                str(actual),
                "--reference",
                str(self.reference),
                "--output",
                str(output if output is not None else self.root / "comparison.json"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def test_accepts_synchronized_media(self) -> None:
        output = self.root / "comparison.json"
        output.write_text("previous report\n")
        result = self._compare(self.reference)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["passed"])
        self.assertTrue(json.loads(output.read_text())["passed"])

    def test_rejects_output_aliases_of_either_input(self) -> None:
        actual = self.root / "actual.mp4"
        shutil.copyfile(self.reference, actual)
        original = self.reference.read_bytes()
        for source in (actual, self.reference):
            for alias in ("direct", "normalized", "symlink", "hardlink"):
                with self.subTest(source=source.name, alias=alias):
                    if alias == "direct":
                        output = source
                    elif alias == "normalized":
                        output = source.parent / ".." / source.parent.name / source.name
                    else:
                        output = self.root / f"{source.stem}-{alias}.json"
                        if alias == "symlink":
                            output.symlink_to(source)
                        else:
                            output.hardlink_to(source)
                    result = self._compare(actual, output)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("--output must not overwrite --actual or --reference", result.stderr)
                    self.assertEqual(actual.read_bytes(), original)
                    self.assertEqual(self.reference.read_bytes(), original)

    def test_rejects_audio_delayed_against_video(self) -> None:
        self._assert_rejects_offset("audio")

    def test_rejects_audio_advanced_against_video(self) -> None:
        self._assert_rejects_offset("video")

    def _assert_rejects_offset(self, delayed_track: str) -> None:
        actual = self.root / f"delayed-{delayed_track}.mp4"
        audio_limit = []
        if delayed_track == "audio":
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a", "-show_streams", "-of", "json", str(self.reference)],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            # Remuxing changes AAC priming handling. Trim one tail packet so
            # duration and PCM-length gates still pass without the offset fix.
            packets = int(json.loads(probe.stdout)["streams"][0]["nb_frames"])
            audio_limit = ["-frames:a", str(packets - 1)]
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(self.reference),
                "-itsoffset",
                "0.5",
                "-i",
                str(self.reference),
                "-map",
                "1:v" if delayed_track == "video" else "0:v",
                "-map",
                "1:a" if delayed_track == "audio" else "0:a",
                *audio_limit,
                "-c",
                "copy",
                str(actual),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        result = self._compare(actual)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Audio/video start times differ by more than one sample", result.stderr)


if __name__ == "__main__":
    unittest.main()
