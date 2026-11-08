# Copyright 2026 The xLLM Authors.
# SPDX-License-Identifier: Apache-2.0

"""Decode and compare fixed H3 MP4 outputs at the same TP/SP layout.

Requires ffmpeg/ffprobe on PATH and NumPy. The media gates are engineering
regressions, not a substitute for prompt-quality review or latent comparisons.
"""

import argparse
import hashlib
import json
import math
import re
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np


def _validate_media(path: Path) -> dict[str, Any]:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(probe.stdout)["streams"]
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    audio = next(stream for stream in streams if stream["codec_type"] == "audio")
    if (
        (int(video["width"]), int(video["height"]), int(video["nb_read_frames"])) != (1344, 768, 124)
        or Fraction(video["avg_frame_rate"]) != 24
        or (int(audio["sample_rate"]), int(audio["channels"])) != (32000, 2)
        or abs(float(video["duration"]) - 124 / 24) > 1e-5
        or abs(float(audio["duration"]) - float(video["duration"])) > 1 / 24
    ):
        raise ValueError(f"Media does not match the fixed H3 workload: {path}")
    # Decoding PCM discards the presentation offset between the two tracks.
    if abs(Fraction(audio["start_time"]) - Fraction(video["start_time"])) > Fraction(1, int(audio["sample_rate"])):
        raise ValueError(f"Audio/video start times differ by more than one sample: {path}")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
        check=True,
        capture_output=True,
    )
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "video": video, "audio": audio}


def _read_audio(path: Path) -> np.ndarray:
    decoded = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(path), "-vn", "-f", "f32le", "-"],
        check=True,
        capture_output=True,
    )
    audio = np.frombuffer(decoded.stdout, dtype="<f4").reshape(-1, 2).astype(np.float64)
    if not np.isfinite(audio).all():
        raise ValueError(f"Decoded audio is not finite: {path}")
    return audio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    for source in (args.actual, args.reference):
        if output == source.resolve() or (output.exists() and output.samefile(source)):
            parser.error("--output must not overwrite --actual or --reference")
    report = {"actual": _validate_media(args.actual), "reference": _validate_media(args.reference)}
    compared = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "info",
            "-xerror",
            "-filter_complex_threads",
            "1",
            "-i",
            str(args.actual),
            "-i",
            str(args.reference),
            "-filter_complex",
            "[0:v]split[a][b];[1:v]split[c][d];[a][c]psnr[p];[b][d]ssim[s]",
            "-map",
            "[p]",
            "-map",
            "[s]",
            "-an",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    psnr = float(re.search(r"average:([0-9.inf]+)", compared.stderr)[1])
    ssim = float(re.search(r"SSIM .*All:([0-9.]+)", compared.stderr)[1])
    actual, reference = _read_audio(args.actual), _read_audio(args.reference)
    common_samples = min(len(actual), len(reference))
    # AAC may pad the final packet. Compare the common origin without lag search.
    if not common_samples or abs(len(actual) - len(reference)) > 1024:
        raise ValueError("Decoded audio lengths differ by more than one AAC frame")
    scores = []
    for left, right in zip(actual[:common_samples].T, reference[:common_samples].T):
        left, right = left - left.mean(), right - right.mean()
        reference_energy = np.dot(right, right)
        if reference_energy == 0:
            score = math.inf if np.array_equal(left, right) else -math.inf
        else:
            projection = right * (np.dot(left, right) / reference_energy)
            error_energy = np.square(left - projection).sum()
            projection_energy = np.square(projection).sum()
            if projection_energy == 0:
                score = -math.inf
            elif error_energy == 0:
                score = math.inf
            else:
                score = 10 * math.log10(projection_energy / error_energy)
        scores.append(score)
    report.update(
        psnr_db=psnr if math.isfinite(psnr) else str(psnr),
        ssim=ssim,
        si_sdr_db=[score if math.isfinite(score) else str(score) for score in scores],
        compared_audio_samples=common_samples,
        passed=psnr >= 30 and ssim >= 0.95 and all(score >= 20 for score in scores),
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
