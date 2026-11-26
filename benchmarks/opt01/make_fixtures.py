# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.
"""Create deterministic visual inputs with simple, inspectable answers."""

import hashlib
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent / "fixtures"
root.mkdir(exist_ok=True)
for size, color in ((448, "red"), (896, "red"), (448, "blue")):
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((size // 4, size // 4, 3 * size // 4, 3 * size // 4), fill=color)
    image.save(root / f"{color}_{size}.png")
container = av.open(str(root / "moving_red_circle.mp4"), mode="w")
stream = container.add_stream("mpeg4", rate=2)
stream.width, stream.height, stream.pix_fmt = 448, 448, "yuv420p"
for index in range(4):
    image = Image.new("RGB", (448, 448), "white")
    draw = ImageDraw.Draw(image)
    x = 35 + 80 * index
    draw.ellipse((x, 160, x + 80, 240), fill="red")
    frame = av.VideoFrame.from_ndarray(np.array(image), format="rgb24")
    for packet in stream.encode(frame):
        container.mux(packet)
for packet in stream.encode():
    container.mux(packet)
container.close()
manifest = [
    {"file": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for path in sorted(root.iterdir())
    if path.is_file()
]
(root / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest))
