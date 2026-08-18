# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import base64
from io import BytesIO
from typing import Any

from PIL import Image
from xllm_export import MMData


def _bytes_to_data_url(data: bytes) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image;base64,{encoded}"


def _pil_to_data_url(image: Image.Image) -> str:
    buf = BytesIO()
    fmt = image.format or "PNG"
    image.save(buf, format=fmt)
    return _bytes_to_data_url(buf.getvalue())


def normalize_vllm_style_inputs(
    prompts: Any,
) -> tuple[list[str], list[MMData] | None, list[list[str]] | None]:
    if isinstance(prompts, dict):
        requests = [prompts]
        return _parse_vllm_style_requests(requests)
    if isinstance(prompts, list) and prompts and all(isinstance(x, dict) for x in prompts):
        return _parse_vllm_style_requests(prompts)

    raise TypeError(
        "VLM-style inputs must be dict/List[dict] with key 'prompt', e.g. "
        "{'prompt': '...', 'multi_modal_data': {'image': image}}"
    )


def _parse_vllm_style_requests(
    requests: list[dict[str, Any]],
) -> tuple[list[str], list[MMData] | None, list[list[str]] | None]:
    prompts: list[str] = []
    mm_datas: list[MMData] = []
    image_urls: list[list[str]] = []
    use_mm_data: bool | None = None

    for req in requests:
        if "prompt" not in req:
            raise ValueError("Each request dict must contain key 'prompt'")

        prompt = req["prompt"]
        if not isinstance(prompt, str):
            raise TypeError("request['prompt'] must be a string")
        prompts.append(prompt)

        if "multi_modal_data" not in req:
            if use_mm_data is True:
                raise TypeError("Cannot mix MMData and empty multi_modal_data in one batch")
            use_mm_data = False
            image_urls.append([])
            continue

        payload = req["multi_modal_data"]
        if isinstance(payload, MMData):
            if use_mm_data is False:
                raise TypeError("Cannot mix MMData and image inputs in one batch")
            use_mm_data = True
            mm_datas.append(payload)
        else:
            if use_mm_data is True:
                raise TypeError("Cannot mix MMData and image inputs in one batch")
            use_mm_data = False
            image_urls.append(_to_image_urls(payload))

    if use_mm_data:
        return prompts, mm_datas, None
    return prompts, None, image_urls


def _to_image_urls(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        raise TypeError("multi_modal_data must be dict or MMData")

    if "image" in payload:
        images = payload["image"]
        return _normalize_images(images)
    if "video" in payload:
        raise NotImplementedError("video multi_modal_data is not supported yet")
    if "audio" in payload:
        raise NotImplementedError("audio multi_modal_data is not supported yet")

    raise ValueError("Unsupported multi_modal_data format. Expected {'image': ...} or MMData.")


def _normalize_images(images: Any) -> list[str]:
    if isinstance(images, (list, tuple)):
        if len(images) == 0:
            raise ValueError("multi_modal_data['image'] cannot be empty")
        return [_to_image_url(img) for img in images]
    return [_to_image_url(images)]


def _to_image_url(image: Any) -> str:
    if isinstance(image, str):
        return image
    if isinstance(image, Image.Image):
        return _pil_to_data_url(image.convert("RGB"))
    if isinstance(image, (bytes, bytearray)):
        return _bytes_to_data_url(bytes(image))

    raise TypeError("image must be image path/url string, PIL.Image, bytes, or a list of these")
