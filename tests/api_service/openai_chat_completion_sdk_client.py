#!@Python3_EXECUTABLE@
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

"""Official Chat Completions SDK client of the model-free HTTP fixture."""

import json
import sys

import httpx
from openai import OpenAI, RateLimitError


def _main() -> None:
    base_url, phase = sys.argv[1:]
    assert phase in ("success", "limited")
    expected_error = {
        "message": "The number of concurrent requests has reached the limit.",
        "type": "RateLimitError",
        "param": None,
        "code": 429,
    }
    with OpenAI(
        api_key="fixture",
        base_url=base_url,
        max_retries=0,
        timeout=5.0,
        http_client=httpx.Client(trust_env=False),
    ) as client:
        for stream in (False, True):
            try:
                response = client.chat.completions.create(
                    model="fixture",
                    messages=[{"role": "user", "content": "hi"}],
                    stream=stream,
                )
            except RateLimitError as error:
                assert phase == "limited", error
                assert error.status_code == 429
                assert error.body == expected_error
                assert error.type == "RateLimitError"
                assert error.param is None
                assert str(error.code) == "429"
                assert error.response.headers["content-type"] == "application/json"
                assert "retry-after" not in error.response.headers
                sys.stdout.write(
                    json.dumps(
                        {
                            "stream": stream,
                            "status": error.status_code,
                            "content_type": error.response.headers["content-type"],
                            "body": error.response.json(),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                continue
            assert phase == "success", "RateLimitError was not raised by create()"
            if stream:
                with response:
                    chunks = list(response)
                assert chunks
                text = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
                assert text == "hi"
                assert chunks[-1].choices[0].finish_reason == "stop"
            else:
                assert response.choices[0].message.content == "hi"
                assert response.choices[0].finish_reason == "stop"


if __name__ == "__main__":
    _main()
