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

"""Official Messages SDK client of the model-free HTTP fixture."""

import json
import sys

from anthropic import Anthropic, DefaultHttpxClient, RateLimitError


def _main() -> None:
    base_url, phase = sys.argv[1:]
    assert phase in ("success", "limited")
    expected_error = {
        "type": "rate_limit_error",
        "message": "The number of concurrent requests has reached the limit.",
    }
    with Anthropic(
        api_key="fixture",
        base_url=base_url,
        max_retries=0,
        timeout=5.0,
        http_client=DefaultHttpxClient(trust_env=False),
    ) as client:
        for operation in ("json", "stream", "count_tokens"):
            stream = operation == "stream"
            try:
                if operation == "count_tokens":
                    response = client.messages.count_tokens(
                        model="fixture",
                        messages=[{"role": "user", "content": "hi"}],
                    )
                else:
                    response = client.messages.create(
                        model="fixture",
                        max_tokens=8,
                        messages=[{"role": "user", "content": "hi"}],
                        stream=stream,
                    )
            except RateLimitError as error:
                assert phase == "limited", error
                assert error.status_code == 429
                body = error.response.json()
                assert body == {
                    "type": "error",
                    "error": expected_error,
                    "request_id": error.response.headers["request-id"],
                }
                assert body["request_id"]
                assert error.body == body
                assert error.request_id == body["request_id"]
                assert error.response.headers["content-type"] == "application/json"
                assert "retry-after" not in error.response.headers
                sys.stdout.write(
                    json.dumps(
                        {
                            "operation": operation,
                            "status": error.status_code,
                            "content_type": error.response.headers["content-type"],
                            "body": body,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                continue
            assert phase == "success", "RateLimitError was not raised by create()"
            if operation == "count_tokens":
                assert response.input_tokens == 1
            elif stream:
                with response:
                    events = list(response)
                assert events[0].type == "message_start"
                assert events[-1].type == "message_stop"
                assert sum(event.type == "message_stop" for event in events) == 1
                text = "".join(
                    event.delta.text
                    for event in events
                    if event.type == "content_block_delta" and event.delta.type == "text_delta"
                )
                assert text == "hi"
                assert any(event.type == "message_delta" and event.delta.stop_reason == "end_turn" for event in events)
            else:
                assert response.type == "message"
                assert response.role == "assistant"
                assert response.content[0].type == "text"
                assert response.content[0].text == "hi"
                assert response.stop_reason == "end_turn"


if __name__ == "__main__":
    _main()
