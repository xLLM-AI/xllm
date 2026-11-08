<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# Base utilities

`core/util` contains reusable infrastructure and generic tensor tools with no
knowledge of xLLM models, inference policies, or request orchestration. Examples
include thread pools, queues, locks, timers, byte payloads, networking, shared
memory, hashing, tensor operations, tensor serialization, and Python tensor
bindings.

Source includes and library dependencies must not point to other xLLM source
layers. Standard-library and external infrastructure libraries, Torch, Python,
and generated protobuf support are allowed. Generic tensor tools belong here;
model-specific tensor policies belong with their model or feature. Do not add
forwarding headers that pull domain components back into this directory.

Place domain code with its owner:

- Model identity, model paths, and model input metadata: `core/framework/model`.
- Model discovery and loading: `core/framework/model_loader`.
- Speculative decoding algorithms and caches: `core/framework/speculative`.
- Model-specific multimodal download policy: `core/framework/multimodal`.
- Configuration parsing and defaults: `core/framework/config`.

Pass configuration into generic components at their callers. For example, a
base hasher accepts a seed, and an HTTP downloader accepts default headers;
neither reads a model or inference configuration singleton.

Run the dependency boundary checks from the repository root:

```sh
python3 -m unittest discover -s tests/scripts -p test_util_dependencies.py
```
