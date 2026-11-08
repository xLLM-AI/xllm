<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# Base utilities

`core/util` contains reusable infrastructure components with no knowledge of
xLLM models, inference policies, requests, tensor frameworks, or xLLM wire protocols.
Examples include thread pools, queues, locks, timers, byte payloads, networking,
shared memory, hashing, and generic container algorithms.

Project includes and library dependencies must stay within `core/util`.
Standard-library and external infrastructure libraries are allowed. Do not add
forwarding headers that pull domain components back into this directory.

Place domain code with its owner:

- Model identity, model paths, and model input metadata: `core/framework/model`.
- Model discovery and loading: `core/framework/model_loader`.
- Speculative decoding algorithms and caches: `core/framework/speculative`.
- Tensor operations, tensor serialization, and Python tensor boundaries:
  `core/framework/tensor`.
- Model-specific multimodal download policy: `core/framework/multimodal`.
- Configuration parsing and defaults: `core/framework/config`.

Pass configuration into generic components at their callers. For example, a
base hasher accepts a seed, and an HTTP downloader accepts default headers;
neither reads a model or inference configuration singleton.

Run the dependency boundary checks from the repository root:

```sh
python3 -m unittest discover -s tests/scripts -p test_util_dependencies.py
```
