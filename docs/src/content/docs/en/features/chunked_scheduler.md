---
title: "ChunkedPrefill Scheduler"
sidebar:
  order: 21
---
## Feature Introduction
xLLM supports the chunked prefill scheduling strategy. Chunked prefill is a technique that optimizes large language model inference by splitting long prompts into smaller chunks for batch processing, rather than processing the entire prompt at once.
This method can effectively reduce peak GPU memory usage, improve device utilization, and better schedule and mix processing with requests from the decode stage.

## Usage
The aforementioned strategy has been implemented in xLLM and is exposed through gflags parameters to control the feature's on/off state.

- Enable chunked prefill and set the chunked size, if not set chunked size, its default value is equal to max_tokens_per_batch.
```bash
--enable_chunked_prefill=true
--max_tokens_per_chunk_for_prefill=20480 # optional
```

## Performance Impact
After enabling chunked prefill, on the Qwen3-8B model with a TPOT constraint of 50ms, the TTFT latency **decreased by 46%**.
