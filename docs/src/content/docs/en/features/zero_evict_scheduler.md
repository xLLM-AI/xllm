---
title: "Zero Evict Scheduler"
sidebar:
  order: 22
---
## Feature Introduction
xLLM supports the zero evict scheduling strategy. The zero evict scheduling strategy is an algorithm designed to minimize request eviction rates, reducing the need for prefill computation on evicted requests and consequently improving TPOT (Time Per Output Token).
This scheduling algorithm employs simulation rounds to detect whether a request can be scheduled without causing the eviction of other requests.

## Usage
The aforementioned strategy has been implemented in xLLM and is exposed through gflags parameters to control the feature's on/off state.

- Enable the zero evict strategy and set the maximum decode tokens per sequence.
```
--use_zero_evict=true
--max_decode_token_per_sequence=256
```

## Performance Impact
After enabling zero evict, on the Qwen3-8B model with an E2E latency constraint, the TPOT latency **decreased by 27%**.
