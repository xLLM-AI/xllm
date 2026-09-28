# Copyright 2025-2026 The xLLM Authors.
#
#
#
#
#
#
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
#
#
#
#
#
"""KDA / MTP spec-verify configuration constants, shared by NpuPagedAttentionBackend and the KDA mixin."""

import os

# Spec-verify state-commit scheme: "lazy" (default) commits exactly the
# confirmed tokens - each verify step stashes its raw qkv/gate rows and the
# next step advances the live state by the kv delta (the count the C++
# actually committed: draft on accept, corrected argmax on reject), so the
# linear state tracks the true token stream exactly (verified offline
# against a plain-decode oracle: per-step output divergence decays to zero,
# vs a persistent ~6e-3 error for the alternative). "full"
# (GLM5_MTP_COMMIT=full) commits both verify rows every step - the
# re-processed row0 write is near-idempotent under the delta rule, but a
# rejected draft leaves a phantom write, a persistent per-step error.

_MTP_FULL_COMMIT = os.environ.get("GLM5_MTP_COMMIT", "lazy") == "full"
# Seq-wise KDA recurrent dispatch for spec-verify batches: split the flattened
# multi-sequence call into one single-segment call per sequence so concurrent
# verify runs the exact op-call shapes of single-request verify (see the
# read-only branch below for the divergence rationale).
_KDA_SEQWISE = os.environ.get("GLM5_KDA_SEQWISE", "1") == "1"
# Debug: disable the batched cross-layer verify advance coordinator so each
# layer advances itself (per-layer [B, 2] varlen call - the same call shape
# as the V2 path). Used to A/B the coordinator's flattened multi-segment
# recurrent call (L segments, in-process tiling -> 1-ULP state drift).
_KDA_NO_COORD = os.environ.get("GLM5_KDA_NO_COORD", "0") == "1"
# Graph-shaped MTP spec-verify for KDA layers (GLM5_KDA_VERIFY_V2=1): replaces
# the host lazy-advance state machine with fixed-shape ops and device tensors
# so the same code is ACL-graph capturable. See docs/mtp_graph_verify_design.md
# for the full protocol; the two key invariants are (a) the advance consumes
# the stashed [b, d] rows as a fixed 2-rows-per-seq varlen recurrent call with
# row1 masked by (m == 2) — a gate=0/beta=0 row is a bit-exact state no-op
# (verified on NPU) — and (b) conv state is staged as dual tails
# [after-b, after-bd] per slot, the next step gathering the tail selected by
# that step's committed-row count m (= kv_seq_lens growth over the previous
# verify step; m=1 on rejection, m=2 on acceptance).
_KDA_VERIFY_V2 = os.environ.get("GLM5_KDA_VERIFY_V2", "0") == "1"
# Fused multi-slot MTP spec-verify (GLM5_KDA_VERIFY_V3=1): replaces V2's host
# m state machine + 6-buffer slot stash with a persistent combined [base|draft]
# state pool and a single fused recurrent_kda call per layer (vllm-ascend's
# in-kernel-spec contract). See _spec_verify_v3. V3 is OFF by default in this
# PR (eager-only); it is the MTP spec-verify path and is enabled in the MTP
# PR along with its graph snapshot/restore wiring. Set GLM5_KDA_VERIFY_V3=1
# to opt in, or GLM5_KDA_VERIFY_V2=1 to fall back to the legacy V2 path (which
# forces V3 off to avoid the two competing).
_KDA_VERIFY_V3 = (not _KDA_VERIFY_V2) and (os.environ.get("GLM5_KDA_VERIFY_V3", "0") == "1")

# Spec-verify acceptance instrumentation (MTP_TRACE=1): layer-0 counts
# verify steps and non-prefill plain steps (each reject inserts one plain
# bootstrap step, so the plain share of decode steps is the reject rate).
_MTP_TRACE = os.environ.get("MTP_TRACE", "") == "1"
