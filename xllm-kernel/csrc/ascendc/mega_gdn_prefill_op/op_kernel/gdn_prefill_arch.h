/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#pragma once

#if !defined(GDN_PREFILL_TARGET_A2A3) && !defined(GDN_PREFILL_TARGET_A5)
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#define GDN_PREFILL_TARGET_A2A3 1
#elif defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
#define GDN_PREFILL_TARGET_A5 1
#endif
#endif

#if (defined(GDN_PREFILL_TARGET_A2A3) + defined(GDN_PREFILL_TARGET_A5)) != 1
#error \
    "XllmKernelMegaGdnPrefillOp requires exactly one target architecture family"
#endif

#if defined(GDN_PREFILL_TARGET_A2A3)
#define GDN_PREFILL_ARCH_A2A3
#define GDN_PREFILL_ARCH_ID 0u
#else
#define GDN_PREFILL_ARCH_A5
#define GDN_PREFILL_ARCH_ID 1u
#endif

#if defined(__NPU_ARCH__)
#if defined(GDN_PREFILL_ARCH_A2A3) && (__NPU_ARCH__ != 2201)
#error "A2/A3 XllmKernelMegaGdnPrefillOp must compile for the C220 architecture"
#endif
#if defined(GDN_PREFILL_ARCH_A5) && (__NPU_ARCH__ != 3510)
#error "A5 XllmKernelMegaGdnPrefillOp must compile for the C3510 architecture"
#endif
#endif
