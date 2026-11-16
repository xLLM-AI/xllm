# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Complete migrated operation set required before model import/capture.

Each family migration adds its public operations here in the same change as
its callers. Legacy operations remain outside the prepared catalog until
that migration; catalog discovery is not a declaration of worker usage.
"""

NPU_WORKER_OPERATIONS = ("rms_norm",)
