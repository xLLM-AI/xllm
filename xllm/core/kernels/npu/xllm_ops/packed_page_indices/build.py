"""Build the host-side C++ planning extension.

Run once before verification:

    python3 build.py

The extension is also built lazily by ``solution.prepare`` so a fresh checkout
works without a manual step; this entry exists so the build is explicit and its
log can be captured.
"""

from __future__ import annotations

from host.build_host import load_host


def main() -> None:
    module = load_host()
    print(f"built {module.__name__} at {module.__file__}")


if __name__ == "__main__":
    main()
