# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger

PROC = Path("/proc")

def _identity(pid: int) -> tuple[int, str]:
    fields = (PROC / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    return int(fields[1]), fields[19]  # ppid, starttime

def stop_installation(executables: list[str]) -> None:
    targets = {str(Path(p).resolve()) for p in executables}
    snapshot, selected = {}, set()
    for entry in PROC.glob("[0-9]*"):
        pid = int(entry.name)
        try:
            snapshot[pid] = _identity(pid)
            exe = str(Path(os.readlink(entry / "exe").removesuffix(" (deleted)")).resolve())
            if exe in targets and pid != os.getpid():
                selected.add(pid)
        except (OSError, ValueError, IndexError):
            continue
    while True:
        children = {pid for pid, (ppid, _) in snapshot.items() if ppid in selected}
        if children <= selected:
            break
        selected |= children
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in sorted(selected, reverse=True):
            try:
                if _identity(pid)[1] != snapshot[pid][1]:
                    continue
                logger.info(f"Stopping installation process {pid}: {sig.name}")
                os.kill(pid, sig)
            except (ProcessLookupError, FileNotFoundError):
                continue
        if selected and sig == signal.SIGTERM:
            time.sleep(5)

if __name__ == "__main__":
    stop_installation(sys.argv[1:])
