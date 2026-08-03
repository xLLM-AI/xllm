# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import NoReturn, Sequence, TextIO

from scripts.logger import logger


@dataclass
class ServerProcess:
    rank: int
    process: subprocess.Popen
    log_file: TextIO | None


def _package_binary_path() -> str:
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "xllm")


def _installed_binary_path() -> str | None:
    # When `xllm serve` runs from the repo root, `import xllm` is shadowed by
    # the source-tree `xllm/` package (cwd precedes site-packages on sys.path).
    # That directory has no compiled binary -- it only exists in the installed
    # wheel -- so _package_binary_path() misses. Scan sys.path for an installed
    # `xllm/xllm` executable so the command still works from the source tree.
    this_dir = os.path.dirname(os.path.realpath(__file__))
    for entry in sys.path:
        package_dir = os.path.realpath(os.path.join(entry or os.getcwd(), "xllm"))
        if package_dir == this_dir:
            continue
        candidate = os.path.join(package_dir, "xllm")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _resolve_binary_path(binary_path: str | None) -> str:
    if binary_path:
        path = os.path.realpath(os.path.expanduser(binary_path))
    else:
        path = _package_binary_path()
        if not os.path.isfile(path):
            fallback = _installed_binary_path()
            if fallback is not None:
                logger.info(
                    "xllm binary not found next to the source-tree package; "
                    "using the installed binary at %s.",
                    fallback,
                )
                path = fallback
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"xllm server binary was not found: {path}. "
            "Build and install the wheel before using `xllm serve`."
        )
    if not os.access(path, os.X_OK):
        raise PermissionError(f"xllm server binary is not executable: {path}")
    return path


def _format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _ensure_python_model_path() -> None:
    # The Python model executor (--model_impl=python) imports the 'xllm.python'
    # subpackage. --python_model_path (or XLLM_PYTHON_MODEL_PATH when the flag
    # is empty) must point at the directory containing the 'xllm' package. For
    # a wheel install that is site-packages — the parent of this launcher's
    # directory. The embedded interpreter does not reliably pick up venv
    # site-packages on its own, so default the env var explicitly; an explicit
    # --python_model_path or a pre-set env var still takes precedence.
    os.environ.setdefault(
        "XLLM_PYTHON_MODEL_PATH",
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"{os.path.basename(sys.argv[0]) or 'xllm'} serve",
        description=(
            "Launch the packaged xLLM server binary. Unknown arguments are "
            "forwarded to the xllm binary unchanged."
        ),
        allow_abbrev=False,
        # Handle -h/--help ourselves so we can also print the server binary's
        # own help (xllm --help) instead of stopping at the launcher options.
        add_help=False,
    )
    parser.add_argument(
        "-h",
        "--help",
        dest="show_help",
        action="store_true",
        help="Show this launcher help and the xllm server options, then exit.",
    )
    parser.add_argument(
        "--config_json_file",
        "--config-json-file",
        dest="config_json_file",
        default=None,
        help=(
            "JSON config file forwarded to xllm. port and nnodes are used by "
            "this launcher."
        ),
    )
    parser.add_argument(
        "--enable-auto-tuning-gflags",
        "--enable_auto_tuning_gflags",
        dest="enable_auto_tuning",
        action="store_true",
        help=(
            "Generate an optimal JSON config for the model's model_type and "
            "launch with it. The tuned config is written to the current "
            "working directory and forwarded via --config_json_file. Mutually "
            "exclusive with --config_json_file."
        ),
    )
    parser.add_argument(
        "--port",
        "--start-port",
        "--start_port",
        dest="start_port",
        type=int,
        default=8010,
        help="Base service port. Local multi-rank launch uses port + rank.",
    )
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="Total number of nodes (one node per device) across all machines.",
    )
    parser.add_argument(
        "--machine_rank",
        "--machine-rank",
        dest="machine_rank",
        type=int,
        default=None,
        help=(
            "Index of this machine (0-based) in a multi-machine launch. This "
            "machine launches one node per local device (the local device "
            "count), and the launcher assigns each the global card rank "
            "machine_rank * device_count + local_index. If omitted, all nnodes "
            "nodes are launched locally on a single machine."
        ),
    )
    parser.add_argument(
        "--log-dir",
        "--log_dir",
        dest="log_dir",
        default="log",
        help="Directory for per-rank logs. Use --no-log-files to inherit the console.",
    )
    parser.add_argument(
        "--no-log-files",
        "--no_log_files",
        dest="log_dir",
        action="store_const",
        const=None,
        help="Do not redirect server stdout/stderr to log files.",
    )
    parser.add_argument(
        "--binary-path",
        "--binary_path",
        default=None,
        help="Override the packaged xllm binary path. Mainly useful for development.",
    )
    parser.add_argument(
        "--dry-run",
        "--dry_run",
        action="store_true",
        help="Print the commands that would be launched and exit.",
    )
    return parser


def _load_config_json(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> dict[str, object] | None:
    if args.config_json_file is None or args.config_json_file == "":
        return None

    config_path = os.path.realpath(os.path.expanduser(args.config_json_file))
    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            config_json = json.load(config_file)
    except FileNotFoundError:
        parser.error(f"--config_json_file does not exist: {config_path}")
    except json.JSONDecodeError as error:
        parser.error(f"failed to parse --config_json_file {config_path}: {error}")
    except OSError as error:
        parser.error(f"failed to read --config_json_file {config_path}: {error}")

    if not isinstance(config_json, dict):
        parser.error("--config_json_file must contain a JSON object")

    args.config_json_file = str(config_path)
    return config_json


def _read_json_int(
    parser: argparse.ArgumentParser,
    config_json: dict[str, object],
    key: str,
    default_value: int,
) -> int:
    if key not in config_json or config_json[key] is None:
        return default_value

    value = config_json[key]
    if isinstance(value, bool) or not isinstance(value, int):
        parser.error(f"--config_json_file field `{key}` must be an integer")
    return value


def _apply_config_json_overrides(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    config_json: dict[str, object] | None,
) -> None:
    if config_json is None:
        return

    args.start_port = _read_json_int(parser, config_json, "port", args.start_port)
    args.nnodes = _read_json_int(parser, config_json, "nnodes", args.nnodes)


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.nnodes < 1:
        parser.error("--nnodes must be greater than 0")
    if args.start_port < 1 or args.start_port > 65535:
        parser.error("--port/--start-port must be in range [1, 65535]")
    if args.machine_rank is not None and args.machine_rank < 0:
        parser.error("--machine-rank must be greater than or equal to 0")

    if args.machine_rank is None and args.nnodes > 1:
        if args.start_port + args.nnodes - 1 > 65535:
            parser.error("--port + --nnodes - 1 must be less than or equal to 65535")


def _reject_managed_flags(
    parser: argparse.ArgumentParser,
    extra_args: Sequence[str],
) -> None:
    """Reject a manual --node_rank, which the launcher assigns per process.

    The launcher generates each node's global card rank and forwards it to the
    binary as --node_rank, so a user-supplied one would only collide.
    """
    for arg in extra_args:
        flag = arg.split("=", 1)[0]
        if flag in ("--node_rank", "--node-rank"):
            parser.error(
                "--node_rank is assigned automatically by the launcher; use "
                "--machine-rank for multi-machine launches, or run the xllm "
                "binary directly (without `serve`) to set --node_rank manually."
            )


def _detect_device_count(parser: argparse.ArgumentParser) -> int:
    """Number of local nodes (cards) on this machine, from the device count.

    Each machine hosts one node per visible device (matching multi_machine.md),
    and the visible set is controlled by the accelerator's *_VISIBLE_DEVICES
    env mask, which Platform.get_device_count() honors.
    """
    # Imported lazily so the common launch path keeps a light import surface
    # and only touches the auto_config package (and its device probing) when a
    # multi-machine launch actually needs the local device count.
    from xllm.auto_config.utils import Platform

    device_count = Platform.get_device_count()
    if device_count is None or device_count < 1:
        parser.error(
            "could not detect the local device count for a multi-machine "
            "launch (--machine-rank is set); set the accelerator's "
            "*_VISIBLE_DEVICES env mask (e.g. ASCEND_RT_VISIBLE_DEVICES / "
            "CUDA_VISIBLE_DEVICES) so the device count is discoverable."
        )
    return device_count


def _validate_machine_topology(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    device_count: int,
) -> None:
    base_rank = args.machine_rank * device_count
    last_rank = base_rank + device_count - 1
    if last_rank >= args.nnodes:
        parser.error(
            f"--machine-rank {args.machine_rank} with {device_count} local "
            f"nodes maps to global card ranks [{base_rank}, {last_rank}], which "
            f"is outside [0, {args.nnodes}); check --nnodes and the visible "
            f"device count."
        )
    if args.start_port + device_count - 1 > 65535:
        parser.error(
            "--port plus the number of local nodes minus 1 must be less than "
            "or equal to 65535"
        )


def _resolve_launch_targets(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> list[tuple[int, int]]:
    """Map each launched process to its (global card rank, service port).

    The launcher generates the global card rank and passes it to the binary as
    --node_rank. Two modes:
    - --machine-rank M: launch this machine's slice of nodes, global ranks
      [M * device_count, (M + 1) * device_count), ports start_port + local_index.
    - no --machine-rank: single-machine launch of all nnodes nodes, global
      ranks 0..nnodes-1, ports start_port + rank.
    """
    if args.machine_rank is not None:
        device_count = _detect_device_count(parser)
        _validate_machine_topology(parser, args, device_count)
        base_rank = args.machine_rank * device_count
        return [
            (base_rank + local_index, args.start_port + local_index)
            for local_index in range(device_count)
        ]
    return [(rank, args.start_port + rank) for rank in range(args.nnodes)]


def _build_command(
    binary_path: str,
    args: argparse.Namespace,
    rank: int,
    port: int,
    extra_args: Sequence[str],
) -> list[str]:
    command = [binary_path]
    if args.config_json_file is not None:
        command.append(f"--config_json_file={args.config_json_file}")
    command.append(f"--port={port}")
    command.append(f"--nnodes={args.nnodes}")
    command.append(f"--node_rank={rank}")
    command.extend(extra_args)
    return command


def _open_log_file(log_dir: str | None, rank: int) -> TextIO | None:
    if log_dir is None:
        return None
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"node_{rank}.log")
    return open(log_path, "w", encoding="utf-8")


def _start_process(command: Sequence[str], rank: int, log_dir: str | None) -> ServerProcess:
    log_file = _open_log_file(log_dir, rank)
    try:
        process = subprocess.Popen(
            list(command),
            stdout=log_file if log_file is not None else None,
            stderr=subprocess.STDOUT if log_file is not None else None,
        )
    except BaseException:
        if log_file is not None:
            log_file.close()
        raise
    return ServerProcess(rank, process, log_file)


def _terminate_processes(processes: Sequence[ServerProcess]) -> None:
    for server_process in processes:
        if server_process.process.poll() is None:
            server_process.process.terminate()

    deadline = time.time() + 15
    for server_process in processes:
        process = server_process.process
        if process.poll() is not None:
            continue
        timeout = max(0.0, deadline - time.time())
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _close_logs(processes: Sequence[ServerProcess]) -> None:
    for server_process in processes:
        if server_process.log_file is not None:
            server_process.log_file.close()


def _probe_server_ready(
    port: int,
    stop_event: threading.Event,
    poll_interval_s: float = 2.0,
) -> None:
    # Only rank 0 (the master node) serves the HTTP API and /health, so this is
    # the readiness signal for the whole cluster: /health returns 200 once the
    # model is loaded and all workers report healthy. Poll indefinitely; a slow
    # model load must not be reported as a failure. The thread is a daemon and
    # also stops as soon as the main loop signals process exit.
    health_url = f"http://127.0.0.1:{port}/health"
    while not stop_event.is_set():
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    logger.info(
                        "xllm server started successfully, serving on port %s "
                        "(health: %s).",
                        port,
                        health_url,
                    )
                    return
        except (urllib.error.URLError, OSError):
            # Not accepting connections yet, or /health still reporting 503
            # (workers connecting / model loading). Keep waiting.
            pass
        stop_event.wait(poll_interval_s)


def _start_readiness_probe(port: int) -> tuple[threading.Thread, threading.Event]:
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_probe_server_ready,
        args=(port, stop_event),
        name="xllm-readiness-probe",
        daemon=True,
    )
    thread.start()
    return thread, stop_event


def _wait_for_processes(processes: Sequence[ServerProcess]) -> int:
    try:
        while True:
            for server_process in processes:
                return_code = server_process.process.poll()
                if return_code is None:
                    continue
                if len(processes) > 1:
                    logger.warning(
                        "xllm rank %s exited with code %s; terminating "
                        "remaining ranks.",
                        server_process.rank,
                        return_code,
                    )
                    _terminate_processes(processes)
                return return_code
            time.sleep(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted; terminating xllm server processes.")
        _terminate_processes(processes)
        return 130


def _install_signal_handlers() -> None:
    def _raise_keyboard_interrupt(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)


def _print_binary_help(parser: argparse.ArgumentParser, binary_path: str) -> None:
    # Flush our buffered stdout first: when piped, Python stdout is block
    # buffered while the child writes straight to the fd, which would otherwise
    # print the binary help before the launcher help.
    sys.stdout.flush()
    try:
        subprocess.run([binary_path, "--help"], check=False)
    except OSError as error:
        parser.error(f"failed to run xllm binary help {binary_path}: {error}")


def launch_server(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args, extra_args = parser.parse_known_args(argv)

    if args.show_help:
        parser.print_help()
        # Also surface the server binary's own options (HelpFormatter output).
        try:
            binary_path = _resolve_binary_path(args.binary_path)
        except (FileNotFoundError, PermissionError) as error:
            parser.exit(status=0, message=f"\n{error}\n")
        print("\nxllm server options:\n")
        _print_binary_help(parser, binary_path)
        return 0

    if args.enable_auto_tuning:
        if args.config_json_file:
            parser.error(
                "--enable-auto-tuning-gflags and --config_json_file are "
                "mutually exclusive"
            )
        # Imported lazily so the common launch path keeps a light import
        # surface and never touches the auto_config package.
        from xllm.auto_config.utils import AutoTuningError, generate_tuned_config

        try:
            args.config_json_file = generate_tuned_config(extra_args, os.getcwd())
        except AutoTuningError as error:
            parser.error(f"auto-tuning failed: {error}")

    config_json = _load_config_json(parser, args)
    _apply_config_json_overrides(parser, args, config_json)
    _validate_args(parser, args)
    _reject_managed_flags(parser, extra_args)

    binary_path = _resolve_binary_path(args.binary_path)

    _ensure_python_model_path()

    targets = _resolve_launch_targets(parser, args)
    commands = [
        _build_command(binary_path, args, rank, port, extra_args)
        for rank, port in targets
    ]

    for (rank, port), command in zip(targets, commands):
        logger.info("rank %s (port %s): %s", rank, port, _format_command(command))

    if args.dry_run:
        return 0

    _install_signal_handlers()
    processes: list[ServerProcess] = []
    readiness_stop_event: threading.Event | None = None
    try:
        for (rank, _port), command in zip(targets, commands):
            processes.append(_start_process(command, rank, args.log_dir))
            if args.log_dir is not None:
                logger.info(
                    "rank %s log: %s",
                    rank,
                    os.path.join(args.log_dir, f"node_{rank}.log"),
                )
        # Only rank 0 (the master node) serves the HTTP API, so probe its port
        # for readiness when this machine hosts it.
        rank_0_port = next((port for rank, port in targets if rank == 0), None)
        if rank_0_port is not None:
            _, readiness_stop_event = _start_readiness_probe(rank_0_port)
        return _wait_for_processes(processes)
    except BaseException:
        _terminate_processes(processes)
        raise
    finally:
        if readiness_stop_event is not None:
            readiness_stop_event.set()
        _close_logs(processes)


def _exec_binary(argv: Sequence[str]) -> NoReturn:
    # Replace this process with the packaged xllm binary so `xllm <args>`
    # behaves exactly like invoking the binary directly (same pid, signals,
    # stdio, and exit code). Any argument other than the `serve` subcommand is
    # forwarded verbatim, which keeps `xllm --model ...` working for users who
    # start the server through the binary directly.
    try:
        binary_path = _resolve_binary_path(None)
    except (FileNotFoundError, PermissionError) as error:
        logger.error("%s", error)
        raise SystemExit(1)

    _ensure_python_model_path()
    os.execv(binary_path, [binary_path, *argv])


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)

    if args and args[0] == "serve":
        raise SystemExit(launch_server(args[1:]))

    # Everything else is handed straight to the xllm binary, including no args
    # and -h/--help, so `xllm` is a transparent alias for the server binary.
    _exec_binary(args)


if __name__ == "__main__":
    main()
