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
import time
from typing import Sequence, TextIO

from scripts.logger import logger


@dataclass
class ServerProcess:
    rank: int
    process: subprocess.Popen
    log_file: TextIO | None


def _package_binary_path() -> str:
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "xllm")


def _resolve_binary_path(binary_path: str | None) -> str:
    path = (
        os.path.realpath(os.path.expanduser(binary_path))
        if binary_path
        else _package_binary_path()
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"xllm server binary was not found: {path}. "
            "Build and install the wheel before using `python -m xllm.launch_server`."
        )
    if not os.access(path, os.X_OK):
        raise PermissionError(f"xllm server binary is not executable: {path}")
    return path


def _format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the packaged xLLM server binary. Unknown arguments are "
            "forwarded to the xllm binary unchanged."
        ),
        allow_abbrev=False,
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
        help="Total number of xllm ranks.",
    )
    parser.add_argument(
        "--node_rank",
        "--node-rank",
        dest="node_rank",
        type=int,
        default=None,
        help="Launch only this rank. If omitted, local ranks 0..nnodes-1 are launched.",
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
    if args.node_rank is not None and (
        args.node_rank < 0 or args.node_rank >= args.nnodes
    ):
        parser.error("--node-rank must be in range [0, nnodes)")

    launches_all_local_ranks = args.node_rank is None
    if launches_all_local_ranks and args.nnodes > 1:
        if args.start_port + args.nnodes - 1 > 65535:
            parser.error("--port + --nnodes - 1 must be less than or equal to 65535")


def _resolve_ranks(args: argparse.Namespace) -> list[int]:
    if args.node_rank is not None:
        return [args.node_rank]
    return list(range(args.nnodes))


def _resolve_port(
    args: argparse.Namespace,
    rank: int,
    launches_all_local_ranks: bool,
) -> int:
    if launches_all_local_ranks:
        return args.start_port + rank
    return args.start_port


def _build_command(
    binary_path: str,
    args: argparse.Namespace,
    rank: int,
    extra_args: Sequence[str],
    launches_all_local_ranks: bool,
) -> list[str]:
    port = _resolve_port(args, rank, launches_all_local_ranks)

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


def launch_server(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args, extra_args = parser.parse_known_args(argv)
    config_json = _load_config_json(parser, args)
    _apply_config_json_overrides(parser, args, config_json)
    _validate_args(parser, args)

    binary_path = _resolve_binary_path(args.binary_path)

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

    launches_all_local_ranks = args.node_rank is None
    ranks = _resolve_ranks(args)
    commands = [
        _build_command(binary_path, args, rank, extra_args, launches_all_local_ranks)
        for rank in ranks
    ]

    for rank, command in zip(ranks, commands):
        logger.info("rank %s: %s", rank, _format_command(command))

    if args.dry_run:
        return 0

    _install_signal_handlers()
    processes: list[ServerProcess] = []
    try:
        for rank, command in zip(ranks, commands):
            processes.append(_start_process(command, rank, args.log_dir))
            if args.log_dir is not None:
                logger.info(
                    "rank %s log: %s",
                    rank,
                    os.path.join(args.log_dir, f"node_{rank}.log"),
                )
        return _wait_for_processes(processes)
    except BaseException:
        _terminate_processes(processes)
        raise
    finally:
        _close_logs(processes)


def main() -> None:
    raise SystemExit(launch_server())


if __name__ == "__main__":
    main()
