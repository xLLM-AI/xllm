import base64
import os
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from scripts.build_support.env import set_npu_envs
from scripts.logger import logger

_MOONCAKE_DEPENDENCIES_COMMAND = "bash scripts/build_support/install_dependencies.sh"
_MOONCAKE_GO_BIN_DIR = "/usr/local/go/bin"
_GO_DEFAULT_PROXY = "https://proxy.golang.org,direct"
_MOONCAKE_GO_PROXY = "https://goproxy.cn|https://goproxy.io|direct"


def _b64(s: str) -> str:
    return base64.b64decode(s).decode()


# get cpu architecture
def get_cpu_arch() -> str:
    arch = platform.machine()
    if "x86" in arch or "amd64" in arch:
        return "x86"
    elif "arm" in arch or "aarch64" in arch:
        return "arm"
    else:
        raise ValueError(f"❌ Unsupported architecture: {arch}")


def get_ascend_platform() -> str:
    if not hasattr(get_ascend_platform, "_cached_platform"):
        set_npu_envs()
        import acl

        acl.init()
        soc_name = acl.get_soc_name().upper()
        if _b64("OTEwQg==") in soc_name:
            platform = "a2"
        elif _b64("QVNDRU5EOTEwXzkz") in soc_name or _b64("OTEwQw==") in soc_name:
            platform = "a3"
        elif _b64("QVNDRU5EOTUw") in soc_name:
            platform = "a5"
        elif _b64("OTEw") in soc_name:
            platform = "a2"
        else:
            raise ValueError(f"Unsupported Ascend SoC for TileLang wheel: {soc_name}")
        get_ascend_platform._cached_platform = platform
    return get_ascend_platform._cached_platform


# get device type
def get_device_type() -> str:
    import torch

    if torch.cuda.is_available():
        try:
            from torch.utils.cpp_extension import HIP_HOME

            if HIP_HOME and "dtk" in HIP_HOME.lower():
                return "dcu"
        except Exception:
            pass

    if torch.cuda.is_available():
        try:
            import ixformer

            return "ilu"
        except ImportError:
            return "cuda"

    try:
        import torch_musa

        if torch.musa.is_available():
            return "musa"
    except ImportError:
        pass

    try:
        import torch_mlu

        if torch.mlu.is_available():
            return "mlu"
    except ImportError:
        pass

    try:
        import torch_npu

        if torch.npu.is_available():
            return "npu"
    except ImportError:
        pass

    logger.error("❌ Unsupported device type, please check what device you are using.")
    exit(1)


def get_base_dir() -> str:
    helper_path = Path(__file__).resolve()
    for parent in helper_path.parents:
        if all((parent / marker).exists() for marker in ("setup.py", "version.txt", "CMakeLists.txt")):
            return str(parent)

    fallback_index = min(2, len(helper_path.parents) - 1)
    return str(helper_path.parents[fallback_index])


def _join_path(*paths: str) -> str:
    return os.path.join(get_base_dir(), *paths)


# return the python version as a string like "310" or "311" etc
def get_python_version() -> str:
    return sysconfig.get_python_version().replace(".", "")


def get_torch_version(device: str) -> str | None:
    try:
        import torch

        if device == "cuda":
            return torch.__version__
        return torch.__version__.split("+")[0]
    except ImportError:
        return None


def get_torch_cmake_prefix_path() -> str:
    import torch

    return torch.utils.cmake_prefix_path


def get_version() -> str:
    # first read from environment variable
    version: str | None = os.getenv("XLLM_VERSION")
    if not version:
        # then read from version file
        with open(_join_path("version.txt")) as f:
            version = f.read().strip()

    # strip the leading 'v' if present
    if version and version.startswith("v"):
        version = version[1:]

    if not version:
        raise RuntimeError("❌ Unable to find version string.")

    version_suffix = os.getenv("XLLM_VERSION_SUFFIX")
    if version_suffix:
        version += version_suffix
    return version


def read_readme() -> str:
    p = _join_path("README.md")
    if os.path.isfile(p):
        return open(p, encoding="utf-8").read()
    else:
        return ""


def get_cmake_dir() -> str:
    plat_name = sysconfig.get_platform()
    python_version = get_python_version()
    dir_name = f"cmake.{plat_name}-{sys.implementation.name}-{python_version}"
    cmake_dir = os.path.join(get_base_dir(), "build", dir_name)
    os.makedirs(cmake_dir, exist_ok=True)
    return cmake_dir


# A single translation unit that pulls in torch is tens of megabytes of object
# code, so the ccache default limit (5 GB) is evicted faster than it fills and
# the hit rate collapses on a clean or near-clean tree.
_CCACHE_MAXSIZE = "50G"


def configure_ccache(env: dict[str, str]) -> None:
    """Make ccache usable before a CMake build, or say why it will not be.

    Values already present in the environment win, so a caller that mounts a
    pre-warmed cache elsewhere can override any of them.
    """
    ccache = shutil.which("ccache")
    if ccache is None:
        logger.warning(
            "ccache not found: every rebuild recompiles every file and a clean "
            "build is a fresh full build. Install ccache to fix this."
        )
        return

    env.setdefault("CCACHE_MAXSIZE", _CCACHE_MAXSIZE)
    # Keeping paths relative to the source tree lets one cache serve several
    # build directories (build/cmake.<platform>-<impl>-<version>) and, in CI,
    # runners whose checkout lives under a different absolute path.
    env.setdefault("CCACHE_BASEDIR", get_base_dir())
    cache_dir = env.get("CCACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "ccache")
    logger.info(f"ccache: {ccache}")
    logger.info(f"ccache: cache dir {cache_dir}, max size {env['CCACHE_MAXSIZE']}")
    logger.info(
        "ccache: this directory must be persisted between CI runs, or mounted "
        "into the build container, to have any effect there."
    )


def check_and_install_pre_commit() -> None:
    # check if .git is a directory
    if not os.path.isdir(".git"):
        return

    if not os.path.exists(".git/hooks/pre-commit"):
        ok, _, _ = _run_command(["pre-commit", "install"], check=True)
        if not ok:
            logger.error("❌ Run 'pre-commit install' failed. Please install pre-commit: pip install pre-commit")
            exit(1)


def _run_command(
    args: list[str],
    cwd: str | None = None,
    check: bool = True,
    input_text: str | None = None,
    passthrough_output: bool = False,
) -> tuple[bool, str, str]:
    try:
        if passthrough_output:
            result = subprocess.run(
                args,
                cwd=cwd,
                check=False,
                input=input_text,
                text=True,
            )
        else:
            result = subprocess.run(
                args,
                cwd=cwd,
                check=False,
                input=input_text,
                capture_output=True,
                text=True,
            )
    except OSError as e:
        return False, "", str(e)

    if passthrough_output:
        if check and result.returncode != 0:
            return False, "", f"exit code {result.returncode}"
        return result.returncode == 0, "", ""

    if check and result.returncode != 0:
        return False, result.stdout.strip(), (result.stderr or result.stdout).strip()

    return result.returncode == 0, result.stdout.strip(), (result.stderr or "").strip()


def _print_manual_check_commands(commands: list[str]) -> None:
    logger.info("🔎 You can run these commands to inspect manually:")
    for cmd in commands:
        logger.info(f"   {cmd}")


def _run_shell_command(
    command: str,
    cwd: str | None = None,
    check: bool = True,
    passthrough_output: bool = False,
) -> bool:
    ok, _, err = _run_command(
        shlex.split(command),
        cwd=cwd,
        check=check,
        passthrough_output=passthrough_output,
    )
    if not ok:
        logger.error(f"❌ Run shell command '{command}' failed: {err}")
        return False
    return True


def _run_git_command(repo_root: str, args: list[str]) -> tuple[bool, str]:
    ok, output, err = _run_command(["git"] + args, cwd=repo_root, check=True)
    if not ok and "No such file or directory" in err:
        logger.error(f"❌ Failed to run git command in {repo_root}: git {' '.join(args)}")
        logger.error(f"   {err}")
        return False, ""
    if not ok:
        logger.error(f"❌ Git command failed in {repo_root}: git {' '.join(args)}")
        if err:
            logger.error(f"   {err}")
        return False, ""
    return True, output


def _collect_submodule_init_issues(repo_root: str) -> dict[str, str]:
    ok, output = _run_git_command(repo_root, ["submodule", "status", "--recursive"])
    if not ok:
        logger.error("❌ Failed to inspect submodule status.")
        _print_manual_check_commands(
            [
                f"cd {repo_root}",
                "git submodule status --recursive",
                "git submodule update --init --recursive",
            ]
        )
        exit(1)

    issues: dict[str, str] = {}
    for line in output.splitlines():
        if not line:
            continue
        state = line[0]
        content = line[1:].strip()
        parts = content.split()
        if len(parts) < 2:
            continue
        path = parts[1]
        commit = parts[0]

        if state == "-":
            issues[path] = f"uninitialized (expected commit starts with {commit})"
        elif state == "+":
            issues[path] = f"commit mismatch (checked-out commit starts with {commit})"
        elif state == "U":
            issues[path] = "merge conflict"

    return issues


def _is_dependency_installed(candidate_files: list[str]) -> bool:
    """Return whether any distribution-specific dependency marker exists."""
    normalized_candidates = [os.path.abspath(os.path.expanduser(file_path)) for file_path in candidate_files]
    return any(os.path.isfile(file_path) for file_path in normalized_candidates)


def _get_yalantinglibs_prefix() -> str:
    return os.path.abspath(os.path.expanduser(os.getenv("YALANTINGLIBS_PREFIX", "/usr/local")))


def _get_mooncake_source_dir(script_path: str) -> str:
    return os.path.join(script_path, "third_party", "Mooncake")


def _get_mooncake_go_version(mooncake_source_dir: str) -> str | None:
    dependencies_script = os.path.join(mooncake_source_dir, "dependencies.sh")
    try:
        with open(dependencies_script, encoding="utf-8") as script_file:
            for line in script_file:
                key, separator, value = line.strip().partition("=")
                if separator and key == "GOVER":
                    return value.strip("\"'")
    except OSError as error:
        logger.error(f"❌ Failed to read Mooncake Go version from {dependencies_script}: {error}")
    return None


def _get_go_version(go_binary: str) -> str | None:
    ok, output, _ = _run_command([go_binary, "version"], check=False)
    if not ok:
        return None

    for token in output.split():
        if token.startswith("go") and token[2:3].isdigit():
            return token[2:]
    return None


def _go_version_satisfies(version: str | None, required_version: str) -> bool:
    if version is None:
        return False
    try:
        return tuple(map(int, version.split("."))) >= tuple(map(int, required_version.split(".")))
    except ValueError:
        return False


def _export_mooncake_go_path(mooncake_source_dir: str) -> bool:
    required_version = _get_mooncake_go_version(mooncake_source_dir)
    if required_version is None:
        return False
    if _go_version_satisfies(_get_go_version("go"), required_version):
        return True

    cache_bin_dir = os.path.join(
        os.getenv("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "xllm", f"go{required_version}", "bin"
    )
    for go_bin_dir in (cache_bin_dir, _MOONCAKE_GO_BIN_DIR):
        go_binary = os.path.join(go_bin_dir, "go")
        if not os.path.isfile(go_binary) or not os.access(go_binary, os.X_OK):
            continue
        if not _go_version_satisfies(_get_go_version(go_binary), required_version):
            continue
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        os.environ["PATH"] = os.pathsep.join([go_bin_dir, *(path for path in path_entries if path != go_bin_dir)])
        logger.info(f"✅ Export Mooncake Go toolchain to PATH: {go_bin_dir}")
        return True
    return False


def _export_mooncake_go_proxy() -> None:
    if os.environ.get("GOPROXY"):
        return

    ok, go_proxy, error = _run_command(["go", "env", "GOPROXY"], check=False)
    if not ok:
        logger.error(f"Failed to read Mooncake Go proxy configuration: {error}")
        exit(1)
    if go_proxy and go_proxy != _GO_DEFAULT_PROXY:
        return

    os.environ["GOPROXY"] = _MOONCAKE_GO_PROXY
    logger.info(f"Export Mooncake GOPROXY to environment: {_MOONCAKE_GO_PROXY}")


def _ensure_git_safe_directory_or_exit(directory: str) -> None:
    safe_directory = os.path.realpath(directory)
    count = int(os.environ.get("GIT_CONFIG_COUNT", "0"))
    if any(
        os.environ.get(f"GIT_CONFIG_KEY_{i}") == "safe.directory"
        and os.environ.get(f"GIT_CONFIG_VALUE_{i}") == safe_directory
        for i in range(count)
    ):
        return
    os.environ.update(
        {
            f"GIT_CONFIG_KEY_{count}": "safe.directory",
            f"GIT_CONFIG_VALUE_{count}": safe_directory,
            "GIT_CONFIG_COUNT": str(count + 1),
        }
    )


def _get_required_dependency_files() -> dict[str, list[str]]:
    install_prefix = _get_yalantinglibs_prefix()
    library_dirs = [
        "/usr/lib",
        "/usr/lib64",
        "/usr/local/lib",
        "/usr/local/lib64",
    ]
    multiarch = sysconfig.get_config_var("MULTIARCH")
    if multiarch:
        library_dirs.extend(
            [
                os.path.join("/usr/lib", multiarch),
                os.path.join("/usr/local/lib", multiarch),
            ]
        )

    return {
        "yalantinglibs": [
            os.path.join(
                install_prefix,
                "lib",
                "cmake",
                "yalantinglibs",
                "config.cmake",
            ),
        ],
        "zstd-header": [
            "/usr/include/zstd.h",
            "/usr/local/include/zstd.h",
        ],
        "zstd-library": [os.path.join(path, "libzstd.so") for path in library_dirs],
        "xxhash-header": [
            "/usr/include/xxhash.h",
            "/usr/local/include/xxhash.h",
        ],
        "xxhash-library": [os.path.join(path, "libxxhash.so") for path in library_dirs],
        "msgpack-cxx": [
            "/usr/include/msgpack.hpp",
            "/usr/local/include/msgpack.hpp",
        ],
    }


def _collect_missing_dependencies(
    dependency_files: dict[str, list[str]],
) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for name, candidate_files in dependency_files.items():
        normalized_candidates = [os.path.abspath(os.path.expanduser(file_path)) for file_path in candidate_files]
        if not _is_dependency_installed(normalized_candidates):
            missing[name] = normalized_candidates
    return missing


def _export_cmake_prefix_paths(prefix_paths: list[str]) -> None:
    existing = os.environ.get("CMAKE_PREFIX_PATH", "")
    merged_paths: list[str] = []

    for path in existing.split(os.pathsep):
        if not path:
            continue
        normalized_path = os.path.abspath(os.path.expanduser(path))
        if normalized_path and normalized_path not in merged_paths:
            merged_paths.append(normalized_path)

    for path in prefix_paths:
        if not path:
            continue
        normalized_path = os.path.abspath(os.path.expanduser(path))
        if normalized_path and normalized_path not in merged_paths:
            merged_paths.append(normalized_path)

    if not merged_paths:
        return

    os.environ["CMAKE_PREFIX_PATH"] = os.pathsep.join(merged_paths)
    logger.info(f"✅ Export CMAKE_PREFIX_PATH to environment: {os.environ['CMAKE_PREFIX_PATH']}")


def _run_dependencies_script_or_exit(script_path: str) -> None:
    if not _run_shell_command(
        _MOONCAKE_DEPENDENCIES_COMMAND,
        cwd=script_path,
        passthrough_output=True,
    ):
        logger.error(f"❌ Run shell command '{_MOONCAKE_DEPENDENCIES_COMMAND}' failed!")
        _print_manual_check_commands(
            [
                f"cd {script_path}",
                _MOONCAKE_DEPENDENCIES_COMMAND,
            ]
        )
        exit(1)


def _validate_submodules_or_exit(repo_root: str) -> None:
    issues = _collect_submodule_init_issues(repo_root)

    if issues:
        logger.error("❌ Submodule commit check failed. Repositories not correctly initialized:")
        for path in sorted(issues):
            logger.error(f"   - {path}: {issues[path]}")
        logger.error("Please align submodules and try again:")
        logger.error("   git submodule update --init --recursive [-f|--force]")
        exit(1)


def _ensure_prebuild_dependencies_installed(
    script_path: str,
    force_install: bool = False,
) -> None:
    dependency_files = _get_required_dependency_files()
    if force_install:
        os.environ["YALANTINGLIBS_PREFIX"] = _get_yalantinglibs_prefix()
        _run_dependencies_script_or_exit(script_path)
    missing_dependencies = _collect_missing_dependencies(dependency_files)
    if missing_dependencies:
        logger.error("Missing third-party build dependencies: %s", ", ".join(sorted(missing_dependencies)))
        _print_manual_check_commands([f"cd {script_path}", _MOONCAKE_DEPENDENCIES_COMMAND])
        sys.exit(1)
    _export_cmake_prefix_paths([_get_yalantinglibs_prefix()])


def ensure_mooncake_go_toolchain(script_path: str, install: bool = False) -> None:
    """Check Go only after CMake has selected a Go-backed HA runtime."""
    mooncake_source_dir = _get_mooncake_source_dir(script_path)
    if not _export_mooncake_go_path(mooncake_source_dir):
        command = "bash scripts/build_support/install_dependencies.sh --ensure-go"
        if install and not _run_shell_command(command, cwd=script_path, passthrough_output=True):
            sys.exit(1)
        if not install or not _export_mooncake_go_path(mooncake_source_dir):
            logger.error(
                "Go %s or newer is required by the enabled Mooncake HA backend.",
                _get_mooncake_go_version(mooncake_source_dir),
            )
            _print_manual_check_commands([f"cd {script_path}", command])
            sys.exit(1)
    _export_mooncake_go_proxy()


def _get_cmake_cache_path() -> str:
    plat_name = sysconfig.get_platform()
    dir_name = f"cmake.{plat_name}-{sys.implementation.name}-{get_python_version()}"
    return os.path.join(get_base_dir(), "build", dir_name, "CMakeCache.txt")


def _get_xllm_ops_marker_path() -> str:
    opp_root = os.getenv("ASCEND_OPP_PATH")
    if not opp_root:
        logger.error("ASCEND_OPP_PATH is not initialized for NPU build.")
        _print_manual_check_commands(["echo $ASCEND_OPP_PATH"])
        exit(1)

    opp_root = os.path.abspath(os.path.expanduser(opp_root))
    # Adapt to the DeepSeekV4 / CANN 9.0 change where xllm_ops installs under the
    # "custom_xllm_math" vendor instead of "xllm". The marker must be read from the
    # same vendor directory CMakeLists.txt writes it to; otherwise the read and
    # write paths diverge and the incremental-build cache never hits. Detection
    # mirrors CMakeLists.txt exactly (same ASCEND_OPP_PATH root and probe path).
    vendor = "xllm"
    if os.path.isdir(os.path.join(opp_root, "vendors", "custom_xllm_math", "op_api", "include", "aclnnop")):
        vendor = "custom_xllm_math"
    return os.path.join(opp_root, "vendors", vendor, ".xllm_ops_git_head")


def _clear_xllm_ops_cache_git_head(cache_path: str) -> bool:
    if not os.path.isfile(cache_path):
        return False

    cache_prefix = "XLLM_OPS_GIT_HEAD_CACHED:"
    with open(cache_path, encoding="utf-8") as cache_file:
        old_lines = cache_file.readlines()

    new_lines = [line for line in old_lines if not line.startswith(cache_prefix)]
    if new_lines == old_lines:
        return False

    temp_file_path = f"{cache_path}.tmp"
    with open(temp_file_path, "w", encoding="utf-8") as cache_file:
        cache_file.writelines(new_lines)
    os.replace(temp_file_path, cache_path)
    return True


def _get_xllm_ops_source_git_head() -> str | None:
    repo_root = _join_path("third_party", "xllm_ops")
    ok, output, err = _run_command(
        ["git", "-c", f"safe.directory={repo_root}", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
    )
    if ok:
        return output.strip() or None
    logger.error(f"❌ Failed to get git HEAD for {repo_root}.")
    if err:
        logger.error(f"   {err}")
    return None


def _read_version_info(file_path: str) -> str | None:
    if not os.path.isfile(file_path):
        return None

    with open(file_path, encoding="utf-8") as version_info:
        for line in version_info:
            if line.startswith("Version="):
                version = line.split("=", 1)[1].strip()
                return version or None
    return None


def _get_cann_version_info() -> tuple[str, str | None]:
    marker_path = _get_xllm_ops_marker_path()
    opp_root = os.path.dirname(os.path.dirname(os.path.dirname(marker_path)))
    version_info_path = os.path.join(
        os.path.dirname(opp_root),
        "compiler",
        "version.info",
    )
    return version_info_path, _read_version_info(version_info_path)


def _parse_major_minor_version(version: str) -> tuple[int, int] | None:
    version_parts = version.strip().split(".")
    if len(version_parts) < 2:
        return None

    try:
        return int(version_parts[0]), int(version_parts[1])
    except ValueError:
        return None


def _ensure_xllm_ops_rebuild_state(device: str) -> None:
    if device != "npu":
        return

    set_npu_envs()
    source_git_head = _get_xllm_ops_source_git_head()
    if source_git_head is None:
        logger.error("❌ Failed to determine third_party/xllm_ops git HEAD.")
        _print_manual_check_commands([f"git -C {_join_path('third_party', 'xllm_ops')} rev-parse HEAD"])
        exit(1)

    marker_path = _get_xllm_ops_marker_path()
    installed_git_head = None
    if os.path.isfile(marker_path):
        with open(marker_path, encoding="utf-8") as marker_file:
            installed_git_head = marker_file.readline().strip() or None
    if installed_git_head == source_git_head:
        os.environ["XLLM_OPS_GIT_HEAD_CACHED"] = source_git_head
        logger.info(f"✅ xllm_ops is already installed in {os.path.dirname(marker_path)} with a matching git HEAD.")
        return

    os.environ.pop("XLLM_OPS_GIT_HEAD_CACHED", None)
    if _clear_xllm_ops_cache_git_head(_get_cmake_cache_path()):
        logger.info("✅ Cleared XLLM_OPS_GIT_HEAD_CACHED from CMake cache to trigger xllm_ops rebuild.")

    if installed_git_head is None:
        logger.info(f"ℹ️ xllm_ops marker not found at {marker_path}. A rebuild is required.")
    else:
        logger.info("ℹ️ Installed xllm_ops marker does not match third_party/xllm_ops HEAD. A rebuild is required.")
        logger.info(f"   installed git HEAD: {installed_git_head}")
        logger.info(f"   source git HEAD:    {source_git_head}")

    version_info_path, cann_version = _get_cann_version_info()
    if cann_version is None:
        logger.error("❌ Cannot find CANN version.info. xllm_ops compilation requires CANN 8.5.")
        _print_manual_check_commands([f"test -f {version_info_path}"])
        exit(1)

    parsed_version = _parse_major_minor_version(cann_version)
    if parsed_version is None:
        logger.error(f"❌ Failed to parse CANN version from {version_info_path}: {cann_version}")
        exit(1)

    if parsed_version < (8, 5):
        logger.error(
            f"❌ xllm_ops compilation requires CANN 8.5 or higher, but detected {cann_version} "
            f"from {version_info_path},"
            f"please use the latest image."
        )
        exit(1)


def pre_build(device: str, force_dependencies: bool = False) -> None:
    script_path = get_base_dir()

    _ensure_git_safe_directory_or_exit(_get_mooncake_source_dir(script_path))
    _validate_submodules_or_exit(script_path)
    _ensure_prebuild_dependencies_installed(script_path, force_install=force_dependencies)
    _ensure_xllm_ops_rebuild_state(device)
