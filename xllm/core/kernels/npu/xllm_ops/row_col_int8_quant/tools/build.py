"""Stage compilation in an ASCII path: CANN's generator cannot quote Unicode paths."""
from pathlib import Path
import hashlib, os, shutil, subprocess, importlib.util

root = Path(__file__).resolve().parents[1]
cann = os.environ.get("ASCEND_HOME_PATH")
if not cann: raise SystemExit("Source CANN set_env.sh first")
key = hashlib.sha256(str(root).encode()).hexdigest()[:16]
base = Path("/tmp/cann-task-build") / key
source, build = base/"src", base/"build"
if base.exists(): shutil.rmtree(base)
source.mkdir(parents=True, exist_ok=True)
for name in ("op_kernel", "op_host", "include", "examples"):
    if (root/name).exists():
        shutil.copytree(root/name, source/name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("._*", "__pycache__"))
shutil.copy2(root/"CMakeLists.txt",source/"CMakeLists.txt")
subprocess.run(["cmake","-S",str(source),"-B",str(build),"-DCMAKE_BUILD_TYPE=Release",
                "-DACLNN_ROOT="+str(root/"aclnn_install/vendors/community_task2"),
                "-DSOC_VERSION="+os.environ.get("SOC_VERSION","Ascend910_9382"),
                "-DASCEND_CANN_PACKAGE_PATH="+cann,
                "-DTORCH_ROOT="+str(Path(importlib.util.find_spec("torch").origin).parent),
                "-DTORCH_NPU_ROOT="+str(Path(importlib.util.find_spec("torch_npu").origin).parent)],check=True)
subprocess.run(["cmake","--build",str(build),"--parallel",os.environ.get("BUILD_JOBS","2")],check=True)
libraries=list(build.rglob("libtask2_kernels.so"))
if not libraries: raise RuntimeError("Compiled shared library not found")
(root/"build").mkdir(exist_ok=True)
shutil.copy2(libraries[0],root/"build/libtask2_kernels.so")
(root/"build/libtask2_kernels.so").chmod(0o755)
print("BUILT",root/"build/libtask2_kernels.so")

shutil.copy2(build/"run_example",root/"build/run_example")
(root/"build/run_example").chmod(0o755)

shutil.copy2(build/"libtask2_torch.so",root/"build/libtask2_torch.so")
(root/"build/libtask2_torch.so").chmod(0o755)

shutil.copy2(build/"run_aclnn_example",root/"build/run_aclnn_example")
(root/"build/run_aclnn_example").chmod(0o755)
