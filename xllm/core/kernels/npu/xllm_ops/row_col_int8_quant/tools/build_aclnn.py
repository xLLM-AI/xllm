"""Build and install the ACLNN package in this answer's private directory."""
from pathlib import Path
import hashlib,os,shutil,subprocess
root=Path(__file__).resolve().parents[1]
if not os.environ.get('ASCEND_HOME_PATH'):raise SystemExit('Source CANN set_env.sh')
base=Path('/tmp/cann-aclnn-build')/hashlib.sha256(str(root).encode()).hexdigest()[:16]
if base.exists():shutil.rmtree(base)
shutil.copytree(root/'aclnn',base,ignore=shutil.ignore_patterns('build_out','__pycache__'))
subprocess.run(['cmake','-S',str(base),'-B',str(base/'build_out'),'--preset=default'],cwd=base,check=True)
subprocess.run(['cmake','--build',str(base/'build_out'),'--target','binary','package','--parallel',os.environ.get('BUILD_JOBS','2')],check=True)
packages=list((base/'build_out').glob('*.run'))
if len(packages)!=1:raise RuntimeError(f'Expected one package, found {packages}')
dest=root/'build';dest.mkdir(exist_ok=True)
shutil.copy2(packages[0],dest/packages[0].name)
install=base/'installed';install.mkdir()
subprocess.run(['bash',str(packages[0]),'--install-path='+str(install)],check=True)
target=root/'aclnn_install'
if target.exists():shutil.rmtree(target)
shutil.copytree(install,target)
print('ACLNN_INSTALL_OK',target)
