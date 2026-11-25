import os,sys,subprocess,importlib.util,importlib.metadata as package_metadata
from pathlib import Path
COLAB_MODE=bool(importlib.util.find_spec('google.colab')) if importlib.util.find_spec('google') else False
if COLAB_MODE:
    subprocess.check_call([sys.executable,'-m','pip','install','-q','numpy>=1.26','pandas>=2.2','scipy>=1.13','statsmodels>=0.14','scikit-learn>=1.5','xgboost>=2.1','matplotlib>=3.9','prophet>=1.1'])
    try:
        current_torch=tuple(int(x) for x in package_metadata.version('torch').split('+')[0].split('.')[:2])
    except package_metadata.PackageNotFoundError:
        current_torch=(0,0)
    if current_torch<(2,4):
        subprocess.check_call([sys.executable,'-m','pip','install','-q','torch>=2.4','--index-url','https://download.pytorch.org/whl/cpu'])
print('CPU-only experiment; no paid services. Colab mode:',COLAB_MODE)
