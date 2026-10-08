"""Reproducibility manifests contain references and hashes, never trial weights."""
import hashlib
import json
import platform
from pathlib import Path


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def environment():
    import numpy,pandas,sklearn,scipy
    return dict(python=platform.python_version(),numpy=numpy.__version__,pandas=pandas.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__)


def code_hashes():
    return {name:file_hash(Path(__file__).with_name(name)) for name in ['learning.py','learning_input.py','model_recipe.py','period.py']}


def seal(recipe):
    value=dict(recipe);value.pop('recipe_id',None)
    value['recipe_id']=hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    return value


def preserve_code():
    # One source copy per content hash, shared by every experiment using it.
    from .config import RUNS
    import shutil
    source=Path(__file__).with_name('learning.py');digest=file_hash(source)
    dest=RUNS/'model_code'/digest/'learning.py'
    if not dest.exists():
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    return str(dest)


def learner_path(recipe):
    from .config import RUNS
    import re
    digest=recipe['code']['learning.py']
    if not re.fullmatch('[a-f0-9]{64}',digest):raise ValueError('학습 코드 해시 형식 오류')
    saved=RUNS/'model_code'/digest/'learning.py'
    path=saved if saved.exists() else Path(__file__).with_name('learning.py')
    if file_hash(path)!=digest:raise ValueError('보존된 학습 코드가 없거나 변경됨')
    return path


def saved_learner(recipe):
    import importlib.util
    path=learner_path(recipe)
    spec=importlib.util.spec_from_file_location('autofolio._sealed_learner_'+recipe['code']['learning.py'][:16],path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def verify(recipe):
    if seal(recipe)['recipe_id'] != recipe.get('recipe_id'):raise ValueError('모델 학습 명세 해시 불일치')
    if file_hash(recipe['input']['path']) != recipe['input']['sha256']:raise ValueError('동결 입력 데이터 변경됨')
    if recipe['libraries'] != environment():raise ValueError('학습 라이브러리 버전 변경됨')
    # Frozen input content is verified above. Input-building and reporting code
    # are provenance, not dependencies of the terminal estimator retraining.
    learner_path(recipe)
    return True
