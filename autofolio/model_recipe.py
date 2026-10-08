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


def environment(neural=False):
    import numpy,pandas,sklearn,scipy
    result=dict(python=platform.python_version(),numpy=numpy.__version__,pandas=pandas.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__)
    if neural:
        import torch
        result['torch']=torch.__version__
    return result


def code_hashes():
    return {name:file_hash(Path(__file__).with_name(name)) for name in ['learning.py','neural_models.py','execution.py','learning_input.py','feature_sources.py','model_recipe.py','period.py','evaluation.py']}


def seal(recipe):
    value=dict(recipe);value.pop('recipe_id',None)
    value['recipe_id']=hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    return value


def preserve_code():
    # One source copy per content hash, shared by every experiment using it.
    from .config import RUNS
    import shutil,os,tempfile
    result=None
    for name in ('learning.py','neural_models.py','execution.py'):
        source=Path(__file__).with_name(name);digest=file_hash(source)
        dest=RUNS/'model_code'/digest/name
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True)
            fd,temporary=tempfile.mkstemp(dir=dest.parent,prefix='source-',suffix='.tmp');os.close(fd)
            try:
                shutil.copy2(source,temporary)
                if file_hash(temporary)!=digest:raise ValueError('보존 중 학습 코드 변경 감지')
                os.replace(temporary,dest)
            finally:
                Path(temporary).unlink(missing_ok=True)
        if name=='learning.py':result=str(dest)
    return result


def learner_path(recipe):
    from .config import RUNS
    import re
    digest=recipe['code']['learning.py']
    if not re.fullmatch('[a-f0-9]{64}',digest):raise ValueError('학습 코드 해시 형식 오류')
    saved=RUNS/'model_code'/digest/'learning.py'
    path=saved if saved.exists() else Path(__file__).with_name('learning.py')
    if file_hash(path)!=digest:raise ValueError('보존된 학습 코드가 없거나 변경됨')
    return path


def saved_component(recipe,name):
    import importlib.util,sys,re
    from .config import RUNS
    digest=recipe['code'][name]
    if not re.fullmatch('[a-f0-9]{64}',digest):raise ValueError('학습 코드 해시 형식 오류')
    path=RUNS/'model_code'/digest/name
    if not path.exists():path=Path(__file__).with_name(name)
    if file_hash(path)!=digest:raise ValueError('보존된 학습 의존 코드가 없거나 변경됨')
    key='autofolio._sealed_'+name.removesuffix('.py')+'_'+digest[:16]
    if key in sys.modules:return sys.modules[key]
    spec=importlib.util.spec_from_file_location(key,path);module=importlib.util.module_from_spec(spec)
    sys.modules[key]=module
    try:spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(key,None);raise
    return module


def saved_learner(recipe):
    learner_path(recipe)
    module=saved_component(recipe,'learning.py')
    if 'neural_models.py' in recipe['code'] and recipe.get('definition',{}).get('model') in ('neural_mlp','residual_mlp','lstm','gru','tcn','transformer'):
        module._NEURAL_MODULE=saved_component(recipe,'neural_models.py')
    if 'execution.py' in recipe['code']:module._EXECUTION_MODULE=saved_component(recipe,'execution.py')
    return module


def verify(recipe):
    if seal(recipe)['recipe_id'] != recipe.get('recipe_id'):raise ValueError('모델 학습 명세 해시 불일치')
    if file_hash(recipe['input']['path']) != recipe['input']['sha256']:raise ValueError('동결 입력 데이터 변경됨')
    if recipe['libraries'] != environment(neural='torch' in recipe['libraries']):raise ValueError('학습 라이브러리 버전 변경됨')
    # Frozen input content is verified above. Input-building and reporting code
    # are provenance, not dependencies of the terminal estimator retraining.
    learner_path(recipe)
    for source in recipe.get('auxiliary_inputs',[]):
        if file_hash(source['path'])!=source['sha256']:raise ValueError('동결 보조 입력 데이터 변경됨')
    for name in ('neural_models.py','execution.py'):
        if name in recipe['code']:
            from .config import RUNS
            path=RUNS/'model_code'/recipe['code'][name]/name
            if not path.exists():path=Path(__file__).with_name(name)
            if file_hash(path)!=recipe['code'][name]:raise ValueError('보존된 학습 의존 코드 변경됨')
    return True
