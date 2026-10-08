import pytest
from autofolio import model_recipe


def test_sealed_recipe_rejects_mutation(tmp_path):
    p=tmp_path/'data';p.write_text('fixed')
    r=model_recipe.seal(dict(input=dict(path=str(p),sha256=model_recipe.file_hash(p)),libraries=model_recipe.environment(),code=model_recipe.code_hashes()))
    assert model_recipe.verify(r)
    broken=dict(r,weights_retained=True)
    with pytest.raises(ValueError,match='명세'):model_recipe.verify(broken)
    p.write_text('changed')
    with pytest.raises(ValueError,match='데이터'):model_recipe.verify(r)


def test_saved_training_source_survives_application_changes(tmp_path,monkeypatch):
    from autofolio import config,model_recipe
    monkeypatch.setattr(config,'RUNS',tmp_path)
    source=b'def identity(): return 42\n'
    import hashlib
    digest=hashlib.sha256(source).hexdigest()
    path=tmp_path/'model_code'/digest/'learning.py';path.parent.mkdir(parents=True);path.write_bytes(source)
    recipe={'code':{'learning.py':digest}}
    assert model_recipe.saved_learner(recipe).identity()==42
    path.write_bytes(b'def identity(): return 0\n')
    import pytest
    with pytest.raises(ValueError):model_recipe.saved_learner(recipe)


def test_archived_neural_and_execution_restore_after_module_unload(tmp_path,monkeypatch):
    import sys
    import numpy as np
    import joblib
    from autofolio import config
    monkeypatch.setattr(config,'RUNS',tmp_path)
    model_recipe.preserve_code()
    recipe={'code':model_recipe.code_hashes(),'definition':{'model':'neural_mlp'}}
    learner=model_recipe.saved_learner(recipe)
    assert learner._EXECUTION_MODULE.__name__.startswith('autofolio._sealed_execution_')
    module=learner._NEURAL_MODULE
    model=module.NeuralRegressor(input_features=1,sequence_length=1)
    x=np.array([[0.],[1.],[2.],[3.]])
    model.fit(x,np.array([.1,.2,.3,.4]))
    expected=model.predict(x)
    path=tmp_path/'weights.joblib';joblib.dump(model,path)
    sys.modules.pop(module.__name__)
    model_recipe.saved_learner(recipe)
    np.testing.assert_array_equal(joblib.load(path).predict(x),expected)
    archive=tmp_path/'model_code'/recipe['code']['execution.py']/'execution.py'
    archive.write_text('raise RuntimeError("changed")')
    with pytest.raises(ValueError,match='변경'):
        model_recipe.saved_learner(recipe)
