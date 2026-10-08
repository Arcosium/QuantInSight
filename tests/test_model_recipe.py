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
