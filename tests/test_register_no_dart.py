import ast
import re
from pathlib import Path


def test_dart_module_uses_only_env_config_key():
    src = Path("tools/dart_disclosure.py").read_text(encoding="utf-8")
    assert "from config import OPENDART_API_KEY" in src
    # crtfc_key 에 20자+ 리터럴이 직접 박히면(키 유출) 차단
    assert not re.search(r'crtfc_key["\']\s*:\s*["\'][A-Za-z0-9]{20,}', src)


def test_application_cannot_register_new_users():
    from server.app import app
    assert "/api/register" not in {r.path for r in app.routes}
