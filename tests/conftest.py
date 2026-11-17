import pytest

from filingsqa.config import get_settings


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Keep tests independent of the developer's real `.env` and environment."""
    monkeypatch.chdir(tmp_path)  # no .env in tmp_path
    for var in ("ANTHROPIC_API_KEY", "LLM_MODEL", "JUDGE_MODEL", "SEC_USER_AGENT"):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
