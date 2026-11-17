import pytest
from pydantic import ValidationError

from filingsqa.config import Settings


def test_defaults_load_without_env():
    settings = Settings()

    assert settings.llm_effort == "low"
    assert settings.anthropic_api_key is None


def test_judge_is_configured_independently_of_generator(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "generator-model")
    monkeypatch.setenv("JUDGE_MODEL", "judge-model")

    settings = Settings()

    assert settings.llm_model == "generator-model"
    assert settings.judge_model == "judge-model"


def test_user_agent_without_email_is_rejected(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "FilingsQA")

    with pytest.raises(ValidationError, match="contact email"):
        Settings()


@pytest.mark.parametrize("rps", [0, 11])
def test_sec_rate_limit_must_stay_within_sec_policy(monkeypatch, rps):
    monkeypatch.setenv("SEC_MAX_REQUESTS_PER_SECOND", str(rps))

    with pytest.raises(ValidationError):
        Settings()


def test_api_key_is_not_leaked_in_repr(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-value")

    settings = Settings()

    assert "sk-secret-value" not in repr(settings)
