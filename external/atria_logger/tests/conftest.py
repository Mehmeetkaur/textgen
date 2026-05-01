# ----------------------------
# Helper to reset env vars
# ----------------------------
import pytest


@pytest.fixture(autouse=True)
def reset_env(monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore
    monkeypatch.delenv("ATRIA_LOG_LEVEL", raising=False)
    monkeypatch.delenv("ATRIA_RANK", raising=False)
    monkeypatch.delenv("ATRIA_LOG_FILE", raising=False)
    yield
