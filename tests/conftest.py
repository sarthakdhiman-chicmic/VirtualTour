import pytest

from tour_worker.config import reset_settings


@pytest.fixture(autouse=True)
def _fresh_settings():
    """Settings are read from the environment once per process; tests that monkeypatch the
    environment get them re-read."""
    reset_settings()
    yield
    reset_settings()
