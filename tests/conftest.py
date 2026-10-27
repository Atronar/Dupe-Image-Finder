from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def df():
    """Импортирует библиотеку внутри теста, чтобы падение импорта не ломало сбор тестов."""
    import duplicatefinder
    return duplicatefinder
