from __future__ import annotations

import pytest

from indexer.database.connection import DatabaseManager
from indexer.database.repository import IndexRepository


TEST_DB_URL = "postgresql://localhost:5432/research_intel"


@pytest.fixture
def repo() -> IndexRepository:
    db = DatabaseManager(TEST_DB_URL)
    repository = IndexRepository(db)
    repository.init_schema()
    return repository
