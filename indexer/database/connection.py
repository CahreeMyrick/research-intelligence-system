from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator
import psycopg
from pgvector.psycopg import register_vector

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages PostgreSQL connections and transaction context."""
    def __init__(self, database_url: str):
        self.database_url = database_url

    def get_connection(self) -> psycopg.Connection:
        """Create a new connection with pgvector registered."""
        conn = psycopg.connect(self.database_url, autocommit=False)
        register_vector(conn)
        return conn

    @contextmanager
    def connection(self) -> Generator[psycopg.Connection, None, None]:
        """Context manager yielding an open connection."""
        conn = self.get_connection()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Generator[psycopg.Connection, None, None]:
        """Context manager executing inside an atomic transaction."""
        conn = self.get_connection()
        try:
            with conn.transaction():
                yield conn
        finally:
            conn.close()
