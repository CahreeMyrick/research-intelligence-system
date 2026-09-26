import psycopg


def get_connection() -> psycopg.Connection:
    """
    Create a PostgreSQL connection.

    Output:
        psycopg.Connection:
            Active database connection.
    """

    return psycopg.connect(
        "postgresql://localhost:5432/document_intelligence"
    )


