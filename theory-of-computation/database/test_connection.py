from connection import get_connection


with get_connection() as connection:
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1")

        result = cursor.fetchone()

        print(result)
