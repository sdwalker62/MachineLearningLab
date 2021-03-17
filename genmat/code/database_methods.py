import sqlite3 as sql


def create_connection(path: str) -> sql.Connection:
    """
    Creates a database connection
    :param path: str
        path to database object
    :return sql.Connection
        a connection to the database
    """
    try:
        conn = sql.connect(path)
        print('connection successful')
        return conn
    except sql.Error as e:
        print(e)
