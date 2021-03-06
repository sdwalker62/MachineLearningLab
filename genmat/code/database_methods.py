import sqlite3 as sql


def create_connection(db_file: str) -> sql.Connection:
    """
    Creates a database connection
    :param db_file: str
        path to database object
    :return sql.Connection
        a connection to the database
    """
    try:
        conn = sql.connect(db_file)
        print('connection successful')
        return conn
    except sql.Error as e:
        print(e)
