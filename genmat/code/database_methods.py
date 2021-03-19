import sqlite3 as sql
import pandas as pd
import os


def database_builder(path: str) -> pd.DataFrame():
    print('Constructing main DataFrame...')
    (_, _, files) = next(os.walk(path))
    head = 'database/'
    df = pd.DataFrame()
    sql_query = 'SELECT * FROM logs'
    data = []
    for f in files:
        conn = create_connection(head + f)
        d = pd.read_sql_query(sql_query, conn)
        data.append(d)
    print('...construction complete!')
    return pd.concat(data)


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
        return conn
    except sql.Error as e:
        print(e)

