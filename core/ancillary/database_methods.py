import sqlite3 as sql
import pandas as pd
import os
import logging


logging.basicConfig(format='%(asctime)s %(levelname)s | %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def database_builder(path: str) -> pd.DataFrame():
    logger.info('Building DataFrame ...')
    (_, _, files) = next(os.walk(path))
    sql_query = 'SELECT * FROM logs'
    data = []
    for f in files:
        if '.db' in f:
            conn = create_connection(path + '\\' + f)
            d = pd.read_sql_query(sql_query, conn)
            data.append(d)
    logger.info('...complete!')
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
        logger.info('Connected to database ' + path)
        return conn
    except sql.Error as e:
        logger.warning(e)


