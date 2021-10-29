import pandas as pd
import logging
import sys
import os


logging.basicConfig(
    format='%(asctime)s %(levelname)s | %(message)s',
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def get_data(path: str) -> pd.DataFrame():
    ret_data = dict()
    logger.info('Building data dictionary ...')
    _, dirs, _ = next(os.walk(path))
    for directory in dirs:
        logging.info(f'directory found: {directory}')
        _, _, files = next(os.walk(f'{path}/{directory}'))
        for file in files:
            accpt_params = ['.csv', 'base', 'Logs']
            unaccpt_params = ['.~lock', 'OLD']
            if (all(p in file for p in accpt_params) and
                all(p not in file for p in unaccpt_params)):
                logging.info(f'file found: {file}')
                data = pd.read_csv(f'{path}/{directory}/{file}')
                #data = clean_logs('log', data)                           # clean logs by removing internal timestamps
                data['timestamp'] = pd.to_numeric(data['timestamp'])
                data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')  # replace time since unix time with actual datetime obj
                data.set_index('timestamp', inplace=True)   # set index to be the timestamp
                data.sort_index(inplace=True)
                ret_data[directory] = data
    logger.info('...complete!')
    return ret_data