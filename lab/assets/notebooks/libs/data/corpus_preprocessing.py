import pandas as pd
from copy import deepcopy
import sys
import logging 
import re 
from tqdm.notebook import tqdm


logging.basicConfig(
    format='%(asctime)s %(levelname)s | %(message)s',
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def clean_logs(logs: pd.DataFrame, column_name: str) -> pd.DataFrame:
    logger.info('Applying log cleaner:')
    
    logs_copy = deepcopy(logs)
    
    logger.info('Cleaning timestamps ...')
    regex_dict = {'timestamps': re.compile(
                    r"""
                    (?:               # Match all enclosed
                    \d{4}-\d{2}-\d{2} # YYYY-MM-DD
                    [\sT]             # Accept either a space or T
                    \d{2}:\d{2}:\d{2} # HH:MM:SS
                    ([.,]\d{3}|\s)    # Accept either a space or milliseconds
                    )                 # End timestamp match
                    | (?:\s{2,})      # Remove double spaces
                    """, re.X),       
                 'unwanted symbols:': re.compile(r'[^./:$a-zA-Z\d\s]', re.X),
                 'unwanted spaces': re.compile(r'\s+', re.X),
                 'isolated symbols': re.compile(
                    r"""
                    ((?<=\s)|(?<=\A)) # Look behind for whitespace (\s) or start of string (\A)
                    [a-zA-Z]{1}       # Match single alphabet chracter both upper and lowercase
                    ((?=\s)|(?=\Z))   # Look ahead for whitespace or end of string (\Z)
                    """, re.X),
                 'unwanted solr o.a.s...': re.compile(r'(o.a.s.(.\.)+)', re.X)}
    
    for key in tqdm(regex_dict.keys(), leave=False):
        logger.info(f'Cleaning {key} ...')
        logs_copy.loc[:, column_name].replace(
            to_replace=regex_dict[key],
            value=' ',
            regex=True,
            inplace=True)
    

    logger.info(f'... Finished cleaning logs, {len(logs_copy.index)} logs processed.')
    return logs_copy


def data_preprocessing_pipeline(data: dict) -> pd.DataFrame:
    logger.info('Starting preprocessing pipeline ...')
    
    ret_dict = dict()
    
    drop_keys = ['filebeat']
    
    for key in tqdm(data.keys(), desc='Walking through database...'):
        logs = data[key]
        out_logs = clean_logs(logs, 'log')
        mask = out_logs[out_logs['container_name'].isin(drop_keys)]
        out_logs.drop(mask.index, inplace=True)
        ret_dict[key] = out_logs
    
    logger.info('... Finished log preprocessing pipeline.')
    return ret_dict