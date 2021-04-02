import logging
import pandas as pd
import string
import re

import drain3 as d3


logging.basicConfig(format='%(asctime)s %(levelname)s | %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class LogPreprocessor:

    def __init__(self, logs: pd.DataFrame):
        self.logs = logs
        self.Drain = d3.TemplateMiner()
        self.cleaned_logs = pd.DataFrame
        self.clusters = {}
        self.n_clusters = 0

    @staticmethod
    def clean_solr_logs(s: str) -> str:
        if len(s) == 33 or len(s) == 32:
            if 'zoo' in s or 'solr' in s:
                s = s[:8] + ' ' + s[9:22] + ' ' + s[22:]

        return s

    def standardize(self, logs: pd.DataFrame) -> pd.DataFrame:
        fmt = '%Y-%m-%dT%H:%M:%S.%f'
        logs['timestamp'] = pd.to_datetime(logs['timestamp'], format=fmt)

        logger.info('Standardizing log documents ...')

        # remove timestamps
        logs['log'] = logs['log'].replace(to_replace=r'(?:\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}([.,]\d{3}|\s))',
                                          value='',
                                          regex=True)
        logs['log'] = logs['log'].apply(lambda log: self.clean_solr_logs(log))

        # remove punctuation
        #logs['log'] = logs['log'].replace(to_replace=r'[^\w\s]',
        #                                  value=' ',
        #                                  regex=True)

        logger.info('...complete!')

        return logs

    def generate_clusters(self) -> list:
        self.cleaned_logs = self.standardize(self.logs)
        logger.info('Generating Drain model ...')

        for row in self.cleaned_logs.itertuples():
            self.Drain.add_log_message(row.log)

        logger.info('...complete!')

        self.clusters = self.Drain.drain.clusters
        self.n_clusters = len(self.Drain.drain.clusters)

        return [cluster.get_template() for cluster in self.Drain.drain.clusters]

