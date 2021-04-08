import logging
import pandas as pd
import re
import os
import joblib

from preprocessing.word2vec import Word2Vec
from drain3 import TemplateMiner


logging.basicConfig(format='%(asctime)s %(levelname)s | %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class LogPreprocessor:

    def __init__(self, logs: pd.DataFrame):
        self.logs = logs
        self.template_miner = TemplateMiner()
        self.cleaned_logs = pd.DataFrame
        self.clusters = {}
        self.results = {}
        self.n_clusters = 0
        self.word_2_vec = Word2Vec()

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
        #logs['log'] = logs['log'].apply(lambda log: self.clean_solr_logs(log))

        # remove punctuation
        #logs['log'] = logs['log'].replace(to_replace=r'[^\w\s]',
        #                                  value=' ',
        #                                  regex=True)

        logger.info('...complete!')

        return logs

    def generate_clusters(self):
        self.cleaned_logs = self.standardize(self.logs)
        logger.info('Generating log templates ...')

        for idx, row in enumerate(self.cleaned_logs.itertuples()):
            self.results[idx] = self.template_miner.add_log_message(row.log)

        self.clusters = self.template_miner.drain.clusters
        self.n_clusters = len(self.template_miner.drain.clusters)

        # cleaned_clusters = [re.sub(pattern=r'[^\w\s]',
        #                            repl=' ',
        #                            string=cluster.get_template())
        #                     for cluster in self.Drain.drain.clusters]

        cleaned_clusters = [re.sub(pattern=r' +',
                                   repl=' ',
                                   string=cluster.get_template())
                            for cluster in self.template_miner.drain.clusters]

        logger.info('...complete!')
        joblib.dump(cleaned_clusters, '/results/clean_clusters.joblib')
        return cleaned_clusters, self.template_miner.drain.clusters

    def generate_word_embeddings(self):
        logger.info('Generating Word Embeddings ...')
        
        if os.environ["GENERATE_NEW_DRAIN"] == "yes":
            clusters, _ = self.generate_clusters()
        else:
            clusters = joblib.load('/results/clean_clusters.joblib')

        self.word_2_vec.corpus = clusters
        self.word_2_vec.generate_embeddings()

