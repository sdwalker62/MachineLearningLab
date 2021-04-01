import string
import pandas as pd
import logging
import re

from core.training.abstract.word_embeddings import LogPreprocessor


logging.basicConfig(
    format='%(asctime)s %(levelname)s | %(message)s',
    level=logging.INFO)

logger = logging.getLogger('log-preprocessor')


class TransformerLogPreprocessor(LogPreprocessor):

    def standardize(self, logs: pd.DataFrame) -> pd.DataFrame:

        reg_exp_log = '|INFO|WARNING|DANGER|ERROR|WARN'
        reg_exp_punctuation = string.punctuation

        logger.info('Standardizing log documents ...')

        logs['log'] = logs['log'].apply(lambda log: re.sub(reg_exp_log + reg_exp_punctuation, ' ', log))

        logger.info('... complete!')

        return logs
