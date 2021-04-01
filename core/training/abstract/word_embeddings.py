import pandas as pd
import tensorflow as tf

from abc import ABC, abstractmethod


class LogPreprocessor(ABC):

    @abstractmethod
    def standardize(self, logs: pd.DataFrame) -> pd.DataFrame:
        """
        A majority of the logs will need to be scrubbed of unwanted characters
        in preparation for generating log templates.
        :param logs: a pandas dataframe of logs from the collections database
        :return: a pandas dataframe containing the cleaned logs
        """
        pass


class TemplateGenerator(ABC):

    @abstractmethod
    def generate(self, logs: pd.DataFrame) -> list:
        """
        This will more than likely be handled by Drain but I want to leave this open for
        better models in the future. It is possible to training a new transformer which
        outputs to a CFG that will avoid mis-clustering, but that is highly theoretical.
        :param logs: a pandas dataframe of logs cleaned by the LogPreprocessor
        :return: a list of templates which will be used to generate word embeddings
        """
        pass


class WordEmbedder(ABC):

    @abstractmethod
    def generate_sequences(self, templates: list) -> tf.data.Dataset:
        pass
