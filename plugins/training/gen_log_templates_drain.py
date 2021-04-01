import logging
import pandas as pd

from core.training.abstract.word_embeddings import TemplateGenerator
from drain3.drain import Drain

logging.basicConfig(
    format='%(asctime)s %(levelname)s | %(message)s',
    level=logging.INFO)

logger = logging.getLogger('drain')


class TransformerTemplateGenerator(TemplateGenerator):

    def __init__(self):
        self.clusters = {}
        self.n_clusters = 0
        self.cluster_list = []

    def generate(self, logs: pd.DataFrame) -> list:
        drain_model = Drain()

        logger.info('Generating Drain model ...')

        for row in logs.itertuples():
            drain_model.add_log_message(row.log)

        logger.info('...complete!')

        self.clusters = drain_model.clusters
        self.n_clusters = len(drain_model.clusters)

        return [cluster.get_template() for cluster in drain_model.clusters]
