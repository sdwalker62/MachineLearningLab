import logging
from Word2Vec import Word2Vec

logging.basicConfig(format='%(asctime)s %(levelname)s | %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class WordEmbeddings:
    def __init__(self, clusters):
        self.clusters = clusters
        self.word_2_vec = Word2Vec()

    def generate_word_embeddings(self):
        logger.info('Generating Word Embeddings ...')

        self.word_2_vec.corpus = self.clusters
        self.word_2_vec.generate_embeddings()

        logger.info('...complete!')

        return self.word_2_vec.embeddings