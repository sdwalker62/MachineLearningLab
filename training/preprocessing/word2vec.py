import numpy as np
import pandas as pd
import tensorflow as tf
import os 
import re

from tensorflow.random import log_uniform_candidate_sampler as negative_skipgrams
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dot, Embedding, Flatten
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

tf.debugging.set_log_device_placement(True)

window_size = int(os.environ["W2V_WINDOW_SIZE"])
embed_size = int(os.environ["W2V_EMBED_SIZE"])

number_negative_sampling = 4

# https://www.tensorflow.org/tutorials/text/word2vec#compile_all_steps_into_one_function
class FullyConnectedNN(Model):
    def __init__(self, vocab_size, embedding_dim):
        super(FullyConnectedNN, self).__init__()
        self.target_embedding = Embedding(vocab_size,
                                        embedding_dim,
                                        input_length=1,
                                        name="w2v_embedding")
        self.context_embedding = Embedding(vocab_size,
                                        embedding_dim,
                                        input_length=number_negative_sampling+1)
        self.dots = Dot(axes=(3, 2))
        self.flatten = Flatten()

    def call(self, pair):
        target, context = pair
        we = self.target_embedding(target)
        ce = self.context_embedding(context)
        dots = self.dots([ce, we])
        return self.flatten(dots)

class Word2Vec:
    def __init__(self, corpus=None, window=window_size, embed_size=embed_size):

        __slots__ = ['corpus', 'word_list', 'vectorized_logs', 'vocabulary', 
                     'inverse_vocabulary', 'embeddings', 'size', 'window_size',
                     'embed_size']

        self.corpus = corpus

        self.word_list = []
        self.vectorized_logs = []
        
        self.vocabulary = {}
        self.inverse_vocabulary = {}
        self.embeddings = {}

        self.targets = []
        self.contexts = []
        self.labels = []

        self.size = 1
        self.window_size = window
        self.embed_size = embed_size

    def collect_vocabulary(self):
        idx = 1
        self.vocabulary['<pad>'] = 0

        # Collect Logging Unique Vocabulary
        for log in self.corpus:
            words = re.split(r'[,\s]', log)
            word_set = list(set(words))
            for token in word_set:
                if token not in self.vocabulary:
                    self.vocabulary[token] = idx
                    idx += 1

        # Generate Inverse Vocabulary
        self.inverse_vocabulary = {idx: token for token, idx in self.vocabulary.items()}

        print(f'The vocabulary size is {len(self.vocabulary)}')

    def vectorize_logs(self):
        for log in self.corpus:
            words = re.split(r'[,\s]', log)
            word_set = list(set(words))
            sequence = [self.vocabulary[word] for word in word_set]
            self.vectorized_logs.append(sequence)

    def find_word_context(self):

        # Build the sampling table for vocab_size tokens.
        sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(len(self.vocabulary))

        for sequence in self.vectorized_logs:

            positive_skip_grams, _ = skipgrams(sequence,
                                               vocabulary_size=len(self.vocabulary),
                                               sampling_table=sampling_table,
                                               window_size=self.window_size,
                                               negative_samples=0)

            for target_word, context_word in positive_skip_grams:
                context_class = tf.expand_dims(
                    tf.constant([context_word], dtype='int64'), 1
                )

                negative_sampling_candidates, _, _ = negative_skipgrams(
                    true_classes=context_class,
                    num_true=1,
                    num_sampled=number_negative_sampling,
                    unique=True,
                    range_max=len(self.vocabulary),
                    seed=42,
                    name="negative_sampling"
                )

                negative_sampling_candidates = tf.expand_dims(
                    negative_sampling_candidates, 1
                )

                context = tf.concat([context_class, negative_sampling_candidates], 0)
                label = tf.constant([1] + [0]*number_negative_sampling, dtype='int64')

                self.targets.append(target_word)
                self.contexts.append(context)
                self.labels.append(label)


    def train_embeddings(self):
        NN = FullyConnectedNN(len(self.vocabulary), 10)

        BATCH_SIZE = 1024
        BUFFER_SIZE = 10000
        dataset = tf.data.Dataset.from_tensor_slices(((self.targets, self.contexts), self.labels))
        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
        dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
        NN.compile(optimizer='adam',
                   loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

        NN.fit(dataset, epochs=2)

    def generate_embeddings(self):
        self.collect_vocabulary()
        self.vectorize_logs()
        self.find_word_context()
        self.train_embeddings()