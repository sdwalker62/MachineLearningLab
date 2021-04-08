import numpy as np
import pandas as pd
from keras.models import Input, Model
from keras.layers import Dense

class Word2Vec:
    def __init__(self, corpus=None, window=2, embed_size=2):
        self.corpus = corpus
        self.word_list = []
        self.unique_words = {}
        self.embeddings = {}
        self.size = 1
        self.window_size = window
        self.embed_size = embed_size

    # def find_word_context(self):
    #     for text in self.corpus:


