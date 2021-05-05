from database_methods.database_methods import database_builder
from LogPreprocessor import LogPreprocessor
from WordEmbeddings import WordEmbeddings
import os
import joblib
import time

if __name__ == '__main__':

    # collect logs from database into pandas dataframe
    df = database_builder('/database')


    # create LogPreprocessor object and clean logs and generate templates
    log_preprocessor = LogPreprocessor(df)

    if os.environ["GENERATE_NEW_DRAIN"] == "yes":
        clusters, _ = log_preprocessor.generate_clusters()
    else:
        clusters = joblib.load('/results/clean_clusters.joblib')

    word_embeddings = WordEmbeddings(clusters)
    
    embeddings = word_embeddings.generate_word_embeddings()