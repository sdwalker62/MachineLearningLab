from core.ancillary.database_methods import database_builder
from core.training.LogPreprocessor import LogPreprocessor

import string

if __name__ == '__main__':

    # collect logs from database into pandas dataframe
    df = database_builder(r'C:\Users\Samue\Dropbox\Work\LAT\database')

    # create LogPreprocessor object and clean logs and generate templates

    log_preprocessor = LogPreprocessor(df)
    templates_list = log_preprocessor.generate_clusters()
    print(templates_list)
    print(log_preprocessor.n_clusters)

