from core.ancillary.database_methods import database_builder
from core.training.LogPreprocessor import LogPreprocessor

import string

if __name__ == '__main__':

    # collect logs from database into pandas dataframe
    df = database_builder(r'/Users/dalton/PycharmProjects/log-analyzer/database')

    # create LogPreprocessor object and clean logs and generate templates

    log_preprocessor = LogPreprocessor(df)
    templates_list = log_preprocessor.generate_clusters()

    for template in templates_list:
        print(template + '\n')

    print(log_preprocessor.n_clusters)

