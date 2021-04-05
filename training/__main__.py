from database_methods.database_methods import database_builder
from preprocessing.LogPreprocessor import LogPreprocessor

if __name__ == '__main__':

    # collect logs from database into pandas dataframe
    df = database_builder('/database')


    # create LogPreprocessor object and clean logs and generate templates
    log_preprocessor = LogPreprocessor(df)
    templates_list = log_preprocessor.generate_clusters()

    for template in templates_list:
        if len(template) < 150:
            print(template + '\n')

    print(log_preprocessor.n_clusters)

