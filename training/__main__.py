from database_methods.database_methods import database_builder
from preprocessing.LogPreprocessor import LogPreprocessor

import re

if __name__ == '__main__':

    # collect logs from database into pandas dataframe
    df = database_builder('/database')


    # create LogPreprocessor object and clean logs and generate templates
    log_preprocessor = LogPreprocessor(df)
    log_preprocessor.generate_word_embeddings()

    # "change_type": change_type,
    # "cluster_id": cluster.cluster_id,
    # "cluster_size": cluster.size,
    # "cluster_example": log_message,
    # "template_mined": cluster.get_template(),
    # "cluster_count": len(self.drain.clusters)
    
    # for result in log_preprocessor.results.values():
    #     template = result["template_mined"]
    #     if len(template) < 500:
    #         if re.search(r'(<\*>\s{0,}){1,}', template):
    #             print(f'Cluster Id: {result["cluster_id"]} \n' \
    #                 f'Cluster Example: {result["cluster_example"]} \n' \
    #                 f'Cluster Template: {template} \n\n')

    # for cluster in clusters:
    #     idx = cluster.cluster_id
    #     print(f'template: {cluster.get_template()}\n log: {log_preprocessor} \n\n')