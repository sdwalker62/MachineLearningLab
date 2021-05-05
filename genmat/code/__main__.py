import numpy as np
import pandas as pd
import math as math
import os
import joblib
import time
from genmat.mod_drain.drain import Drain
from datetime import timedelta
from collections import namedtuple
from genmat.database_methods import database_builder
from tabulate import tabulate


def genmat(df: pd.DataFrame, tf: timedelta):
    """
    Each cluster will
    :param
        df: pd.DataFrame
            Pandas dataframe where each row is an example with time stamp, container name, log, and label as columns
        tf: datetime.timedelta
            Time frame for clustering arrivals

    :return
        ret_dict: dict
            Dictionary of integers representing the number of observations for each
            (label, container, cluster) coordinate of the arrivals dictionary falling
            within the timeframe determined by time_frame.
    """

    fmt = '%Y-%m-%dT%H:%M:%S.%f'
    df['timestamp'] = pd.to_datetime(df['timestamp'], format=fmt)
    labels = set(df['label'].unique().tolist())

    all_containers = set(df['container_name'].unique().tolist())
    forbidden_containers = {'solr'}
    considered_containers = all_containers - forbidden_containers

    # the goal here is to make a drain model for each container that has clusters for all observations
    # from all collections so that we may see which receive messages during specific collection runs

    dd = {container: Drain() for container in considered_containers}

    total_number_of_logs = 0
    for log in df.itertuples():
        if log.container_name not in forbidden_containers:
            dd[log.container_name].add_log_message(log.log)
            total_number_of_logs += 1

    # now that we have these drain models we can analyze each collection:

    # pd.DataFrame with True in the index where a label has changed and false otherwise
    transitions = df['label'] != df['label'].shift(axis=0)

    # numpy.ndarray of indices where the transition occurred, i.e. where the True values are in the
    # transitions dataframe
    transitions_indices = np.where(transitions)[0]

    # determine where each collection begins
    ce = list(transitions_indices[::len(labels)])
    ce.append(len(df.index))
    n_collections = len(ce)

    # split the dataframe into sub-dataframes by collection
    dfs = {idx: df.iloc[ce[idx] + 1:ce[idx + 1] + 1, :].sort_values(by='timestamp') for idx in range(n_collections - 1)}


    col_lens = []
    col_count = 0 
    total_number_of_logs = 0
    for df in dfs.values():
        col_count += 1
        col_lens.append([col_count, len(df.index)])
        total_number_of_logs += len(df[df['container_name'] !='solr'].index)

    headers = [['Collection', '#(logs)']]
    col_lens = headers + col_lens
    col_lens += [['########','########']]
    col_lens += [['Total', total_number_of_logs]]

    print(tabulate(col_lens, headers=("firstrow"), missingval="-"))

    # build the cluster count dictionary
    had = {idx: {label: {container: np.array for container in considered_containers} for label in labels}
           for idx in dfs.keys()}

    hd = {idx: {label: {container: {} for container in considered_containers} for label in labels}
          for idx in dfs.keys()}

    for idx, di in dfs.items():
        for label in labels:
            dl = di[di['label'] == label]
            start_time = dl['timestamp'].iloc[0]
            end_time = dl['timestamp'].iloc[-1]
            n_intervals = math.ceil((end_time - start_time) / tf)
            if n_intervals < 1:
                print('number of intervals is less than one!')
                break
            for container in considered_containers:
                logs = dl[dl['container_name'] == container]
                drain_model = dd[container]
                counter_array = np.zeros((len(drain_model.clusters), n_intervals))
                for log in logs.itertuples():
                    time_stamp = log.timestamp
                    # instead of iterating over each time period we can calculate which period the log arrived
                    col = math.floor((time_stamp - start_time) / tf)
                    cluster = drain_model.find_cluster(log.log)
                    if not cluster is None:
                        counter_array[cluster.cluster_id - 1, col] += 1
                    else:
                        print(f'Could not find matching cluster for log: {log.log}')
                    if cluster not in hd[idx][label][container].keys():
                        hd[idx][label][container][cluster] = [cluster.cluster_id, 0]
                    hd[idx][label][container][cluster][1] += 1
                had[idx][label][container] = counter_array

    empty_cluster = False
    total_number_of_hits = 0 
    num_of_empty_clusters = 0
    for idx in dfs.keys():
        for label in labels:
            for container in considered_containers:
                for cluster_vec in hd[idx][label][container].values():
                    if cluster_vec[1] == 0:
                        empty_cluster = True 
                        num_of_empty_clusters += 1
                    total_number_of_hits += cluster_vec[1]

    if empty_cluster:
        print('Empty clusters found!')
        print(f'Number of empty clusters: {num_of_empty_clusters}')
    else:
        print('No empty clusters found!')    
    
    print(f'Number of total cluster hits: {total_number_of_hits}')

    return labels, considered_containers, dfs.keys(), had, hd, dd


def main():
    path = os.environ["DB_PATH"]
    epoch = os.environ["TIME_FRAME"]

    df = database_builder(path)
    tf = timedelta(minutes=int(epoch))

    print("Starting matrix creation...")
    labels, containers, collections, had, hd, dd = genmat(df, tf)
    print("... matrix creation complete!")

    print("Saving results...")
    joblib.dump(labels, "/code/results/labels.joblib")
    joblib.dump(containers, "/code/results/containers.joblib")
    joblib.dump(set(collections), "/code/results/collections.joblib")
    joblib.dump(had, "/code/results/matrices_dict.joblib")
    joblib.dump(hd, "/code/results/hits_dict.joblib")
    joblib.dump(dd, "/code/results/drain_dict.joblib")
    print("... saving complete!")


if __name__ == '__main__':
    start = time.time()
    main()
    print(f"Runtime: {time.time()-start}")
