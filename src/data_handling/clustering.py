import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, OPTICS, MeanShift
import logging
logging.basicConfig(level=logging.INFO)
import hdbscan
from collections import defaultdict


def k_means_clustering(config):
    pca_transformed_df = pd.read_csv(config["output_directory"] + "pca_transformed_data.csv")
    columns_for_clustering = [e for e in list(pca_transformed_df.columns) if e not in ('Unnamed: 0', "video_id", "timestamp")]

    logging.info("Perform KMeans clustering...")
    kmeans = KMeans(n_clusters=config["k_means_config"]["clusters"],
        n_init=config["k_means_config"]["n_init"], 
        max_iter=config["k_means_config"]["max_iter"], 
        random_state=config["k_means_config"]["random_state"])

    kmeans = kmeans.fit(pca_transformed_df[columns_for_clustering])
    pca_transformed_df.loc[:,"cluster"] = kmeans.labels_

    only_cluster_df = pca_transformed_df.drop(columns=columns_for_clustering)
    #only_cluster_df = pca_transformed_df

    logging.info("Finished Clustering")
    logging.info(only_cluster_df["cluster"].value_counts())

    return only_cluster_df, columns_for_clustering

def dbscan_clustering(config): 
    pca_transformed_df = pd.read_csv(config["output_directory"] + "pca_transformed_data.csv")
    columns_for_clustering = [e for e in list(pca_transformed_df.columns) if e not in ('Unnamed: 0', "video_id", "timestamp")]
    
    logging.info("Perform DBScan clustering...")

    dbscan = DBSCAN(eps=config["dbscan_config"]["eps"], 
        min_samples=config["dbscan_config"]["min_samples"], algorithm= "kd_tree").fit(pca_transformed_df[columns_for_clustering])

    pca_transformed_df.loc[:,"cluster"] = dbscan.labels_

    only_cluster_df = pca_transformed_df

    logging.info("Finished Clustering")
    logging.info(only_cluster_df["cluster"].value_counts())

    only_cluster_df = only_cluster_df[only_cluster_df.cluster != -1]
    only_cluster_df.to_csv(config["output_directory"] + "clustered_data.csv")

    #print(only_cluster_df)

def optics_clustering(config):
    pca_transformed_df = pd.read_csv(config["output_directory"] + "pca_transformed_data.csv")
    columns_for_clustering = [e for e in list(pca_transformed_df.columns) if e not in ('Unnamed: 0', "video_id", "timestamp")]
    
    logging.info("Perform Optics clustering...")

    dbscan = OPTICS(min_samples=config["dbscan_config"]["min_samples"]).fit(pca_transformed_df[columns_for_clustering])

    pca_transformed_df.loc[:,"cluster"] = dbscan.labels_

    only_cluster_df = pca_transformed_df

    logging.info("Finished Clustering")
    logging.info(only_cluster_df["cluster"].value_counts())

    only_cluster_df = only_cluster_df[only_cluster_df.cluster != -1]
    only_cluster_df.to_csv(config["output_directory"] + "clustered_data.csv")

def meanshift_clustering(config):
    pca_transformed_df = pd.read_csv(config["output_directory"] + "pca_transformed_data.csv")
    columns_for_clustering = [e for e in list(pca_transformed_df.columns) if e not in ('Unnamed: 0', "video_id", "timestamp")]
    
    logging.info("Perform MeanShift clustering...")

    clustering = MeanShift().fit(pca_transformed_df[columns_for_clustering])

    pca_transformed_df.loc[:,"cluster"] = clustering.labels_

    only_cluster_df = pca_transformed_df

    logging.info("Finished Clustering")
    logging.info(only_cluster_df["cluster"].value_counts())

    only_cluster_df.to_csv(config["output_directory"] + "clustered_data.csv")

def reattach_noise(pca_transformed_df, columns_for_clustering, clustering_fit, config):

    pca_transformed_df["altered"] = 0
    pca_transformed_df["altered_probab"] = 0

    noise = pca_transformed_df[pca_transformed_df["cluster"] == -1]
    noise_pca_columns = noise[columns_for_clustering]
    noise_pca_columns_trans = np.array([tuple(row) for row in noise_pca_columns.to_records(index=False)])

    mem_vector = hdbscan.membership_vector(clustering_fit, noise_pca_columns_trans)
    mem_vector = [[(round(value * 100, 2), idx) for idx, value in enumerate(sublist)] for sublist in mem_vector]
    length_mem = len(mem_vector)

    for idx, point in enumerate(mem_vector):

        if idx % 1000 == 0:
            print("{}/{}".format(str(idx),str(length_mem)),flush=True)

        sorted_list = sorted(point, key=lambda x: x[0], reverse=True)
        if sorted_list[0][0] > config["hdbscan_config"]["similarity_threshold"]:
            video_id = str(noise.iloc[idx]["video_id"])
            timestamp = float(noise.iloc[[idx]]["timestamp"])

            pca_transformed_df.loc[(pca_transformed_df['video_id'] == video_id) & 
                (pca_transformed_df['timestamp'] == timestamp), "cluster"] = int(sorted_list[0][1])

            pca_transformed_df.loc[(pca_transformed_df['video_id'] == video_id) & 
                (pca_transformed_df['timestamp'] == timestamp), "altered"] = 1

            pca_transformed_df.loc[(pca_transformed_df['video_id'] == video_id) & 
                (pca_transformed_df['timestamp'] == timestamp), "altered_probab"] = sorted_list[0][0]
    
    return pca_transformed_df

def hdbscan_clustering(config):
    
    pca_transformed_df = pd.read_csv(config["output_directory"] + "pca_transformed_data.csv")
    columns_for_clustering = [e for e in list(pca_transformed_df.columns) if e not in ('Unnamed: 0', "video_id", "timestamp")]

    #clustering = hdbscan.HDBSCAN(
    #    min_cluster_size=config["hdbscan_config"]["min_cluster_size"],
    #    min_samples=config["hdbscan_config"]["min_samples"]).fit_predict(pca_transformed_df[columns_for_clustering])

    clustering_fit = hdbscan.HDBSCAN(
        min_cluster_size=config["hdbscan_config"]["min_cluster_size"],
        min_samples=config["hdbscan_config"]["min_samples"],
        prediction_data=True).fit(pca_transformed_df[columns_for_clustering])

    pca_transformed_df.loc[:,"cluster"] = clustering_fit.labels_

    logging.info(pca_transformed_df["cluster"].value_counts())

    clustered_df = reattach_noise(pca_transformed_df, columns_for_clustering, clustering_fit, config)
    clustered_df = clustered_df.drop(columns=columns_for_clustering)

    logging.info("Finished Clustering")
    logging.info(clustered_df["cluster"].value_counts())

    return clustered_df, columns_for_clustering