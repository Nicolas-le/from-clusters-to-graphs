import pandas as pd
import os
from datetime import datetime
import json

import logging
logging.basicConfig(level=logging.INFO)

import data_preprocessing
import pca
import clustering
from graph import ClusterGraph

def handle_config():
    config = {
        "source": "tagesschau",
        "embeddings": "./resources/embeddings/",
        "images": "./resources/images/",
        "output_directory": "./outputs/" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + "/",
        "initial_transformation": False,
        "principal_components": 30,
        "clustering": "hdbscan",
        "k_means_config": {
            "clusters": 92,
            "n_init": 3,
            "max_iter": 3000,
            "random_state": 1
        },
        "dbscan_config": {
            "eps": 1, # 2*dim
            "min_samples": 15 # larger dataset higher, >= dims of data (pca)
        },
        "hdbscan_config":{
            "min_cluster_size": 100,
            "min_samples": 100,
            "similarity_threshold": 10
        }
    }

    os.mkdir(config["output_directory"])
    with open(config["output_directory"]+"/config.json", 'w') as convert_file:
        convert_file.write(json.dumps(config))
    
    return config

if __name__ == "__main__":
    config = handle_config()

    logging.info(str(config))

    if config["initial_transformation"]:
        logging.info("Data Transformation...")
        data_preprocessing.get_embeddings(config)
    
    transformed_data_path = "./resources/transformed_embeddings/"+ config["source"] + ".csv"
    transformed_data = pd.read_csv(transformed_data_path).drop(columns=["Unnamed: 0"])

    logging.info("{} Embeddings to cluster.".format(str(len(transformed_data))))
       
    logging.info("Start PCA...")
    pca.pca_main(transformed_data, config)

    logging.info("Start Clustering...")
    if config["clustering"] == "kmeans":
        clustered_df, columns_for_clustering = clustering.k_means_clustering(config)
    elif config["clustering"] == "dbscan":
        clustering.dbscan_clustering(config)
    elif config["clustering"] == "mean_shift":
        clustering.meanshift_clustering(config)
    elif config["clustering"] == "hdbscan":
        clustered_df, columns_for_clustering = clustering.hdbscan_clustering(config)
        clustered_df = clustered_df[clustered_df["cluster"] != -1]

    
    clustered_df.to_csv(config["output_directory"] + "clustered_data.csv")

    logging.info("Finished Clustering for all videos...")
    
    g = ClusterGraph(config["output_directory"],
        low_cluster_filter = 5,
        community_resolution = 1.2,
        edge_threshold=2
        )
    
    g.save_to_json()
