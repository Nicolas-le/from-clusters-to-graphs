import networkx as nx
from networkx.algorithms import community
from collections import defaultdict
import pandas as pd
from itertools import chain
import os
import json
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)


class ClusterGraph():

    def __init__(self,
    frames_with_clusters,
    low_cluster_filter,
    community_resolution,
    #columns_for_clustering,
    edge_threshold_corr,
    edge_threshold_count):

        self.run_path = frames_with_clusters
        self.frames_with_clusters = pd.read_csv(frames_with_clusters + "clustered_data.csv")
        self.frames_with_clusters = self.frames_with_clusters[self.frames_with_clusters["altered"]!= 1]
        with open(self.run_path + 'correlations.json') as json_file:
            self.correlations = json.load(json_file)

        self.edge_threshold_corr  = edge_threshold_corr
        self.edge_threshold_count  = edge_threshold_count

        self.low_cluster_filter = low_cluster_filter
        self.community_resolution = community_resolution
        #self.columns_for_clustering = columns_for_clustering
        self.listOfEdges_count, self.list_of_edges_corr = self.transform_data()
        self.networkx_graph_count, self.networkx_graph_corr  = self.create_networkx_graph()
        self.communities_count, self.communities_corr = self.get_communities()

    def transform_data(self):

        clusters = list(self.frames_with_clusters['cluster'].unique())
        videos = list(self.frames_with_clusters['video_id'].unique())
        self.frames_with_clusters = self.frames_with_clusters.drop(columns=["Unnamed: 0.1","Unnamed: 0", "timestamp"])

        self.filter_low_cluster_counts(videos)
        cluster_videos_listing = self.attach_videos_to_clusters(clusters)
        same_appearance_counter = self.count_intersection_between_clusters(cluster_videos_listing)

        list_of_edges_count, list_of_edges_corr = self.transform_counts_to_edges(same_appearance_counter)
        list_of_edges_count_norm_weights = self.normalize_weights(list_of_edges_count)

        return list_of_edges_count_norm_weights,  list_of_edges_corr

    def filter_low_cluster_counts(self,videos):
        for video_id in videos:
            removed_altered = self.frames_with_clusters.drop(columns=["altered","altered_probab"])
            cluster_counts = removed_altered[removed_altered["video_id"] == video_id].value_counts()

            for identifier, count in cluster_counts.items():
                #if identifier[1] == 57:
                #    print(cluster_counts)
                if count < self.low_cluster_filter:
                    self.frames_with_clusters = self.frames_with_clusters.drop(self.frames_with_clusters[(self.frames_with_clusters.video_id == identifier[0]) & (self.frames_with_clusters.cluster == identifier[1])].index)

    def attach_videos_to_clusters(self,clusters):
        cluster_videos_listing = defaultdict(list)
        for cluster_id in clusters:
            cluster_videos_listing[cluster_id] = list(self.frames_with_clusters["video_id"][self.frames_with_clusters["cluster"]==cluster_id])
        
        return cluster_videos_listing

    def count_intersection_between_clusters(self, cluster_videos_listing):
        counter = defaultdict(lambda: defaultdict(int))
        for cluster_id, videos in cluster_videos_listing.items():
            for cluster_id2, videos2 in cluster_videos_listing.items():
                if cluster_id == cluster_id2:
                    continue
                counter[cluster_id][cluster_id2] = len(list(set(videos).intersection(videos2)))
        
        return counter

    def transform_counts_to_edges(self, same_appearance_counter):
        list_of_edges_count = []
        list_of_edges_corr = []

        for cluster, counter_dict in same_appearance_counter.items():
            for link_cluster, count in counter_dict.items():
                # no intersection = now edge
                if count == 0.0 or count < 2:
                    continue
                
                list_of_edges_count.append((cluster,link_cluster,count))

                corr_weight = self.correlations[str(cluster)][str(link_cluster)]
                if corr_weight < self.edge_threshold_corr:
                    continue
                list_of_edges_corr.append((cluster,link_cluster,corr_weight))

        return list_of_edges_count, list_of_edges_corr

    def normalize_weights(self,listOfEdges):
        
        weights = []
        for edge in listOfEdges:
            weights.append(edge[2])

        minimum, maximum = min(weights), max(weights)

        normalized_list = []
        for edge in listOfEdges:
            norm_w = (edge[2]-minimum) / (maximum-minimum)
            #if norm_w == 0.0 or norm_w < self.edge_threshold:
            if edge[2] <= self.edge_threshold_count:
                continue
            normalized_list.append((edge[0],edge[1], norm_w))

        return normalized_list
        
    def create_networkx_graph(self):
        graph_count, graph_corr = nx.DiGraph(), nx.DiGraph()

        edges_count, edges_corr = self.listOfEdges_count, self.list_of_edges_corr

        graph_count.add_nodes_from(self.frames_with_clusters['cluster'].unique())
        graph_count.add_weighted_edges_from(edges_count)

        graph_corr.add_nodes_from(self.frames_with_clusters['cluster'].unique())
        graph_corr.add_weighted_edges_from(edges_corr)

        return graph_count, graph_corr

    def get_analytics(self):
        return {
            "degreeC": nx.degree_centrality(self.networkx_graph),
            "closenessC": nx.closeness_centrality(self.networkx_graph),
            "betweenessC": nx.betweenness_centrality(self.networkx_graph),
            "pagerank": nx.pagerank(self.networkx_graph, weight="weight")
        }

    def get_communities(self):
        communities_count = community.greedy_modularity_communities(self.networkx_graph_count,
            weight="weight",
            resolution=self.community_resolution)

        self.networkx_graph_count = self.attach_communities_to_graph(communities_count, self.networkx_graph_count)

        communities_corr = community.greedy_modularity_communities(self.networkx_graph_corr,
            weight="weight",
            resolution=self.community_resolution)

        self.networkx_graph_corr = self.attach_communities_to_graph(communities_corr, self.networkx_graph_corr)

        return communities_count, communities_corr

    def attach_communities_to_graph(self, communities, graph):
        for comm_number, nodes in enumerate(communities):
            for node in nodes:
                graph.nodes[node].update({"community": comm_number})

        return graph


    def save_to_json(self):
        with open(self.run_path + "graph_communities_count.json", "w") as f:
            json.dump(nx.node_link_data(self.networkx_graph_count), f, cls=NpEncoder)

        with open(self.run_path + "graph_communities_corr.json", "w") as f:
            json.dump(nx.node_link_data(self.networkx_graph_corr), f, cls=NpEncoder)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)




