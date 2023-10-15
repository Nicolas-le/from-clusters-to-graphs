# From Clusters to Graphs – Toward a Scalable Viewing of News Videos

Nicolas Ruth, Bernhard Liebl and Manuel Burghardt - **Computational Humanities Group Leipzig**

CHR 2023: Computational Humanities Research Conference, December 6-8, 2023, Paris, France

**Abstract**
In this short paper, we present a novel approach that combines density-based clustering and graph modeling to create a scalable viewing application for the exploration of similarity patterns in news videos.
Unlike most existing video analysis tools that focus on individual videos, our approach allows for an overview of a larger collection of videos, which can be further examined based on their connections or communities. By utilizing scalable reading, specific subgraphs can be selected from the overview and their respective clusters can be explored in more detail on the video frame level.

---

## Visualization
Related code in: ```/src/app```. The code is based on https://github.com/vasturiano/3d-force-graph.

### Video

https://github.com/Nicolas-le/from-clusters-to-graphs/assets/40566913/3fa33411-f39e-4cef-a65b-8b611db0d7ab


Due to copyright reasons of the news video data which was analyzed in the paper cannot be made public. However, the source code of the visualization should help to reproduce it.

To visualize the graph you need a data structure similar to ```/src/app/graph_communities_corr.json``` which is a saved [NetworkX](https://networkx.org/) graph. How the graph is exported can be found in ```/src/data_handling/graph.py```.

Additionaly, you should locate up to 80 images for each of the clusters in the folder ```/src/app/hdbscan_clusters/```. File names should be structure like "(clusterID)_(increamenting number).jpg" (for cluster 0: 0_0.jpg, 0_1.jpg, ...).

Afterwards you could adapt ```/src/app/backend.py``` with your file path and start backend.py in an environment with the dependcies installed.

## Creating the data structure

Related code in: ```/src/data_handling```

The prototype should be perceived as a conceptual framework rather than a pre-packaged tool that is immediately usable for all video collections. Nevertheless, both the concept and the code in this repository are flexible and can be tailored to meet individual requirements. The concept was originally conceived to facilitate the practicality of scalable viewing for various application scenarios.

Before starting to implement this solution, you should think about the characteristics of your dataset. Does it make sense to examine the data for similarities in motifs and content across their videos? Does the data exhibit enough similarity between the videos for this purpose? You should check the data properties and how they're represented using CLIP embeddings by actively involving the described methodology, including PCA, clustering, and setting threshold values for the graph. It's essential to keep in mind that the choice of parameters for these steps depends heavily on the data itself, and adjustments must be made interactively to achieve an effective visualization.

```/src/data_handling/main.py``` contains every major parameter. The code works with previously calculated CLIP embeddings of the video corpus and their associated images, in this case BASE64 encoded.

The general procedure to generate the final visualization is as follows

1. Calculate CLIP-Embeddings for frames ever x seconds of each of your videos!
2. Convert your embeddings into a csv with the following columns "video_id", "timestamp", "CLIP1", ..., "CLIP512"!
3. Use ```/src/data_handling/main.py``` to perform the clustering and build of the graph structure!

