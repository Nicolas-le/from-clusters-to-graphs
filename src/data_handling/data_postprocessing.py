import pandas as pd
import json
import base64
from PIL import Image
from io import BytesIO
import random

def show_images_cluster(cluster, cluster_video_timestamp, output_dir,  sample_size=5):

    cluster_df = cluster_video_timestamp[cluster_video_timestamp["cluster"]==cluster]
    try:
        sampled_df = cluster_df.sample(n=sample_size, random_state=1)
    except ValueError:
        sampled_df = cluster_df
    counter = 0
    
    for index, image in sampled_df.iterrows():
        #print(counter)
        thumbnails_path = "./resources/tagesschau/" + str(image["video_id"]) + "/thumbnails.json"
        
        with open(thumbnails_path) as json_file:
            data = json.load(json_file)
            
            for time in data:
                if time["t"] == image["timestamp"]:
                    image_base64 = time["image"]

            with open(output_dir+str(cluster)+"_"+str(counter)+".jpg", "wb") as fh:
                fh.write(base64.b64decode(image_base64))
        
        counter += 1



cluster_video_timestamp = pd.read_csv("./outputs/15_07_2023_23_45_08/clustered_data.csv")

for cluster in [28, 57, 151]:
    print(cluster)
    show_images_cluster(cluster, cluster_video_timestamp, "./resources/", sample_size=50)
