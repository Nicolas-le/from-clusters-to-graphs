from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

def split_Xy_scaling(transformed_data):

    X  = transformed_data.drop(columns=["video_id", "timestamp"])
    y = transformed_data.loc[:, "video_id":"timestamp"]

    scale = StandardScaler()
    X_scaled = pd.DataFrame(scale.fit_transform(X.values), columns=X.columns, index=X.index)

    return X, y, X_scaled

def perform_pca(x_scaled, y, principal_component_count):

    principal_components_names = ["PCA" + str(i) for i in range(1,principal_component_count+1)]
    pca = PCA(n_components=principal_component_count)
    pca_features = pca.fit_transform(x_scaled)
    pca_df = pd.DataFrame(data=pca_features,columns=principal_components_names)

    y = y.reset_index(drop=True)

    pca_df["video_id"] =  y["video_id"]
    pca_df["timestamp"] =  y["timestamp"]
 
    return pca_df

def pca_main(transformed_data, config):
    logging.info("Scaling data...")
    _, y, X_scaled = split_Xy_scaling(transformed_data)

    logging.info("Perform PCA...")
    pca_data = perform_pca(X_scaled, y, config["principal_components"])

    pca_data.to_csv(config["output_directory"] + "pca_transformed_data.csv")

