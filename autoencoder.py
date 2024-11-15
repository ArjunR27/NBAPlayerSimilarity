import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import random
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import plotly.express as px

def main():
    # Ensuring reproductibility by setting the random seeds
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    df = pd.read_csv('player_data1.csv')
    df.drop(['awards', 'pos', 'team_name_abbr'], axis=1, inplace=True)
    df = df.dropna()
    player_names = df['name_display'].values
    df.drop(['name_display'], axis=1, inplace=True)
    
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    
    # Trying to find optimal number of k clusters

    # 1) using elbow method, k = 6-8 clusters
    
    # elbow_graph(df_scaled)
    ae, latent_representation = create_autoencoder(df_scaled)

    """dbscan = DBSCAN(eps=3, min_samples=3)
    cluster_labels = dbscan.fit_predict(latent_representation)"""
    

    kmeans = KMeans(n_clusters=6, random_state=42)
    cluster_labels = kmeans.fit_predict(latent_representation)

    cluster_df = pd.DataFrame()
    cluster_df['Cluster'] = cluster_labels
    cluster_df['Name'] = player_names

    player_clusters = pd.DataFrame({'Player': player_names, 'Cluster': cluster_labels})
    player_clusters.to_csv('player_clusters.csv')
    print(player_clusters)


    # This is used to reduce the 11-dimension to a further reduced 2 so that we can visualize it using a graph
    # I feel like we shouldnt be using this because it may defeat the purpose of the autoencoder
    # Maybe look into finding out which factors are most important for finding player similairty and graph using that instead of pca?
    # pca = PCA(n_components=2)
    # latent_representation = pca.fit_transform(latent_representation)

    # 2 dimensional plot
    c_df = pd.DataFrame(latent_representation, columns=['PCA Component 1', 'PCA Component 2'])
    c_df['Cluster'] = cluster_labels
    c_df['Name'] = player_names

    fig = px.scatter(c_df, x='PCA Component 1', y='PCA Component 2', color='Cluster', hover_name='Name', 
                    title="K-Means Clusters", 
                    labels={'Feature 1': 'PCA Component 1', 'Feature 2': 'PCA Component 2'})


    fig.show()


    
    """# 3 dimensional plot
    # pca = PCA(n_components=3)
    # educed_latent = pca.fit_transform(latent_representation)
    c_df = pd.DataFrame(latent_representation, columns=['PCA Component 1', 'PCA Component 2', 'PCA Component 3'])
    c_df['Cluster'] = cluster_labels
    c_df['Name'] = player_names


    fig = px.scatter_3d(c_df, x='PCA Component 1', y='PCA Component 2', z='PCA Component 3', color='Cluster', hover_name='Name', 
                    title="K-Means Clusters", 
                    labels={'Feature 1': 'PCA Component 1', 'Feature 2': 'PCA Component 2', 'Feature 3': 'PCA Component 3'})"""


    fig.show()

def elbow_graph(data):
    n_inputs = data.shape[1]

    ae, latent_prediction= create_autoencoder(data)

    sum_squared_distances = []
    for i in range(2, n_inputs):
        km = KMeans(n_clusters=i).fit(latent_prediction)
        sum_squared_distances.append([int(i), km.inertia_])
    sum_squared_distances = np.array(sum_squared_distances).reshape(-1, 2)
    plt.plot(sum_squared_distances  [:, 0], sum_squared_distances[:, 1], 'bo-')
    plt.xlabel("num clusters")
    plt.ylabel('sum of squared sitance')
    plt.title('elbow method')
    plt.show()

def create_autoencoder(data):
    n_inputs = data.shape[1]

    # Encoder with gradual reduction
    input_data = Input(shape=(n_inputs, ))
    encoded = Dense(int(n_inputs / 4), activation='relu')(input_data)
    encoded = Dense(int(n_inputs / 6), activation='relu')(encoded)
    latent = Dense(2, activation='relu')(encoded)    # Latent space now reduced to 2 dimensions

    # Decoder with gradual expansion
    decoded = Dense(10, activation='relu')(latent)
    decoded = Dense(int(n_inputs / 6), activation='relu')(decoded)
    decoded = Dense(int(n_inputs / 4), activation='relu')(decoded)
    output_data = Dense(n_inputs, activation='sigmoid')(decoded)

    autoencoder = Model(input_data, output_data)
    encoder = Model(input_data, latent)

    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(data, data, epochs=35)

    latent_representation = encoder.predict(data)
    
    return autoencoder, latent_representation



if __name__ == "__main__":
    main()