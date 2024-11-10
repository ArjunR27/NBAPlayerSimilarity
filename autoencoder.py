import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import plotly.express as px

def main():
    df = pd.read_csv('player_data.csv')
    df.drop(['awards', 'pos', 'team_name_abbr'], axis=1, inplace=True)
    df = df.dropna()
    player_names = df['name_display'].values
    df.drop(['name_display'], axis=1, inplace=True)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    # Trying to find optimal number of k clusters

    # 1) using elbow method, k = 6-8 clusters
    
    # elbow_graph(df_scaled)
    ae, latent_representation = create_autoencoder(df_scaled)

    kmeans = KMeans(n_clusters=8, random_state=42)
    cluster_labels = kmeans.fit_predict(latent_representation)

    cluster_df = pd.DataFrame()
    cluster_df['Cluster'] = cluster_labels
    cluster_df['Name'] = player_names

    player_clusters = pd.DataFrame({'Player': player_names, 'Cluster': cluster_labels})
    print(player_clusters)


    # This is used to reduce the 11-dimension to a further reduced 2 so that we can visualize it using a graph
    pca = PCA(n_components=2)
    reduced_latent = pca.fit_transform(latent_representation)

    # 2 dimensional plot
    c_df = pd.DataFrame(reduced_latent, columns=['PCA Component 1', 'PCA Component 2'])
    c_df['Cluster'] = cluster_labels
    c_df['Name'] = player_names

    fig = px.scatter(c_df, x='PCA Component 1', y='PCA Component 2', color='Cluster', hover_name='Name', 
                    title="K-Means Clusters", 
                    labels={'Feature 1': 'PCA Component 1', 'Feature 2': 'PCA Component 2'})


    fig.show()


    
    """# 3 dimensional plot
    pca = PCA(n_components=3)
    reduced_latent = pca.fit_transform(latent_representation)
    c_df = pd.DataFrame(reduced_latent, columns=['PCA Component 1', 'PCA Component 2', 'PCA Component 3'])
    c_df['Cluster'] = cluster_labels
    c_df['Name'] = player_names


    fig = px.scatter_3d(c_df, x='PCA Component 1', y='PCA Component 2', z='PCA Component 3', color='Cluster', hover_name='Name', 
                    title="K-Means Clusters", 
                    labels={'Feature 1': 'PCA Component 1', 'Feature 2': 'PCA Component 2', 'Feature 3': 'PCA Component 3'})


    fig.show()"""

def elbow_graph(data):
    tf.random.set_seed(42)
    np.random.seed(42)
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
    tf.random.set_seed(42)
    np.random.seed(42)
    n_inputs = data.shape[1]

    # Encoder
    input_data = Input(shape=(n_inputs, ))
    encoded = Dense(int(n_inputs/2), activation='relu')(input_data)
    encoded = Dense(int(n_inputs/4), activation='relu')(encoded)
    latent = Dense(11, activation='relu')(encoded)

    # Decoder
    decoded = Dense(int(n_inputs/4), activation='relu')(latent)
    decoded = Dense(int(n_inputs/2), activation='relu')(decoded)
    output_data = Dense(n_inputs, activation='sigmoid')(decoded)

    autoencoder = Model(input_data, output_data)
    encoder = Model(input_data, latent)

    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(data, data, epochs=50)

    latent_representation = encoder.predict(data)
    
    return autoencoder, latent_representation


if __name__ == "__main__":
    main()