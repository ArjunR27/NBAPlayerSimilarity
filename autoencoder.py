import tensorflow as tf
import numpy as np
import pandas as pd
import random
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import plotly.express as px

def main():
    # Ensuring reproductibility by setting the random seeds
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Player Averages
    df = pd.read_csv('player_data1.csv')
    df.drop(['awards', 'pos', 'team_name_abbr'], axis=1, inplace=True)
    df = df.dropna()
    player_names = df['name_display'].values
    df.drop(['name_display'], axis=1, inplace=True)
    
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    ae, latent_representation = create_autoencoder(df_scaled)
    
    kmeans = KMeans(n_clusters=6, random_state=42)
    cluster_labels = kmeans.fit_predict(latent_representation)

    cluster_df = pd.DataFrame()
    cluster_df['Cluster'] = cluster_labels
    cluster_df['Name'] = player_names

    player_clusters = pd.DataFrame({'Player': player_names, 'Cluster': cluster_labels})
    player_clusters.to_csv('player_clusters.csv')
    print(player_clusters)

    # 2 dimensional plot
    plot_2d(latent_representation, cluster_labels, player_names)

    # 3 dimensional plot
    # plot_3d(latent_representation, cluster_labels, player_names)
    
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

def plot_2d(latent_representation, cluster_labels, player_names):
    c_df = pd.DataFrame(latent_representation, columns=['LR1', 'LR2'])
    c_df['Cluster'] = cluster_labels
    c_df['Name'] = player_names

    fig = px.scatter(c_df, x='LR1', y='LR2', color='Cluster', hover_name='Name', 
                    title="K-Means Clusters", 
                    labels={'Feature 1': 'LR1', 'Feature 2': 'LR2'})


    fig.show()

def plot_3d(latent_representation, cluster_labels, player_names):
    c_df = pd.DataFrame(latent_representation, columns=['LR1', 'LR2', 'LR3'])
    c_df['Cluster'] = cluster_labels
    c_df['Name'] = player_names
    fig = px.scatter_3d(c_df, x='LR1', y='LR2', z='LR3', color='Cluster', hover_name='Name', 
                    title="K-Means Clusters", 
                    labels={'Feature 1': 'LR1', 'Feature 2': 'LR2', 'Feature 3': 'LR3'})
    fig.show()
    
def create_autoencoder(data):
    n_inputs = data.shape[1]

    input_data = Input(shape=(n_inputs, ))
    encoded = Dense(int(n_inputs / 4), activation='relu')(input_data)
    encoded = Dense(int(n_inputs / 6), activation='relu')(encoded)
    latent = Dense(2, activation='relu')(encoded) 

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