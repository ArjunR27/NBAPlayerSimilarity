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
import matplotlib.pyplot as plt 
def main():
    df = pd.read_csv('player_data.csv')
    df.drop(['awards', 'name_display', 'pos', 'team_name_abbr'], axis=1, inplace=True)
    df = df.dropna()
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    # Trying to find optimal number of k clusters

    # 1) using elbow method, k = 6-8 clusters
    
    elbow_graph(df_scaled)

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

    autoencoder.fit(data, data, epochs=45, shuffle=True)

    latent_representation = encoder.predict(data)
    
    return autoencoder, latent_representation

def cluster(data):

    return


if __name__ == "__main__":
    main()