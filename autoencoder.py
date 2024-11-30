import tensorflow as tf
import numpy as np
import pandas as pd
import random
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt 
import plotly.express as px
import math
import seaborn as sns
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
c_df2 = None
df_with_names = None
player_names = None

def initialize(year):
    print("initializing")
    global c_df2, df_with_names, player_names
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Player Averages
    df = pd.read_csv(f'./season_data/player_data_{year}.csv')
    df.drop(['age', 'awards', 'pos', 'team_name_abbr'], axis=1, inplace=True)
    df = df.dropna()
    player_names = df['name_display'].values
    df.drop(['name_display'], axis=1, inplace=True)

    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    ae, latent_representation = create_autoencoder(df_scaled)

    # elbow_graph(df_scaled)
    
    num_clusters = 7
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(latent_representation)

    cluster_df = pd.DataFrame()
    cluster_df['Cluster'] = cluster_labels
    cluster_df['Name'] = player_names


    # 2-dimensional plot
    c_df2 = pd.DataFrame(latent_representation, columns=['LR1', 'LR2'])
    c_df2['Cluster'] = cluster_labels
    c_df2['Name'] = player_names
    # plot_2d(c_df2)

    player_names_series = pd.Series(player_names, name='Name')
    df_with_names = df.copy()
    df_with_names['Name'] = player_names_series.values

    latent_df = pd.DataFrame(latent_representation, columns=['LR1', 'LR2'])
    
    # Calculates the correlations between the Latent Representation Values and Features
    # LR1 = measures (negatively correlated) performance (shot attempts, games played, minutes played) --> low LR1 means higher performance player, high LR1 means lower performance player
    # LR2 = measures volume (scoring, points, assists) --> low LR2 means less volumetric stats, high LR2 means high volumetric stats
    correlations = pd.DataFrame(
        {
            'Feature': df.columns,
            'LR1 Correlation': [np.corrcoef(df[col], latent_df['LR1'])[0, 1] for col in df.columns],
            'LR2 Correlation': [np.corrcoef(df[col], latent_df['LR2'])[0, 1] for col in df.columns],
        }
    )
    correlations = correlations.sort_values(by=['LR1 Correlation', 'LR2 Correlation'], ascending=False)
    print(correlations)
    

    # Loop to repeatedly ask for similar players
    # while True:
    #     player_name = input("What player do you want to find similar players for? ")
    #     if player_name not in player_names:
    #         print(f"Player '{player_name}' not found. Please try again.")
    #         continue

    #     try:
    #         num_players = int(input("How many similar players are you looking for? "))
    #     except ValueError:
    #         print("Invalid input. Please enter a number.")
    #         continue

    #     find_similar_players(c_df2, df_with_names, player_name, num_players)

    #     cont = input("Do you want to find similar players for another player? (yes/no): ").strip().lower()
    #     if cont != 'yes':
    #         print("Exiting")
    #         break


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
    plt.ylabel('sum of squared distance')
    plt.title('elbow method')
    plt.show()

def calculate_distance(row, target_player):
    return math.sqrt(math.pow(target_player['LR1']-row['LR1'], 2) + (math.pow(target_player['LR2']-row['LR2'], 2)))

@app.route('/predict', methods=['POST'])
def predict():
    
    data = request.json  # Get JSON input
    player_name = data.get('playerName')
    num_players = data.get('numberPlayers')
    curr_c_df2 = pd.DataFrame(data.get('c_df2'))
    curr_df_with_names = pd.DataFrame(data.get('df_with_names'))
    
    if curr_c_df2.empty or player_name not in player_names or num_players < 1:
        return jsonify([])
    similar_players = find_similar_players(curr_c_df2, curr_df_with_names, player_name, num_players)
    print(similar_players)
    result = similar_players['Name'].tolist()[1:]

    return jsonify(result)

@app.route('/build', methods=['POST'])
def build():
    global c_df2, df_with_names
    data = request.json  # Get JSON input
    print(data)
    year = data.get('year')
    if year < 2000 or year > 2024:
        return jsonify([])
    initialize(year)
    return jsonify([c_df2.to_dict(orient='list'), df_with_names.to_dict(orient='list')])


def find_similar_players(c_df, stat_df, player_name, num_players):
    stat_df_copy = stat_df.copy()

    merged_df = pd.merge(c_df, stat_df_copy, on='Name')
    merged_df.drop(['games', 'games_started'], axis=1, inplace=True)

    stat_df_copy.drop(['Name', 'games', 'games_started'], axis=1, inplace=True)
    stat_cols = [col for col in stat_df_copy.columns]
    target_player = merged_df[merged_df['Name'] == player_name].iloc[0]

    for index, row in merged_df.iterrows():
        dist = calculate_distance(row, target_player)
        merged_df.loc[index, 'Distance'] = dist

    closest_players = merged_df.sort_values('Distance').iloc[:num_players + 1]

    print(closest_players)
    return closest_players
    # melted_df = closest_players.melt(id_vars=['Name', 'Cluster'], value_vars=stat_cols, var_name='Stat', value_name='Value')

    # fig = px.box(melted_df,
    #              x='Stat',
    #              y='Value',
    #              color='Name',
    #              title=f'Comparison: Top {num_players} Similar Players to {player_name}',
    #              color_discrete_sequence=px.colors.qualitative.Set2,
    #              points='all')
    # fig.update_layout(
    #     xaxis_title='Statistic',
    #     yaxis_title='Value',
    #     legend_title='Clusters',
    #     xaxis=dict(tickangle=45),
    #     height=600,
    #     width=1000
    # )

    # fig.show()


def search_player(stats, autoencoder):
    # stats would be a list of player stats
    # we can then run this 'fake' player through the latent representation and
    # 'fake a point' and then find players closest/ which cluster the player is in
    return NotImplementedError


def describe_clusters(c_df, stat_df):
    # Using averages per cluster for each stat
    merged_df = pd.merge(c_df, stat_df, on='Name')
    merged_df.drop(['games', 'games_started'], axis=1, inplace=True)
    
    stat_df.drop(['Name','games', 'games_started'], axis=1, inplace=True)
    stat_cols = [col for col in stat_df.columns]

    cluster_means = merged_df.groupby('Cluster')[stat_cols].mean().reset_index()

    cluster_vals = cluster_means.melt(id_vars=['Cluster'],
                                  var_name='Stat',
                                  value_name='Avg Value')
    fig = px.box(cluster_vals,
                 x='Stat',
                 y='Avg Value',
                 color='Cluster',
                 title='Average Player Stat by Cluster',
                 color_discrete_sequence=px.colors.qualitative.Set1,
                 points='all')
    fig.update_layout(
        xaxis_title='Statistic',
        yaxis_title='Value',
        legend_title='Clusters',
        xaxis=dict(tickangle=45),
        height=600,
        width=1000
    )

    fig.show()

    """
    Look at which clusters excel at what
    Create some basic text descriptions of what each cluster describes
    high points = scorer
    high shots = volume shot taker
    high mins = high usage
    etc.
    """

def plot_2d(c_df):
    fig = px.scatter(c_df, x='LR1', y='LR2', color='Cluster', hover_name='Name', 
                     title="K-Means Clusters", 
                     labels={'Feature 1': 'LR1', 'Feature 2': 'LR2'})
    fig.show()


def plot_3d(c_df):
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

    decoded = Dense(8, activation='relu')(latent)
    decoded = Dense(int(n_inputs / 6), activation='relu')(decoded)
    decoded = Dense(int(n_inputs / 4), activation='relu')(decoded)
    output_data = Dense(n_inputs, activation='sigmoid')(decoded)

    autoencoder = Model(input_data, output_data)
    encoder = Model(input_data, latent)

    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(data, data, epochs=75)

    latent_representation = encoder.predict(data)

    return autoencoder, latent_representation



if __name__ == "__main__":
    app.run()
    