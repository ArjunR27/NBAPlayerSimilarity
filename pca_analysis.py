import pandas as pd
from sklearn.decomposition import PCA
import numpy
import matplotlib.pyplot as plot

# Using this to try and find output dimensions for the encoder
df = pd.read_csv('player_data.csv')

df.drop(['awards', 'name_display', 'pos', 'team_name_abbr'], axis=1, inplace=True)

df = df.dropna()

from sklearn.preprocessing import StandardScaler

numerical_data = df.select_dtypes(include=[numpy.number])

scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

pca = PCA()
pca.fit(scaled_data)

explained_variance = pca.explained_variance_ratio_

cumulative_explained_variance = numpy.cumsum(explained_variance)
plot.figure(figsize=(8, 6))
plot.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plot.title('Cumulative Explained Variance by Principal Components')
plot.xlabel('Principal Components')
plot.ylabel('Cumulative Explained Variance')
plot.show()

threshold = 0.95
num_components = numpy.argmax(cumulative_explained_variance >= threshold) + 1
print(f"Number of components that explain {threshold*100}% variance: {num_components}")
