import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.signal import medfilt
import matplotlib.pyplot as plt

def find_optimal_clusters(data):
    silhouette_scores = []

    # Try different numbers of clusters
    for i in range(2, 7):
        print(i)
        kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300)
        cluster_labels = kmeans.fit_predict(data.values.reshape(-1, 1))
        silhouette_avg = silhouette_score(data.values.reshape(-1, 1), cluster_labels, n_jobs=-1)
        silhouette_scores.append(silhouette_avg)

    # Find the index of the maximum silhouette score
    optimal_clusters_index = np.argmax(silhouette_scores)

    # Return the optimal number of clusters
    return optimal_clusters_index + 2  # Adding 2 because indexing starts from 0

# Read the CSV file
df = pd.read_csv()

# Convert the 'timestamp' column to datetime format
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='ISO8601')

# Round the timestamp to the nearest 200ms, as the data is sampled in <200ms intervals
df['ROUNDED_TIMESTAMP'] = df['TIMESTAMP'].dt.round('200ms')

# Step 1: Input is the aggregate power consumption signal
result = df.groupby('ROUNDED_TIMESTAMP').agg({
    'POW': 'last',
})

# Step 2: Apply median filter
result['POW'] = medfilt(result['POW'])

# Remove all NaN values
result = result.dropna(subset=['POW'])

# Step 3: Apply silhouette method to determine the number of clusters
# optimal_clusters = find_optimal_clusters(result['POW'])
optimal_clusters = 4

# Step 4: Perform KMeans clustering with the optimal number of clusters
optimal_kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++')
result['Cluster'] = optimal_kmeans.fit_predict(result['POW'].values.reshape(-1, 1))

event_indices = []
for i in range(1, len(result['Cluster']) - 1):
    if result['Cluster'][i] != result['Cluster'][i - 1] and result['Cluster'][i] == result['Cluster'][i + 1]:
        event_indices.append(i-1)

print(result.index[event_indices].strftime('%H:%M:%S.%f').tolist())

# Step 5: Plot the power aggregation signal with colored clusters and event indices
plt.figure(figsize=(10, 6))

# Plot the power aggregation signal
plt.plot(result.index, result['POW'], label='Power Aggregation Signal', color='blue')

# Highlight the event indices with colors according to the cluster
for event_index in event_indices:
    cluster_num = result['Cluster'].iloc[event_index]
    plt.scatter(result.index[event_index], result['POW'].iloc[event_index], c=f'C{cluster_num}', marker='x', s=100, label=f'Event {event_index} (Cluster {cluster_num})')

# Add labels and legend
plt.xlabel('Timestamp')
plt.ylabel('Power')
plt.title(f'Power Aggregation Signal with {optimal_clusters} Clusters and Event Indices')

# Show the plot
plt.show()


# To print the timestamps:
# print(result.index[event_indices].strftime('%H:%M:%S.%f').tolist())
