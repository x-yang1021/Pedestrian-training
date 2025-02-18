import pandas as pd
import numpy as np
from tslearn.metrics import dtw
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

shorest_traj = 9

df = pd.read_csv('./Cluster dataset.csv')


# record the raw data
ID = df.iloc[0]['ID']
trajectory = 1
original_series = []
traj = []

for i in range(df.shape[0]):
    if df.iloc[i]['ID'] != ID and traj:
        if len(traj) >= shorest_traj:
            original_series.append(traj)
        ID = df.iloc[i]['ID']
        traj = []
    else:
        if df.iloc[i]['Trajectory'] != trajectory and traj:
            if len(traj) >= shorest_traj:
                original_series.append(traj)
            trajectory = df.iloc[i]['Trajectory']
            traj = []
    traj.append([df.iloc[i]['Speed Change'], df.iloc[i]['Direction Change'], df.iloc[i]['Speed']])
original_series.append(traj)

df.pop('Speed')

#normalization

df['Speed Change'] = 2 * (df['Speed Change'] - df['Speed Change'].min()) / (df['Speed Change'].max() - df['Speed Change'].min()) - 1
df['Direction Change'] = 2 * (df['Direction Change'] - df['Direction Change'].min()) / (df['Direction Change'].max() - df['Direction Change'].min()) - 1

# cluster based on normalized data
ID = df.iloc[0]['ID']
trajectory = 1
series = []
traj = []
for i in range(df.shape[0]):
    if df.iloc[i]['ID'] != ID and traj:
        if len(traj) >= shorest_traj + 1:
            series.append(traj)
        ID = df.iloc[i]['ID']
        traj = []
    else:
        if df.iloc[i]['Trajectory'] != trajectory and traj:
            if len(traj) >= shorest_traj + 1:
                series.append(traj)
            trajectory = df.iloc[i]['Trajectory']
            traj = []
    if not traj:
        traj.append([ID, trajectory])
    traj.append([df.iloc[i]['Speed Change'], df.iloc[i]['Direction Change']])
series.append(traj)

time_series_data = [np.array(ts) for ts in series]

first_columns = [ts[0] for ts in time_series_data]  # Store the first column
processed_series = [ts[1:] for ts in time_series_data]
# print(processed_series[879])

# Step 2: Compute the pairwise DTW distance matrix using only the second column
n = len(processed_series)
dtw_distances = np.zeros((n, n))

for i in range(n):
    for j in range(i + 1, n):
        distance = dtw(processed_series[i], processed_series[j])
        dtw_distances[i, j] = distance
        dtw_distances[j, i] = distance

# Convert the distance matrix to a condensed distance matrix format
dtw_distances_condensed = squareform(dtw_distances)

# Step 3: Perform hierarchical clustering using the precomputed distance matrix
Z = linkage(dtw_distances_condensed, method='ward')

# Step 4: Optional: Visualize the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z, no_labels=True)
# plt.title("Dendrogram of Hierarchical Clustering with DTW")
plt.ylabel("Distance")
plt.savefig('Dendrogram.png')
plt.show()

exit()

# Step 5: Determine the optimal number of clusters using the Silhouette score
range_n_clusters = range(2, 10)  # Example range to check
best_n_clusters = None
best_silhouette_score = -1

for n_clusters in range_n_clusters:
    cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    silhouette_avg = silhouette_score(dtw_distances, cluster_labels, metric='precomputed')
    print(f"For n_clusters = {n_clusters}, the Silhouette score is {silhouette_avg:.4f}")

    if silhouette_avg > best_silhouette_score:
        best_silhouette_score = silhouette_avg
        best_n_clusters = n_clusters

print(f"\nOptimal number of clusters: {best_n_clusters} with a Silhouette score of {best_silhouette_score:.4f}")

# Step 6: Use the optimal number of clusters to finalize clustering
cluster_labels = fcluster(Z, t= best_n_clusters, criterion='maxclust')

# Step 7: Concatenate the first column back to the clustered data


clustered_dataset = []

for i in range(n):
    length = len(original_series[i])
    clustered_data = pd.DataFrame({
        'ID':[first_columns[i][0]] * length,
        'Trajectory':[first_columns[i][1]] * length,
        'Speed Change':[change[0] for change in original_series[i]],
        'Direction Change':[change[1] for change in original_series[i]],
        'Speed':[change[2] for change in original_series[i]],
        'Cluster':[cluster_labels[i]] * length
    })
    clustered_dataset.append(clustered_data)

clustered_dataset = pd.concat(clustered_dataset, ignore_index=True)

clustered_dataset.to_csv('./clustered.csv', index = False)

cluster_2 = 0
for i in range(len(cluster_labels)):
    if cluster_labels[i] == 2:
        cluster_2+=1
print(cluster_2, len(cluster_labels))

