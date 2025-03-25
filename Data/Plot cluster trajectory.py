import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

cluster = 1
mapping = { 'ID':0,
           'Trajectory':1,
           'Positionx':2,
           'Positiony':3,
            'Distance':4,
            'Up':5,
            'Right':6,
            'Down':7,
            'Left':8,
            'Speed':9,
            'Speed Change':10,
            'Direction':11,
            'Direction Change':12}
width = len(mapping)
traj_length = 15
distance_threshold = 2.58
# distance_threshold = 9
df = pd.read_csv('../Data/clustered.csv')
df_all = pd.read_csv('./entrie dataset.csv')
df_all = df_all[['ID','Positionx','Positiony','Trajectory','Speed','Speed Change','Distance']]
df_all['Cluster'] = np.nan


Cluster_IDs = []
Cluster_Trajectories_num = {}
df_cluster = df[df['Cluster']==cluster]
ID = df_cluster.iloc[0]['ID']
Cluster_Trajectories = defaultdict(list)
trajectory = 0
traj_num = 1
lengths = []
length = 0
for j in range(df_cluster.shape[0]):
    if ID not in Cluster_IDs:
        Cluster_IDs.append(ID)
    if df_cluster.iloc[j]['ID'] != ID:
        Cluster_Trajectories_num[ID] = traj_num
        if length:
            lengths.append(length)
        ID = df_cluster.iloc[j]['ID']
        Cluster_Trajectories[ID] = []
        traj_num = 0
        length = 0
        trajectory = 0
    if df_cluster.iloc[j]['Trajectory'] != trajectory:
        trajectory = df_cluster.iloc[j]['Trajectory']
        traj_num+=1
        Cluster_Trajectories[ID].append(trajectory)
        if length:
            lengths.append(length)
            length = 0
    length += 1
Cluster_Trajectories_num[ID] = traj_num
Cluster_Trajectories[ID].append(trajectory)

for i in range(df_all.shape[0]):
    if pd.isna(df_all.at[i, 'ID']):
        continue
    ID = df_all.at[i, 'ID']
    if df_all.at[i, 'Trajectory'] in Cluster_Trajectories[ID]:
        df_all.at[i, 'Cluster'] = cluster
    else:
        df_all.at[i, 'Cluster'] = cluster+1

df_plot = df_all[df_all['Distance']<=distance_threshold].copy()

cols = ['Positionx', 'Positiony', 'Speed', 'Speed Change']

# Drop rows with missing values in the selected columns
df_plot = df_plot.dropna(subset=cols).copy()

# Compute mean and std
standardization_stats = {
    col: {
        'mean': df_plot[col].mean(),
        'std': df_plot[col].std()
    }
    for col in cols
}

# Print results
for col, stats in standardization_stats.items():
    print(f"{col}: mean = {stats['mean']:.4f}, std = {stats['std']:.4f}")

exit()

grouped = df_plot.groupby('Trajectory')
fig, ax = plt.subplots(figsize=(8, 6))
for name, group in grouped:
    ax.plot(group['Positionx'], group['Positiony'], marker='o', linestyle='-', markersize=2)

ax.set_xlabel('Position X')
ax.set_ylabel('Position Y')
ax.legend()
plt.show()

