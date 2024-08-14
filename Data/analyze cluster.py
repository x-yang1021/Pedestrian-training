import numpy as np
import pandas as pd

df = pd.read_csv('./clustered.csv')

IDs = []
Trajectories = {}
ID = df.iloc[0]['ID']
trajectory = 1

for i in range(df.shape[0]):
    if ID not in IDs:
        IDs.append(ID)
    if df.iloc[i]['ID'] != ID:
        Trajectories[ID] = trajectory
        ID = df.iloc[i]['ID']
    if df.iloc[i]['Trajectory'] != trajectory:
        trajectory = df.iloc[i]['Trajectory']
Trajectories[ID] = trajectory

n_cluster = df['Cluster'].max()

for cluster in range(1, n_cluster+1):
    Cluster_IDs = []
    Cluster_Trajectories = {}
    df_cluster = df[df['Cluster']==cluster]
    ID = df_cluster.iloc[0]['ID']
    trajectory = 0
    traj_num = 0
    lengths = []
    length = 0
    dists = []
    dist = 0
    print(f'cluster {cluster}',df_cluster['Speed Change'].abs().mean(), df_cluster['Speed Change'].var(),df_cluster['Direction Change'].abs().mean(),df_cluster['Direction Change'].var())
    for j in range(df_cluster.shape[0]):
        if ID not in Cluster_IDs:
            Cluster_IDs.append(ID)
        if df_cluster.iloc[j]['ID'] != ID:
            Cluster_Trajectories[ID] = traj_num
            if length:
                lengths.append(length)
            if dist:
                dists.append(dist)
            ID = df_cluster.iloc[j]['ID']
            traj_num = 0
            length = 0
            dist = 0
        if df_cluster.iloc[j]['Trajectory'] != trajectory:
            trajectory = df_cluster.iloc[j]['Trajectory']
            traj_num+=1
            if length:
                lengths.append(length)
                length = 0
            if dist:
                dists.append(dist)
                dist = 0
        length += 1
        dist += df_cluster.iloc[j]['Speed'] * 0.5
    Cluster_Trajectories[ID] = trajectory
    # for ID in Cluster_IDs:
    #     print(cluster, ID, Cluster_Trajectories[ID], Trajectories[ID], Cluster_Trajectories[ID]/Trajectories[ID])
    print(f'cluster {cluster}', np.sum(lengths), len(lengths), np.mean(dists), np.mean(np.array(dists)/np.array(lengths)/0.5))

# exam number of useful traj
df_2 = df[df['Cluster'] == 2]

ID = df_2.iloc[0]['ID']
trajectory = df_2.iloc[0]['Trajectory']
i = 0
count = 0
useful_traj = 0
while i < df_2.shape[0]:
    if df_2.iloc[i]['ID'] == ID and df_2.iloc[i]['Trajectory'] == trajectory:
        count += 1
    else:
        count = 0
    if count == 8:
        useful_traj += 1
        count = 0
    ID = df_2.iloc[i]['ID']
    trajectory = df_2.iloc[i]['Trajectory']
    i += 1

print(useful_traj)