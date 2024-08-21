import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv('./clustered.csv')

IDs = []
ID = df.iloc[0]['ID']
Trajectories = {ID:[]}
trajectory = 1

for i in range(df.shape[0]):
    if ID not in IDs:
        IDs.append(ID)
    if df.iloc[i]['ID'] != ID:
        ID = df.iloc[i]['ID']
        Trajectories[ID] = []
    if df.iloc[i]['Trajectory'] != trajectory:
        trajectory = df.iloc[i]['Trajectory']
        Trajectories[ID].append(trajectory)
Trajectories[ID].append(trajectory)

n_cluster = df['Cluster'].max()

for cluster in range(1, n_cluster+1):
    Cluster_IDs = []
    Cluster_Trajectories_num = {}
    df_cluster = df[df['Cluster']==cluster]
    ID = df_cluster.iloc[0]['ID']
    Cluster_Trajectories = {ID:[]}
    trajectory = 0
    traj_num = 1
    lengths = []
    length = 0
    dists = []
    dist = 0
    print(f'cluster {cluster}',
          'Average Speed Change', df_cluster['Speed Change'].abs().mean(),
          'Variance of speed change', df_cluster['Speed Change'].var(),
          'Average Direction Change', df_cluster['Direction Change'].abs().mean(),
          'Variance of direction change', df_cluster['Direction Change'].var())
    for j in range(df_cluster.shape[0]):
        if ID not in Cluster_IDs:
            Cluster_IDs.append(ID)
        if df_cluster.iloc[j]['ID'] != ID:
            Cluster_Trajectories_num[ID] = traj_num
            if length:
                lengths.append(length)
            if dist:
                dists.append(dist)
            ID = df_cluster.iloc[j]['ID']
            Cluster_Trajectories[ID] = []
            traj_num = 0
            length = 0
            dist = 0
            trajectory = 0
        if df_cluster.iloc[j]['Trajectory'] != trajectory:
            trajectory = df_cluster.iloc[j]['Trajectory']
            traj_num+=1
            Cluster_Trajectories[ID].append(trajectory)
            if length:
                lengths.append(length)
                length = 0
            if dist:
                dists.append(dist)
                dist = 0
        length += 1
        dist += df_cluster.iloc[j]['Speed'] * 0.5
    Cluster_Trajectories_num[ID] = traj_num
    Cluster_Trajectories[ID].append(trajectory)
    print(f'cluster {cluster}',
          'Total data points',np.sum(lengths),
          'Number of trajectories', len(lengths),
          'Average Travel Distance',np.mean(dists),
          'Average Speed', np.mean(np.array(dists)/np.array(lengths)/0.5))

# Typical_traj = 0
# total_traj = 0
# for ID in Cluster_IDs:
#     if Cluster_Trajectories_num[ID]/len(Trajectories[ID]) > 0:
#         Typical_traj += Cluster_Trajectories_num[ID]
#         print(ID)
#     total_traj += Cluster_Trajectories_num[ID]
# print(Typical_traj, total_traj)
#
# exit()

Far_speed_change = []
Near_speed_change = []
Far_direction_change = []
Near_direction_change = []
df_all = pd.read_csv('./entrie dataset.csv')
# All_IDs = []
# ID = df_all.iloc[0]['ID']
# All_trajectories = {ID:[]}
# for i in range(df_all.shape[0]):
#     if ID not in All_IDs:
#         All_IDs.append(ID)
#     if df_all.iloc[i]['ID'] != ID:
#         ID = df_all.iloc[i]['ID']
#         All_trajectories[ID] = []
#     if df_all.iloc[i]['Trajectory'] != trajectory:
#         trajectory = df_all.iloc[i]['Trajectory']
#         All_trajectories[ID].append(trajectory)
# All_trajectories[ID].append(trajectory)
# Cluster_IDs = All_IDs
# Cluster_Trajectories = All_trajectories
df_total = df_all.dropna(subset=['Direction Change'])
df_total.reset_index(drop=True, inplace=True)
for i in range(df_total.shape[0]):
    if df_total.iloc[i]['ID'] in Cluster_IDs:
        trajectory = Cluster_Trajectories[df_total.iloc[i]['ID']]
        if df_total.iloc[i]['Trajectory'] in trajectory:
            if df_total.iloc[i]['Distance'] > 2.58:
                Far_speed_change.append(df_total.iloc[i]['Speed Change'])
                Far_direction_change.append(df_total.iloc[i]['Direction Change'])
            else:
                Near_speed_change.append(df_total.iloc[i]['Speed Change'])
                Near_direction_change.append(df_total.iloc[i]['Direction Change'])

tt_stat, p_value = stats.mannwhitneyu(np.array(Near_speed_change),np.array(Far_speed_change))

print(p_value, 'Near Speed Change')

tt_stat, p_value = stats.mannwhitneyu(np.array(Near_direction_change),np.array(Far_direction_change))

print(p_value, 'Near Direction Change')


# exam number of useful traj

dfs = []

df_total = df_total[df_total['Distance']<=9]
total_traj = 0
for ID in Cluster_IDs:
    df_temp = df_total[df_total['ID'] == ID]
    for traj in Cluster_Trajectories[ID]:
        dfs.append(df_temp[df_temp['Trajectory']==traj])
    total_traj += Cluster_Trajectories_num[ID]

print('number of trajectories', total_traj)

df_2 = pd.concat(dfs,ignore_index=True)

ID = df_2.iloc[0]['ID']
trajectory = df_2.iloc[0]['Trajectory']
i = 0
count = 0
useful_traj = []
while i < df_2.shape[0]:
    if df_2.iloc[i]['ID'] == ID and df_2.iloc[i]['Trajectory'] == trajectory:
        count += 1
    else:
        useful_traj.append(count)
        count = 0
    ID = df_2.iloc[i]['ID']
    trajectory = df_2.iloc[i]['Trajectory']
    i += 1
print(max(useful_traj), min(useful_traj), sum(useful_traj))

df_2['Speed Change'] = df_2['Speed Change'].abs()
print(df_2['Speed'].max(), df_2['Speed'].min())

