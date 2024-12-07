import numpy as np
import torch
from env.utils import getUsefulPositions
import pandas as pd
from sklearn.model_selection import train_test_split


cluster = 2
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
distance_threshold = 9 #2.58

df = pd.read_csv('../Data/clustered.csv')


Cluster_IDs = []
Cluster_Trajectories_num = {}
df_cluster = df[df['Cluster']==cluster]
ID = df_cluster.iloc[0]['ID']
Cluster_Trajectories = {ID:[]}
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

df_1 = pd.read_csv('../env/Rush_Data/Experiment 1.csv')
df_2 = pd.read_csv('../env/Rush_Data/Experiment 2.csv')
df_3 = pd.read_csv('../env/Rush_Data/Experiment 3.csv')
dfs = [df_1,df_2,df_3]
positions = []
for dataset in dfs:
    for col in range(2, dataset.shape[1], width):
        data_col = dataset.iloc[:,col:col+width]
        data_temp = data_col.dropna()
        if data_temp.shape[0] == 0:
            print(dataset,col)
        ID = data_temp.iloc[0,mapping['ID']]
        if ID not in Cluster_IDs:
            continue
        timestep = 0
        while timestep < dataset.shape[0] - traj_length:
            if pd.isna(data_col.iloc[timestep,mapping['Direction']]):
                timestep += 1
                continue
            if data_col.iloc[timestep,mapping['Trajectory']] not in Cluster_Trajectories[ID]:
                timestep += 1
                continue
            elif pd.isna(data_col.iloc[timestep + traj_length - 1,mapping['Trajectory']]):
                timestep += 1
                continue
            elif data_col.iloc[timestep,mapping['Trajectory']] != data_col.iloc[timestep + traj_length - 1,
                mapping['Trajectory']]:
                timestep += 1
                continue
            elif data_col.iloc[timestep,mapping['Distance']] > distance_threshold:
                timestep += 1
                continue
            data_traj = data_col.iloc[timestep:timestep + traj_length]
            self_position = []
            other_position = []
            for i in range(data_traj.shape[0]):
                x1 = data_traj.iloc[i,mapping['Positionx']]
                y1 = data_traj.iloc[i,mapping['Positiony']]
                self_position.append([x1,y1])
                other_position.append(getUsefulPositions(dataset, timestep, ID, width))
            positions.append([self_position,other_position])
            timestep += traj_length - 1

train_position, test_position = train_test_split(positions, test_size=0.2, random_state=42)

torch.save(train_position,'./Impatient/train_position.pt')
torch.save(test_position, './Impatient/test_position.pt')