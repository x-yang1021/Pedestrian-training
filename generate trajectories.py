import imitation.data.types as types
from imitation.data import serialize
import numpy as np
from env.utils import withinSight, getPositions,getDensity, getFront
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
traj_length = 9
distance_threshold = 9

df = pd.read_csv('./Data/clustered.csv')


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

df_1 = pd.read_csv('./env/Rush_Data/Experiment 1.csv')
df_2 = pd.read_csv('./env/Rush_Data/Experiment 2.csv')
df_3 = pd.read_csv('./env/Rush_Data/Experiment 3.csv')
dfs = [df_1,df_2,df_3]

obs = []
acts = []
infos = []
trajectories = []
for dataset in dfs:
    for col in range(2, dataset.shape[1], width):
        data_col = dataset.iloc[:,col:col+width]
        data_temp = data_col.dropna()
        if data_temp.shape[0] == 0:
            print(dataset,col)
        ID = data_temp.iloc[0][mapping['ID']]
        if ID not in Cluster_IDs:
            continue
        timestep = 0
        while timestep < dataset.shape[0] - traj_length:
            if pd.isna(data_col.iloc[timestep][mapping['Speed']]):
                timestep += 1
                continue
            if data_col.iloc[timestep][mapping['Trajectory']] not in Cluster_Trajectories[ID]:
                timestep += 1
                continue
            elif pd.isna(data_col.iloc[timestep+traj_length-1][mapping['Trajectory']]):
                timestep += 1
                continue
            elif data_col.iloc[timestep][mapping['Trajectory']] != data_col.iloc[timestep+traj_length-1][mapping['Trajectory']]:
                timestep += 1
                continue
            elif data_col.iloc[timestep][mapping['Distance']] > distance_threshold:
                timestep += 1
                continue
            data_traj = data_col.iloc[timestep:timestep+traj_length]
            for i in range(data_traj.shape[0]):
                ob = []
                x1 = data_traj.iloc[i][mapping['Positionx']]
                y1 = data_traj.iloc[i][mapping['Positiony']]
                ob.append(x1)
                ob.append(y1)
                ob.append(data_traj.iloc[traj_length-1][mapping['Positionx']])
                ob.append(data_traj.iloc[traj_length-1][mapping['Positiony']])
                ob.append(data_traj.iloc[i][mapping['Distance']])
                ob.append(data_traj.iloc[i][mapping['Speed']])
                direction = data_traj.iloc[i][mapping['Direction']]
                ob.append(direction)
                positions = getPositions(dataset, timestep+i, ID, width)
                front = getFront(dataset,timestep+i,width,positions,x1,y1,direction)
                density = getDensity(positions,x1,y1,direction)
                ob.append(front[0])
                ob.append(front[1])
                ob.append(density)
                up = data_traj.iloc[i][mapping['Up']]
                right = data_traj.iloc[i][mapping['Right']]
                down = data_traj.iloc[i][mapping['Down']]
                left = data_traj.iloc[i][mapping['Left']]
                ob.append(up)
                ob.append(right)
                ob.append(down)
                ob.append(left)
                obs.append(ob)
                if i != traj_length-1:
                    act = [data_traj.iloc[i][mapping['Speed Change']],
                           data_traj.iloc[i][mapping['Direction Change']]]
                    acts.append(act)
                    infos.append([dataset.iloc[0][0], ID, timestep])
            trajectory = types.Trajectory(obs=np.array(obs), acts=np.array(acts), infos=np.array(infos), terminal=True)
            trajectories.append(trajectory)
            obs = []
            acts = []
            infos = []
            timestep += traj_length-1

train_traj, test_traj = train_test_split(trajectories, test_size=0.2, random_state=42)

serialize.save('./env/Rush_Data/Training Trajectories', train_traj)
serialize.save('./env/Rush_Data/Testing Trajectories', test_traj)