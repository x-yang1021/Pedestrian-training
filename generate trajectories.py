import imitation.data.types as types
from imitation.data import serialize
import numpy as np
from env.rush_env import withinSight, getPositions,getDensity, getFront
import pandas as pd


cluster = 2
width = 11
length = 8

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
        if data_col.iloc[0]['ID'] not in Cluster_IDs:
            continue
        ID = data_col.iloc[0]['ID']
        timestep = 0
        while timestep < dataset.shape[0] - length:
            if pd.isna(data_col.iloc[timestep]['Speed']):
                timestep += 1
                continue
            if data_col.iloc[timestep]['Trajectory'] not in Cluster_Trajectories[ID]:
                timestep += 1
                continue
            elif pd.isna(data_col.iloc[timestep+length]['Trajectory']):
                timestep += 1
                continue
            elif data_col.iloc[timestep]['Trajectory'] != data_col.iloc[timestep+length]['Trajectory']:
                timestep += 1
                continue
            data_traj = data_col.iloc[timestep:timestep+length]
            for i in range(data_traj.shape[0]):
                ob = []
                x1 = data_traj.iloc[i]['Positionx']
                y1 = data_traj.iloc[i]['Positiony']
                ob.append(x1)
                ob.append(y1)
                ob.append(data_traj.iloc[timestep+length-1]['Positionx'])
                ob.append(data_traj.iloc[timestep+length-1]['Positiony'])
                ob.append(data_traj.iloc[i]['Distance'])
                ob.append(data_traj.iloc[i]['Speed'])
                direction = data_traj.iloc[i]['Direction']
                ob.append(direction)
                positions = getPositions(dataset, timestep+i, ID, width)
                front = getFront(dataset,timestep+i,width,positions,x1,y1,direction)
                density = getDensity(positions,x1,y1,direction)
                ob.append(front)
                ob.append(density)
                up = data_traj.iloc[i]['Up']
                right = data_traj.iloc[i]['Right']
                down = data_traj.iloc[i]['Down']
                left = data_traj.iloc[i]['Left']
                ob.append([up,right,down,left])
                obs.append(ob)
                if i != length-1:
                    act = [data_traj.iloc[i]['Speed Change'],
                           data_traj.iloc[i]['Direction Change']]
                    acts.append(acts)
            info = [dataset.iloc[0]['Experiment'], ID, timestep * 0.5]
            trajectory = types.Trajectory(obs=np.array(obs), acts=np.array(acts), infos=np.array(infos), terminal=True)
            trajectories.append(trajectory)
            obs = []
            acts = []
            infos = []
            timestep += length-1
print(trajectories)











serialize.save('', trajectories)