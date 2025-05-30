import imitation.data.types as types
from imitation.data import serialize
import numpy as np
from env.utils import withinSight, getPositions,getDensity, getFront
import pandas as pd
from sklearn.model_selection import train_test_split

Patient = True

if not Patient:
    cluster = 2
    distance_threshold = 9
else:
    cluster = 1
    distance_threshold = 2.58
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
traj_length = 10

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
            if pd.isna(data_col.iloc[timestep][mapping['Direction']]):
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
                x1 = data_traj.iloc[i][mapping['Positionx']]
                y1 = data_traj.iloc[i][mapping['Positiony']]
                speed = data_traj.iloc[i][mapping['Speed']]
                direction = data_traj.iloc[i][mapping['Direction']]

                # Get additional data
                positions = getPositions(dataset, timestep + i, ID, width)
                front = getFront(dataset, timestep + i, width, positions, x1, y1, direction)
                density = getDensity(positions, x1, y1, direction)
                up = data_traj.iloc[i][mapping['Up']]
                right = data_traj.iloc[i][mapping['Right']]
                down = data_traj.iloc[i][mapping['Down']]
                left = data_traj.iloc[i][mapping['Left']]
                if left or right or down:
                    rest = 1
                else:
                    rest = 0
                # Flattened observation
                ob = np.concatenate([
                    np.array([x1, y1]),  # position
                    np.array([data_traj.iloc[i][mapping['Distance']]]),  # distance
                    np.array([speed, direction]),  # self movement
                    np.array(front),  # front movement
                    np.array([density]),  # density
                    np.array([up, rest])  # contact
                ])

                obs.append(ob)
                if i != traj_length-1:
                    act = [data_traj.iloc[i][mapping['Speed Change']],
                           data_traj.iloc[i][mapping['Direction Change']]]
                    acts.append(act)
                    info = {
                        'experiment': dataset.iloc[0][0],
                        'ID': ID,
                        'timestep': timestep
                    }
                    infos.append(info)
            trajectory = types.Trajectory(obs=np.array(obs), acts=np.array(acts), infos=np.array(infos), terminal=True)
            trajectories.append(trajectory)
            obs = []
            acts = []
            infos = []
            timestep += traj_length-1

train_traj, test_traj = train_test_split(trajectories, test_size=0.2, random_state=42)

if not Patient:
    serialize.save('./env/Rush_Data/Impatient/Training Trajectories', train_traj)
    serialize.save('./env/Rush_Data/Impatient/Testing Trajectories', test_traj)
else:
    serialize.save('./env/Rush_Data/Patient/Training Trajectories', train_traj)
    serialize.save('./env/Rush_Data/Patient/Testing Trajectories', test_traj)