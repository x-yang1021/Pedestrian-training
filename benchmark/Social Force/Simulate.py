import pandas as pd
import numpy as np
import random
from collections import defaultdict
from Social_Force_model import Pedestrian
import sys
sys.path.append('../../')
from env.utils import withinSight, getPositions,getDensity, getFront, getContact_new
from utils import retrieveOriginalTrajectory

impatient = True
simulation_steps = 500
seed = 42
distance_threshold = 2.58
goal = [0,0]

if not impatient:
    df_shap = pd.read_csv('./Patient_Average_SHAP.csv')
    cluster = 1
else:
    df_shap = pd.read_csv('./Impatient_Average_SHAP.csv')
    cluster = 2

SHAP = dict()

for i in range(df_shap.shape[0]):
    SHAP[df_shap.iloc[i]['Feature']] = -df_shap.iloc[i]['Abs Mean'] * np.sign(df_shap.iloc[i]['Mean'])

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

df = pd.read_csv('../../Data/clustered.csv')


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

df_1 = pd.read_csv('../../env/Rush_Data/Experiment 1.csv')
df_2 = pd.read_csv('../../env/Rush_Data/Experiment 2.csv')
df_3 = pd.read_csv('../../env/Rush_Data/Experiment 3.csv')
dfs = [df_1,df_2,df_3]

step = 0
rng = np.random.default_rng(seed)
ADEs = []
FDEs = []
while step < simulation_steps:
    agent_seed = rng.integers(0, 2 ** 32)
    random.seed(seed)
    df = random.choice(dfs)
    time_col = df.iloc[:, 1]
    df_core = df.iloc[:, 2:]
    agent_count = df_core.shape[1] // width
    col = rng.integers(0, agent_count)*width
    ped = df_core.iloc[:,col:col+width]
    ped = pd.concat([ped, time_col],axis=1)
    ID = ped.iloc[0,mapping['ID']]
    traj_num = ped.iloc[:,mapping['Trajectory']].max()
    traj = rng.integers(1,traj_num+1)
    if not Cluster_Trajectories[ID] or traj not in Cluster_Trajectories[ID]:
        continue
    ped = ped[ped.iloc[:, mapping['Distance']] <= distance_threshold]
    ped = ped.dropna(subset=[ped.columns[mapping['Direction Change']]])
    if ped.shape[0] < traj_length:
        continue

    max_start_idx = ped.shape[0] - traj_length
    valid_starts = [i for i in range(max_start_idx + 1)
                    if ped.iloc[i:i + traj_length].shape[0] == traj_length]

    if not valid_starts:
        continue

    start_idx = random.choice(valid_starts)
    ped = ped.iloc[start_idx:start_idx + traj_length]
    initial_x = ped.iloc[0,mapping['Positionx']]
    initial_y = ped.iloc[0,mapping['Positiony']]
    initial_pos = np.array([initial_x, initial_y])
    initial_vel = ped.iloc[0,mapping['Speed']]
    initial_heading = ped.iloc[0,mapping['Direction']]
    v_pref = ped.iloc[0,mapping['Distance']]/(ped.shape[0] * 0.5)
    ego = Pedestrian(initial_pos, initial_vel, goal, SHAP, v_pref, initial_heading)
    time = ped.iloc[0, -1] / 0.5
    time_step = 1
    sim_traj = [initial_pos]
    real_traj = [initial_pos]
    while time_step < traj_length:
        f_drive = ego.compute_driving_force()
        positions = getPositions(df, time_step, ID, width)
        direction_flags = getContact_new(positions, ego.pos[0], ego.pos[1] ,ego.direction)
        f_repulsion = ego.compute_repulsion_force(direction_flags)
        density = getDensity(positions, ego.pos[0], ego.pos[1], ego.direction)
        f_density = ego.compute_density_force(density)
        total_force = (f_drive + f_repulsion + f_density) * 0.01
        ego.update(total_force)
        sim_traj.append(ego.pos.copy())
        real_x = ped.iloc[time_step, mapping['Positionx']]
        real_y = ped.iloc[time_step, mapping['Positiony']]
        real_traj.append(np.array([real_x, real_y]))
        time_step += 1
    sim_traj = np.array(sim_traj)
    real_trah = np.array(real_traj)
    ade = np.mean(np.linalg.norm(sim_traj - real_traj, axis=1))
    fde = np.linalg.norm(sim_traj[-1] - real_traj[-1])
    ADEs.append(ade)
    FDEs.append(fde)
    step += 1

print(np.mean(ADEs), np.mean(FDEs))

# impatient 0.44 1.37
# patient 0.30 1.31







