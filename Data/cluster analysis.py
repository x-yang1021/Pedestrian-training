import pandas as pd
import numpy as np


df = pd.read_csv('./Cluster dataset.csv')


#normalization

df['Speed Change'] = 2 * (df['Speed Change'] - df['Speed Change'].min()) / (df['Speed Change'].max() - df['Speed Change'].min()) - 1
df['Direction Change'] = 2 * (df['Direction Change'] - df['Direction Change'].min()) / (df['Direction Change'].max() - df['Direction Change'].min()) - 1

ID = df.iloc[0]['ID']
trajectory = 1
series = []
traj = []
for i in range(df.shape[0]):
    if df.iloc[i]['ID'] != ID and traj:
        series.append(traj)
        ID = df.iloc[i]['ID']
        traj = []
    else:
        if df.iloc[i]['Trajectory'] != trajectory and traj:
            series.append(traj)
            trajectory = df.iloc[i]['Trajectory']
            traj = []
        if not traj:
            traj.append([ID, trajectory])
        traj.append([df.iloc[i]['Speed Change'], df.iloc[i]['Direction Change']])





