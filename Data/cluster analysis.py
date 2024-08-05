import pandas as pd
import numpy as np


df = pd.read_csv('./Cluster dataset.csv')


print(df['Speed Change'].max())
print(df[df['Speed Change'] == df['Speed Change'].max()].index)
print(df['Speed Change'].min())
print(df[df['Speed Change'] == df['Speed Change'].min()].index)
print(df['Direction Change'].max())
print(df[df['Direction Change'] == df['Direction Change'].max()].index)
print(df['Direction Change'].min())
print(df[df['Direction Change'] == df['Direction Change'].min()].index)

exit()

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

# print(series)



