import imitation.data.types as types
from imitation.data import serialize
import numpy as np
from env.xinjiekou_env import withinSight, get_surrounding, get_wall, get_green_space
import pandas as pd


sun_unity = np.array([29375,8785])
green_start1 = (np.array([29536, 9352]) - sun_unity)/10
green_end1 = (np.array([29536,10510]) - sun_unity)/10
edge_start1 = (np.array([29920,9547]) - sun_unity)/10
edge_end1 = (np.array([29688,9734]) - sun_unity)/10
wall_start1 = (np.array([29688,9734]) - sun_unity)/10
wall_end1 = (np.array([29688,10392]) - sun_unity)/10
wall_start2 = (np.array([29735,10778]) - sun_unity)/10
wall_end2 = (np.array([29738,12794]) - sun_unity)/10
wall_mid1 = (np.array([29688,10076]) - sun_unity)/10
wall_mid2 = (np.array([29738,12525])- sun_unity)/10
episode_length = 5

data = pd.ExcelFile("./processed data.xlsx")
data2 = pd.read_excel(data, 'Group 2')
data3 = pd.read_excel(data, "Group 3")

dataset = pd.ExcelFile("./raw data.xlsx")
dataset2 = pd.read_excel(dataset, 'Group 2')
dataset3 = pd.read_excel(dataset, "Group 3")

obs = []
acts = []
infos = []
prev = None
number = 0
trajectories = []
for col in range(1, dataset2.shape[1], 5):
    data_col = data2.iloc[:,col:col+5]
    for timestep in range(dataset2.shape[0]):
        data_row = dataset2.iloc[timestep:timestep + 1, 1:]
        if pd.notna(data_col.iloc[timestep, 0]) and (prev is None or (timestep - prev) == 1):
            prev = timestep
            x1 = data_col.iloc[timestep,1]
            y1 = data_col.iloc[timestep,2]
            speed = data_col.iloc[timestep,3]
            direction = data_col.iloc[timestep,4]
            obs.append([x1,y1])
            if len(obs) > 1:
                acts.append([speed,direction])
                infos.append([2, number,timestep])
            if len(obs) == episode_length:
                for i in range(len(obs)):
                    x1 = obs[i][0]
                    y1 = obs[i][1]
                    # add destination
                    obs[i].append(obs[-1][0])
                    obs[i].append(obs[-1][1])
                    #add surrounding
                    surrounding = get_surrounding(x1, y1, direction, timestep, dataset2, number)
                    obs[i].append(surrounding[0])
                    obs[i].append(surrounding[1])
                    wall = get_wall(y1)
                    obs[i].append(wall[0])
                    green = get_green_space(y1)
                    obs[i].append(green[0])
                trajectory = types.Trajectory(obs = np.array(obs),acts = np.array(acts), infos = np.array(infos), terminal = True)
                trajectories.append(trajectory)
                obs = []
                acts = []
                infos = []
                prev = None
        else:
            obs = []
            acts = []
            infos = []
            prev = None
    number += 1

serialize.save('', trajectories)