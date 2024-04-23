import imitation.data.types as types
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

data = pd.ExcelFile("./processed data.xlsx")
data2 = pd.read_excel(data, 'Group 2')
data3 = pd.read_excel(data, "Group 3")

dataset = pd.ExcelFile("./raw data.xlsx")
dataset2 = pd.read_excel(dataset, 'Group 2')
dataset3 = pd.read_excel(dataset, "Group 3")

obs = []
acts = []
infos = []
for col in range(1, dataset2.shape[1], 5):
    data_col = data.iloc[:,col:col+5]
    number = col%5
    for timestep in range(dataset2.shape[0]):
        data_row = dataset.iloc[timestep:timestep + 1, :]
        if data_col.iloc[timestep:timestep + 1, 0:1] is not None and (timestep - infos[-1][-1] == 1):
            x1 = data_col.iloc[timestep:timestep + 1,1:2]
            y1 = data_col.iloc[timestep:timestep + 1,2:3]
            speed = data_col.iloc[timestep:timestep + 1, 3:4]
            direction = data_col.iloc[timestep:timestep + 1, 4:5]
            obs.append([x1,y1])
            if len(obs) > 1:
                acts.append([speed,direction])
                infos.append([number,timestep])
            if len(obs) == 5:
                surrounding = get_surrounding(x1, y1, direction, timestep, dataset2, number)
                wall = get_wall(y1)
                green = get_green_space(y1)



trajectory = types.Trajectory(obs = obs,acts = acts, infos = None, terminal = terminal)

print(trajectory)