import numpy as np
import pandas as pd
import glob
import torch
from env.utils import get_transparency, get_wall_distance
import imitation.data.types as types
from imitation.data import serialize
from sklearn.model_selection import train_test_split

origin = [-455,52322]
North_wall = [(abs(-474 - origin[0]),52322-origin[1]), (abs(-474 - origin[0]),52468-origin[1])]
North_green = [(-455 - origin[0],52322-origin[1]), (-455 - origin[0],52468-origin[1])]
North_transparent = [(52337-origin[1],52344-origin[1]), (52401-origin[1], 52407-origin[1])]

South_origin = [-455, 52546]
South_wall = [(abs(-467 - South_origin[0]),52546-South_origin[1]), (abs(-467 - South_origin[0]), 52612-South_origin[1])]
South_green = [(-455-South_origin[0], 52546-South_origin[1]), (-455-South_origin[0], 52612-South_origin[1])]
South_transparent = [(52546-South_origin[1], 52581-South_origin[1])]

North = False
step_length = 4
episode_length = 20
# Load the data
if North:
    path = '../Data/Xinjiekou/North'
    wall = North_wall
    green = North_green
    transparencies = North_transparent
    origin = origin
else:
    path = '../Data/Xinjiekou/South'
    wall = South_wall
    green = South_green
    transparencies = South_transparent
    origin = South_origin


position_high = np.array([wall[1][0], wall[1][1]], dtype=np.float32)
position_low = np.array([green[0][0], green[0][1]], dtype=np.float32)

trajectories = []

all_files = glob.glob(path + "/*.txt")
for file in all_files:
    df = pd.read_csv(file,sep="\t", header=None)
    if df.shape[0] < episode_length*step_length:
        continue
    distance = np.sqrt((df.iloc[-1, 2] - df.iloc[0, 2]) ** 2 + (df.iloc[-1, 4] - df.iloc[0, 4]) ** 2)
    if distance < 1:
        continue
    heading = int(df.iloc[-1, 4] - df.iloc[0, 4] > 0)
    cycle = 0
    while cycle < step_length:
        i = cycle
        prev_x = abs(df.iloc[i, 2] - origin[0])
        prev_y = df.iloc[i, 4] - origin[1]
        traj = []
        while i < df.shape[0] - step_length*episode_length:
            x = abs(df.iloc[i, 2] - origin[0])
            y = df.iloc[i, 4] - origin[1]
            if x < position_low[0] or x > position_high[0] or y < position_low[1] or y > position_high[1]:
                i += step_length
                traj = []
                continue
            traj.append([x,y])
            if len(traj) == episode_length:
                trajectories.append(traj)
                traj = []
            i += step_length
        cycle += 1

train_traj, test_traj = train_test_split(trajectories, test_size=0.2, random_state=1)

exit()

if North:
    torch.save(train_traj, './North/train_trajectory.pt')
    torch.save(test_traj, './North/test_trajectory.pt')
else:
    torch.save(train_traj, './South/train_trajectory.pt')
    torch.save(test_traj, './South/test_trajectory.pt')





