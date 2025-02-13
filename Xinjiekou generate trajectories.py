import numpy as np
import pandas as pd
import glob
from env.utils import get_transparency, get_wall_distance
import imitation.data.types as types
from imitation.data import serialize
from sklearn.model_selection import train_test_split

origin = [-455,52322]
North_wall = [(abs(-474 - origin[0]),52322-origin[1]), (abs(-474 - origin[0]),52468-origin[1])]
North_green = [(-455 - origin[0],52322-origin[1]), (-455 - origin[0],52468-origin[1])]
North_transparent = [(52337-origin[1],52344-origin[1]), (52401-origin[1], 52407-origin[1])]

South_origin = [-455, 52546]
South_wall = [(abs(-467 - South_origin[0]),52546-South_origin[1]), (abs(-467 - South_origin[0]), 52512-South_origin[1])]
South_green = [(-455-South_origin[0], 52546-South_origin[1]), (-455-South_origin[0], 52612-South_origin[1])]
South_transparent = [(52546-South_origin[1], 52581-South_origin[1])]

North = True
step_length = 4
episode_length = 13
# Load the data
if North:
    path = './Data/Xinjiekou/North'
    wall = North_wall
    green = North_green
    transparencies = North_transparent
    origin = origin
else:
    path = './Data/Xinjiekou/South'
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
    if df.shape[0] < 48:
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
        obs = []
        acts = []
        while i < df.shape[0] - 48:
            x = abs(df.iloc[i, 2] - origin[0])
            y = df.iloc[i, 4] - origin[1]
            if x < position_low[0] or x > position_high[0] or y < position_low[1] or y > position_high[1]:
                i += step_length
                obs = []
                acts = []
                continue
            wall_distance = get_wall_distance(x, wall)
            transparency = get_transparency(y, transparencies)
            ob = np.concatenate([np.array([x, y]), np.array([wall_distance]), np.array([transparency]), np.array([heading])])
            obs.append(ob)
            if len(obs) > 1:
                dist = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                speed = dist / 0.4
                direction = np.arctan2(y - prev_y, x - prev_x)
                act = np.array([speed, direction])
                acts.append(act)
            if len(obs) == episode_length:
                infos = [{}] * (len(obs)-1)
                trajectory = types.Trajectory(obs=np.array(obs), acts=np.array(acts), infos=np.array(infos),
                                              terminal=True)
                trajectories.append(trajectory)
                obs = []
                acts = []
            prev_x = x
            prev_y = y
            i += step_length
        cycle += 1

train_traj, test_traj = train_test_split(trajectories, test_size=0.2, random_state=42)

if North:
    serialize.save('./env/Xinjiekou_Data/North/Training Trajectories', train_traj)
    serialize.save('./env/Xinjiekou_Data/North/Testing Trajectories', test_traj)
else:
    serialize.save('./env/Xinjiekou_Data/South/Training Trajectories', train_traj)
    serialize.save('./env/Xinjiekou_Data/South/Testing Trajectories', test_traj)





