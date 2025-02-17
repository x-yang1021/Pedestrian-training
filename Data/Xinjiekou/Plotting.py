import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from env.utils import get_transparency, get_wall_distance
import imitation.data.types as types
from imitation.data import serialize
from sklearn.model_selection import train_test_split

origin = [-455,52322]
North_wall = [(-(-474 - origin[0]),52322-origin[1]), (-(-474 - origin[0]),52468-origin[1])]
North_green = [(-455 - origin[0],52322-origin[1]), (-455 - origin[0],52468-origin[1])]
North_transparent = [(52337-origin[1],52344-origin[1]), (52401-origin[1], 52407-origin[1])]

South_origin = [-455, 52612]
South_wall = [(-(-467 - South_origin[0]),52612-South_origin[1]), (-(-467 - South_origin[0]), -(52546-South_origin[1]))]
South_green = [(-455-South_origin[0], 52612-South_origin[1]), (-455-South_origin[0], -(52546-South_origin[1]))]
South_transparent = [(-(52581-South_origin[1]),-(52546-South_origin[1]))]

North = False
step_length = 4
episode_length = 13
# Load the data
if North:
    path = './North'
    wall = North_wall
    green = North_green
    transparencies = North_transparent
    origin = origin
else:
    path = './South New'
    wall = South_wall
    green = South_green
    transparencies = South_transparent
    origin = South_origin

position_high = np.array([wall[1][0], wall[1][1]], dtype=np.float32)
position_low = np.array([green[0][0], green[0][1]], dtype=np.float32)

plt.figure(figsize=(10, 8))

useful_traj = 0
all_files = glob.glob(path + "/*.txt")
for file in all_files:
    df = pd.read_csv(file,sep="\t", header=None)
    if df.shape[0] < 48:
        continue
    distance = np.sqrt((df.iloc[-1, 2] - df.iloc[0, 2]) ** 2 + (df.iloc[-1, 4] - df.iloc[0, 4]) ** 2)
    if distance < 1:
        continue
    if North:
        x = -(df.iloc[:, 2] - origin[0])
        y = df.iloc[:, 4] - origin[1]
    else:
        x = -(df.iloc[:, 2] - origin[0])
        y = -(df.iloc[:, 4] - origin[1])
    if y.max() > position_high[1] or y.min() < position_low[1] or x.max() > position_high[0] or x.min() < position_low[0]:
        continue
    useful_traj += 1
    #     print(file,df)
    # print(np.mean(y))

    plt.plot(x,y, marker='o', linestyle='-', alpha=0.5)
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.title("All Trajectories")
plt.show()
print(useful_traj)