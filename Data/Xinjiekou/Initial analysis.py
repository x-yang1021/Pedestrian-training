import numpy as np
import pandas as pd
import glob

total_traj = 0
short_traj = 0
non_moving = 0
times = []
distances = []
useful_traj = 0

# Load the data
path = './North'
all_files = glob.glob(path + "/*.txt")
for file in all_files:
    df = pd.read_csv(file,sep="\t", header=None)
    total_traj += 1
    length = df.shape[0]
    if length < 45:
        short_traj += 1
        continue
    distance = np.sqrt((df.iloc[-1,1] - df.iloc[0,1])**2 + (df.iloc[-1,2] - df.iloc[0,2])**2)
    useful_traj += length // 45
    if distance < 1:
        non_moving += 1
    else:
        time = df.iloc[-1, 0] - df.iloc[0, 0]
        times.append(time / 10)
        distances.append(distance)

print(f'Total trajectories: {total_traj}')
print(f'Short trajectories: {short_traj}')
print(f'Non-moving trajectories: {non_moving}')
print(f'Average time: {np.mean(times)}')
print(f'Average distance: {np.mean(distances)}')
print(f'Useful trajectories: {useful_traj}')


