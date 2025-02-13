import torch
import torch.nn as nn
import torch.optim as optim
from Scene_LSTM import SceneLSTM


origin = [-455,52322]
North_wall = [(abs(-474 - origin[0]),52322-origin[1]), (abs(-474 - origin[0]),52468-origin[1])]
North_green = [(-455 - origin[0],52322-origin[1]), (-455 - origin[0],52468-origin[1])]
North_transparent = [(52337-origin[1],52344-origin[1]), (52401-origin[1], 52407-origin[1])]

South_origin = [-455, 52546]
South_wall = [(abs(-467 - South_origin[0]),52546-South_origin[1]), (abs(-467 - South_origin[0]), 52512-South_origin[1])]
South_green = [(-455-South_origin[0], 52546-South_origin[1]), (-455-South_origin[0], 52612-South_origin[1])]
South_transparent = [(52546-South_origin[1], 52581-South_origin[1])]

# Hyperparameters
North = True
obs_len = 8   # Length of observed trajectory
pred_len = 12  # Length of predicted trajectory

if North:
    wall = North_wall
    green = North_green
    transparencies = North_transparent
else:
    wall = South_wall
    green = South_green
    transparencies = South_transparent

num_cols = wall[0][0] + 1
num_rows = wall[1][1] + 1
grid_size = (num_rows, num_cols)

grid = torch.zeros(num_rows, num_cols, dtype=torch.float32)
grid[:,0] = 1
grid[:,-1] = 2
for transparent in transparencies:
    grid[transparent[0]:transparent[1]+1, -1] = 3

model = SceneLSTM(
    grid_size=grid_size,
    observed=obs_len,
    predict=pred_len
)
