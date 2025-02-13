import torch
import torch.nn as nn
import torch.optim as optim
from Scene_LSTM import SceneLSTM


origin = [-455,52322]
North_wall = [(abs(-474 - origin[0]),52322-origin[1]), (abs(-474 - origin[0]),52468-origin[1])]
North_green = [(-455 - origin[0],52322-origin[1]), (-455 - origin[0],52468-origin[1])]
North_transparent = [(52337-origin[1],52344-origin[1]), (52401-origin[1], 52407-origin[1])]

South_origin = [-455, 52546]
South_wall = [(abs(-467 - South_origin[0]),52546-South_origin[1]), (abs(-467 - South_origin[0]), 52612-South_origin[1])]
South_green = [(-455-South_origin[0], 52546-South_origin[1]), (-455-South_origin[0], 52612-South_origin[1])]
South_transparent = [(52546-South_origin[1], 52581-South_origin[1])]

# Hyperparameters
North = False
obs_len = 8   # Length of observed trajectory
pred_len = 12  # Length of predicted trajectory
epochs = 4
learning_rate = 0.001

if North:
    wall = North_wall
    green = North_green
    transparencies = North_transparent
    model_path = './North/scene_lstm.pth'
    test_data = torch.load('./North/test_trajectory.pt')
else:
    wall = South_wall
    green = South_green
    transparencies = South_transparent
    model_path = './South/scene_lstm.pth'
    test_data = torch.load('./South/test_trajectory.pt')

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

model.load_state_dict(torch.load(model_path))
model.eval()  # Switch to evaluation mode

avg_mses = []
final_mses = []
with torch.no_grad():
    for observed_trajectory in test_data:
        # Convert lists to tensors
        observed_trajectory = torch.tensor(observed_trajectory, dtype=torch.float)
        actual_traj = observed_trajectory[obs_len:].numpy()
        observed_trajectory = observed_trajectory.unsqueeze(1)

        # Forward pass through the model
        # Set train=False to ensure it uses only the observed steps to predict the future
        predict_traj = model(observed_trajectory, grid, train=False)
        predict_traj = predict_traj.squeeze(1).numpy()
        avg_mse = np.mean(np.sqrt(np.sum((predict_traj - actual_traj) ** 2, axis=1)))
        final_mse = np.sqrt(np.sum((predict_traj[-1] - actual_traj[-1]) ** 2))
        avg_mses.append(avg_mse)
        final_mses.append(final_mse)

print('Average MSE:', 'Mean', np.mean(avg_mses), 'Min', np.min(avg_mses), 'Max', np.max(avg_mses), '25th', np.percentile(avg_mses, 25), '75th', np.percentile(avg_mses, 75))
print('Final MSE:', 'Mean', np.mean(final_mses), 'Min', np.min(final_mses), 'Max', np.max(final_mses), '25th', np.percentile(final_mses, 25), '75th', np.percentile(final_mses, 75))
