import torch
import numpy as np
import torch.nn as nn
from Sophie import Sophie

# Hyperparameters
obs_len = 6
pred_len = 9
input_size = 2
hidden_size = 64
dropout = 0.1
grid_size = (4, 4)
map_channels = 4

patient = True
use_scene = False
use_social = True

# Load model and data
if not patient:
    model_path = '../Impatient/sophie.pth'
    test_data = torch.load('../Impatient/test_position.pt')
else:
    model_path = '../Patient/sophie.pth'
    test_data = torch.load('../Patient/test_position.pt')

# Instantiate the model
model = Sophie(
    input_size=input_size,
    hidden_size=hidden_size,
    grid_size=grid_size,
    map_channels=map_channels,
    use_scene=use_scene,
    use_social=use_social,
    dropout=dropout,
    observed=obs_len,
    predict=pred_len
)
model.load_state_dict(torch.load(model_path))
model.eval()

avg_mses = []
final_mses = []

with torch.no_grad():
    for sample in test_data:
        observed_trajectory_target, observed_trajectory_others = sample

        # Convert to tensors
        observed_trajectory_target = torch.tensor(observed_trajectory_target, dtype=torch.float)
        observed_trajectory_target = observed_trajectory_target.unsqueeze(1)
        observed_trajectory_others = torch.tensor(observed_trajectory_others, dtype=torch.float)

        actual_traj = observed_trajectory_target[obs_len:].squeeze(1).numpy()

        # Predict future
        predicted = model(observed_trajectory_target[:obs_len], observed_trajectory_others[:obs_len], scene_map=None, train=False)
        predicted = predicted[obs_len:].squeeze(1).numpy()

        avg_mse = np.mean(np.sqrt(np.sum((predicted - actual_traj) ** 2, axis=1)))
        final_mse = np.sqrt(np.sum((predicted[-1] - actual_traj[-1]) ** 2))
        avg_mses.append(avg_mse)
        final_mses.append(final_mse)

print('Average MSE:', 'Mean', np.mean(avg_mses), 'Min', np.min(avg_mses), 'Max', np.max(avg_mses),
      '25th', np.percentile(avg_mses, 25), '75th', np.percentile(avg_mses, 75))
print('Final MSE:', 'Mean', np.mean(final_mses), 'Min', np.min(final_mses), 'Max', np.max(final_mses),
      '25th', np.percentile(final_mses, 25), '75th', np.percentile(final_mses, 75))

#impatient 0.73 1.17
#patient 0.33 0.56


















