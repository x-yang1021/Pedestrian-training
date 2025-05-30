import torch
import numpy as np
import torch.nn as nn
from Social_LSTM import SocialLSTM  # Assuming you've defined your model here

# Hyperparameters (must match those used during training)
obs_len = 6
pred_len = 9
input_size = 2
hidden_size = 64
grid_size = (4, 4)
neighborhood_size = 4.0
dropout = 0.1

patient = True
# Load the trained model
if not patient:
    model_path = '../Impatient/social_lstm.pth'
    test_data = torch.load('../Impatient/test_position.pt')
else:
    model_path = '../Patient/social_lstm.pth'
    test_data = torch.load('../Patient/test_position.pt')

model = SocialLSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    grid_size=grid_size,
    neighborhood_size=neighborhood_size,
    dropout=dropout,
    observed=obs_len,
    predict=pred_len
)
model.load_state_dict(torch.load(model_path))
model.eval()  # Switch to evaluation mode



avg_mses = []
final_mses = []
with torch.no_grad():
    for sample in test_data:
        observed_trajectory_target, observed_trajectory_others = sample

        # Convert lists to tensors
        observed_trajectory_target = torch.tensor(observed_trajectory_target, dtype=torch.float)
        actual_traj = observed_trajectory_target[obs_len:].numpy()
        observed_trajectory_others = torch.tensor(observed_trajectory_others, dtype=torch.float)
        observed_trajectory_target = observed_trajectory_target.unsqueeze(1)

        # Forward pass through the model
        # Set train=False to ensure it uses only the observed steps to predict the future
        predict_traj = model(observed_trajectory_target, observed_trajectory_others, train=False)
        predict_traj = predict_traj.squeeze(1).numpy()
        avg_mse = np.mean(np.sqrt(np.sum((predict_traj - actual_traj) ** 2, axis=1)))
        final_mse = np.sqrt(np.sum((predict_traj[-1] - actual_traj[-1]) ** 2))
        avg_mses.append(avg_mse)
        final_mses.append(final_mse)

print('Average MSE:', 'Mean', np.mean(avg_mses), 'Min', np.min(avg_mses), 'Max', np.max(avg_mses), '25th', np.percentile(avg_mses, 25), '75th', np.percentile(avg_mses, 75))
print('Final MSE:', 'Mean', np.mean(final_mses), 'Min', np.min(final_mses), 'Max', np.max(final_mses), '25th', np.percentile(final_mses, 25), '75th', np.percentile(final_mses, 75))

# impatient 0.77 1.21
# Patient 0.41 0.64