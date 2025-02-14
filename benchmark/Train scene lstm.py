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
epochs = 200
learning_rate = 0.0001

if North:
    wall = North_wall
    green = North_green
    transparencies = North_transparent
    train_data =  torch.load('./North/train_trajectory.pt')
else:
    wall = South_wall
    green = South_green
    transparencies = South_transparent
    train_data = torch.load('./South/train_trajectory.pt')

num_cols = (wall[0][0] + 1)//2
num_rows = (wall[1][1] + 1)//2
grid_size = (num_rows, num_cols)

grid = torch.zeros(num_rows, num_cols, dtype=torch.float32)
grid[:,0] = 1
grid[:,-1] = 2
for transparent in transparencies:
    grid[transparent[0]//2:(transparent[1]+1)//2, -1] = 3

model = SceneLSTM(
    grid_size=grid_size,
    observed=obs_len,
    predict=pred_len
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model.train()  # Set the model to training mode

for epoch in range(epochs):
    epoch_loss = 0.0

    # Here we assume train_data is already a list of samples.
    # If you want to batch them, you'd need to create batches from train_data.
    # For simplicity, let's iterate sample by sample.
    for observed_trajectory in train_data:
        observed_trajectory = torch.tensor(observed_trajectory, dtype=torch.float)
        observed_trajectory = observed_trajectory.unsqueeze(1)

        optimizer.zero_grad()

        # Forward pass
        # The model now expects:
        #   observed_trajectory_target: (obs_len, 1, 2) input steps
        #   observed_trajectory_others: (obs_len, N_others, 2) input steps
        # Make sure your data includes obs_len steps for observed and potentially pred_len for future prediction.
        # The model will produce (obs_len+pred_len, 1, 2) predictions for the target agent.
        outputs = model(observed_trajectory, grid, train=True)

        # outputs shape: (obs_len+pred_len, 1, 2)
        # If observed_trajectory_target includes the full (obs_len+pred_len) trajectory,
        # we can compute loss directly:
        loss = criterion(outputs, observed_trajectory)

        # Backpropagation
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_data)
    print(f"Epoch [{epoch + 1}/{epochs}] - Average Loss: {avg_loss:.4f}")

# Save the model
if North:
    torch.save(model.state_dict(), './North/scene_lstm.pth')
else:
    torch.save(model.state_dict(), './South/scene_lstm.pth')
print("Training complete.")
