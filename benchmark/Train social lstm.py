import torch
import torch.nn as nn
import torch.optim as optim
from Social_LSTM import SocialLSTM

# Hyperparameters
obs_len = 6   # Length of observed trajectory
pred_len = 9  # Length of predicted trajectory
input_size = 2
hidden_size = 64
grid_size = (4, 4)
neighborhood_size = 4.0
dropout = 0.1
epochs = 4
learning_rate = 0.001

patient = True

# Load train_data, which is a list of samples
# Each element in train_data is of the form:
# [observed_trajectory_target, observed_trajectory_others]
# observed_trajectory_target: (obs_len+pred_len, 1, 2)
# observed_trajectory_others: (obs_len+pred_len, N_others, 2)
if not patient:
    train_data = torch.load('./Impatient/train_position.pt')
else:
    train_data = torch.load('./Patient/train_position.pt')

# Instantiate the model
model = SocialLSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    grid_size=grid_size,
    neighborhood_size=neighborhood_size,
    dropout=dropout,
    observed=obs_len,
    predict=pred_len
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model.train()  # Set the model to training mode
# model.cuda()

for epoch in range(epochs):
    epoch_loss = 0.0

    # Here we assume train_data is already a list of samples.
    # If you want to batch them, you'd need to create batches from train_data.
    # For simplicity, let's iterate sample by sample.
    for sample in train_data:
        observed_trajectory_target, observed_trajectory_others = sample

        # Convert lists to tensors
        observed_trajectory_target = torch.tensor(observed_trajectory_target, dtype=torch.float)
        observed_trajectory_others = torch.tensor(observed_trajectory_others, dtype=torch.float)
        observed_trajectory_target = observed_trajectory_target.unsqueeze(1)

        # Ensure the data is on the correct device if using GPU
        # observed_trajectory_target = observed_trajectory_target.cuda()
        # observed_trajectory_others = observed_trajectory_others.cuda()

        optimizer.zero_grad()

        # Forward pass
        # The model now expects:
        #   observed_trajectory_target: (obs_len, 1, 2) input steps
        #   observed_trajectory_others: (obs_len, N_others, 2) input steps
        # Make sure your data includes obs_len steps for observed and potentially pred_len for future prediction.
        # The model will produce (obs_len+pred_len, 1, 2) predictions for the target agent.
        outputs = model(observed_trajectory_target, observed_trajectory_others, train=True)

        # outputs shape: (obs_len+pred_len, 1, 2)
        # If observed_trajectory_target includes the full (obs_len+pred_len) trajectory,
        # we can compute loss directly:
        loss = criterion(outputs, observed_trajectory_target)

        # Backpropagation
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_data)
    print(f"Epoch [{epoch + 1}/{epochs}] - Average Loss: {avg_loss:.4f}")

# Save the model
if not patient:
    torch.save(model.state_dict(), './Impatient/social_lstm.pth')
else:
    torch.save(model.state_dict(), './Patient/social_lstm.pth')
print("Training complete.")
