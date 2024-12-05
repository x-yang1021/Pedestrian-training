import torch
import torch.nn as nn
import torch.optim as optim
from Social_LSTM import SocialLSTM

# Assuming the SocialLSTM class from your provided code is already defined above this line.
# For reference, we use the code you finalized earlier.

# Hyperparameters
obs_len = 6  # Length of the observed trajectory
pred_len = 8  # Number of steps to predict
input_size = 2  # (x, y)
hidden_size = 64
grid_size = (4, 4)
neighborhood_size = 4.0
dropout = 0.1
batch_size = 32
num_batches = 10  # Just a small number of batches for demonstration
epochs = 5
learning_rate = 0.001

# Total length of the trajectory (observed + predicted)
total_len = obs_len + pred_len


# Create random synthetic data for testing
# Shape: (total_len, batch_size, 2)
# For demonstration, we create multiple batches of synthetic data.
def generate_random_trajectory(total_len, batch_size, input_size):
    # For simplicity, random points in a 10x10 area
    return torch.rand(total_len, batch_size, input_size) * 10.0


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

for epoch in range(epochs):
    epoch_loss = 0.0

    for batch_i in range(num_batches):
        # Generate a new random batch
        observed_trajectory = generate_random_trajectory(total_len, batch_size, input_size)

        # Move data to GPU if available (optional)
        # observed_trajectory = observed_trajectory.cuda()
        # model.cuda()

        optimizer.zero_grad()

        # Forward pass
        # The model expects (total_len, batch_size, 2) and returns predictions
        # If train=True, it will output predictions for both observed and future steps
        outputs = model(observed_trajectory, train=True)
        # outputs shape: (obs_len + pred_len, batch_size, 2)

        # For scenario A:
        # We generated full ground truth trajectory (observed + future)
        # We can compute the loss on all steps:
        loss = criterion(outputs, observed_trajectory)

        # Backpropagation
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / num_batches
    print(f"Epoch [{epoch + 1}/{epochs}] - Average Loss: {avg_loss:.4f}")

print("Training complete.")
