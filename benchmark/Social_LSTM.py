import torch
import torch.nn as nn
import torch.nn.functional as F

class SocialLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, grid_size=(4, 4), neighborhood_size=4.0, dropout=0.1, observed = 6, predict = 8):
        super().__init__()
        self.input_size = input_size  # Typically x and y coordinates
        self.hidden_size = hidden_size
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        self.obs_len = observed
        self.pred_len = predict
        # LSTM for individual trajectories
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # MLP for social pooling
        self.social_pool_mlp = nn.Sequential(
            nn.Linear(hidden_size * grid_size[0] * grid_size[1], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # Output layer to predict the next position
        self.output_layer = nn.Linear(hidden_size, input_size)

    def get_social_grid(self, hidden_states, positions, batch_size):
        """
        Constructs the social pooling grid.

        Args:
            hidden_states: Tensor of shape (batch_size, hidden_size)
            positions: Tensor of shape (batch_size, 2)
            batch_size: int

        Returns:
            social_tensor: Tensor of shape (batch_size, hidden_size * grid_cells)
        """
        grid_cells = self.grid_size[0] * self.grid_size[1]
        social_tensor = torch.zeros(batch_size, grid_cells, self.hidden_size).to(hidden_states.device)

        # Calculate cell indices
        min_x = positions[:, 0].min()
        min_y = positions[:, 1].min()
        max_x = positions[:, 0].max()
        max_y = positions[:, 1].max()

        cell_width = (max_x - min_x) / self.grid_size[0]
        cell_height = (max_y - min_y) / self.grid_size[1]

        cell_x = ((positions[:, 0] - min_x) / cell_width).long()
        cell_y = ((positions[:, 1] - min_y) / cell_height).long()

        cell_x = torch.clamp(cell_x, 0, self.grid_size[0] - 1)
        cell_y = torch.clamp(cell_y, 0, self.grid_size[1] - 1)

        cell_indices = cell_y * self.grid_size[0] + cell_x  # Flatten 2D grid to 1D index

        for i in range(batch_size):
            idx = cell_indices[i]
            social_tensor[i, idx] += hidden_states[i]

        social_tensor = social_tensor.view(batch_size, -1)
        return social_tensor

    def forward(self, observed_trajectory, train = True):
        """
        Forward pass of the Social LSTM for multi-step prediction.

        Args:
            observed_trajectory: Tensor of shape (total_step, batch_size, 2)
            Train: If True, include prediction during the observed trajectory

        Returns:
            outputs: Tensor of shape (total_len, batch_size, 2)
        """
        total_len, batch_size, _ = observed_trajectory.size()

        # Initialize hidden and cell states
        hidden_states = torch.zeros(batch_size, self.hidden_size).to(observed_trajectory.device)
        cell_states = torch.zeros(batch_size, self.hidden_size).to(observed_trajectory.device)

        outputs = []

        # Process the observed trajectory
        for t in range(self.obs_len):
            position = observed_trajectory[t]

            # Individual LSTM step
            hidden_states, cell_states = self.lstm_cell(position, (hidden_states, cell_states))
            hidden_states = self.dropout(hidden_states)

            # Social pooling
            social_tensor = self.get_social_grid(hidden_states, position, batch_size)
            social_context = self.social_pool_mlp(social_tensor)

            # Combine individual and social context
            combined = hidden_states + social_context

            # Predict next position
            if train:
                output = self.output_layer(combined)
                outputs.append(output.unsqueeze(0))

        # Predict future steps
        for t in range(self.pred_len):
            if total_len > self.obs_len:
                # Use ground truth position if within observed trajectory length
                position = observed_trajectory[t+self.obs_len]
            else:
                # Use the last predicted position
                position = output.detach()

            # Individual LSTM step
            hidden_states, cell_states = self.lstm_cell(position, (hidden_states, cell_states))
            hidden_states = self.dropout(hidden_states)

            # Social pooling
            social_tensor = self.get_social_grid(hidden_states, position, batch_size)
            social_context = self.social_pool_mlp(social_tensor)

            # Combine individual and social context
            combined = hidden_states + social_context

            # Predict next position
            output = self.output_layer(combined)
            outputs.append(output.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs
