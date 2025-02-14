import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneLSTM(nn.Module):
    def __init__(self, grid_size, input_size=2, hidden_size=64, scene_embedding_dim=64, dropout=0.1,
                 observed=6, predict=8):
        super().__init__()
        self.input_size = input_size  # Typically (x, y) coordinates
        self.hidden_size = hidden_size
        self.grid_size = grid_size  # Defines the spatial representation of the environment
        self.scene_embedding_dim = scene_embedding_dim
        self.obs_len = observed
        self.pred_len = predict

        # LSTM for individual agent's trajectory
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # CNN-based scene embedding
        self.scene_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(grid_size[0] * grid_size[1] * 64, scene_embedding_dim),
            nn.ReLU()
        )

        # Scene-aware fusion module (combines agent's LSTM state + scene embedding)
        self.scene_fusion_mlp = nn.Sequential(
            nn.Linear(hidden_size + scene_embedding_dim, hidden_size),
            nn.ReLU()
        )

        # Output layer to predict next position
        self.output_layer = nn.Linear(hidden_size, input_size)

    def extract_scene_features(self, scene_grid):
        """
        Extracts scene features from a given environment representation (e.g., occupancy grid).
        """
        scene_grid = scene_grid.unsqueeze(0).unsqueeze(0)  # Add batch & channel dimensions: (1, 1, H, W)
        scene_embedding = self.scene_encoder(scene_grid)     # (1, scene_embedding_dim)
        return scene_embedding

    def forward(self, observed_trajectory, scene_grid, train=True):
        """
        Forward pass of Scene-LSTM for single-agent motion prediction in a static environment.

        Args:
            observed_trajectory: Tensor of shape (total_len, 1, 2) where total_len = obs_len + pred_len.
                (During inference, observed_trajectory should contain at least the observed part.)
            scene_grid: Tensor of shape (grid_size[0], grid_size[1]) -> Encoded environment.
            train: bool - If True, uses ground truth during prediction steps (teacher forcing).

        Returns:
            outputs: Tensor of shape (total_len, 1, 2) when train is True, or
                     (pred_len, 1, 2) when train is False.
        """
        # Extract scene features (remains constant during the trajectory)
        scene_embedding = self.extract_scene_features(scene_grid)  # (1, scene_embedding_dim)

        # Initialize LSTM hidden and cell states
        hidden_state = torch.zeros(1, self.hidden_size, device=observed_trajectory.device)
        cell_state = torch.zeros(1, self.hidden_size, device=observed_trajectory.device)

        observed_outputs = []  # To store outputs from the observed steps (only used if train=True)

        # Process the observed trajectory
        for t in range(self.obs_len):
            pos = observed_trajectory[t, 0, :].unsqueeze(0)  # (1, 2)

            # LSTM step
            hidden_state, cell_state = self.lstm_cell(pos, (hidden_state, cell_state))
            hidden_state = self.dropout(hidden_state)

            # Fuse scene information into the LSTM hidden state
            combined_state = torch.cat([hidden_state, scene_embedding], dim=1)
            fused_state = self.scene_fusion_mlp(combined_state)

            output = self.output_layer(fused_state)

            if train:
                # Store the output of the observed steps only during training
                observed_outputs.append(output.unsqueeze(0))

        predicted_outputs = []  # To store predicted future outputs

        # Predict future steps
        for t in range(self.pred_len):
            future_step = t + self.obs_len
            if train:
                # Teacher forcing: use the ground truth future position as input
                pos = observed_trajectory[future_step, 0, :].unsqueeze(0)
            else:
                # Inference: use the last predicted position as input (autoregressive)
                pos = output.detach()

            # LSTM update
            hidden_state, cell_state = self.lstm_cell(pos, (hidden_state, cell_state))
            hidden_state = self.dropout(hidden_state)

            # Fuse scene features
            combined_state = torch.cat([hidden_state, scene_embedding], dim=1)
            fused_state = self.scene_fusion_mlp(combined_state)

            output = self.output_layer(fused_state)
            predicted_outputs.append(output.unsqueeze(0))

        if train:
            # Return full sequence: observed + predicted
            outputs = torch.cat(observed_outputs + predicted_outputs, dim=0)  # (obs_len + pred_len, 1, 2)
        else:
            # Return only the predicted future trajectory
            outputs = torch.cat(predicted_outputs, dim=0)  # (pred_len, 1, 2)

        return outputs
