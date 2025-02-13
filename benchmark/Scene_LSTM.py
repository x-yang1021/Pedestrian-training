import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneLSTM(nn.Module):
    def __init__(self, grid_size, input_size=2, hidden_size=64, scene_embedding_dim=64, dropout=0.1,
                 observed=8, predict=12):
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

        # CNN-based scene embedding (alternative: use learned embeddings)
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
        scene_grid = scene_grid.unsqueeze(0).unsqueeze(0)  # Add batch & channel dimensions
        scene_embedding = self.scene_encoder(scene_grid)
        return scene_embedding

    def forward(self, observed_trajectory, scene_grid, train=True):
        """
        Forward pass of Scene-LSTM for single-agent motion prediction in a static environment.

        Args:
            observed_trajectory: Tensor of shape (obs_len, 1, 2) -> (x, y) for a single agent
            scene_grid: Tensor of shape (grid_size[0], grid_size[1]) -> Encoded environment
            train: bool - If True, uses ground truth during prediction steps.

        Returns:
            outputs: Tensor of shape (total_len, 1, 2) -> Predicted trajectory
        """
        obs_len, _, _ = observed_trajectory.size()

        # Extract scene features
        scene_embedding = self.extract_scene_features(scene_grid)  # (1, scene_embedding_dim)

        # Initialize LSTM hidden and cell states
        hidden_state = torch.zeros(1, self.hidden_size, device=observed_trajectory.device)
        cell_state = torch.zeros(1, self.hidden_size, device=observed_trajectory.device)

        outputs = []

        # Process observed trajectory
        for t in range(self.obs_len):
            pos = observed_trajectory[t, 0, :].unsqueeze(0)  # (1, 2)

            # LSTM step
            hidden_state, cell_state = self.lstm_cell(pos, (hidden_state, cell_state))
            hidden_state = self.dropout(hidden_state)

            # Fuse scene information into the LSTM hidden state
            combined_state = torch.cat([hidden_state, scene_embedding], dim=1)
            fused_state = self.scene_fusion_mlp(combined_state)

            output = self.output_layer(fused_state)
            outputs.append(output.unsqueeze(0))  # (1, 1, 2)

        # Predict future steps
        for t in range(self.pred_len):
            future_step = t + self.obs_len
            if train:
                pos = observed_trajectory[future_step, 0, :].unsqueeze(0)
            else:
                # Use last predicted position
                pos = output.detach()

            # LSTM update
            hidden_state, cell_state = self.lstm_cell(pos, (hidden_state, cell_state))
            hidden_state = self.dropout(hidden_state)

            # Fuse scene features
            combined_state = torch.cat([hidden_state, scene_embedding], dim=1)
            fused_state = self.scene_fusion_mlp(combined_state)

            output = self.output_layer(fused_state)
            outputs.append(output.unsqueeze(0))  # (1, 1, 2)

        outputs = torch.cat(outputs, dim=0)  # (total_len, 1, 2)
        return outputs
