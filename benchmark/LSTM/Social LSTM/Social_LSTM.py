import torch
import torch.nn as nn
import torch.nn.functional as F


class SocialLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, grid_size=(4, 4), neighborhood_size=4.0, dropout=0.1,
                 observed=6, predict=8):
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

    def get_social_grid(self, hidden_states, positions, batch_size, reference_pos):
        """
        Constructs the social pooling grid for the target agent.
        Only includes agents that fall within the neighborhood defined by self.neighborhood_size.
        """
        grid_cells = self.grid_size[0] * self.grid_size[1]
        social_tensor = torch.zeros(1, grid_cells, self.hidden_size, device=hidden_states.device)

        # Dimensions of each cell in the grid
        cell_width = self.neighborhood_size / self.grid_size[0]
        cell_height = self.neighborhood_size / self.grid_size[1]

        half_grid_x = self.grid_size[0] // 2
        half_grid_y = self.grid_size[1] // 2

        ref_pos = reference_pos.squeeze(0)  # shape: (2,)

        # Compute relative positions of other agents with respect to the target
        rel_positions = positions - ref_pos  # shape: (N_others, 2)

        for j in range(batch_size):
            rel_x, rel_y = rel_positions[j]

            # Check if the other agent is within the neighborhood region
            if (abs(rel_x) > self.neighborhood_size / 2) or (abs(rel_y) > self.neighborhood_size / 2):
                # Skip this agent as it lies outside the defined neighborhood
                continue

            # Compute which cell this agent falls into
            cell_x = (rel_x / cell_width).long() + half_grid_x
            cell_y = (rel_y / cell_height).long() + half_grid_y

            if 0 <= cell_x < self.grid_size[0] and 0 <= cell_y < self.grid_size[1]:
                idx = cell_y * self.grid_size[0] + cell_x
                social_tensor[0, idx] += hidden_states[j]

        # Flatten to (1, hidden_size * grid_cells)
        social_tensor = social_tensor.view(1, -1)
        return social_tensor

    def forward(self, observed_trajectory_target, observed_trajectory_others, train=True):
        """
        Forward pass of the Social LSTM for multi-step prediction of the target agent.

        Args:
            observed_trajectory_target: Tensor of shape (obs_len, 1, 2)
            observed_trajectory_others: Tensor of shape (obs_len, N_others, 2)
            train: bool - If True, uses ground truth during prediction steps.

        Returns:
            outputs: Tensor of shape (total_len, 1, 2)
        """
        obs_len, _, _ = observed_trajectory_target.size()
        _, N_others, _ = observed_trajectory_others.size()

        # Initialize hidden and cell states for the target agent
        hidden_target = torch.zeros(1, self.hidden_size, device=observed_trajectory_target.device)
        cell_target = torch.zeros(1, self.hidden_size, device=observed_trajectory_target.device)

        # Initialize hidden and cell states for the other agents
        if N_others > 0:
            hidden_others = torch.zeros(N_others, self.hidden_size, device=observed_trajectory_others.device)
            cell_others = torch.zeros(N_others, self.hidden_size, device=observed_trajectory_others.device)
        else:
            hidden_others = None
            cell_others = None

        outputs = []

        # Process the observed trajectory
        for t in range(self.obs_len):
            target_pos = observed_trajectory_target[t, 0, :].unsqueeze(0)  # (1,2)

            # Target agent LSTM step
            hidden_target, cell_target = self.lstm_cell(target_pos, (hidden_target, cell_target))
            hidden_target = self.dropout(hidden_target)

            # Other agents LSTM step
            if N_others > 0:
                others_pos = observed_trajectory_others[t, :, :]  # (N_others, 2)
                hidden_others, cell_others = self.lstm_cell(others_pos, (hidden_others, cell_others))
                hidden_others = self.dropout(hidden_others)

                # Social pooling
                social_tensor = self.get_social_grid(hidden_others, others_pos, N_others, reference_pos=target_pos)
                social_context = self.social_pool_mlp(social_tensor)
            else:
                social_context = torch.zeros_like(hidden_target)

            combined = hidden_target + social_context
            output = self.output_layer(combined)

            if train:
                outputs.append(output.unsqueeze(0))  # (1,1,2)

        # Predict future steps
        for t in range(self.pred_len):
            future_step = t + self.obs_len
            if train:
                target_pos = observed_trajectory_target[future_step, 0, :].unsqueeze(0)
            else:
                # Use the last predicted position
                target_pos = output.detach()

            # Target agent update
            hidden_target, cell_target = self.lstm_cell(target_pos, (hidden_target, cell_target))
            hidden_target = self.dropout(hidden_target)

            # Other agents update as if ground truth is known
            if N_others > 0:
                # Use the corresponding future_step for others
                others_pos = observed_trajectory_others[future_step, :, :]  # (N_others, 2)

                hidden_others, cell_others = self.lstm_cell(others_pos, (hidden_others, cell_others))
                hidden_others = self.dropout(hidden_others)

                social_tensor = self.get_social_grid(hidden_others, others_pos, N_others, reference_pos=target_pos)
                social_context = self.social_pool_mlp(social_tensor)
            else:
                social_context = torch.zeros_like(hidden_target)

            combined = hidden_target + social_context
            output = self.output_layer(combined)
            outputs.append(output.unsqueeze(0))  # (1,1,2)

        outputs = torch.cat(outputs, dim=0)  # (total_len, 1, 2)
        return outputs
