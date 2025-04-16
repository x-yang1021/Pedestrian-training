import torch
import torch.nn as nn
import torch.nn.functional as F

class SocialLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, grid_size=(4, 4), neighborhood_size=4.0, dropout=0.1, observed=15):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        self.obs_len = observed

        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.agent_attention = nn.MultiheadAttention(hidden_size, num_heads=1, batch_first=True)
        self.grid_q_proj = nn.Linear(hidden_size, hidden_size)
        self.grid_k_proj = nn.Linear(hidden_size, hidden_size)
        self.grid_v_proj = nn.Linear(hidden_size, hidden_size)

        self.social_pool_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.classifier = nn.Linear(hidden_size, 2)

    def get_social_grid_attention(self, hidden_states, hidden_target, positions, batch_size, reference_pos, mask=None):
        grid_cells = self.grid_size[0] * self.grid_size[1]
        cell_width = self.neighborhood_size / self.grid_size[0]
        cell_height = self.neighborhood_size / self.grid_size[1]
        half_grid_x = self.grid_size[0] // 2
        half_grid_y = self.grid_size[1] // 2

        ref_pos = reference_pos.squeeze(0)
        rel_positions = positions - ref_pos

        cell_contents = [[] for _ in range(grid_cells)]

        for j in range(batch_size):
            if mask is not None and mask[j] == 0:
                continue

            rel_x, rel_y = rel_positions[j]

            if (abs(rel_x) > self.neighborhood_size / 2) or (abs(rel_y) > self.neighborhood_size / 2):
                continue

            cell_x = (rel_x / cell_width).long() + half_grid_x
            cell_y = (rel_y / cell_height).long() + half_grid_y

            if 0 <= cell_x < self.grid_size[0] and 0 <= cell_y < self.grid_size[1]:
                idx = cell_y * self.grid_size[0] + cell_x
                cell_contents[idx].append(hidden_states[j].unsqueeze(0))

        cell_outputs = []
        for agents in cell_contents:
            if len(agents) > 0:
                agent_tensor = torch.cat(agents, dim=0).unsqueeze(0)  # shape [1, num_agents_in_cell, hidden_size]
                q = hidden_target.unsqueeze(0).unsqueeze(1)
                attn_out, _ = self.agent_attention(q, agent_tensor, agent_tensor)
                cell_outputs.append(attn_out.squeeze(0))  # [1, hidden_size]
            else:
                cell_outputs.append(torch.zeros(1, self.hidden_size, device=hidden_states.device))

        cell_stack = torch.cat(cell_outputs, dim=0)  # [grid_cells, hidden_size]
        # Project keys and values from cell outputs
        K = self.grid_k_proj(cell_stack)  # [N, D]
        V = self.grid_v_proj(cell_stack)  # [N, D]

        # Query: target agent's hidden state
        q = self.grid_q_proj(hidden_target).unsqueeze(0)  # [1, D]

        # Compute scaled dot-product attention
        d = self.hidden_size ** 0.5
        attn_scores = (q @ K.T) / d  # [1, N]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [1, N]

        # Weighted sum of values
        grid_context = attn_weights @ V

        return grid_context

    def forward(self, observed_trajectory_target, observed_trajectory_others, neighbor_mask=None):
        obs_len = len(observed_trajectory_others)

        hidden_target = torch.zeros(1, self.hidden_size, device=observed_trajectory_target[0].device)
        cell_target = torch.zeros(1, self.hidden_size, device=observed_trajectory_target[0].device)

        for t in range(obs_len):
            target_input = observed_trajectory_target[t].unsqueeze(0)
            target_pos = target_input[:, :2]

            hidden_target, cell_target = self.lstm_cell(target_input, (hidden_target, cell_target))
            hidden_target = self.dropout(hidden_target)

            others_input = observed_trajectory_others[t]
            if others_input.shape[0] > 0:
                others_pos = others_input[:, :2]
                hidden_others = torch.zeros(others_input.shape[0], self.hidden_size, device=others_input.device)
                cell_others = torch.zeros(others_input.shape[0], self.hidden_size, device=others_input.device)

                hidden_others, cell_others = self.lstm_cell(others_input, (hidden_others, cell_others))
                hidden_others = self.dropout(hidden_others)

                mask_t = neighbor_mask[t] if neighbor_mask is not None else None
                social_context = self.get_social_grid_attention(hidden_others, hidden_target, others_pos, others_input.shape[0], target_pos, mask=mask_t)
            else:
                social_context = torch.zeros_like(hidden_target)

            combined = hidden_target + social_context

        return self.classifier(combined)
