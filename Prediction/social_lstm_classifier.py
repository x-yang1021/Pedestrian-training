import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class SocialLSTMClassifier(nn.Module):
    def __init__(
        self, input_size, hidden_size=64,
        dropout=0.0, observed=15,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.obs_len = observed

        # LSTM for the target trajectory
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # Normalize input
        self.input_norm = nn.LayerNorm(input_size)

        # Embed neighbors to hidden size
        self.neighbor_fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Attention over temporal features
        self.temporal_attn = nn.MultiheadAttention(hidden_size, num_heads=1, batch_first=True)

        # GAT for social interaction modeling
        self.gat_conv = GATConv(hidden_size, hidden_size, heads=1, concat=False)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, observed_trajectory_target, observed_trajectory_others):
        # observed_trajectory_target: [B, T, F]
        # observed_trajectory_others: list of list of [N_t, F] for each batch item

        device = observed_trajectory_target.device
        B, T, F = observed_trajectory_target.shape

        # Normalize input
        obs_target_norm = self.input_norm(observed_trajectory_target)

        # Encode target sequence using LSTM
        lstm_out, _ = self.lstm(obs_target_norm)  # [B, T, H]

        combined_history = []
        for b in range(B):
            combined_steps = []
            for t in range(self.obs_len):
                hidden_target = lstm_out[b, t].unsqueeze(0)  # [1, H]
                others_t = observed_trajectory_others[b][t]  # [N_t, F] or empty
                if others_t.size(0) == 0:
                    combined = hidden_target
                else:
                    others_norm = self.input_norm(others_t)  # [N_t, F]
                    hidden_others = self.neighbor_fc(others_norm)  # [N_t, H]
                    all_nodes = torch.cat([hidden_target, hidden_others], dim=0)  # [1+N_t, H]
                    edge_index = torch.tensor([
                        [0] * hidden_others.size(0) + list(range(1, 1 + hidden_others.size(0))),
                        list(range(1, 1 + hidden_others.size(0))) + [0] * hidden_others.size(0)
                    ], dtype=torch.long, device=device)
                    updated = self.gat_conv(all_nodes, edge_index)
                    combined = updated[0].unsqueeze(0)  # [1, H]

                combined_steps.append(combined.unsqueeze(1))  # [1, 1, H]
            combined_seq = torch.cat(combined_steps, dim=1)  # [1, T, H]
            combined_history.append(combined_seq)

        combined_batch = torch.cat(combined_history, dim=0)  # [B, T, H]
        query = combined_batch[:, -1:, :]  # [B, 1, H]
        attended, _ = self.temporal_attn(query, combined_batch, combined_batch)
        attended = attended.squeeze(1)  # [B, H]
        logits = self.classifier(attended)  # [B, 2]
        return logits
