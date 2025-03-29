import torch
import torch.nn as nn
import torch.nn.functional as F

class Sophie(nn.Module):
    def __init__(self,
                 input_size=2,
                 hidden_size=64,
                 grid_size=(4, 4),
                 map_channels=4,
                 use_scene=True,
                 use_social=True,
                 dropout=0.1,
                 observed=6,
                 predict=8):
        super().__init__()
        self.use_scene = use_scene
        self.use_social = use_social
        self.obs_len = observed
        self.pred_len = predict
        self.grid_size = grid_size
        self.hidden_size = hidden_size

        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Scene encoding
        if use_scene:
            self.scene_embedding = nn.Embedding(num_embeddings=10, embedding_dim=map_channels)
            self.scene_cnn = nn.Sequential(
                nn.Conv2d(map_channels, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.scene_fc = nn.Linear(32, hidden_size)

        # Social attention
        if use_social:
            attn_dim = hidden_size
            self.query_layer = nn.Linear(hidden_size, attn_dim)
            self.key_layer = nn.Linear(hidden_size, attn_dim)
            self.value_layer = nn.Linear(hidden_size, attn_dim)
            self.temperature = attn_dim ** 0.5

        self.output_layer = nn.Linear(hidden_size, input_size)

    def get_scene_feature(self, scene_map):
        embedded = self.scene_embedding(scene_map.long().unsqueeze(0))  # (1, H, W, C)
        embedded = embedded.permute(0, 3, 1, 2)  # (1, C, H, W)
        scene_feature = self.scene_cnn(embedded)  # (1, 32, 1, 1)
        scene_feature = scene_feature.view(1, -1)  # (1, 32)
        return self.scene_fc(scene_feature)  # Project to hidden_size

    def get_social_attention(self, hidden_target, hidden_others):
        q = self.query_layer(hidden_target)        # (1, D)
        k = self.key_layer(hidden_others)          # (N, D)
        v = self.value_layer(hidden_others)        # (N, D)
        attn_weights = F.softmax(torch.matmul(q, k.T) / self.temperature, dim=1)  # (1, N)
        attended = torch.matmul(attn_weights, v)    # (1, D)
        return attended

    def forward(self, observed_trajectory_target, observed_trajectory_others, scene_map=None, train=True):
        obs_len, _, _ = observed_trajectory_target.size()
        _, N_others, _ = observed_trajectory_others.size()

        hidden_target = torch.zeros(1, self.hidden_size, device=observed_trajectory_target.device)
        cell_target = torch.zeros(1, self.hidden_size, device=observed_trajectory_target.device)

        if self.use_social and N_others > 0:
            hidden_others = torch.zeros(N_others, self.hidden_size, device=observed_trajectory_others.device)
            cell_others = torch.zeros(N_others, self.hidden_size, device=observed_trajectory_others.device)

        outputs = []

        for t in range(self.obs_len + self.pred_len):
            if t < self.obs_len:
                target_pos = observed_trajectory_target[t, 0, :].unsqueeze(0)
            elif train:
                target_pos = observed_trajectory_target[t, 0, :].unsqueeze(0)
            else:
                target_pos = output.detach()

            hidden_target, cell_target = self.lstm_cell(target_pos, (hidden_target, cell_target))
            hidden_target = self.dropout(hidden_target)

            # Social context
            if self.use_social and N_others > 0:
                others_pos = observed_trajectory_others[min(t, obs_len - 1), :, :]
                hidden_others, cell_others = self.lstm_cell(others_pos, (hidden_others, cell_others))
                hidden_others = self.dropout(hidden_others)
                social_context = self.get_social_attention(hidden_target, hidden_others)
            else:
                social_context = torch.zeros_like(hidden_target)

            # Scene feature
            if self.use_scene and scene_map is not None:
                scene_feat = self.get_scene_feature(scene_map)
            else:
                scene_feat = torch.zeros_like(hidden_target)

            combined = hidden_target + social_context + scene_feat
            output = self.output_layer(combined)
            outputs.append(output.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs
