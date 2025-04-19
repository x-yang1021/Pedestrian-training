import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import random
from social_lstm_classifier import SocialLSTMClassifier  # ensure this path is correct

# Configuration
TRAIN_PATH = "./data/train_social_lstm_classifier.pt"
TEST_PATH  = "./data/test_social_lstm_classifier.pt"
VAL_PATH   = "./data/val_social_lstm_classifier.pt"
LR         = 1e-3
EPOCHS     = 200
BATCH_SIZE = 16
val_step   = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feature definitions
trajectory_features = [
    'Positionx', 'Positiony', 'Distance',
    'Speed', 'Speed Change', 'Direction', 'Direction Change'
]
feature_combos = {
    # 'xy_2':   ['Positionx', 'Positiony'],
    'sd_4':   ['Speed', 'Direction','Speed Change', 'Direction Change'],
    # 'xyd_3':  ['Positionx', 'Positiony', 'Distance'],
    # 'xysd_5': ['Positionx', 'Positiony', 'Speed', 'Direction', 'Distance'],
    # 'xysd_4': ['Positionx', 'Positiony', 'Speed', 'Direction'],
    # 'xyds_4': ['Positionx', 'Positiony', 'Distance', 'Speed'],
    # 'xydd_4': ['Positionx', 'Positiony', 'Distance', 'Direction'],
    # 'xys_3':  ['Positionx', 'Positiony', 'Speed'],
    # 'xydi_3': ['Positionx', 'Positiony', 'Direction']
}

# Load data
train_data = torch.load(TRAIN_PATH)
test_data  = torch.load(TEST_PATH)
val_data = torch.load(VAL_PATH)

# train_data = random.sample(train_data, 10)

# class0 = [sample for sample in train_data if sample['cluster'] == 1]
# class1 = [sample for sample in train_data if sample['cluster'] == 2]
# min_len = min(len(class0), len(class1))
# class0_down = random.sample(class0, min_len)
# class1_down = random.sample(class1, min_len)
#
# balanced_data = class0_down + class1_down
# random.shuffle(balanced_data)
#
# train_data = balanced_data

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha)
            else:
                self.alpha = alpha
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, reduction='none')  # [B]
        pt = torch.exp(-ce_loss)  # [B]

        if self.alpha is not None:
            if logits.is_cuda:
                self.alpha = self.alpha.to(logits.device)
            at = self.alpha.gather(0, labels.view(-1))  # alpha per sample
            loss = at * (1 - pt) ** self.gamma * ce_loss
        else:
            loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

results = []

for name, feats in feature_combos.items():
    print(f"\n*** Running feature set: {name} ***")
    index_map = [trajectory_features.index(f) for f in feats]
    input_size = len(feats)
    obs_len = train_data[0]['trajectory'].shape[0]

    model = SocialLSTMClassifier(
        input_size=input_size,
        hidden_size=128,
        dropout=0.0,
        observed=obs_len
    ).to(device)

    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(alpha=torch.tensor([0.35,0.65]), gamma=2.0, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        random.shuffle(train_data)

        for i in tqdm(range(0, len(train_data), BATCH_SIZE), desc=f"{name} Epoch {epoch}/{EPOCHS}"):
            batch = train_data[i:i+BATCH_SIZE]
            B = len(batch)

            obs_traj = torch.stack([sample['trajectory'][:, index_map] for sample in batch]).to(device)  # [B, T, F]
            neighbor_traj = []
            for sample in batch:
                neighbors_seq = []
                for t in range(obs_len):
                    if len(sample['neighbors']) > t:
                        n = sample['neighbors'][t][:, index_map]
                        neighbors_seq.append(n.to(device))
                    else:
                        neighbors_seq.append(torch.empty(0, input_size, device=device))
                neighbor_traj.append(neighbors_seq)  # List[T x N_t x F]

            labels = torch.tensor([sample['cluster'] - 1 for sample in batch], dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits = model(obs_traj, neighbor_traj)  # [B, 2]
            # print("Logits:", logits)
            # print("Softmax:", torch.softmax(logits, dim=1))
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: grad norm = {param.grad.norm().item()}")
            optimizer.step()
            total_loss += loss.item() * B

        avg_loss = total_loss / len(train_data)
        print(f"[{name}] Epoch {epoch}/{EPOCHS} — Avg Loss: {avg_loss:.4f}")

        # === Validation step ===
        if epoch % val_step == 0:
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for i in range(0, len(val_data), BATCH_SIZE):
                    batch = val_data[i:i + BATCH_SIZE]
                    B = len(batch)

                    obs_traj = torch.stack([sample['trajectory'][:, index_map] for sample in batch]).to(device)
                    neighbor_traj = []
                    for sample in batch:
                        neighbors_seq = []
                        for t in range(obs_len):
                            if len(sample['neighbors']) > t:
                                n = sample['neighbors'][t][:, index_map]
                                neighbors_seq.append(n.to(device))
                            else:
                                neighbors_seq.append(torch.empty(0, input_size, device=device))
                        neighbor_traj.append(neighbors_seq)

                    labels = torch.tensor([sample['cluster'] - 1 for sample in batch], dtype=torch.long, device=device)
                    logits = model(obs_traj, neighbor_traj)
                    preds = torch.argmax(logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            acc = accuracy_score(all_labels, all_preds)
            print(f"[{name}] Validation @ Epoch {epoch}: Accuracy = {acc:.4f}")
            print(classification_report(all_labels, all_preds, digits=4))

    torch.save(model.state_dict(), f"./model/{name}_social_lstm_classifier.pth")

print("\n=== Feature Combination Results ===")
df = pd.DataFrame(results)
print(df)
df.to_csv("./results/feature_combination_results.csv", index=False)