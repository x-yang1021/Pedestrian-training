import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# CONFIG
CSV_PATH = "./dataset_with_cluster_masked.csv"
OUT_TRAIN = "./data/train_social_lstm_classifier.pt"
OUT_VAL   = "./data/val_social_lstm_classifier.pt"
OUT_TEST  = "./data/test_social_lstm_classifier.pt"

trajectory_features = [
    'Positionx', 'Positiony', 'Distance',
    'Speed', 'Speed Change', 'Direction', 'Direction Change'
]
traj_length = 6
distance_threshold = 5.0

# 1. LOAD & CLEAN
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=['ID'])
df = df.dropna(subset=trajectory_features + ['Trajectory', 'Time', 'Cluster', 'exp_num'])
df['Time_rounded'] = df['Time'].round(2)

# 2. NEIGHBOR HELPER
def get_neighbors_tensor(df, time_seq, target_id, traj_id, exp_num, features):
    df = df.copy()
    df[features] = df[features].apply(pd.to_numeric, errors='coerce')
    df['Time_rounded'] = df['Time'].round(2)

    neighbors_list = []
    mask_list = []

    for t in time_seq:
        frame = df[
            (df['Time_rounded'] == round(t, 2)) &
            (df['exp_num'] == exp_num)
        ].dropna(subset=features + ['ID', 'Trajectory'])

        frame = frame[
            (frame['ID'] != target_id) |
            (frame['Trajectory'] != traj_id)
        ]

        feats = []
        m = []
        for _, row in frame.iterrows():
            vals = row[features].values.astype(float)
            feats.append(torch.tensor(vals, dtype=torch.float32))
            m.append(1.0)
        if feats:
            neighbors_list.append(torch.stack(feats))
            mask_list.append(torch.tensor(m))
        else:
            neighbors_list.append(torch.zeros(0, len(features)))
            mask_list.append(torch.zeros(0))

    return neighbors_list, mask_list

# 3. BUILD SAMPLES
data = []
grouped = df.groupby(['ID', 'Trajectory', 'exp_num'])

for (pid, traj_id, exp_num), group in grouped:
    group = group.sort_values('Time').reset_index(drop=True)

    # Skip leading rows where Distance > 5
    start_idx = group[group['Distance'] <= distance_threshold].index.min()
    if pd.isna(start_idx):
        continue  # never enters crowd radius

    group = group.loc[start_idx:].reset_index(drop=True)

    traj_np = group[trajectory_features].values
    time_seq = group['Time'].tolist()
    n_segs = len(traj_np) // traj_length

    for i in range(n_segs):
        seg_slice = slice(i * traj_length, (i + 1) * traj_length)
        segment = traj_np[seg_slice]
        times = time_seq[seg_slice]

        if segment.shape[0] < traj_length:
            continue

        traj_tensor = torch.tensor(segment, dtype=torch.float32)
        neighs, mask = get_neighbors_tensor(
            df, times, pid, traj_id, exp_num, trajectory_features
        )

        if all(n.size(0) == 0 for n in neighs):
            continue

        sample = {
            'trajectory': traj_tensor,
            'neighbors': neighs,
            'neighbor_mask': mask,
            'cluster': int(group['Cluster'].iloc[0]),
        }
        data.append(sample)

print(f"Built {len(data)} samples (starting after entering crowd radius <= {distance_threshold}m)")

# 4. SPLIT INTO TRAIN / VAL / TEST (80/10/10)
train_data, temp_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=[s['cluster'] for s in data]
)

val_data, test_data = train_test_split(
    temp_data,
    test_size=0.5,
    random_state=42,
    stratify=[s['cluster'] for s in temp_data]
)

print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

# 5. SAVE
torch.save(train_data, OUT_TRAIN)
torch.save(val_data,   OUT_VAL)
torch.save(test_data,  OUT_TEST)
