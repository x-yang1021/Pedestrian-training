from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from social_lstm_classifier import SocialLSTMClassifier

# Configuration
TRAIN_PATH = "./data/train_social_lstm_classifier.pt"
TEST_PATH  = "./data/test_social_lstm_classifier.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feature definitions
trajectory_features = [
    'Positionx', 'Positiony', 'Distance',
    'Speed', 'Speed Change', 'Direction', 'Direction Change'
]

feature_combos = {
    # 'xy_2':   ['Positionx', 'Positiony'],
    'sd_4': ['Speed', 'Direction', 'Speed Change', 'Direction Change'],
    # 'xyd_3':  ['Positionx', 'Positiony', 'Distance'],
    # 'xysd_5': ['Positionx', 'Positiony', 'Speed', 'Direction', 'Distance'],
    # 'xysd_4': ['Positionx', 'Positiony', 'Speed', 'Direction'],
}

# Load test data
test_data = torch.load(TEST_PATH)

for name, feats in feature_combos.items():
    results = []
    index_map = [trajectory_features.index(f) for f in feats]
    input_size = len(feats)
    obs_len = test_data[0]['trajectory'].shape[0]

    # Instantiate model
    model = SocialLSTMClassifier(
        input_size=input_size,
        hidden_size=128,
        dropout=0.0,
        observed=obs_len
    ).to(device)
    model.load_state_dict(torch.load(f"./model/{name}_social_lstm_classifier.pth"))
    model.eval()

    y_true, y_prob = [], []
    with torch.no_grad():
        for sample in tqdm(test_data, desc=f"Evaluating {name}"):
            traj_full = sample['trajectory'][:, index_map].to(device)
            neighbors = []
            for t in range(obs_len):
                if len(sample['neighbors']) > t:
                    neighbors.append(sample['neighbors'][t][:, index_map].to(device))
                else:
                    neighbors.append(torch.empty(0, input_size, device=device))

            obs_traj = traj_full.unsqueeze(0)  # [1, T, F]
            neighbors_batch = [neighbors]  # batch of one

            logits = model(obs_traj, neighbors_batch)  # [1, 2]
            prob = torch.softmax(logits, dim=1)[0, 1].item()
            y_prob.append(prob)
            y_true.append(sample['cluster'] - 1)

    # Threshold tuning
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    # Final predictions using best threshold
    y_pred = [1 if p > best_threshold else 0 for p in y_prob]

    acc = accuracy_score(y_true, y_pred)
    rep = classification_report(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print(f"\nBest Threshold: {best_threshold:.3f}, Best F1: {best_f1:.3f}")
    print(f"\nResults for feature set {name}:")
    print("Accuracy: {:.3f}".format(acc))
    print("Classification Report:\n", rep)
    print("Confusion Matrix:\n", cm)

    # Plot F1 vs threshold
    plt.plot(thresholds, f1_scores[:-1])
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title(f"F1 Score vs. Threshold for {name}")
    plt.grid(True)
    plt.show()