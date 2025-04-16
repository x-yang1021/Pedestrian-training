import torch
import torch.nn as nn
from Sophie import Sophie

impatient = False
OBS_LEN = 6
PRED_LEN = 9
EPOCHS = 10
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if impatient:
    train_data = torch.load("../Impatient/train_position.pt")
else:
    train_data = torch.load("../Patient/train_position.pt")



model = Sophie(use_scene=False, observed=OBS_LEN, predict=PRED_LEN).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()


for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for sample in train_data:
        full_target_traj, observed_trajectory_others = sample

        full_target_traj = torch.tensor(full_target_traj, dtype=torch.float).to(DEVICE)
        full_target = full_target_traj.unsqueeze(1)
        observed_trajectory_others = torch.tensor(observed_trajectory_others, dtype=torch.float)


        optimizer.zero_grad()
        pred = model(full_target, observed_trajectory_others, scene_map=None, train=True)
        loss = criterion(pred, full_target)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

if impatient:
    torch.save(model.state_dict(), "../Impatient/sophie.pth")
else:
    torch.save(model.state_dict(), "../Patient/sophie.pth")
print("Training done!")
