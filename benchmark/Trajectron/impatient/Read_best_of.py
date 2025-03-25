import os
import pandas as pd

base = "/home/xiangmin/Documents/GitHub/Pedestrian-training/benchmark/Trajectron/impatient/results"
ade_file = os.path.join(base, "impatient_rush_ade_full.csv")
fde_file = os.path.join(base, "impatient_rush_fde_full.csv")

for file in [ade_file, fde_file]:
    if not os.path.exists(file):
        print(f"❌ File not found: {file}")
        exit(1)

ade_mean = pd.read_csv(ade_file)['value'].mean()
fde_mean = pd.read_csv(fde_file)['value'].mean()

print(f"✅ ADE (best of): {ade_mean:.4f}")
print(f"✅ FDE (best of): {fde_mean:.4f}")
