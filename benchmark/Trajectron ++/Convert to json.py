import pandas as pd
import json

# Load all CSV files
df_1 = pd.read_csv('../env/Rush_Data/Experiment 1.csv')
df_2 = pd.read_csv('../env/Rush_Data/Experiment 2.csv')
df_3 = pd.read_csv('../env/Rush_Data/Experiment 3.csv')
dfs = [df_1, df_2, df_3]

# List to store all scenes
all_scenes = []

scene_id = 1  # Start scene IDs from 1

for df in dfs:
    # Drop rows without valid agent IDs
    df = df.dropna(subset=["ID"]).copy()
    df["ID"] = df["ID"].astype(int)

    # Rename columns for consistency
    df = df.rename(columns={
        "Time": "timestep",
        "Positionx": "x",
        "Positiony": "y"
    })

    # Build scene structure
    scene = {"id": scene_id, "objects": []}
    for agent_id, group in df.groupby("ID"):
        group = group.sort_values("timestep")
        obj = {
            "id": str(agent_id),
            "type": "pedestrian",
            "timesteps": group["timestep"].tolist(),
            "x": group["x"].tolist(),
            "y": group["y"].tolist(),
        }
        scene["objects"].append(obj)

    all_scenes.append(scene)
    scene_id += 1  # Increment for the next scene

# Final JSON structure
trajectron_data = {"scenes": all_scenes}

# Save combined JSON
json_path = "../env/Rush_Data/trajectron_combined.json"
with open(json_path, "w") as f:
    json.dump(trajectron_data, f, indent=4)

