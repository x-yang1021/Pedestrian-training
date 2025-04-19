import pandas as pd
import numpy as np

# Load clustered data
clsuter_path = '../Data/clustered.csv'
df_clsuter = pd.read_csv(clsuter_path)
df_clsuter = df_clsuter.dropna(subset=['ID', 'Trajectory', 'Cluster'])

# Ensure unique cluster assignment
grouped_cluster = df_clsuter.groupby(['ID', 'Trajectory'])['Cluster'].nunique().reset_index()
non_unique_cluster = grouped_cluster[grouped_cluster['Cluster'] > 1]
print(non_unique_cluster)  # Should be empty

unique_clusters = df_clsuter.groupby(['ID', 'Trajectory'])['Cluster'].first().reset_index()

# Load experiment files
exp1 = '../Data/Experiment 1.csv'
df_exp1 = pd.read_csv(exp1)

exp2 = '../Data/Experiment 2.csv'
df_exp2 = pd.read_csv(exp2)

exp3 = '../Data/Experiment 3.csv'
df_exp3 = pd.read_csv(exp3)

dfs = [df_exp1, df_exp2, df_exp3]
cluster_map = (
    unique_clusters
    .dropna(subset=['ID', 'Trajectory', 'Cluster'])
    .groupby(['ID', 'Trajectory'])['Cluster']
    .first()
    .reset_index()
)

# Merge clusters and compute speed/direction change
for i, df in enumerate(dfs):
    exp_num = i + 1

    df['ID'] = pd.to_numeric(df['ID'], errors='coerce')
    df['Trajectory'] = pd.to_numeric(df['Trajectory'], errors='coerce')
    cluster_map['ID'] = pd.to_numeric(cluster_map['ID'], errors='coerce')
    cluster_map['Trajectory'] = pd.to_numeric(cluster_map['Trajectory'], errors='coerce')

    df = df.merge(cluster_map, on=['ID', 'Trajectory'], how='left')

    df['Speed Change'] = df.groupby(['ID', 'Trajectory'])['Speed'].diff()
    df['Direction Change'] = df.groupby(['ID', 'Trajectory'])['Direction'].diff()

    df['exp_num'] = exp_num

    dfs[i] = df

combined_df = pd.concat(dfs, ignore_index=True)
combined_df.loc[combined_df['Cluster'].isna(), combined_df.columns.difference(['Time'])] = np.nan


# Label crowd radius function
def label_crowd_radius(df):
    crowd_radius = 2.58
    numeric_columns = df.select_dtypes(include=['object']).columns
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    df['Crowd_Radius_Label'] = np.nan

    for (participant_id, trajectory) in df[['ID', 'Trajectory']].dropna().drop_duplicates().itertuples(index=False):
        trajectory_section = df[(df['ID'] == participant_id) & (df['Trajectory'] == trajectory)].copy()

        df.loc[
            trajectory_section.index[trajectory_section['Distance'] <= crowd_radius], 'Crowd_Radius_Label'] = 'Inside'
        df.loc[
            trajectory_section.index[trajectory_section['Distance'] > crowd_radius], 'Crowd_Radius_Label'] = 'Outside'

        last_outside_idx = (
            trajectory_section[trajectory_section['Distance'] > crowd_radius].index[-1]
            if not trajectory_section[trajectory_section['Distance'] > crowd_radius].empty
            else None
        )

        if last_outside_idx is not None:
            inside_before_last_outside = trajectory_section.loc[
                (trajectory_section.index <= last_outside_idx) & (trajectory_section['Distance'] <= crowd_radius)
                ].index

            df.loc[inside_before_last_outside, 'Crowd_Radius_Label'] = 'Inside-Out'

    for (participant_id, trajectory) in df[['ID', 'Trajectory']].dropna().drop_duplicates().itertuples(index=False):
        trajectory_section = df[(df['ID'] == participant_id) & (df['Trajectory'] == trajectory)].copy()

        first_idx = trajectory_section.index[0]
        if trajectory_section.loc[first_idx, 'Crowd_Radius_Label'] == 'Outside':
            non_outside_rows = trajectory_section[~trajectory_section['Crowd_Radius_Label'].isin(['Outside'])]
            if not non_outside_rows.empty:
                first_non_outside_idx = non_outside_rows.index.min()
                cols_to_mask = [col for col in df.columns if col != 'Time']
                df.loc[trajectory_section.loc[first_non_outside_idx:].index, cols_to_mask] = np.nan
        else:
            cols_to_mask = [col for col in df.columns if col != 'Time']
            df.loc[trajectory_section.index, cols_to_mask] = np.nan

    return df


# Apply labeling and export
df_all = label_crowd_radius(combined_df)

# Optional: diagnostics
retained_rows = df_all.groupby(['ID', 'Trajectory']).apply(lambda x: x.dropna().shape[0])
retained_df = retained_rows.reset_index()
retained_df.columns = ['ID', 'Trajectory', 'Non_NaN_Count']

merged = unique_clusters.merge(retained_df, on=['ID', 'Trajectory'], how='left')
missing_trajectories = merged[(merged['Non_NaN_Count'].isna()) | (merged['Non_NaN_Count'] == 0)]

print(
    f"Number of (ID, Trajectory) pairs present in unique_cluster but with no retained rows in retained_rows: {missing_trajectories.shape[0]}")
print(missing_trajectories[['ID', 'Trajectory']])

# Cluster distribution
df_valid = df_all.dropna(subset=['ID', 'Trajectory', 'Cluster'])
unique_traj = df_valid[['ID', 'Trajectory', 'Cluster']].drop_duplicates()
cluster_counts = unique_traj.groupby('Cluster').size().reset_index(name='Num_Trajectories')
print("Unique (ID, Trajectory) count per Cluster:")
print(cluster_counts)

# Save output
df_all.to_csv("./dataset_with_cluster_masked.csv", index=False)
print("Saved to dataset_with_cluster_masked.csv")
