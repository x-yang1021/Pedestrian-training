import numpy as np
import pandas as pd
import glob
import os

default_direction = np.arctan2(1,0)

exp_1 = glob.glob(os.path.join('./Data/processed data/1', "*.csv"))
exp_2 = glob.glob(os.path.join('./Data/processed data/2', "*.csv"))
exp_3 = glob.glob(os.path.join('./Data/processed data/3', "*.csv"))

exps = [exp_1,exp_2,exp_3]

for exp in exps:
    if exp == exp_1:
        exp_num = 1
    elif exp == exp_2:
        exp_num = 2
    elif exp == exp_3:
        exp_num = 3
    dfs = []
    for data in exp:
        df = pd.read_csv(data)
        df['Speed Change'] = np.nan
        df['Direction Change'] = np.nan
        df['Experiment'] = exp_num
        for i in range(df.shape[0] - 1):
            if pd.notna(df.iloc[i + 1]['Speed']):
                if pd.notna(df.iloc[i]['Speed']):
                    df.at[i, 'Speed Change'] = df.iloc[i + 1]['Speed'] - df.iloc[i]['Speed']
                else:
                    df.at[i, 'Speed Change'] = df.iloc[i]['Speed']
            if pd.notna(df.iloc[i + 1]['Direction']):
                diff = df.iloc[i + 1]['Direction'] - df.iloc[i]['Direction']
                # Wrap the difference to the range -pi to pi
                if diff > np.pi:
                    diff -= 2 * np.pi
                elif diff < -np.pi:
                    diff += 2 * np.pi
                df.at[i, 'Direction Change'] = diff
            else:
                diff = df.iloc[i + 1]['Direction'] - default_direction
                if diff > np.pi:
                    diff -= 2 * np.pi
                elif diff < -np.pi:
                    diff += 2 * np.pi
                df.at[i, 'Direction Change'] = diff
        new_order = ['Experiment','Time', 'ID', 'Trajectory', 'Positionx','Positionz','Positiony', 'Yaw',
                     'Distance', 'Up', 'Right', 'Down', 'Left','Speed','Speed Change',
                     'Direction','Direction Change']
        df = df[new_order]
        df = df.drop(columns=['Yaw','Positionz'])
        if dfs:
            df = df.drop(columns=['Time','Experiment'])
        dfs.append(df)
    dataset = pd.concat(dfs, axis=1)
    dataset.to_csv('./env/Rush_Data/Experiment %s.csv' %exp_num,index=False)
