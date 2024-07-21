import pandas as pd
import numpy as np
import glob
import os

column_names = ["Time", "ID", "Positionx","Positionz","Positiony", "Yaw", "Up", "Right", "Down", "Left"]
shorest_episode = 10
max_range = 11
teleport_range = 7
speed_limit = 2.5 # per half second

all_files = glob.glob(os.path.join('./Experiment 3 data', "*.csv"))
total_traj = 0
entire_data = pd.DataFrame()
for file in all_files:
    df = pd.read_csv(file,names=column_names)
    df['Positionx'] = df['Positionx'].astype(str).str.replace(r'[()]', '', regex=True)
    df['Positiony'] = df['Positiony'].astype(str).str.replace(r'[()]', '', regex=True)
    df['Yaw'] = df['Yaw'].astype(str).str.replace('Yaw:', '').astype(float)
    df[['Positionx', 'Positiony', 'Time','Yaw']] = df[['Positionx', 'Positiony', 'Time','Yaw']].apply(pd.to_numeric)
    ID = df['ID'][1]
    df['Trajectory'] = int(1)
    df['Distance'] = np.nan
    df['Speed'] = np.nan
    print(df.shape[0], ID, df.iloc[-1]['Time'] - df.iloc[0]['Time'])
    origin = df.iloc[0, 0]
    traj_count = 1
    #add speed, distance, remove
    for i in range(df.shape[0]):
        #specifically for 53
        if df.iloc[i]['Positionx'] == -1000:
            df.iloc[i, 1:] = np.nan
        df.iloc[i, 0] = df.iloc[i, 0] - origin
        if pd.notna(df.iloc[i]['Positionx']):
            df.at[i,'Distance'] = np.sqrt((df.iloc[i]['Positionx']-0)**2
                                             +(df.iloc[i]['Positiony']-0)**2)
        if i > 0:
            if pd.notna(df.iloc[i]['Positionx']) and pd.notna(df.iloc[i-1]['Positionx']):
                curr_coor = np.array([df.iloc[i]['Positionx'], df.iloc[i]['Positiony']])
                prev_coor = np.array([df.iloc[i-1]['Positionx'], df.iloc[i-1]['Positiony']])
                last_move = np.linalg.norm(curr_coor-prev_coor)
                df.at[i, 'Speed'] = (last_move)/(df.iloc[i]['Time']-df.iloc[i-1]['Time'])
            else:
                last_move = df.iloc[i]['Distance']
            if teleport_range > last_move > speed_limit:
                df.iloc[i - 1:i, 1:] = np.nan
            elif last_move > teleport_range and df.iloc[i-1]['Distance']:
                df.iloc[i-6:i,1:] = np.nan
                traj_count += 1
        if pd.notna(df.iloc[i]['Trajectory']):
            df.at[i,'Trajectory'] = traj_count
    df.to_csv('./Experiment 3 data/processed data/%s old.csv' % (ID), index=False)

    consec_zero = 0
    # zeros = []
    for i in range(df.shape[0]):
        #remove point that is too far away
        if df.iloc[i]['Distance'] > max_range:
            df.iloc[i,1:] = np.nan
        #remove last zeros
        if pd.notna(df.iloc[i]['Speed']):
            if df.iloc[i]['Speed'] == 0 :
                consec_zero += 1
            else:
                consec_zero = 0
        else:
            consec_zero = 0
        if consec_zero and i == df.shape[0]-1:
            df.iloc[i-consec_zero+1:i+1, 1:] = np.nan
    #         elif df.iloc[i]['Speed'] != 0 and consec_zero >= stop_time and df.iloc[i]['Distance'] > 7:
    #             if df.iloc[i]['Distance'] < 8:
    #                 print(df.iloc[i]['Distance'], ID)
    #             df.iloc[i - consec_zero:i, 1:] = np.nan
    #             zeros.append(consec_zero)
    #             consec_zero = 0
    #         else:
    #             consec_zero = 0
    # for i in range(df.shape[0]-1):
    #     if pd.notna(df.iloc[i]['ID']) and pd.isna(df.iloc[i+1]['ID']):
    #         if df.iloc[i]['Distance'] > longest_step: #traj not properly end in gate
    #             abandon_traj = df.iloc[i]['Trajectory']
    #             df.loc[df['Trajectory'] == abandon_traj, df.columns[1:]] = np.nan
    # #remove the last traj if not properly end in gate
    # if df.iloc[-1]['Distance'] > longest_step:
    #     abandon_traj = df.iloc[-1]['Trajectory']
    #     df.loc[df['Trajectory'] == abandon_traj, df.columns[1:]] = np.nan
    #remove zeros at the beginning and too short traj
    initial_zeros = 0
    traj_length = 0
    curr_traj = 0
    for i in range(df.shape[0]):
        if pd.isna(df.iloc[i]['Trajectory']):
            if traj_length < shorest_episode and traj_length >0:
                df.iloc[i - traj_length:i, 1:] = np.nan
            if initial_zeros > 0:
                df.iloc[i - initial_zeros - 1:i-1, 1:] = np.nan
            initial_zeros = 0
            traj_length = 0
            continue
        if pd.notna(df.iloc[i]['Trajectory']):
            if traj_length == 0:  # at the beginning of trajectory
                curr_traj = df.iloc[i]['Trajectory']
                traj_length += 1
            else:
                if traj_length == 1 and df.iloc[i]['Speed'] < 0.05 and df.iloc[i]['Distance'] > 4:
                    initial_zeros = 1
                elif initial_zeros > 0:
                    if df.iloc[i]['Speed'] < 0.05:
                        initial_zeros += 1
                    else:
                        df.iloc[i - initial_zeros-1:i-1, 1:] = np.nan
                        traj_length -= initial_zeros
                        initial_zeros = 0
                traj_length += 1
    if traj_length < shorest_episode and traj_length > 0:
        df.loc[df['Trajectory'] == curr_traj, df.columns[1:]] = np.nan
    if initial_zeros > 0:
        df.iloc[i - initial_zeros - 1:i-1, 1:] = np.nan
    curr_traj = 0
    traj_length = 0
    traj_map = {}
    for i in range(df.shape[0]):
        if pd.isna(df.iloc[i]['Trajectory']):
            traj_length = 0
            continue
        if pd.notna(df.iloc[i]['Trajectory']):
            if traj_length == 0:  # at the beginning of trajectory
                df.at[i, 'Speed'] = np.nan
                curr_traj += 1
                old_traj = df.iloc[i]['Trajectory']
                if old_traj not in traj_map:
                    traj_map[old_traj] = curr_traj
                traj_length += 1
            else:
                traj_length += 1
    # Apply the trajectory mapping
    df['Trajectory'] = df['Trajectory'].map(traj_map)
    total_traj += curr_traj
    df.to_csv('./Experiment 3 data/processed data/%s new.csv' % (ID), index=False)
    entire_data = pd.concat([entire_data,df], ignore_index = True)
print(total_traj)
entire_data.to_csv('./Experiment 3 data/processed data/Experiment 3.csv', index=False)