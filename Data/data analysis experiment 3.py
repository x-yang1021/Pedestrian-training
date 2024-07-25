import pandas as pd
import numpy as np
import glob
import os

column_names = ["Time", "ID", "Positionx","Positionz","Positiony", "Yaw", "Up", "Right", "Down", "Left"]
shorest_episode = 10
speed_limit = 2.5
max_range = 11
teleport_range = 7
data_length = 1343
step_length = 0.5
congestion_range = 4

all_files = glob.glob(os.path.join('./Experiment 3 data', "*.csv"))
total_traj = 0
entire_data = pd.DataFrame()
for file in all_files:
    df = pd.read_csv(file,names=column_names)
    ID = df['ID'][1]
    if ID == 184:
        n_rows = int((df.iloc[-1]['Time'] - df.iloc[0]['Time']) // step_length) + 1
        origin = df.iloc[0]['Time']
        for i in range(df.shape[0]):
            df.at[i, 'Time'] = df.at[i, 'Time'] - origin + step_length * (data_length - n_rows)
            # technical issue
        last_time = 0
        new_start = []
        for i in range(data_length - n_rows):
            new_row = pd.DataFrame({"Time": [step_length * i + last_time],
                                    "ID": [-1000],
                                    "Positionx": [-1000],
                                    "Positionz": [-1000],
                                    "Positiony": [-1000],
                                    "Yaw": [-1000],
                                    "Up": [-1000],
                                    "Right": [-1000],
                                    "Down": [-1000],
                                    "Left": [-1000]})
            new_start.append(new_row)
        new_start_df = pd.concat(new_start, ignore_index=True)
        df = pd.concat([new_start_df, df], axis=0, ignore_index=True)
    df['Positionx'] = df['Positionx'].astype(str).str.replace(r'[()]', '', regex=True)
    df['Positiony'] = df['Positiony'].astype(str).str.replace(r'[()]', '', regex=True)
    df['Yaw'] = df['Yaw'].astype(str).str.replace('Yaw:', '').astype(float)
    df[['Positionx', 'Positiony', 'Time','Yaw']] = df[['Positionx', 'Positiony', 'Time','Yaw']].apply(pd.to_numeric)
    df['Trajectory'] = int(1)
    df['Distance'] = np.nan
    df['Speed'] = np.nan
    origin = df.iloc[0, 0]
    # add speed, distance, remove
    for i in range(df.shape[0]):
        # specifically for 53
        if df.iloc[i]['Positionx'] == -1000:
            df.iloc[i, 1:] = np.nan
        df.iloc[i, 0] = df.iloc[i, 0] - origin
        if pd.notna(df.iloc[i]['Positionx']):
            df.at[i, 'Distance'] = np.sqrt((df.iloc[i]['Positionx'] - 0) ** 2
                                           + (df.iloc[i]['Positiony'] - 0) ** 2)
        if i > 0:
            if pd.notna(df.iloc[i]['Positionx']) and pd.notna(df.iloc[i - 1]['Positionx']):
                curr_coor = np.array([df.iloc[i]['Positionx'], df.iloc[i]['Positiony']])
                prev_coor = np.array([df.iloc[i - 1]['Positionx'], df.iloc[i - 1]['Positiony']])
                last_move = np.linalg.norm(curr_coor - prev_coor)
                df.at[i, 'Speed'] = (last_move) / (df.iloc[i]['Time'] - df.iloc[i - 1]['Time'])
        # remove point that is too far away
        if df.iloc[i]['Distance'] > max_range:
            df.iloc[i, 1:] = np.nan

    # break into trajectories
    traj_count = 1
    prev_coor = np.array([df.iloc[0]['Positionx'], df.iloc[0]['Positiony']])
    for i in range(1, df.shape[0]):
        if pd.notna(df.iloc[i]['Positionx']):
            curr_coor = np.array([df.iloc[i]['Positionx'], df.iloc[i]['Positiony']])
            last_move = np.linalg.norm(curr_coor - prev_coor)
            # if teleport_range > last_move > speed_limit:
            #     df.iloc[i - 1:i, 1:] = np.nan
            if last_move > teleport_range:
                df.iloc[i - 5:i, 1:] = np.nan
                traj_count += 1
            prev_coor = curr_coor
        if pd.notna(df.iloc[i]['Trajectory']):
            df.at[i, 'Trajectory'] = traj_count
    df.to_csv('./Experiment 3 data/processed data/%s old.csv' % (ID), index=False)

    #linear interpolation
    correct_time = step_length
    added_rows = []
    pre_time = 0
    pre_x = df.iloc[0]['Positionx'] if df.iloc[0]['Positionx'] else np.nan
    pre_y = df.iloc[0]['Positiony'] if df.iloc[0]['Positiony'] else np.nan
    for i in range(1, df.shape[0]):
        time_diff = df.iloc[i]['Time'] - correct_time
        if abs(time_diff) < step_length:
            if pd.notna(df.iloc[i]['Positionx']):#make modification
                if pd.notna(df.iloc[i-1]['Positionx']):#exclude first line of trajectory
                    curr_x = df.iloc[i]['Positionx']
                    curr_y = df.iloc[i]['Positiony']
                    #the current position is based on recorded previous position, not corrected position
                    df.at[i,'Positionx'] = round(((curr_x-pre_x) * (step_length/(df.iloc[i]['Time'] - pre_time)) + pre_x),2)
                    df.at[i,'Positiony'] = round(((curr_y-pre_y) * (step_length/(df.iloc[i]['Time'] - pre_time)) + pre_y),2)
                    curr_coor = np.array([df.iloc[i]['Positionx'],df.iloc[i]['Positiony']])
                    df.at[i, 'Distance'] = np.linalg.norm([curr_coor, np.array([0, 0])])
                    #calculate the new speed
                    prev_coor = np.array([df.iloc[i - 1]['Positionx'], df.iloc[i - 1]['Positiony']])
                    last_move = np.linalg.norm(curr_coor - prev_coor)
                    df.at[i, 'Speed'] = (last_move) / step_length
                    pre_x = curr_x
                    pre_y = curr_y
                else:
                    pre_x = df.iloc[i]['Positionx']
                    pre_y = df.iloc[i]['Positiony']
            pre_time = df.iloc[i]['Time']
            df.at[i, 'Time'] = correct_time
            correct_time += step_length
        else: # indicating missing rows above
            n_rows = 1
            while abs(time_diff) > step_length:
                if pd.isna(df.iloc[i]['Positionx']) or pd.isna(df.iloc[i-1]['Positionx']):
                    new_row = pd.DataFrame({"Time":[correct_time],
                                "ID":[np.nan],
                                "Positionx":[np.nan],
                                "Positionz":[np.nan],
                                "Positiony":[np.nan],
                                "Yaw":[np.nan],
                                "Up":[np.nan],
                                "Right":[np.nan],
                                "Down":[np.nan],
                                "Left":[np.nan],
                                "Trajectory":[np.nan],
                                "Distance":[np.nan],
                                "Speed":[np.nan]}) # insert an empty row
                else:
                    t = df.iloc[i]['Time'] - pre_time
                    x = round((pre_x + (df.iloc[i]['Positionx']-pre_x) * (step_length/t)),2)
                    y = round((pre_y + (df.iloc[i]['Positiony']-pre_y) * (step_length/t)),2)
                    curr_coor = np.array([x,y])
                    dist = np.linalg.norm([curr_coor, np.array([0, 0])])
                    # calculate the new speed
                    if n_rows == 1:
                        prev_coor = np.array([df.iloc[i - 1]['Positionx'], df.iloc[i - 1]['Positiony']])
                    last_move = np.linalg.norm(curr_coor - prev_coor)
                    speed = (last_move) / step_length
                    new_row = pd.DataFrame({"Time":[correct_time + (n_rows-1) * step_length],
                                "ID":[ID],
                                "Positionx":[x],
                                "Positionz":[float(0.0)],
                                "Positiony":[y],
                                "Yaw":[df.iloc[i]['Yaw']],
                                "Up":[df.iloc[i]['Up']],
                                "Right":[df.iloc[i]['Right']],
                                "Down":[df.iloc[i]['Down']],
                                "Left":[df.iloc[i]['Left']],
                                "Trajectory": [df.iloc[i]['Trajectory']],
                                "Distance": [dist],
                                "Speed": [speed]})
                    prev_coor = np.array([x,y])
                time_diff -= step_length
                n_rows += 1
                added_rows.append(new_row)
            if pd.notna(df.iloc[i]['Positionx']):#make modification
                if pd.notna(df.iloc[i-1]['Positionx']):#exclude first line of trajectory
                    curr_x = df.iloc[i]['Positionx']
                    curr_y = df.iloc[i]['Positiony']
                    #the current position is based on recorded previous position, not corrected position
                    df.at[i,'Positionx'] = round(((curr_x-pre_x) * (step_length/(df.iloc[i]['Time'] - pre_time)) + pre_x),2)
                    df.at[i,'Positiony'] = round(((curr_y-pre_y) * (step_length/(df.iloc[i]['Time'] - pre_time)) + pre_y),2)
                    curr_coor = np.array([df.iloc[i]['Positionx'],df.iloc[i]['Positiony']])
                    df.at[i, 'Distance'] = np.linalg.norm([curr_coor, np.array([0, 0])])
                    #calculate the new speed
                    prev_coor = np.array([df.iloc[i - 1]['Positionx'], df.iloc[i - 1]['Positiony']])
                    last_move = np.linalg.norm(curr_coor - prev_coor)
                    df.at[i, 'Speed'] = (last_move) / step_length
                    pre_x = curr_x
                    pre_y = curr_y
                else:
                    pre_x = df.iloc[i]['Positionx']
                    pre_y = df.iloc[i]['Positiony']
            pre_time = df.iloc[i]['Time']
            df.at[i, 'Time'] = correct_time + (n_rows-1)*step_length
            correct_time += (n_rows*step_length)

    if added_rows:
        added_rows_df = pd.concat(added_rows, ignore_index=True)
        df = pd.concat([df,added_rows_df],ignore_index=True)
        df = df.sort_values(by='Time')
        df.reset_index(drop=True,inplace=True)

    consec_zero = 0
    for i in range(df.shape[0]):
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
    #remove zeros at the beginning, too short traj and traj with long moving back
    initial_zeros = 0
    traj_length = 0
    curr_traj = 0
    move_back = 0
    remove = False
    for i in range(df.shape[0]):
        if pd.isna(df.iloc[i]['Trajectory']):
            if traj_length < shorest_episode and traj_length >0:
                df.iloc[i - traj_length:i, 1:] = np.nan
            if initial_zeros > 0:
                df.iloc[i - initial_zeros - 1:i-1, 1:] = np.nan
            if move_back > 1: #indicating still moving back at the end of traj
                remove = True
            if remove:
                df.iloc[i - traj_length:i, 1:] = np.nan
                remove = False
            initial_zeros = 0
            traj_length = 0
            move_back = 0
            continue
        if pd.notna(df.iloc[i]['Trajectory']):
            if traj_length == 0:  # at the beginning of trajectory
                curr_traj = df.iloc[i]['Trajectory']
                traj_length += 1
            else:
                if traj_length == 1 and df.iloc[i]['Speed'] < 0.1: # late start in experiment 3
                    initial_zeros = 1
                elif initial_zeros > 0:
                    if df.iloc[i]['Speed'] < 0.05:
                        initial_zeros += 1
                    else:
                        df.iloc[i - initial_zeros-1:i-1, 1:] = np.nan
                        traj_length -= initial_zeros
                        initial_zeros = 0
                if df.iloc[i]['Distance'] - df.iloc[i-1]['Distance'] > 0:
                    move_back += 1
                else:
                    if move_back < shorest_episode:
                        move_back = 0
                    else:
                        remove = True
                        move_back = 0
                traj_length += 1
    if traj_length < shorest_episode and traj_length > 0:
        df.loc[df['Trajectory'] == curr_traj, df.columns[1:]] = np.nan
    if initial_zeros > 0:
        df.iloc[i - initial_zeros - 1:i-1, 1:] = np.nan

    # reset traj number,
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
    #round distance and speed
    df['Distance'] = df['Distance'].round(2)
    df['Speed'] = df['Speed'].round(2)
    # add rows in the end
    added_rows = []
    if df.iloc[-1]['Time'] < step_length * (data_length - 1):
        curr_time = df.iloc[-1]['Time']
        while curr_time < step_length * (data_length - 1):
            new_row = pd.DataFrame({"Time": [curr_time + step_length],
                                    "ID": [np.nan],
                                    "Positionx": [np.nan],
                                    "Positionz": [np.nan],
                                    "Positiony": [np.nan],
                                    "Yaw": [np.nan],
                                    "Up": [np.nan],
                                    "Right": [np.nan],
                                    "Down": [np.nan],
                                    "Left": [np.nan],
                                    "Trajectory": [np.nan],
                                    "Distance": [np.nan],
                                    "Speed": [np.nan]})  # insert an empty row
            curr_time += step_length
            added_rows.append(new_row)
    if added_rows:
        # print(len(added_rows), ID)
        added_rows_df = pd.concat(added_rows, ignore_index=True)
        df = pd.concat([df,added_rows_df],ignore_index=True)
        df = df.sort_values(by='Time')
        df.reset_index(drop=True,inplace=True)
    # Apply the trajectory mapping
    df['Trajectory'] = df['Trajectory'].map(traj_map)
    total_traj += curr_traj
    print(ID, df.shape[0], df.iloc[-1]['Time'])

    # add direction
    df['Direction'] = np.nan
    for i in range(df.shape[0]):
        if pd.notna(df.iloc[i]['Speed']):
            x2 = df.iloc[i]['Positionx']
            y2 = df.iloc[i]["Positiony"]
            x1 = df.iloc[i-1]['Positionx']
            y1 = df.iloc[i-1]["Positiony"]
            theta_x = np.arctan2(y2-y1, x2-x1)
            df.at[i,'Direction'] = round(theta_x,2)
        if pd.notna(df.iloc[i]['Yaw']):
            df.at[i,'Yaw'] = np.radians(90 - df.iloc[i]['Yaw'])
    df.to_csv('./Experiment 3 data/processed data/%s new.csv' % (ID), index=False)
    entire_data = pd.concat([entire_data,df], ignore_index = True)
print(total_traj)
entire_data.to_csv('./Experiment 3 data/processed data/Experiment 3.csv', index=False)