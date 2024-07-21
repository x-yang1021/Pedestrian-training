import pandas as pd
import numpy as np
import glob
import os

column_names = ["Time", "ID", "Positionx","Positionz","Positiony", "Yaw", "Up", "Right", "Down", "Left"]
shorest_episode = 10
max_range = 11
teleport_range = 7
speed_limit = 2.5 # per half second
generate_file = True
process_53 = False
data_length = 1269
step_length = 0.507

#concat plyaer 53 due to technical issue
if process_53:
    df_1 = pd.read_csv('./Experiment 1 data/PlayerPositions_53_240425_152535.csv',names=column_names)
    df_1['Positionx'] = df_1['Positionx'].astype(str).str.replace(r'[()]', '', regex=True)
    df_1['Positiony'] = df_1['Positiony'].astype(str).str.replace(r'[()]', '', regex=True)
    df_1['Yaw'] = df_1['Yaw'].astype(str).str.replace('Yaw:', '').astype(float)
    df_1[['Positionx', 'Positiony', 'Yaw', 'Time','ID']] = df_1[['Positionx', 'Positiony', 'Yaw', 'Time','ID']].apply(pd.to_numeric)

    df_2 = pd.read_csv('./Experiment 1 data/PlayerPositions_53_240425_153959.csv',names=column_names)
    df_2['Positionx'] = df_1['Positionx'].astype(str).str.replace(r'[()]', '', regex=True)
    df_2['Positiony'] = df_1['Positiony'].astype(str).str.replace(r'[()]', '', regex=True)
    df_2['Yaw'] = df_1['Yaw'].astype(str).str.replace('Yaw:', '').astype(float)
    df_2[['Positionx', 'Positiony', 'Yaw', 'Time','ID']] = df_2[['Positionx', 'Positiony', 'Yaw', 'Time','ID']].apply(pd.to_numeric)

    time_diff = data_length - df_1.shape[0] - df_2.shape[0]

    origin = df_1.iloc[0,0]

    for i in range(df_1.shape[0]):
        df_1.iloc[i,0] = df_1.iloc[i,0] - origin

    last_time = df_1.iloc[-1]['Time']

    for i in range(1,time_diff+1):
        new_row = pd.DataFrame({"Time":[step_length*i + last_time],
                                "ID":[-1000],
                                "Positionx":[-1000],
                                "Positionz":[-1000],
                                "Positiony":[-1000],
                                "Yaw":[-1000],
                                "Up":[-1000],
                                "Right":[-1000],
                                "Down":[-1000],
                                "Left":[-1000]})
        df_1 = pd.concat([df_1,new_row],axis=0, ignore_index = True)

    df_2.at[0,'Time'] = df_1.iloc[-1]['Time'] + step_length

    for i in range(1, df_2.shape[0]):
        df_2.iloc[i,0] = df_2.iloc[i-1,0] + step_length

    df = pd.concat(objs = [df_1,df_2], axis=0, ignore_index = True)


    # Convert to NumPy array to remove column names
    data = df.to_numpy()
    # Convert back to DataFrame without column names
    df_no_columns = pd.DataFrame(data)

    print(df.shape[0])

    df.to_csv('./Experiment 1 data/53 old.csv', index = False)

    exit()

all_files = glob.glob(os.path.join('./Experiment 1 data', "*.csv"))
total_traj = 0
entire_data = pd.DataFrame()
ID_list = []
for file in all_files:
    df = pd.read_csv(file,names=column_names)
    df['Positionx'] = df['Positionx'].astype(str).str.replace(r'[()]', '', regex=True)
    df['Positiony'] = df['Positiony'].astype(str).str.replace(r'[()]', '', regex=True)
    if file == './Experiment 1 data/53 old.csv':
        # df['Positionx', 'Positiony', 'Time', 'Yaw','ID'] = df['Positionx', 'Positiony', 'Time', 'Yaw','ID'].astype(float)
        df = df.iloc[1:]
        df = df.reset_index(drop=True)
            # print(df.iloc[i]['Yaw'], type(df.iloc[i]['Yaw']))
        df['Yaw'] = df['Yaw'].astype(str).str.replace('Yaw:', '').astype(float)
        df[['Positionx', 'Positiony', 'Time', 'Yaw']] = df[['Positionx', 'Positiony', 'Time', 'Yaw']].apply(
            pd.to_numeric)
        df['ID'] = int(53)
    else:
        df['Yaw'] = df['Yaw'].astype(str).str.replace('Yaw:', '').astype(float)
        df[['Positionx', 'Positiony', 'Time','Yaw']] = df[['Positionx', 'Positiony', 'Time','Yaw']].apply(pd.to_numeric)
    ID = df['ID'][1]
    ID_list.append(ID)
    df['Trajectory'] = int(1)
    df['Distance'] = np.nan
    df['Speed'] = np.nan
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
    if generate_file:
        df.to_csv('./Experiment 1 data/processed data/%s old.csv' % (ID), index=False)
    correct_time = step_length
    added_rows = []
    for i in range(1, df.shape[0]):
        time_diff = df.iloc[i]['Time'] - correct_time
        if time_diff != 0:
            if abs(time_diff) < step_length:
                if pd.notna(df.iloc[i]['Positionx']): #make modification
                    df.at[i,'Positionx'] = df.iloc[i]['Positionx'] * (1+time_diff/step_length)
                    df.at[i,'Positiony'] = df.iloc[i]['Positiony'] * (1+time_diff/step_length)
                    curr_coor = np.array([df.iloc[i]['Positionx'],df.iloc[i]['Positiony']])
                    df.at[i, 'Distance'] = np.linalg.norm([curr_coor,np.array([0,0])])
                    # speed remain the same
                    # if pd.notna(df.iloc[i-1]['Positionx']):
                    #     prev_coor = np.array([df.iloc[i - 1]['Positionx'], df.iloc[i - 1]['Positiony']])
                    #     last_move = np.linalg.norm(curr_coor - prev_coor)
                    #     df.at[i, 'Speed'] = (last_move) / (df.iloc[i]['Time'] - df.iloc[i - 1]['Time'])
                df.at[i, 'Time'] = correct_time
                correct_time += step_length
            else: # indicating a missing row above
                if pd.isna(df.iloc[i]['Positionx']) or pd.isna(df.iloc[i-1]['Positionx']):
                    new_row = pd.DataFrame({"Time":[correct_time + step_length],
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
                    speed = df.iloc[i]['Speed']
                    t = df.iloc[i]['Time'] - df.iloc[i-1]['Time']
                    x = df.iloc[i-1]['Positionx'] + (df.iloc[i]['Positionx']-df.iloc[i-1]['Positionx']) * (step_length/t)
                    y = df.iloc[i-1]['Positiony'] + (df.iloc[i]['Positiony']-df.iloc[i-1]['Positiony']) * (step_length/t)
                    new_row = pd.DataFrame({"Time":[correct_time],
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
                                "Distance": [np.linalg.norm([np.array([x,y]),np.array([0,0])])],
                                "Speed": [df.iloc[i]['Speed']]})
                time_diff -= step_length
                df.at[i,'Positionx'] = df.iloc[i]['Positionx'] * (1+time_diff/step_length)
                df.at[i,'Positiony'] = df.iloc[i]['Positiony'] * (1+time_diff/step_length)
                curr_coor = np.array([df.iloc[i]['Positionx'],df.iloc[i]['Positiony']])
                df.at[i, 'Distance'] = np.linalg.norm([curr_coor,np.array([0,0])])
                df.at[i, 'Time'] = correct_time + step_length
                correct_time += (2*step_length)
                added_rows.append(new_row)
        else:
            correct_time += step_length
    if added_rows:
        print(added_rows, ID)
        added_rows_df = pd.concat(added_rows, ignore_index=True)
        df = pd.concat([df,added_rows_df],ignore_index=True)
        df = df.sort_values(by='Time')
        df.reset_index(drop=True,inplace=True)
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
    if generate_file:
        df.to_csv('./Experiment 1 data/processed data/%s new.csv' % (ID), index=False)
    entire_data = pd.concat([entire_data,df], ignore_index = True)
print(total_traj)
if generate_file:
    entire_data.to_csv('./Experiment 1 data/processed data/Experiment 1.csv', index=False)
print(ID_list)
