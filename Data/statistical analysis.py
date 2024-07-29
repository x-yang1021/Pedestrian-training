import pandas as pd
import numpy as np
from scipy import stats

df_1 = pd.read_csv('./Experiment 1.csv')
df_2 = pd.read_csv('./Experiment 2.csv')
df_3 = pd.read_csv('./Experiment 3.csv')
dfs = [df_1, df_2, df_3]

range_of_interest = 3
range_of_interest2 = 12
time_of_interest = 300
defalut_direction = np.arctan2(-1,0)

curr_traj = 0
traj_length = 0
lengths = []
lost = 0
for df in dfs:
    for i in range(df.shape[0]):
        if (pd.isna(df.iloc[i]['Trajectory']) or df.iloc[i]['Time'] == 0) and traj_length != 0:
            if traj_length < 9:
                # print(df.iloc[i-1]["ID"], df.iloc[i-1]['Trajectory'], traj_length)
                lost += traj_length
            lengths.append(traj_length)
            traj_length = 0
        if pd.notna(df.iloc[i]['Trajectory']):
            if df.iloc[i]['Distance'] <= range_of_interest:
                traj_length += 1
# print(lengths)
# print(lost)
# print(max(lengths), min(lengths))
# print(sum(lengths))
#
# exit()

speed_1 = []
direction_1 = []
change_speed_1 = []
change_direction_1 = []
for i in range(df_1.shape[0]):
    # if df_1.iloc[i]['Time'] < time_of_interest:
    #     continue
    if pd.notna(df_1.iloc[i]['Speed']):
        if df_1.iloc[i]['Distance'] <= range_of_interest:
            speed_1.append(df_1.iloc[i]['Speed'])
        if pd.notna(df_1.iloc[i - 1]['Speed']):
            change_speed_1.append(df_1.iloc[i]['Speed'] - df_1.iloc[i - 1]['Speed'])
        else:
            change_speed_1.append(df_1.iloc[i]['Speed'])
    if pd.notna(df_1.iloc[i]['Direction']):
        direction_1.append(df_1.iloc[i]['Direction'])
        if pd.notna(df_1.iloc[i-1]['Direction']):
            change_direction_1.append(df_1.iloc[i]['Direction'] - df_1.iloc[i-1]['Direction'])
        else:
            change_direction_1.append(df_1.iloc[i]['Direction'] - defalut_direction)

speed_2 = []
direction_2 = []
change_speed_2 = []
change_direction_2 = []
for i in range(df_2.shape[0]):
    # if df_2.iloc[i]['Time'] < time_of_interest:
    #     continue
    if pd.notna(df_2.iloc[i]['Speed']):
        if df_2.iloc[i]['Distance'] <= range_of_interest:
            speed_2.append(df_2.iloc[i]['Speed'])
        if pd.notna(df_2.iloc[i - 1]['Speed']):
            change_speed_2.append(df_2.iloc[i]['Speed'] - df_2.iloc[i - 1]['Speed'])
        else:
            change_speed_2.append(df_2.iloc[i]['Speed'])
    if pd.notna(df_2.iloc[i]['Direction']):
        direction_2.append(df_2.iloc[i]['Direction'])
        if pd.notna(df_2.iloc[i-1]['Direction']):
            change_direction_2.append(df_2.iloc[i]['Direction'] - df_2.iloc[i-1]['Direction'])
        else:
            change_direction_2.append(df_2.iloc[i]['Direction'] - defalut_direction)

speed_3 = []
direction_3 = []
change_speed_3 = []
change_direction_3 = []
for i in range(df_3.shape[0]):
    # if df_3.iloc[i]['Time'] < time_of_interest:
    #     continue
    if pd.notna(df_3.iloc[i]['Speed']):
        if df_3.iloc[i]['Distance'] <= range_of_interest:
            speed_3.append(df_3.iloc[i]['Speed'])
        if pd.notna(df_3.iloc[i - 1]['Speed']):
            change_speed_3.append(df_3.iloc[i]['Speed'] - df_3.iloc[i - 1]['Speed'])
        else:
            change_speed_3.append(df_3.iloc[i]['Speed'])
    if pd.notna(df_3.iloc[i]['Direction']):
        direction_3.append(df_3.iloc[i]['Direction'])
        if pd.notna(df_3.iloc[i - 1]['Direction']):
            change_direction_3.append(df_3.iloc[i]['Direction'] - df_3.iloc[i - 1]['Direction'])
        else:
            change_direction_3.append(df_3.iloc[i]['Direction'] - defalut_direction)



# shapiro_test = stats.shapiro(np.array(change_direction_1))
# print(shapiro_test.pvalue)
#
# shapiro_test = stats.shapiro(np.array(change_direction_2))
# print(shapiro_test.pvalue)
#
# shapiro_test = stats.shapiro(np.array(change_direction_3))
# print(shapiro_test.pvalue)

tt_stat, p_value = stats.kruskal(np.array(speed_1), np.array(speed_2), np.array(speed_3))

print(p_value, 'Speed')

tt_stat, p_value = stats.kruskal(np.array(change_speed_1), np.array(change_speed_2), np.array(change_speed_3))

print(p_value, 'Change Speed')

tt_stat, p_value = stats.kruskal(np.array(direction_1), np.array(direction_2), np.array(direction_3))

print(p_value, 'Direction')

tt_stat, p_value = stats.kruskal(np.array(change_direction_1), np.array(change_direction_2), np.array(change_direction_3))

print(p_value, 'Change Direction')



head_speed = []
head_speed_change = []
tail_speed = []
tail_speed_change = []
near_speed = []
near_speed_change = []
far_speed = []
far_speed_change = []
head_direction = []
head_direction_change = []
tail_direction = []
tail_direction_change = []
near_direction = []
near_direction_change = []
far_direction = []
far_direction_change = []
for df in dfs:
    t = time_of_interest
    for i in range(df.shape[0]):
        if pd.notna(df.iloc[i]['Speed']):
            if df.iloc[i]['Distance'] <= range_of_interest:
                far_speed.append(df.iloc[i]['Speed'])
                if pd.notna(df.iloc[i-1]['Speed']):
                    far_speed_change.append(df.iloc[i]['Speed'] - df.iloc[i-1]['Speed'])
                else:
                    far_speed_change.append(df.iloc[i]['Speed'])
                if df.iloc[i]['Time']<=t:
                    head_speed.append(df.iloc[i]['Speed'])
                    if pd.notna(df.iloc[i-1]['Speed']):
                        head_speed_change.append(df.iloc[i]['Speed'] - df.iloc[i-1]['Speed'])
                    else:
                        head_speed_change.append(df.iloc[i]['Speed'])
                else:
                    tail_speed.append(df.iloc[i]['Speed'])
                    if pd.notna(df.iloc[i-1]['Speed']):
                        tail_speed_change.append(df.iloc[i]['Speed'] - df.iloc[i-1]['Speed'])
                    else:
                        tail_speed_change.append(df.iloc[i]['Speed'])
            elif range_of_interest2 > df.iloc[i]['Distance'] > range_of_interest:
                near_speed.append(df.iloc[i]['Speed'])
                if pd.notna(df.iloc[i-1]['Speed']):
                    near_speed_change.append(df.iloc[i]['Speed'] - df.iloc[i-1]['Speed'])
                else:
                    near_speed_change.append(df.iloc[i]['Speed'])
        if pd.notna(df.iloc[i]['Direction']):
            if df.iloc[i]['Distance'] <= range_of_interest:
                far_direction.append(df.iloc[i]['Direction'])
                if pd.notna(df.iloc[i-1]['Direction']):
                    far_direction_change.append(df.iloc[i]['Direction'] - df.iloc[i-1]['Direction'])
                else:
                    far_direction_change.append(df.iloc[i]['Direction'] - defalut_direction)
                if df.iloc[i]['Time']<=t:
                    head_direction.append(df.iloc[i]['Direction'])
                    if pd.notna(df.iloc[i-1]['Direction']):
                        head_direction_change.append(df.iloc[i]['Direction'] - df.iloc[i-1]['Direction'])
                    else:
                        head_direction_change.append(df.iloc[i]['Direction'] - defalut_direction)
                else:
                    tail_direction.append(df.iloc[i]['Direction'])
                    if pd.notna(df.iloc[i-1]['Direction']):
                        tail_direction_change.append(df.iloc[i]['Direction'] - df.iloc[i-1]['Direction'])
                    else:
                        tail_direction_change.append(df.iloc[i]['Direction'] - defalut_direction)
            elif range_of_interest2 > df.iloc[i]['Distance'] > range_of_interest:
                near_direction.append(df.iloc[i]['Direction'])
                if pd.notna(df.iloc[i-1]['Direction']):
                    near_direction_change.append(df.iloc[i]['Direction'] - df.iloc[i-1]['Direction'])
                else:
                    near_direction_change.append(df.iloc[i]['Direction'] - defalut_direction)

tt_stat, p_value = stats.mannwhitneyu(np.array(head_speed), np.array(tail_speed))

print(p_value, 'head Speed')

tt_stat, p_value = stats.mannwhitneyu(np.array(head_speed_change), np.array(tail_speed_change))

print(p_value, 'head Speed Change')


tt_stat, p_value = stats.mannwhitneyu(np.array(head_direction), np.array(tail_direction))

print(p_value, 'head Direction')

tt_stat, p_value = stats.mannwhitneyu(np.array(head_direction_change), np.array(tail_direction_change))

print(p_value, 'head Direction Change')



tt_stat, p_value = stats.mannwhitneyu(np.array(near_speed), np.array(far_speed))

print(p_value, 'near Speed', np.mean(near_speed), np.mean(far_speed))

tt_stat, p_value = stats.mannwhitneyu(np.array(near_speed_change), np.array(far_speed_change))

print(p_value, 'near Speed Change', np.mean(near_speed_change), np.mean(far_speed_change))

tt_stat, p_value = stats.mannwhitneyu(np.array(near_direction), np.array(far_direction))

print(p_value, 'near Direction')

tt_stat, p_value = stats.mannwhitneyu(np.array(near_direction_change), np.array(far_direction_change))

print(p_value, 'near Direction Change', np.mean(near_direction_change), np.mean(far_direction_change))