import numpy.linalg
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import ruptures as rpt
from ruptures import costs



df_1 = pd.read_csv('./Experiment 1.csv')
df_2 = pd.read_csv('./Experiment 2.csv')
df_3 = pd.read_csv('./Experiment 3.csv')
dfs = [df_1, df_2, df_3]


range_of_interest = 2.58
range_of_interest2 = 12
time_of_interest = 300
default_direction = np.arctan2(1,0)


df_total = pd.DataFrame()
IDs = []
#add change of speed and direction
for df in dfs:
    df['Speed Change'] = np.nan
    df['Direction Change'] = np.nan
    for i in range(df.shape[0]-1):
        if pd.notna(df.iloc[i+1]['Speed']):
                if pd.notna(df.iloc[i]['Speed']):
                    df.at[i,'Speed Change'] = df.iloc[i+1]['Speed'] - df.iloc[i]['Speed']
        if pd.notna(df.iloc[i+1]['Direction']):
            if pd.notna(df.iloc[i]['Direction']):
                diff = df.iloc[i+1]['Direction'] - df.iloc[i]['Direction']
                # Wrap the difference to the range -pi to pi
                if diff > np.pi:
                    diff -= 2 * np.pi
                elif diff < -np.pi:
                    diff += 2 * np.pi
                df.at[i, 'Direction Change'] = diff

        # if abs(df.iloc[i]['Speed Change']) > 10:
        #     print(df.iloc[i]['ID'], df.iloc[i]['Time'], df.iloc[i]['Speed Change'], df.iloc[i]['Trajectory'])
    # plt.plot(df['speed_change_rate'], df['Distance'],  marker='o', linestyle='-', color='b')
    # # plt.plot(df['Direction Change'], df['Distance'],  marker='o', linestyle='-', color='r')
    # plt.show()
    df_total = pd.concat([df_total, df], axis=0)

# df_total.to_csv('entrie dataset.csv', index = False)
# exit()

# df_yaw = df_total.dropna(subset=['Direction'])
# angle_diff = df_yaw['Yaw'] - df_yaw['Direction']
# print(np.mean(angle_diff))
#
# exit()

df_clean = df_total.dropna(subset=['Direction Change']) #include the path that contain both features
df_clean.reset_index(drop=True, inplace=True)


# df_file = df_clean[df_clean['Distance']<= range_of_interest]
# df_file = df_file.dropna(subset=['Direction Change'])
# df_file.reset_index(drop=True, inplace=True)
# df_file = df_file[['ID', 'Trajectory', 'Speed', 'Speed Change', 'Direction Change']]
# df_file.to_csv('Cluster dataset.csv', index=False)
# exit()

# df_clean.loc[:, 'Positionx'] = df_clean['Positionx'].abs()
df_clean =df_clean.sort_values(by=['Distance'])
df_clean = df_clean[df_clean['Distance']<10]
df_clean.reset_index(drop=True, inplace=True)

df_test = df_clean
df_test = pd.DataFrame()
df_test['Speed Change'] = 2 * (df_clean['Speed Change'] - df_clean['Speed Change'].min()) / (df_clean['Speed Change'].max() - df_clean['Speed Change'].min()) - 1
df_test['Direction Change'] = 2 * (df_clean['Direction Change'] - df_clean['Direction Change'].min()) / (df_clean['Direction Change'].max() - df_clean['Direction Change'].min()) - 1
# df_test['Direction Change'] = df_test['Direction Change'].abs()

# signal = df_clean['Distance'].values
signal = df_test[['Speed Change','Direction Change']].values
models = ["l2",'l1','linear','clinear','rank'] # Change point detection model
models=['l2']
for model in models:
    algo = rpt.Window(model=model).fit(signal)
    for i in range(1,2):
        result = algo.predict(n_bkps=1)
        rpt.display(signal, result)
        plt.xlabel('Index')
        # plt.ylabel('Direction Change')
        plt.savefig('Break point.png')
        plt.show()

        print(f'Change points: {result}', model,i)

        for k in range(len(result)-1):

            print(df_clean.iloc[result[k]]['Distance'])

# df_special = df_clean[df_clean['Distance'] == 3.28]
#
# print(df_special['Speed'].mean())

obs, vars = signal.shape
ranks = stats.mstats.rankdata(signal, axis=0)
ranks = ranks - ((obs + 1) / 2)
cov = np.cov(ranks, rowvar=False, bias=True).reshape(vars, vars)
inv_cov = np.linalg.pinv(cov)

mean = np.reshape(np.mean(ranks[:result[0]], axis=0), (-1, 1))

var = signal[:result[0]].var(axis=0)

print(mean, var)

mean = np.reshape(np.mean(ranks[result[0]:], axis=0), (-1, 1))

var = signal[result[0]:].var(axis=0)

print(mean, var)

exit()

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
        if pd.notna(df_1.iloc[i]['Speed Change']):
            change_speed_1.append(df_1.iloc[i]['Speed Change'])
    if pd.notna(df_1.iloc[i]['Direction']):
        direction_1.append(df_1.iloc[i]['Direction'])
        if pd.notna(df_1.iloc[i]['Direction Change']):
            change_direction_1.append(df_1.iloc[i]['Direction Change'])
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
        if pd.notna(df_2.iloc[i]['Speed Change']):
            change_speed_2.append(df_2.iloc[i]['Speed Change'])
    if pd.notna(df_2.iloc[i]['Direction']):
        direction_2.append(df_2.iloc[i]['Direction'])
        if pd.notna(df_2.iloc[i]['Direction Change']):
            change_direction_2.append(df_2.iloc[i]['Direction Change'])
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
        if pd.notna(df_3.iloc[i]['Speed Change']):
            change_speed_3.append(df_3.iloc[i]['Speed Change'])
    if pd.notna(df_3.iloc[i]['Direction']):
        direction_3.append(df_3.iloc[i]['Direction'])
        if pd.notna(df_3.iloc[i]['Direction Change']):
            change_direction_3.append(df_3.iloc[i]['Direction Change'])


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




# Make sure to initialize the lists before the loop
head_speed, head_speed_change, tail_speed, tail_speed_change = [], [], [], []
head_direction, head_direction_change, tail_direction, tail_direction_change = [], [], [], []
near_speed_change, near_direction_change, far_speed_change, far_direction_change = [],[],[],[]

for df in dfs:
    t = time_of_interest
    for i in range(df.shape[0]):
        if pd.notna(df.iloc[i]['Speed']):
            if df.iloc[i]['Distance'] <= range_of_interest:
                if pd.notna(df.iloc[i]['Speed Change']):
                    near_speed_change.append(df.iloc[i]['Speed Change'])
                if df.iloc[i]['Time'] <= t:
                    head_speed.append(df.iloc[i]['Speed'])
                    if pd.notna(df.iloc[i]['Speed Change']):
                        head_speed_change.append(df.iloc[i]['Speed Change'])
                else:
                    tail_speed.append(df.iloc[i]['Speed'])
                    if pd.notna(df.iloc[i]['Speed Change']):
                        tail_speed_change.append(df.iloc[i]['Speed Change'])
            else:
                if pd.notna(df.iloc[i]['Speed Change']):
                    far_speed_change.append(df.iloc[i]['Speed Change'])
        if pd.notna(df.iloc[i]['Direction']):
            if df.iloc[i]['Distance'] <= range_of_interest:
                if pd.notna(df.iloc[i]['Direction Change']):
                    near_direction_change.append(df.iloc[i]['Direction Change'])
                if df.iloc[i]['Time'] <= t:
                    head_direction.append(df.iloc[i]['Direction'])
                    if pd.notna(df.iloc[i]['Direction Change']):
                        head_direction_change.append(df.iloc[i]['Direction Change'])
                else:
                    tail_direction.append(df.iloc[i]['Direction'])
                    if pd.notna(df.iloc[i]['Direction Change']):
                        tail_direction_change.append(df.iloc[i]['Direction Change'])
            else:
                if pd.notna(df.iloc[i]['Direction Change']):
                    far_direction_change.append(df.iloc[i]['Direction Change'])
tt_stat, p_value = stats.mannwhitneyu(np.array(head_speed), np.array(tail_speed))

print(p_value, 'head Speed')

tt_stat, p_value = stats.mannwhitneyu(np.array(head_speed_change), np.array(tail_speed_change))

print(p_value, 'head Speed Change')


tt_stat, p_value = stats.mannwhitneyu(np.array(head_direction), np.array(tail_direction))

print(p_value, 'head Direction')

tt_stat, p_value = stats.mannwhitneyu(np.array(head_direction_change), np.array(tail_direction_change))

print(p_value, 'head Direction Change')



# tt_stat, p_value = stats.mannwhitneyu(np.array(near_speed), np.array(far_speed))
#
# print(p_value, 'near Speed', np.mean(near_speed), np.mean(far_speed))
#
tt_stat, p_value = stats.mannwhitneyu(np.array(near_speed_change), np.array(far_speed_change))

print(p_value, 'near Speed Change', np.mean(near_speed_change), np.mean(far_speed_change))

# tt_stat, p_value = stats.mannwhitneyu(np.array(near_direction), np.array(far_direction))
#
# print(p_value, 'near Direction')
#
tt_stat, p_value = stats.mannwhitneyu(np.array(near_direction_change), np.array(far_direction_change))

print(p_value, 'near Direction Change', np.mean(near_direction_change), np.mean(far_direction_change))


# plt.hist(np.array(far_direction_change), color='lightgreen', ec='black', bins=30)
#
# plt.xlim(-10, 10)
#
# plt.show()