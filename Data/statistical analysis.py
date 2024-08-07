import numpy.linalg
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import ruptures as rpt



df_1 = pd.read_csv('./Experiment 1.csv')
df_2 = pd.read_csv('./Experiment 2.csv')
df_3 = pd.read_csv('./Experiment 3.csv')
dfs = [df_1, df_2, df_3]


range_of_interest = 3.08
range_of_interest2 = 12
time_of_interest = 180
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
                else:
                    df.at[i,'Speed Change'] = df.iloc[i+1]['Speed']
        if pd.notna(df.iloc[i+1]['Direction']):
            diff = df.iloc[i + 1]['Direction'] - df.iloc[i]['Direction']
            # Wrap the difference to the range -pi to pi
            if diff > np.pi:
                diff -= 2 * np.pi
            elif diff < -np.pi:
                diff += 2 * np.pi
            df.at[i, 'Direction Change'] = diff
        else:
            # diff = df.iloc[i + 1]['Direction'] - default_direction
            # if diff > np.pi:
            #     diff -= 2 * np.pi
            # elif diff < -np.pi:
            #     diff += 2 * np.pi
            df.at[i, 'Direction Change'] = np.nan
        # if abs(df.iloc[i]['Speed Change']) > 10:
        #     print(df.iloc[i]['ID'], df.iloc[i]['Time'], df.iloc[i]['Speed Change'], df.iloc[i]['Trajectory'])
    # plt.plot(df['speed_change_rate'], df['Distance'],  marker='o', linestyle='-', color='b')
    # # plt.plot(df['Direction Change'], df['Distance'],  marker='o', linestyle='-', color='r')
    # plt.show()
    df_total = pd.concat([df_total, df], axis=0)

df_clean = df_total.dropna(subset=['Direction Change']) #include the path that contain both features
df_clean.reset_index(drop=True, inplace=True)

# df_file = df_clean[df_clean['Distance']<=2.81]
# df_file = df_file.dropna(subset=['Direction Change'])
# df_file.reset_index(drop=True, inplace=True)
# df_file = df_file[['ID', 'Trajectory', 'Speed Change', 'Direction Change']]
# df_file.to_csv('Cluster dataset.csv', index=False)
# exit()


df_clean =df_clean.sort_values(by=['Distance'])
df_clean = df_clean[df_clean['Distance']<9.2]
df_clean.reset_index(drop=True, inplace=True)

# signal = df_clean['Distance'].values
signal = df_clean[['Speed Change']].values
model = "ar"  # Change point detection model
algo = rpt.Binseg(model=model).fit(signal)
result = algo.predict(pen=10)
rpt.display(signal, result)
plt.xlabel('Index')
# plt.ylabel('Direction Change')
plt.savefig('Break point.png')
plt.show()

print(f'Change points: {result}')

print(df_clean.iloc[result[0]]['Distance'])

# print(df_clean.iloc[result[1]]['Distance'])

exit()

# First segment
X1 = df_clean.iloc[:result[0]]['Speed Change'].values.reshape(-1, 1)  # Reshape to 2D array
y1 = df_clean.iloc[:result[0]]['Direction Change'].values

# Compute the least squares solution for the first segment
slope1, residuals1, _, _ = np.linalg.lstsq(X1, y1, rcond=None)
print(f'Slope for first segment: {slope1[0]}')
print(f'Residuals for first segment: {residuals1}')

# Second segment
X2 = df_clean.iloc[result[0]:]['Speed Change'].values.reshape(-1, 1)  # Reshape to 2D array
y2 = df_clean.iloc[result[0]:]['Direction Change'].values

# Compute the least squares solution for the second segment
slope2, residuals2, _, _ = np.linalg.lstsq(X2, y2, rcond=None)
print(f'Slope for second segment: {slope2[0]}')
print(f'Residuals for second segment: {residuals2}')
#
# p, e = curve_fit(piecewise_linear, df_clean['Distance'], df_clean['Direction Change'])
# print(f'Estimated breakpoint: {p[0]} meters')
# xd = np.linspace(0, 10, 100)
# plt.plot(xd, piecewise_linear(xd, *p), label='Piecewise Linear Fit')
# plt.axvline(x=p[0], color='r', linestyle='--', label=f'Breakpoint at {p[0]:.2f}')
# plt.show()

# lowess = sm.nonparametric.lowess
# df_clean['y_lowess'] = lowess(df_clean['Speed Change'], df_clean['Distance'], frac=0.2)[:, 1]
#
# # Plot data and the LOESS fit
# plt.figure(figsize=(10, 6))
# plt.scatter(df_clean['Distance'], df_clean['Speed Change'], label='Data')
# plt.plot(df_clean['Distance'], df_clean['y_lowess'], color='red', label='LOESS fit')
# plt.xlabel('Distance')
# plt.ylabel('Speed Change')
# plt.title('LOESS Regression Fit')
# plt.legend()
# plt.show()

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
    t = df.iloc[-1]['Time']//2
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


# plt.hist(np.array(far_direction_change), color='lightgreen', ec='black', bins=30)
#
# plt.xlim(-10, 10)
#
# plt.show()