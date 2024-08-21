import pyproj
import pandas as pd
from pyproj import Geod
from pyproj import Transformer
import plotly.express as px
import glob
import numpy as np
import seaborn as sns

# Get a list of all CSV files in a directory
csv_files = glob.glob('./raw data 3/*.csv')
geod = Geod(ellps="WGS84")
transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
sun = [118.77886076,32.04382648]
sun = np.array(transformer.transform(sun[0],sun[1]))
sun_unity = np.array([29375,8785])
green_start1 = (np.array([29536, 9352]) - sun_unity)/10
green_end1 = (np.array([29536,10510]) - sun_unity)/10
edge_start1 = (np.array([29920,9547]) - sun_unity)/10
edge_end1 = (np.array([29688,9734]) - sun_unity)/10
wall_start1 = (np.array([29688,9734]) - sun_unity)/10
wall_end1 = (np.array([29688,10392]) - sun_unity)/10
wall_start2 = (np.array([29735,10778]) - sun_unity)/10
wall_end2 = (np.array([29738,12794]) - sun_unity)/10


def edge1(x):
    slope = (edge_end1[1] - edge_start1[1]) / (edge_end1[0] - edge_start1[0])
    y_intercept = edge_start1[1] - slope * edge_start1[0]
    return slope * x + y_intercept

# Loop through each CSV file and append its contents to the combined dataframe

trajectory = pd.DataFrame()


#convert to XY coordinates
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    df['Speed'] = np.nan
    df['Direction'] = np.nan
    for i in range(len(df)):
        Coordinate = np.array(transformer.transform(df.iloc[i][1], df.iloc[i][2])) - sun
        df.iloc[i,1] = Coordinate[0]
        df.iloc[i,2] = Coordinate[1]
        if i == 0:
            df.iloc[i,3] = 0
            df.iloc[i,4] = np.pi/2
        else:
            df.iloc[i, 3] = np.sqrt((df.iloc[i,1] - df.iloc[i-1,1])**2 + (df.iloc[i,2] - df.iloc[i-1,2])**2)
            df.iloc[i, 4] = np.arctan2((df.iloc[i,2] - df.iloc[i-1,2]),(df.iloc[i,1] - df.iloc[i-1,1]))
    trajectory = pd.concat([trajectory, df], axis=1)

with pd.ExcelWriter('./raw data.xlsx', mode='a') as writer:
    trajectory.to_excel(writer, sheet_name= 'Group 3')


for j in range(0, trajectory.shape[1], 5):
    df = trajectory.iloc[:,j:j+5]
    for i in range(len(df)):
        coordinate = df.iloc[i]
        if coordinate[1] < green_start1[0]:
            # coordinate[1] > edge_start1[0]
            # print(edge_start1[0], green_start1[0], edge_start1[1], Coordinate)
            trajectory.iloc[i:i+1,j:j+5] = np.nan
        if (coordinate[2] < edge_start1[1]) or (coordinate[2] > wall_end1[1] and coordinate[2] < wall_start2[1]) or (coordinate[2] > wall_end2[1]):
            # if Coordinate[0] > edge_end1[0]:
            #     if edge1(Coordinate[0]) < Coordinate[1]:
            #         print(edge1(Coordinate[0]), Coordinate[1], edge_end1[0], Coordinate[0])
            #         continue
            trajectory.iloc[i:i+1,j:j+5] = np.nan
        if coordinate[3] > 2 or coordinate[3] == 0:
            trajectory.iloc[i:i + 1, j:j + 5] = np.nan

with pd.ExcelWriter('./processed data.xlsx', mode='a') as writer:
    trajectory.to_excel(writer, sheet_name= 'Group 3')

exit()
# useful_data = pd.DataFrame()
# for i in range(0, trajectory.shape[1], 3):
#     new_trajectory = trajectory.iloc[:,i:i+3]
#     consecutive = 0
#     prev = 0
#     for j in range(1, len(new_trajectory)):
#         if not pd.isna(new_trajectory.iloc[j,0]):
#             useful_data += 1
#             if new_trajectory.iloc[j,0] - prev > 1:
#                 # extra_data.append(consecutive%5)
#                 consecutive = 0
#             else:
#                 consecutive += 1
#             if consecutive:
#                 dist = np.sqrt((new_trajectory.iloc[j,1] - new_trajectory.iloc[j-1,1])**2 + (new_trajectory.iloc[j,2] - new_trajectory.iloc[j-1,2])**2)
#                 if dist > 2:
#                     new_trajectory.iloc[j,:] = np.nan
#             prev = new_trajectory.iloc[j, 0]
#         else:
#             continue
#     useful_data = pd.concat([useful_data, new_trajectory], axis=1)
#     #     move[j] = dist
#     # keys = list(move.keys())
#     # # get values in the same order as keys, and parse percentage values
#     # vals = [float(move[k]) for k in keys]
#     # sns_plot = sns.lineplot(x=keys, y=vals)
#     # fig = sns_plot.get_figure()
#     # fig.savefig("output %d.png"%i)
#
# with pd.ExcelWriter('./processed data.xlsx', mode='a') as writer:
#     useful_data.to_excel(writer, sheet_name= 'Group 2')





# trajectory.to_csv('dataset.csv', index=False)



# def wgs84_to_meters(latitude, longitude):
#     # Define WGS84 and a projection for meters
#     wgs84 = pyproj.Proj(proj='latlong', datum='WGS84')
#     utm = pyproj.Proj(proj='utm', zone=50, datum='WGS84')


# new_trajectory

fig = px.scatter(new_trajectory, x=0,y=1)

fig.show()

# print(new_trajectory.iloc[[0]])
# print(new_trajectory.iloc[[500]])
# print(new_trajectory.iloc[[1000]])
# print(new_trajectory.iloc[[1500]])
# print(new_trajectory.iloc[[2000]])
# print(new_trajectory.iloc[[2300]])