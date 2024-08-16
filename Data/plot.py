import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# 读取Excel文件
df_1 = pd.read_csv('./Experiment 1.csv')
df_2 = pd.read_csv('./Experiment 2.csv')
df_3 = pd.read_csv('./Experiment 3.csv')
dfs = [df_1, df_2, df_3]
data = pd.concat(dfs, ignore_index=True)

# 清理数据：移除包含缺失值的行
cleaned_data = data.dropna(subset=['Positionx', 'Positiony'])
cleaned_data = cleaned_data[cleaned_data['Distance'] <3.28]

df = data

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(df['Positionx'].min(), df['Positionx'].max())
ax.set_ylim(df['Positiony'].min(), df['Positiony'].max())

# 绘制每个轨迹
grouped = df.groupby('Trajectory')
for name, group in grouped:
    ax.plot(group['Positionx'], group['Positiony'], marker='o', linestyle='-', markersize=2, label=f'Trajectory {name}')

ax.set_xlabel('Position X')
ax.set_ylabel('Position Y')
ax.legend()
plt.show()

# 提取Positionx和Positiony数据
x = cleaned_data['Positionx']
y = cleaned_data['Positiony']

# 计算点密度
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)

# 创建图像
fig, ax = plt.subplots()
scatter = ax.scatter(x, y, c=z, s=10, edgecolor='white')
plt.colorbar(scatter, label='Density')
plt.xlabel('Positionx')
plt.ylabel('Positiony')
plt.title('Density Surface Mapping of Position Data')
plt.show()

# 提取Positionx, Positiony数据
x = cleaned_data['Positionx']
y = cleaned_data['Positiony']

# 绘制轨迹点图
plt.figure(figsize=(10, 8))
plt.scatter(x, y, c='orange', s=1, alpha=0.5)
plt.title('Raw Track Points')
plt.xlabel('Positionx')
plt.ylabel('Positiony')
plt.show()

# 绘制密度表面图
plt.figure(figsize=(10, 8))
sns.kdeplot(x=x, y=y, cmap='Blues', fill=True, bw_adjust=0.5)
plt.title('Density Surface Mapping')
plt.xlabel('Positionx')
plt.ylabel('Positiony')
plt.show()

# 绘制轨迹路径图
plt.figure(figsize=(10, 8))
plt.hist2d(x, y, bins=100, cmap='Blues')
plt.colorbar(label='Density')
plt.title('Density Surface of Trajectory Paths')
plt.xlabel('Positionx')
plt.ylabel('Positiony')
plt.show()

# 彩色密度表面图（2D和3D）
plt.figure(figsize=(10, 8))
sns.kdeplot(x=x, y=y, cmap='Greens', fill=True, bw_adjust=0.5)
plt.title('Colored Density Surface Mapping (2D)')
plt.xlabel('Positionx')
plt.ylabel('Positiony')
plt.show()

# 3D密度图
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
hist, xedges, yedges = np.histogram2d(x, y, bins=50, density=True)

# Construct arrays for the anchor positions of the bars.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the bars.
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color=plt.cm.Greens(dz / dz.max()))

ax.set_xlabel('Positionx')
ax.set_ylabel('Positiony')
ax.set_zlabel('Density')
ax.set_title('Colored Density Surface in 3D')
plt.show()
