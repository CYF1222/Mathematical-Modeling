import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, NearestNDInterpolator, LinearNDInterpolator, CloughTocher2DInterpolator
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

x=np.array([1200,1600,2000,2400,2800,3200,3600,4000])
y=np.array([1200,1600,2000,2400,2800,3200,3600])
x,y=np.meshgrid(x,y)
z=np.array([
    [1130, 1250, 1280, 1230, 1040, 900, 500, 700],
    [1320, 1450, 1420, 1400, 1300, 700, 900, 850],
    [1390, 1500, 1500, 1400, 900, 1100, 1060, 950],
    [1500, 1200, 1100, 1350, 1450, 1200, 1150, 1010],
    [1500, 1200, 1100, 1550, 1600, 1550, 1380, 1070],
    [1500, 1550, 1600, 1550, 1600, 1600, 1600, 1550],
    [1480, 1500, 1550, 1510, 1430, 1300, 1200, 980]
])

x_interp = np.linspace(1200, 4000, 100)
y_interp = np.linspace(1200, 3600, 80)
X_interp, Y_interp = np.meshgrid(x_interp, y_interp)

points = np.column_stack((x.ravel(), y.ravel()))
values = z.ravel()

methods = {
    '最近邻插值': 'nearest',
    '双线性插值': 'linear', 
    '双三次插值': 'cubic'
}

interp_results = {}

for name, method in methods.items():
    Z_interp = griddata(points, values, (X_interp, Y_interp), method=method)
    interp_results[name] = Z_interp
    print(f"{name} 完成")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('山区高程数据插值分析', fontsize=16, fontweight='bold')

colors = ['blue', 'lightblue', 'green', 'yellow', 'orange', 'red', 'purple']
cmap = ListedColormap(colors)
bounds = [500, 800, 1000, 1200, 1400, 1600, 1800, 2000]
norm = plt.Normalize(vmin=500, vmax=2000)

for idx, (name, Z_interp) in enumerate(interp_results.items()):
    # 地貌图
    im = axes[0, idx].contourf(X_interp, Y_interp, Z_interp, levels=50, cmap=cmap, norm=norm)
    axes[0, idx].set_title(f'{name} - 地貌图', fontsize=12, fontweight='bold')
    axes[0, idx].set_xlabel('X坐标')
    axes[0, idx].set_ylabel('Y坐标')
    axes[0, idx].grid(True, alpha=0.3)
    
    # 等高线图
    contour = axes[1, idx].contour(X_interp, Y_interp, Z_interp, levels=15, colors='black', linewidths=0.5)
    axes[1, idx].clabel(contour, inline=True, fontsize=8)
    axes[1, idx].contourf(X_interp, Y_interp, Z_interp, levels=50, cmap=cmap, norm=norm, alpha=0.7)
    axes[1, idx].set_title(f'{name} - 等高线图', fontsize=12, fontweight='bold')
    axes[1, idx].set_xlabel('X坐标')
    axes[1, idx].set_ylabel('Y坐标')
    axes[1, idx].grid(True, alpha=0.3)

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = plt.colorbar(im, cax=cbar_ax)
cbar.set_label('高程 (米)', rotation=270, labelpad=15)

plt.tight_layout(rect=[0, 0, 0.9, 0.95])
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('高度≥1400的区域分析', fontsize=16, fontweight='bold')

from mpl_toolkits.mplot3d import Axes3D

Z_analysis = interp_results['双三次插值']

ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
surf = ax_3d.plot_surface(X_interp, Y_interp, Z_analysis, cmap=cmap, 
                         linewidth=0, antialiased=True, alpha=0.8)
ax_3d.contour(X_interp, Y_interp, Z_analysis, levels=[1400], 
              offset=1300, colors='red', linewidths=3)
ax_3d.set_xlabel('X坐标')
ax_3d.set_ylabel('Y坐标')
ax_3d.set_zlabel('高程 (米)')
ax_3d.set_title('三维地形图(红色为1400米等高线)')

im1 = axes[0, 1].contourf(X_interp, Y_interp, Z_analysis, levels=50, cmap=cmap, norm=norm)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = plt.colorbar(im1, cax=cbar_ax)
cbar.set_label('高程 (米)', rotation=270, labelpad=15)

plt.tight_layout(rect=[0, 0, 0.9, 0.95])
plt.show()
