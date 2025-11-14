import numpy as np
import pandas as pd
import umap  # UMAP降维库（需安装：pip install umap-learn）
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  # 可选：自动聚类

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ----------------------
# 步骤1：加载数据（无标签，仅含编号和光谱）
# ----------------------
data = pd.read_excel('D:/Project/Python_VSCode/Mathematical Modeling/ME/w8/1.xlsx')
print(f"数据形状：{data.shape} | 样本数：{len(data)} | 波数点：{data.shape[1]-1}")  # 减1：仅排除编号列

# 提取关键信息
sample_ids = data.iloc[:, 0]  # 第一列：药材编号
wavenumbers = data.columns[1:].astype(float)  # 波数列（从第二列开始）
spectra = data.iloc[:, 1:].values  # 光谱数据（样本数×波数点）
rows_to_delete = [63, 135, 200]  # 您提供的索引

# 1. 删除光谱数据中的指定行
spectra = np.delete(spectra, rows_to_delete, axis=0) 
sample_ids = sample_ids.drop(sample_ids.index[rows_to_delete]).reset_index(drop=True)

# ----------------------
# 步骤2：光谱预处理（与监督学习相同）
# ----------------------
def preprocess_spectra(spectra, window_length=15, polyorder=2, deriv=1):
    """平滑+导数+标准化"""
    # 1. 平滑+一阶导数（消除噪声和基线漂移）
    processed = savgol_filter(spectra, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)
    # 2. 标准化（Z-score）
    scaler = StandardScaler()
    processed = scaler.fit_transform(processed)
    return processed

processed_spectra = preprocess_spectra(spectra, deriv=1)  # 预处理后的光谱
'''
for i in range(422):
    row_data = processed_spectra[i, :]
    plt.plot(wavenumbers,row_data)

plt.xlabel('x')
plt.ylabel('y')
plt.show()
'''

# ----------------------
# 步骤3：UMAP降维（核心！将高维光谱映射到2D）
# ----------------------
# 初始化UMAP模型（关键参数：n_components降维后维度，n_neighbors控制局部/全局结构）
umap_model = umap.UMAP(
    n_components=2,  # 降维到2D（可视化）
    n_neighbors=15,  # 邻居数：小→关注局部结构，大→关注全局结构（建议5-30）
    min_dist=0.1,    # 簇内点最小距离：小→簇更紧密，大→簇更分散
    random_state=42
)

# 执行降维
umap_2d = umap_model.fit_transform(processed_spectra)  # 输出：(样本数, 2)的二维数组


# ----------------------
# 步骤4：可视化UMAP结果（观察聚类趋势）
# ----------------------
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    umap_2d[:, 0], umap_2d[:, 1],  # UMAP降维后的x、y坐标
    c='blue',  # 暂用单一颜色，后续可结合K-means上色
    s=50, 
    alpha=0.7, 
    edgecolors='k'
)

# 添加标签和标题
plt.xlabel('UMAP 1st Component', fontsize=12)
plt.ylabel('UMAP 2nd Component', fontsize=12)
plt.title('UMAP Visualization of Herb Spectra Data', fontsize=14)
plt.grid(linestyle='--', alpha=0.5)


# ----------------------
# 步骤5（可选）：K-means自动聚类+上色
# ----------------------
# 若UMAP图显示明显簇结构，可用K-means划分簇并上色
from sklearn.cluster import KMeans

# 假设通过UMAP观察到可能有3个簇（需根据实际图形调整n_clusters）
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(umap_2d)  # 基于UMAP结果聚类（也可直接用原始光谱聚类，但计算慢）

# 上色后的UMAP图
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    umap_2d[:, 0], umap_2d[:, 1],
    c=clusters,  # 用K-means聚类结果上色
    cmap='viridis',  # 颜色映射（可选：'rainbow', 'plasma'）
    s=50, 
    alpha=0.7, 
    edgecolors='k'
)
plt.colorbar(scatter, label='Cluster Label')  # 颜色条：簇编号
plt.xlabel('UMAP 1st Component', fontsize=12)
plt.ylabel('UMAP 2nd Component', fontsize=12)
plt.title('UMAP + K-means Clustering (n_clusters=3)', fontsize=14)
plt.grid(linestyle='--', alpha=0.5)
plt.show()

# 1. 获取聚类结果（假设已执行步骤5，clusters为长度=样本数的数组，值为0/1/2等簇编号）
n_clusters = len(np.unique(clusters))  # 自动获取簇数量（无需硬编码）
colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))  # 为每个簇分配颜色（从viridis色图中取）

plt.figure(figsize=(12, 7))  # 更大的画布，避免曲线重叠

# 2. 按簇分组绘制光谱
for cluster_id in range(n_clusters):
    # 筛选当前簇的所有样本索引（例如：簇0的样本索引）
    cluster_indices = np.where(clusters == cluster_id)[0]
    # 遍历当前簇的每个样本，用同一颜色绘制
    for i in cluster_indices:
        row_data = spectra[i, :]  # 第i个样本的光谱数据
        plt.plot(wavenumbers, row_data, linewidth=0.5, alpha=0.6, color=colors[cluster_id])

# 3. 添加图例、标签和标题
plt.xlabel('波数 (cm⁻¹)', fontsize=12)
plt.ylabel('预处理后吸光度 (导数+标准化)', fontsize=12)
plt.title(f'按K-means聚类结果分组的光谱曲线 (n_clusters={n_clusters})', fontsize=14)
plt.grid(linestyle='--', alpha=0.5)

# 添加图例（每个簇对应一种颜色）
handles = [plt.Line2D([0], [0], color=colors[i], label=f'簇 {i}') for i in range(n_clusters)]
plt.legend(handles=handles, title='聚类标签', bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在右侧

plt.tight_layout()  # 自动调整布局，避免标签被截断
plt.show()
