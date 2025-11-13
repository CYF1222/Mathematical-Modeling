import numpy as np
import pandas as pd
from sklearn.manifold import Isomap  # 导入Isomap
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ----------------------
# 步骤1：加载数据（路径已修正）
# ----------------------
data = pd.read_excel('D:/Project/Python_VSCode/Mathematical Modeling/ME/w8/1.xlsx')
print(f"数据形状：{data.shape} | 样本数：{len(data)} | 波数点：{data.shape[1]-1}")

sample_ids = data.iloc[:, 0]  # 药材编号
wavenumbers = data.columns[1:].astype(float)  # 波数列
spectra = data.iloc[:, 1:].values  # 光谱数据


# ----------------------
# 步骤2：光谱预处理（不变）
# ----------------------
def preprocess_spectra(spectra, window_length=15, polyorder=2, deriv=1):
    """平滑+导数+标准化"""
    processed = savgol_filter(spectra, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)
    scaler = StandardScaler()
    processed = scaler.fit_transform(processed)
    return processed

processed_spectra = preprocess_spectra(spectra, deriv=1)


# ----------------------
# 步骤3：Isomap降维（核心修改！）
# ----------------------
# 初始化Isomap模型（关键参数：n_neighbors控制流形重构的局部邻域大小）
isomap_model = Isomap(
    n_neighbors=15,  # 近邻数（建议5-30，过小易过拟合，过大丢失局部结构）
    n_components=2,  # 降维到2D
    eigen_solver='auto',  # 自动选择特征值分解方法（大规模数据可选'dense'）
)

# 执行降维（输入：预处理后的光谱数据）
isomap_2d = isomap_model.fit_transform(processed_spectra)  # 输出：(样本数, 2)


# ----------------------
# 步骤4：Isomap结果可视化（与UMAP类似）
# ----------------------
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    isomap_2d[:, 0], isomap_2d[:, 1],
    c='blue', s=50, alpha=0.7, edgecolors='k'
)
plt.xlabel('Isomap Component 1', fontsize=12)
plt.ylabel('Isomap Component 2', fontsize=12)
plt.title('Isomap Visualization of Herb Spectra Data', fontsize=14)
plt.grid(linestyle='--', alpha=0.5)

# 标记前20个样本编号
for i, txt in enumerate(sample_ids[:20]):
    plt.annotate(txt, (isomap_2d[i, 0], isomap_2d[i, 1]), fontsize=8)

plt.show()


# ----------------------
# 步骤5：K-means聚类+上色（不变）
# ----------------------
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(isomap_2d)  # 基于Isomap结果聚类

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    isomap_2d[:, 0], isomap_2d[:, 1],
    c=clusters, cmap='viridis', s=50, alpha=0.7, edgecolors='k'
)
plt.colorbar(scatter, label='Cluster Label')
plt.xlabel('Isomap Component 1', fontsize=12)
plt.ylabel('Isomap Component 2', fontsize=12)
plt.title('Isomap + K-means Clustering (n_clusters=3)', fontsize=14)
plt.grid(linestyle='--', alpha=0.5)
plt.show()
