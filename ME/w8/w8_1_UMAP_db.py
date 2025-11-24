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
data = pd.read_excel('D:/Project/Python_VSCode/Mathematical Modeling/ME/w8/output_wavelength_as_columns.xlsx')
print(f"数据形状：{data.shape} | 样本数：{len(data)} | 波数点：{data.shape[1]-1}")  # 减1：仅排除编号列

# 提取关键信息
sample_ids = data.iloc[:, 0]  # 第一列：药材编号
wavenumbers = data.columns[1:].astype(float)  # 波数列（从第二列开始）
spectra = data.iloc[:, 1:].values  # 光谱数据（样本数×波数点）

# ----------------------
# 步骤2：光谱预处理（与监督学习相同）
# ----------------------
def preprocess_spectra_with_nan(spectra, window_length=15, polyorder=2, deriv=1):
    """处理含NaN的光谱：仅对非NaN区域平滑+导数，保留NaN位置"""
    processed = np.zeros_like(spectra)
    for i in range(spectra.shape[0]):  # 遍历每个样本
        spectrum = spectra[i, :]
        mask = ~np.isnan(spectrum)  # 非NaN区域的掩码（True=有效数据）
        if np.sum(mask) < window_length:  # 有效数据点不足，跳过处理
            processed[i, :] = spectrum
            continue
        # 仅对有效区域进行平滑+导数
        smoothed = savgol_filter(spectrum[mask], window_length=window_length, 
                                polyorder=polyorder, deriv=deriv)
        # 将处理后的数据回填到原位置（NaN位置保持NaN）
        processed[i, mask] = smoothed
    # 标准化（按样本，忽略NaN）
    scaler = StandardScaler()
    for i in range(processed.shape[0]):
        mask = ~np.isnan(processed[i, :])
        if np.sum(mask) == 0:
            continue
        processed[i, mask] = scaler.fit_transform(processed[i, mask].reshape(-1, 1)).flatten()
    return processed

processed_spectra = preprocess_spectra_with_nan(spectra, deriv=1)  # 预处理后的光谱

'''
# 生成包含NaN的波数序列（在波数间隔较大处断开）
light_con = []
for i in range(len(wavenumbers)-1):
    if wavenumbers[i+1] - wavenumbers[i] > 1:  # 波数间隔大于1时插入NaN
        light_con.append(wavenumbers[i])
        light_con.append(np.nan)  # 插入NaN断开连接
    else:
        light_con.append(wavenumbers[i])
light_con.append(wavenumbers[-1])  # 添加最后一个波数值

# 转换light_con为numpy数组
light_con = np.array(light_con)

# 对每个样本的光谱数据也进行相应的NaN插入
processed_spectra_with_gaps = []
for i in range(processed_spectra.shape[0]):
    spectrum_data = []
    for j in range(len(wavenumbers)-1):
        spectrum_data.append(processed_spectra[i, j])
        if wavenumbers[j+1] - wavenumbers[j] > 1:  # 同样的间隔条件
            spectrum_data.append(np.nan)  # 在光谱数据对应位置也插入NaN
    spectrum_data.append(processed_spectra[i, -1])
    processed_spectra_with_gaps.append(spectrum_data)

processed_spectra_with_gaps = np.array(processed_spectra_with_gaps)

# ----------------------
# 绘制预处理后的光谱（使用light_con逻辑）
# ----------------------
plt.figure(figsize=(12, 6))
for i in range(min(50, len(processed_spectra_with_gaps))):  # 绘制前50个样本避免过于密集
    plt.plot(light_con, processed_spectra_with_gaps[i], linewidth=0.5, alpha=0.6)

plt.xlabel('波数 (cm⁻¹)', fontsize=12)
plt.ylabel('预处理后吸光度', fontsize=12)
plt.title('预处理后光谱曲线（非连续区间断开显示）', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
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

# ----------------------
# 步骤5：按聚类结果绘制光谱（关键修改部分）
# ----------------------
n_clusters = len(np.unique(clusters))
colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))

plt.figure(figsize=(12, 7))

# 核心逻辑：生成包含NaN的波数序列（light_con逻辑）
# 假设我们要在非特征区间断开连接（这里以波数间隔大于10为例）
light_con = []
for i in range(len(wavenumbers)-1):
    if wavenumbers[i+1] - wavenumbers[i] > 10:  # 波数间隔大于10时插入NaN
        light_con.append(np.nan)
    else:
        light_con.append(wavenumbers[i])
light_con.append(wavenumbers[-1])  # 添加最后一个波数值

# 按簇分组绘制光谱
for cluster_id in range(n_clusters):
    cluster_indices = np.where(clusters == cluster_id)[0]
    for i in cluster_indices:
        # 使用light_con作为x轴，光谱数据作为y轴
        plt.plot(light_con, spectra[i, :], linewidth=0.5, alpha=0.6, color=colors[cluster_id])

# 添加图表元素
plt.xlabel('波数 (cm⁻¹)', fontsize=12)
plt.ylabel('原始吸光度', fontsize=12)
plt.title(f'按K-means聚类结果分组的光谱曲线 (n_clusters={n_clusters})', fontsize=14)
plt.grid(linestyle='--', alpha=0.5)

handles = [plt.Line2D([0], [0], color=colors[i], label=f'簇 {i}') for i in range(n_clusters)]
plt.legend(handles=handles, title='聚类标签', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

cluster_results = pd.DataFrame({
    'Sample_ID': sample_ids.values,  # 样本编号
    'Cluster_Label': clusters,      # 聚类标签
    'UMAP_1': umap_2d[:, 0],        # UMAP第一维度
    'UMAP_2': umap_2d[:, 1]         # UMAP第二维度
})

# 将原始光谱数据与聚类结果合并
# 首先创建原始数据的副本（不包含样本ID列）
spectra_df = pd.DataFrame(spectra, columns=wavenumbers)
spectra_df.insert(0, 'Sample_ID', sample_ids.values)  # 在第一列插入样本ID

# 合并聚类结果和光谱数据
final_results = pd.merge(cluster_results, spectra_df, on='Sample_ID', how='left')

# 保存到Excel文件
output_path = 'D:/Project/Python_VSCode/Mathematical Modeling/ME/w8/clustering_results.xlsx'
final_results.to_excel(output_path, index=False)
