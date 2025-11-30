import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import io

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 设置标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

med = pd.read_excel('D:/Project/Python_VSCode/Mathematical Modeling/w8/processed_pca_data.xlsx')
med=med.drop('Sample_ID',axis=1)
cluster=med.Cluster
labels = med.iloc[:, 0].values          #标签
X = med   # 特征：排除首列（类别列）

#标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 标准化后特征（均值=0，标准差=1）

#划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, labels, 
    test_size=0.3,        # 测试集占30%
    random_state=42,      # 固定随机种子，结果可复现
    stratify=labels            # 按类别比例划分，避免样本不平衡
)

# SVM多分类默认使用"一对多（OvR）"策略，直接处理4类标签
svm_model = SVC(
    kernel='rbf',         # 核函数：rbf（非线性数据）/linear（线性数据）
    C=1.0,                # 正则化强度：C越大，对错误样本惩罚越重（防过拟合）
    gamma='scale',        # 核系数：自动适配特征尺度（推荐默认）
    probability=True      # 启用概率预测（可选，用于后续评估）
)
svm_model.fit(X_train, y_train)  # 训练模型

#模型评估
#1.准确率
y_pred = svm_model.predict(X_test)  # 测试集预测结果
print(f"准确率：{accuracy_score(y_test, y_pred):.2f}") #准确率

# 2. 分类报告（精确率、召回率、F1值，支持4类）
print("\n分类报告（4类）：")
print(classification_report(
    y_test, y_pred, 
    target_names=[f'类别{i}' for i in range(4)]  # 自定义类别名称（0-3）
))

# 3. 混淆矩阵（直观展示每类预测效果）
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True,          # 显示数值
    fmt='d',             # 整数格式
    cmap='Blues', 
    xticklabels=[f'预测类别{i}' for i in range(1,5)],
    yticklabels=[f'真实类别{i}' for i in range(1,5)]
)
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.title('SVM多分类混淆矩阵（4类）')
plt.show()
