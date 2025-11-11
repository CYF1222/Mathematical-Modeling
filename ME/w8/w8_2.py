import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import sys
import io

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 设置标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

#读取文件并分测试训练集
med = pd.read_excel('D:/Project/Python_VSCode/Mathematical Modeling/ME/w8/2.xlsx')
test = med[med['OP'].isna()]
train = med[med['OP'].notna()]

#类别，特征和测试序号分开，便于处理
train=train.drop('No',axis=1)
no_test=test['No']
test=test.drop('No',axis=1)
x_train_all=train.drop('OP',axis=1)
y_train_all=train['OP']
x_test=test.drop('OP',axis=1)

x_train, x_val, y_train, y_val = train_test_split(
    x_train_all, y_train_all,
    test_size=0.2,  # 20%训练数据作为验证集（可调整比例）
    random_state=42,  # 固定随机种子，结果可复现
    stratify=y_train_all  # 保持类别比例（分类任务推荐）
)

#标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)  # 训练集拟合并标准化
X_val_scaled = scaler.transform(x_val)  # 验证集标准化
X_test_scaled = scaler.transform(x_test)  # 测试集用训练集的scaler标准化


# SVM多分类默认使用"一对多（OvR）"策略，直接处理标签
svm_model = SVC(
    kernel='rbf',         # 核函数：rbf（非线性数据）/linear（线性数据）
    C=1.0,                # 正则化强度：C越大，对错误样本惩罚越重（防过拟合）
    gamma='scale',        # 核系数：自动适配特征尺度（推荐默认）
    probability=True      # 启用概率预测（可选，用于后续评估）
)
svm_model.fit(X_train_scaled, y_train)  # 使用标准化后的数据训练

y_val_pred = svm_model.predict(X_val_scaled)  # 验证集预测
accuracy = accuracy_score(y_val, y_val_pred)  # 计算准确率
print(f"模型在验证集上的准确率：{accuracy:.4f}\n")  # 输出准确率
print("验证集分类报告：")
print(classification_report(y_val, y_val_pred))  # 详细评估指标

#生成测试结果
y_pred = svm_model.predict(X_test_scaled)

# 重置索引，便于拼接
y_pred_series = pd.Series(y_pred, name='prediction')
no_test_reset = no_test.reset_index(drop=True)
y_pred_series_reset = y_pred_series.reset_index(drop=True)

# 拼接
result_df = pd.concat([no_test_reset, y_pred_series_reset], axis=1)
print(result_df)