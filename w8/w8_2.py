import pandas as pd
from scipy.signal import savgol_filter 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 读取数据
med = pd.read_excel('D:/Project/Python_VSCode/Mathematical Modeling/ME/w8/2.xlsx')
test = med[med['OP'].isna()]
train = med[med['OP'].notna()]

# 数据处理，分开训练和分类集，去除无关列
train = train.drop('No', axis=1)
no_test = test['No']
test = test.drop('No', axis=1)
x_train_all = train.drop('OP', axis=1)
y_train_all = train['OP']
x_test = test.drop('OP', axis=1)

print(f"训练集形状: {x_train_all.shape}")
print(f"测试集形状: {x_test.shape}")

# 数据预处理
X_train_smoothed = savgol_filter(x_train_all, window_length=15, polyorder=3, axis=1)
X_test_smoothed = savgol_filter(x_test, window_length=15, polyorder=3, axis=1)

# 标准化
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_smoothed)
X_test_scaled = scaler.transform(X_test_smoothed)

# 特征工程 - PCA
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 数据分割
x_train, x_val, y_train, y_val = train_test_split(
    X_train_pca, y_train_all,
    test_size=0.2,
    random_state=42,
    stratify=y_train_all
)

# 模型训练 - 随机森林
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(x_train, y_train)

# 模型评估
y_train_pred = rf_model.predict(x_train)
y_val_pred = rf_model.predict(x_val)

train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"训练集准确率: {train_accuracy:.4f}")
print(f"验证集准确率: {val_accuracy:.4f}")

# 测试集预测
y_pred = rf_model.predict(X_test_pca)
result_df = pd.DataFrame({
    'No': no_test.reset_index(drop=True),
    'prediction': y_pred
})

print("\n测试集预测结果:")
print(result_df)