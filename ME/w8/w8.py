import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# 读取Excel文件
med = pd.read_excel('D:/Project/Python_VSCode/Mathematical Modeling/ME/w8/1.xlsx')
med=med.drop('No',axis=1)

# 获取列名（light值）
light = med.columns.tolist()

med=med.to_numpy()

# 将数据转换为数值数组
med=np.delete(med,[63,135,200],axis=0)

#计算极差平均值
sum=0
for i in range(3348):
    single_light=med[:,i]
    single_var=np.var(single_light)
    sum=sum+single_var
ave_light=sum/3348

#提取特征区间
flag_start=0
list_start=[]
flag_finish=0
list_finish=[]
for i in range(3348):
    single_light=med[:,i]
    single_var=np.var(single_light)
    if(single_var>ave_light and flag_start==0):
        flag_start=1
        flag_finish=0
        list_start.append(i)
    if(single_var<=ave_light and flag_finish==0):
        flag_finish=1
        flag_start=0
        list_finish.append(i)

#数据处理
#x轴
light_take=[]
for i in range(7):
    for j in range(list_start[i],list_finish[i]+1):
        light_take.append(j+652)

#去除非特征区间的值，防止出现自动填充的直线
light_con=[]

for i in range(len(light_take)-1):
    if(light_take[i+1]-light_take[i]>1):
        light_con.append(np.nan)
    else:
        light_con.append(light_take[i])
light_con.append(light_take[-1])

#绘图
for i in range(422):
    row_data = med[i, light_take]
    plt.plot(light_con,row_data)

plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 提取特征区间的数据
extracted_data = med[:, light_take]  # 形状为 (422, 886)

# 创建DataFrame，行是样本，列是波长
df = pd.DataFrame(extracted_data, columns=light_con)

# 转置DataFrame，使波长成为行，样本成为列
df_transposed = df.T

# 添加样本编号作为列名
sample_columns = [f'Sample_{i+1}' for i in range(422)]
df_transposed.columns = sample_columns

# 添加波长作为第一列
df_transposed.insert(0, 'Wavelength', df_transposed.index)

# 保存到Excel
excel_path = 'D:/Project/Python_VSCode/Mathematical Modeling/ME/w8/output.xlsx'
df_transposed.to_excel(excel_path, index=False)