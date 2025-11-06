import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# 读取Excel文件
med = pd.read_excel('D:/Project/Python_VSCode/Mathematical Modeling/w8/1.xlsx')
med=med.drop('No',axis=1)

# 获取列名（light值）
light = med.columns.tolist()

med=med.to_numpy()

# 将数据转换为数值数组
med=np.delete(med,[63,135,200],axis=0)

sum=0
for i in range(3348):
    single_light=med[:,i]
    single_var=np.var(single_light)
    sum=sum+single_var
ave_light=sum/3348

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

light_take=[]
for i in range(7):
    for j in range(list_start[i],list_finish[i]+1):
        light_take.append(j+652)

light_con=[]

for i in range(len(light_take)-1):
    if(light_take[i+1]-light_take[i]>1):
        light_con.append(np.nan)
    else:
        light_con.append(light_take[i])
light_con.append(light_take[-1])

for i in range(422):
    row_data = med[i, light_take]
    plt.plot(light_con,row_data)

plt.xlabel('x')
plt.ylabel('y')
plt.show()
