import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([2615,1943,1494,1087,765,538,484,290,226,204])

plt.plot(x, y, 'r-', label='aa')
plt.xlabel('this is x')
plt.ylabel('this is y')
plt.title('this is a demo')
plt.legend()  # 将样例显示出来

def exp_func(x,a,b,c):
    return a*np.exp(b*x)+c

popt, pcov = curve_fit(
    f=exp_func,          # 自定义指数函数
    xdata=x,             # x轴数据
    ydata=y,             # y轴数据（含噪声）
    p0=[0,0,0]
)
a_fit, b_fit, c_fit = popt
plt.plot(x, exp_func(x, a_fit, b_fit, c_fit), 'r-', linewidth=2, 
         label=f'拟合曲线: y={a_fit:.2f}e^({b_fit:.2f}x) + {c_fit:.2f}')

plt.legend()
plt.show()

x_new=np.array([14.5])
y_new=exp_func(x_new,*popt)
print(y_new)
