import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import sympy as sp

T=np.arange(10,81,10)
y=np.array([0.1,0.3,0.7,0.94,0.95,0.68,0.34,0.13])

plt.plot(T,y,'r-',label='原始数据')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

x_new=np.arange(10,80,1)

#1
def func1(x,a,b,c):
    return a*x**2+b*x+c

popt,pcov=curve_fit(
    f=func1,
    xdata=T,
    ydata=y,
    p0=[0,0,0]
)

a1_fit,b1_fit,c1_fit=popt
plt.plot(x_new,func1(x_new,a1_fit,b1_fit,c1_fit),'green',
         label=f'拟合曲线1：{a1_fit:.2}T^2+{b1_fit:.2}T+{c1_fit:.2}'
         )

#计算最值
x1_r=sp.symbols('x1_r')
f1=func1(x1_r,a1_fit,b1_fit,c1_fit)
f1_prime=sp.diff(f1,x1_r)
s1=sp.solve(f1_prime,x1_r)
print(s1)

#2
def func2(x,a,b,c):
    return a/(c+b*(x-45)**2)

popt,pcov=curve_fit(
    f=func2,
    xdata=T,
    ydata=y,
    p0=[1,1,1]
)

a2_fit,b2_fit,c2_fit=popt
plt.plot(x_new,func2(x_new,a2_fit,b2_fit,c2_fit),'blue',
         label=f'拟合曲线2：{a2_fit:.2}/({c2_fit:.2}+{b2_fit:.2}(T-45)^2)'
         )

#计算最值
x2_r=sp.symbols('x2_r')
f2=func2(x2_r,a2_fit,b2_fit,c2_fit)
f2_prime=sp.diff(f2,x2_r)
s2=sp.solve(f2_prime,x2_r)
print(s2)

plt.legend()
plt.show()

print(func2(100,a2_fit,b2_fit,c2_fit))