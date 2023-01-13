from scipy.stats import norm
import numpy as np
import pandas as pd
import sympy as sp
data = pd.read_csv("data.csv")
data = data.to_dict("list")
datax=np.array(data['x'])
datax=datax[0:len(datax)-1]
print(len(datax))
x_mean, x_std = norm.fit(datax)
delt=[]
for i in range(len(data['q'])-1):
    # print((data['q'][i+1]-data['q'][i])/3)
    delt.append((data['q'][i+1]-data['q'][i]))
print(len(delt))
k= sp.symbols('K',positive=True)
d= sp.symbols('d')
# x_std= sp.symbols('x_std')
xstar=sp.symbols('xstar',positive=True)
phi = (1.0/(sp.sqrt(2*sp.pi)*x_std))*sp.exp(-0.5*((xstar-k*d-x_mean)**2/x_std*x_std))#分布函数
L = np.prod([phi.subs(d, i) for i in delt]) # 似然函数
logL = sp.expand_log(sp.log(L))
print(logL)
res= sp.solve([sp.diff(logL, k),sp.diff(logL, xstar)],[k,xstar])#对xstar和K求导，解方程
print(res)