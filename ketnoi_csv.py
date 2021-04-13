import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('healthcare-dataset-stroke-data.csv', usecols=[1]).values
print(pd.read_csv('healthcare-dataset-stroke-data.csv'))
nam = 0
nu = 0
for i in range(len(df)):
    # print(df[i],"  ")
    if (df[i] == 'Male'):
        nam = nam + 1
    elif (df[i] == 'Female'):
        nu = nu + 1
print("tong nam:",nam)
print("tong nu:", nu)
print("tong:", nam+nu)

tong = int(nam+nu)
ptnam=float(nam/tong)*100
ptnu = float(nu / tong) * 100

#giá trị phần trăm
percents = [ptnam, ptnu]
#tên côt các loại
program_langunages = ["Nam", "Nữ"]

explode =[0,0]
plt.pie(percents, labels=program_langunages,autopct='%1.2f%%',
        wedgeprops={'edgecolor':'black','linewidth':1.5},
        explode=explode)
plt.title('Biểu đồ tròn tỷ lệ phần trăm về giới tính')
plt.show()
