# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 20:43:05 2021

@author: kh722
"""
# %%
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    print("can't open ipython !")
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
from sklearn.decomposition import PCA
from glob import glob
from datetime import datetime
plt.close("all")
plt.rcParams["figure.dpi"] = 300
# sns.set()
PATH = 'R_ratio_2022-0316-02-19-36'
data_path = glob(os.path.join(PATH, '*.csv'))
now = datetime.now()
timestr =  now.strftime('%Y-%m%d-%H-%M-%S')
output_path = f'Plot_StO2_{timestr}'
if os.path.exists(output_path):
        print('output folder already exist!')
else:
        os.mkdir(output_path)
        
for id_data, path in enumerate(data_path):
        df = pd.read_csv(path)
        df2 = df.drop(['Unnamed: 0', 'sds', 'stO2'], axis=1)
        x1 = df[df['stO2'] == 0.3]
        x2 = df[df['stO2'] == 0.4]
        x3 = df[df['stO2'] == 0.5]
        x4 = df[df['stO2'] == 0.6]
        x5 = df[df['stO2'] == 0.7]
        x1 = x1.drop(['Unnamed: 0', 'sds', 'stO2'], axis=1)
        x2 = x2.drop(['Unnamed: 0', 'sds', 'stO2'], axis=1)
        x3 = x3.drop(['Unnamed: 0', 'sds', 'stO2'], axis=1)
        x4 = x4.drop(['Unnamed: 0', 'sds', 'stO2'], axis=1)
        x5 = x5.drop(['Unnamed: 0', 'sds', 'stO2'], axis=1)
        x1 = np.array(x1)
        x2 = np.array(x2)
        x3 = np.array(x3)
        x4 = np.array(x4)
        x5 = np.array(x5)
        pca = PCA(n_components=2)
        trans_func = pca.fit(df2)
        y1 = trans_func.transform(x1)
        y2 = trans_func.transform(x2)
        y3 = trans_func.transform(x3)
        y4 = trans_func.transform(x4)
        y5 = trans_func.transform(x5)
        c1 = [30] * len(x1)
        c2 = [40] * len(x2)
        c3 = [50] * len(x3)
        c4 = [60] * len(x4)
        c5 = [70] * len(x5)
        plt.scatter(y1[:, 0], y1[:, 1], c = c1, marker='.', vmin=30, vmax=70, s=25, cmap='jet')
        plt.scatter(y2[:, 0], y2[:, 1], c = c2, marker='.', vmin=30, vmax=70, s=25, cmap='jet')
        plt.scatter(y3[:, 0], y3[:, 1], c = c3, marker='.', vmin=30, vmax=70, s=25, cmap='jet')
        plt.scatter(y4[:, 0], y4[:, 1], c = c4, marker='.', vmin=30, vmax=70, s=25, cmap='jet')
        plt.scatter(y5[:, 0], y5[:, 1], c = c5, marker='.', vmin=30, vmax=70, s=25, cmap='jet')
        plt.colorbar()
        plt.axis('equal')
        plt.title('StO2')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.show()
        plt.savefig(os.path.join(output_path, f'{id_data}'))


# %%

rng = np.random.RandomState(1)
# X = rng.randn(2, 200).T
z = range(200)

cm = plt.cm.get_cmap('jet')
X = np.dot(rng.rand(2, 2) , rng.randn(2, 200)).T
mappable = plt.scatter(X[:, 0], X[:, 1], c=z, marker='o',vmin=0, vmax=100, s=25, cmap=cm)
plt.colorbar(mappable)
plt.axis('equal')

pca = PCA(n_components=2)
trans_func = pca.fit(X)
num = np.array([[100, 50]])
n_f = trans_func.transform(num)
print(n_f)

