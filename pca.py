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
from cProfile import label
from configparser import Interpolation
import os
from pickle import TRUE
from unicodedata import decimal
import numpy as np
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
from sklearn.decomposition import PCA
from glob import glob
from datetime import datetime
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from scipy import spatial
from scipy.interpolate import interpn
from itertools import product
import cv2
from tqdm import tqdm
plt.close("all")
plt.rcParams["figure.dpi"] = 300
# sns.set()
def rmse(y, ytest):
        return np.sqrt(np.sum((ytest - y)**2)/len(y))
def to_percent(temp, position):
        return '%1.0f'%(100*temp) + '%'
PATH = 'R_ratio_2022-0323-16-50-11'
data_path = glob(os.path.join(PATH, '*.csv'))
now = datetime.now()
timestr =  now.strftime('%Y-%m%d-%H-%M-%S')
output_path = f'Plot_StO2_{PATH}'
if os.path.exists(output_path):
        print('Output folder already exist!')
else:
        os.mkdir(output_path)

mus = [[15.89,	11.89,	5.01,],
[15.89,	11.89,	6.455,],
[15.89,	11.89,	7.9,],
[15.89,	17.025,	5.01,],
[15.89,	17.025,	6.455,],
[15.89,	17.025,	7.9,],
[15.89,	22.16,	5.01,],
[15.89,	22.16,	6.455,],
[15.89,	22.16,	7.9,],
[20.6,	11.89,	5.01,],
[20.6,	11.89,	6.455,],
[20.6,	11.89,	7.9,],
[20.6,	17.025,	5.01,],
[20.6,	17.025,	6.455,],
[20.6,	17.025,	7.9,],
[20.6,	22.16,	5.01,],
[20.6,	22.16,	6.455,],
[20.6,	22.16,	7.9,],
[25.31,	11.89,	5.01,],
[25.31,	11.89,	6.455,],
[25.31,	11.89,	7.9,],
[25.31,	17.025,	5.01,],
[25.31,	17.025,	6.455,],
[25.31,	17.025,	7.9,],
[25.31,	22.16,	5.01,],
[25.31,	22.16,	6.455,],
[25.31,	22.16,	7.9]]
mus_ = []
for i in range(27):
        for j in range(1620):
                mus_.append(mus[i])
df_mus = pd.DataFrame(mus_, columns=['mus_skin', 'mus_fat', 'mus_muscle'])

# get data of each mus
for id_data, path in enumerate(data_path):
        if id_data == 0:
                df = pd.read_csv(path)
        else: 
                df_temp = pd.read_csv(path)
                df = pd.concat([df, df_temp], axis = 0)
df_all = pd.concat([df.reset_index(), df_mus], axis = 1)
df_all.to_csv(f'{PATH}.csv')
df13 = pd.read_csv(os.path.join(PATH, '13.csv'))
df_new = pd.DataFrame({'730': df_all['small_730'] / df_all['large_730'],
        '760': df_all['small_760'] / df_all['large_760'],
        '780': df_all['small_780'] / df_all['large_780'],
        '810': df_all['small_810'] / df_all['large_810'],
        '850': df_all['small_850'] / df_all['large_850'],
        'stO2': df_all['stO2'],
        'sds': df_all['sds'],
        'skin': df_all['skin'],
        'fat': df_all['fat'],
        'muscle': df_all['muscle'],
        'mus_skin': df_all['mus_skin'],
        'mus_fat': df_all['mus_fat'],
        'mus_muscle': df_all['mus_muscle'],
        })
df_new1 = df_new[df_new['sds'] == 1].reset_index().drop(['index'],axis=1)
df_new2 = df_new[df_new['sds'] == 8].reset_index().drop(['index'],axis=1)
df_new3 = df_new[df_new['sds'] == 15].reset_index().drop(['index'],axis=1)
df_new_ref = (df_new1[['730', '760', '780', '810', '850']] 
        + df_new2[['730', '760', '780', '810', '850']]
        + df_new3[['730', '760', '780', '810', '850']]).mean(axis=1)
df_new1['730'] /= df_new_ref
df_new1['760'] /= df_new_ref
df_new1['780'] /= df_new_ref
df_new1['810'] /= df_new_ref
df_new1['850'] /= df_new_ref
df_new2['730'] /= df_new_ref
df_new2['760'] /= df_new_ref
df_new2['780'] /= df_new_ref
df_new2['810'] /= df_new_ref
df_new2['850'] /= df_new_ref
df_new3['730'] /= df_new_ref
df_new3['760'] /= df_new_ref
df_new3['780'] /= df_new_ref
df_new3['810'] /= df_new_ref
df_new3['850'] /= df_new_ref
df_new_all = pd.concat([df_new1, df_new2, df_new3])
# df_new_all = 
# seperate each sds
df2 = df[['760', '780', '810', '850', 'sds', 'stO2']]
# df13 = df13[['760', '780', '810', '850', 'sds', 'stO2']]
df_sds1 = df2[df2['sds'] == 1]
df_sds2 = df2[df2['sds'] == 8]
df_sds3 = df2[df2['sds'] == 15]
df13_sds1 = df13[df13['sds'] == 1]
df13_sds2 = df13[df13['sds'] == 8]
df13_sds3 = df13[df13['sds'] == 15]
x1 = [df_sds1[df_sds1['stO2'] == i] for i in np.round(np.linspace(0.3, 0.75, 10), decimals=2)]
x2 = [df_sds2[df_sds2['stO2'] == i] for i in np.round(np.linspace(0.3, 0.75, 10), decimals=2)]
x3 = [df_sds3[df_sds3['stO2'] == i] for i in np.round(np.linspace(0.3, 0.75, 10), decimals=2)]

# muscle -> fat -> skin -> mus_muscle -> mus_fat -> mus_skin
# 'skin': array([0.0465, 0.1367, 0.2269]),
#  'fat': array([0.1098 , 0.11555, 0.1213 ]),
#  'muscle': array([0.013  , 0.01916, 0.02532, 0.03148, 0.03764, 0.0438 ])


mua_skin = [0.038, 0.1115, 0.185]
mua_fat = [0.1015, 0.10425, 0.107 ]
mua_muscle = [0.011, 0.0148, 0.0186, 0.0224, 0.0262, 0.03 ]
mus_skin = [15.89, 20.6, 25.31]
mus_fat	= [11.89, 17.025, 22.16]
mus_muscle = [5.01, 6.455, 7.9]

# %% bound plot mua_skin
outpath = f'bound_mua_skin_'+ PATH
if os.path.exists(outpath):
        print(f'{outpath} already exists !')
else: os.mkdir(outpath)
iter_param = list(product(mua_fat, mua_muscle, mus_skin, mus_fat, mus_muscle))
column = ['mua_fat', 'mua_muscle', 'mus_skin', 'mus_fat', 'mus_muscle']
df_bound_mua = pd.DataFrame(np.array(iter_param), columns=column)
df_bound_mua.to_csv(os.path.join(outpath, 'param.csv'))
for pid, iter in enumerate(tqdm(iter_param)): 
        mask_low = df_all['skin'] == mua_skin[0]
        mask_mid = df_all['skin'] == mua_skin[1]
        mask_up = df_all['skin'] == mua_skin[2]
        mask_sds1 = df_all['sds'] == 1
        mask_sds2 = df_all['sds'] == 8
        mask_sds3 = df_all['sds'] == 15
        mask2 = df_all['fat'] == iter[0]
        mask3 = df_all['muscle'] == iter[1]
        mask4 = df_all['mus_skin'] == iter[2]
        mask5 = df_all['mus_fat'] == iter[3]
        mask6 = df_all['mus_muscle'] == iter[4]
        df_low1 = df_all[(mask_low & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid1 = df_all[(mask_mid & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up1 = df_all[(mask_up & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low2 = df_all[(mask_low & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid2 = df_all[(mask_mid & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up2 = df_all[(mask_up & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low3 = df_all[(mask_low & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid3 = df_all[(mask_mid & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up3 = df_all[(mask_up & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        df_low1.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds1, ub', marker='o', grid=True)
        
        df_low2.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds2, ub', marker='^', grid=True)
        
        df_low3.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds3, ub', marker='x', grid=True)
        xtick = np.round(np.linspace(0.8, 1.3, 11), 2)
        ax[0, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='760 nm')
        ax[0, 1].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='780 nm')
        ax[1, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='810 nm')
        ax[1, 1].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='850 nm')
        lines, labels = fig.axes[-1].get_legend_handles_labels()
        ax[0, 0].get_legend().remove()
        ax[0, 1].get_legend().remove()
        ax[1, 0].get_legend().remove()
        ax[1, 1].get_legend().remove()
        fig.legend(lines, labels, bbox_to_anchor=(1.1, 0.5), loc = 'right')
        ax[0, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[0, 1].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[1, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[1, 1].yaxis.set_major_formatter(FuncFormatter(to_percent))
        # ax[0, 0].plot(df_low['780'], df_low['stO2'], marker='o', label='μa_skin lower bound')
        # ax[0, 0].plot(df_up['780'], df_up['stO2'], marker='^', label='μa_skin upper bound')
        # ax[0, 0].plot(df_mid['780'], df_mid['stO2'], marker='x', label='μa_skin middle')
        fig.suptitle(f'μa_skin {pid}', size=20)
        fig.tight_layout()
        fig.savefig(os.path.join(outpath, f'{pid:05d}.png'), bbox_inches='tight')
        plt.close(fig)

# %% bound plot mua_fat
outpath = 'bound_mua_fat_'+ PATH
if os.path.exists(outpath):
        print(f'{outpath} already exists !')
else: os.mkdir(outpath)
iter_param = list(product(mua_skin, mua_muscle, mus_skin, mus_fat, mus_muscle))
column = ['mua_skin', 'mua_muscle', 'mus_skin', 'mus_fat', 'mus_muscle']
df_bound_mua = pd.DataFrame(np.array(iter_param), columns=column)
df_bound_mua.to_csv(os.path.join(outpath, 'param.csv'))
for pid, iter in enumerate(tqdm(iter_param)): 
        mask_low = df_all['fat'] == mua_fat[0]
        mask_mid = df_all['fat'] ==  mua_fat[1]
        mask_up = df_all['fat'] == mua_fat[2]
        mask_sds1 = df_all['sds'] == 1
        mask_sds2 = df_all['sds'] == 8
        mask_sds3 = df_all['sds'] == 15
        mask2 = df_all['skin'] == iter[0]
        mask3 = df_all['muscle'] == iter[1]
        mask4 = df_all['mus_skin'] == iter[2]
        mask5 = df_all['mus_fat'] == iter[3]
        mask6 = df_all['mus_muscle'] == iter[4]
        df_low1 = df_all[(mask_low & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid1 = df_all[(mask_mid & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up1 = df_all[(mask_up & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low2 = df_all[(mask_low & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid2 = df_all[(mask_mid & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up2 = df_all[(mask_up & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low3 = df_all[(mask_low & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid3 = df_all[(mask_mid & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up3 = df_all[(mask_up & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        df_low1.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds1, ub', marker='o', grid=True)
        
        df_low2.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds2, ub', marker='^', grid=True)
        
        df_low3.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds3, ub', marker='x', grid=True)
        xtick = np.round(np.linspace(0.8, 1.3, 11), 2)
        ax[0, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='760 nm')
        ax[0, 1].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='780 nm')
        ax[1, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='810 nm')
        ax[1, 1].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='850 nm')
        lines, labels = fig.axes[-1].get_legend_handles_labels()
        ax[0, 0].get_legend().remove()
        ax[0, 1].get_legend().remove()
        ax[1, 0].get_legend().remove()
        ax[1, 1].get_legend().remove()
        fig.legend(lines, labels, bbox_to_anchor=(1.1, 0.5), loc = 'right')
        ax[0, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[0, 1].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[1, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[1, 1].yaxis.set_major_formatter(FuncFormatter(to_percent))
        # ax[0, 0].plot(df_low['780'], df_low['stO2'], marker='o', label='μa_skin lower bound')
        # ax[0, 0].plot(df_up['780'], df_up['stO2'], marker='^', label='μa_skin upper bound')
        # ax[0, 0].plot(df_mid['780'], df_mid['stO2'], marker='x', label='μa_skin middle')
        fig.suptitle(f'μa_fat {pid}', size=20)
        fig.tight_layout()
        fig.savefig(os.path.join(outpath, f'{pid:05d}.png'), bbox_inches='tight')
        plt.close(fig)

# %% bound plot mua_muscle
outpath = 'bound_mua_muscle_'+ PATH
if os.path.exists(outpath):
        print(f'{outpath} already exists !')
else: os.mkdir(outpath)
iter_param = list(product(mua_skin, mua_fat, mus_skin, mus_fat, mus_muscle))
column = ['mua_skin', 'mua_fat', 'mus_skin', 'mus_fat', 'mus_muscle']
df_bound_mua = pd.DataFrame(np.array(iter_param), columns=column)
df_bound_mua.to_csv(os.path.join(outpath, 'param.csv'))
for pid, iter in enumerate(tqdm(iter_param)): 
        mask_low1 = df_all['muscle'] == mua_muscle[0]
        mask_low2 = df_all['muscle'] == mua_muscle[1]
        mask_mid1 = df_all['muscle'] == mua_muscle[2]
        mask_mid2 = df_all['muscle'] == mua_muscle[3]
        mask_up1 = df_all['muscle'] == mua_muscle[4]
        mask_up2 = df_all['muscle'] == mua_muscle[5]
        mask_sds1 = df_all['sds'] == 1
        mask_sds2 = df_all['sds'] == 8
        mask_sds3 = df_all['sds'] == 15
        mask2 = df_all['skin'] == iter[0]
        mask3 = df_all['fat'] == iter[1]
        mask4 = df_all['mus_skin'] == iter[2]
        mask5 = df_all['mus_fat'] == iter[3]
        mask6 = df_all['mus_muscle'] == iter[4]
        df_low11 = df_all[(mask_low1 & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low12 = df_all[(mask_low2 & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid11 = df_all[(mask_mid1 & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid12 = df_all[(mask_mid2 & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up11 = df_all[(mask_up1 & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up12 = df_all[(mask_up2 & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low21 = df_all[(mask_low1 & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low22 = df_all[(mask_low2 & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid21 = df_all[(mask_mid1 & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid22 = df_all[(mask_mid2 & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up21 = df_all[(mask_up1 & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up22 = df_all[(mask_up2 & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low31 = df_all[(mask_low1 & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low32 = df_all[(mask_low2 & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid31 = df_all[(mask_mid1 & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid32 = df_all[(mask_mid2 & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up31 = df_all[(mask_up1 & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up32 = df_all[(mask_up2 & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        xtick = np.round(np.linspace(0.8, 1.3, 11), 2)

        df_low11.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds1, lb1', marker='o', grid=True)
        df_low12.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds1, lb2', marker='o', grid=True)
        df_mid11.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds1, mid1', marker='o', grid=True)
        df_mid12.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds1, mid2', marker='o', grid=True)
        df_up11.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds1, ub1', marker='o', grid=True)
        df_up12.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds1, ub2', marker='o', grid=True)
        df_low11.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds1, lb1', marker='o', grid=True)
        df_low12.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds1, lb2', marker='o', grid=True)
        df_mid11.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds1, mid1', marker='o', grid=True)
        df_mid12.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds1, mid2', marker='o', grid=True)
        df_up11.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds1, ub1', marker='o', grid=True)
        df_up12.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds1, ub2', marker='o', grid=True)
        df_low11.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds1, lb1', marker='o', grid=True)
        df_low12.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds1, lb2', marker='o', grid=True)
        df_mid11.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds1, mid1', marker='o', grid=True)
        df_mid12.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds1, mid2', marker='o', grid=True)
        df_up11.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds1, ub1', marker='o', grid=True)
        df_up12.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds1, ub2', marker='o', grid=True)
        df_low11.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds1, lb1', marker='o', grid=True)
        df_low12.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds1, lb2', marker='o', grid=True)
        df_mid11.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds1, mid1', marker='o', grid=True)
        df_mid12.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds1, mid2', marker='o', grid=True)
        df_up11.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds1, ub1', marker='o', grid=True)
        df_up12.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds1, ub2', marker='o', grid=True)
        
        df_low21.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds2, lb1', marker='^', grid=True)
        df_low22.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds2, lb2', marker='^', grid=True)
        df_mid21.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds2, mid1', marker='^', grid=True)
        df_mid22.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds2, mid2', marker='^', grid=True)
        df_up21.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds2, ub1', marker='^', grid=True)
        df_up22.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds2, ub2', marker='^', grid=True)
        df_low21.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds2, lb1', marker='^', grid=True)
        df_low22.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds2, lb2', marker='^', grid=True)
        df_mid21.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds2, mid1', marker='^', grid=True)
        df_mid22.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds2, mid2', marker='^', grid=True)
        df_up21.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds2, ub1', marker='^', grid=True)
        df_up22.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds2, ub2', marker='^', grid=True)
        df_low21.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds2, lb1', marker='^', grid=True)
        df_low22.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds2, lb2', marker='^', grid=True)
        df_mid21.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds2, mid1', marker='^', grid=True)
        df_mid22.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds2, mid2', marker='^', grid=True)
        df_up21.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds2, ub1', marker='^', grid=True)
        df_up22.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds2, ub2', marker='^', grid=True)
        df_low21.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds2, lb1', marker='^', grid=True)
        df_low22.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds2, lb2', marker='^', grid=True)
        df_mid21.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds2, mid1', marker='^', grid=True)
        df_mid22.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds2, mid2', marker='^', grid=True)
        df_up21.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds2, ub1', marker='^', grid=True)
        df_up22.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds2, ub2', marker='^', grid=True)
        
        df_low31.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds3, lb1', marker='x', grid=True)
        df_low32.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds3, lb2', marker='x', grid=True)
        df_mid31.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds3, mid1', marker='x', grid=True)
        df_mid32.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds3, mid2', marker='x', grid=True)
        df_up31.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds3, ub1', marker='x', grid=True)
        df_up32.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds3, ub2', marker='x', grid=True)
        df_low31.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds3, lb1', marker='x', grid=True)
        df_low32.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds3, lb2', marker='x', grid=True)
        df_mid31.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds3, mid1', marker='x', grid=True)
        df_mid32.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds3, mid2', marker='x', grid=True)
        df_up31.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds3, ub1', marker='x', grid=True)
        df_up32.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds3, ub2', marker='x', grid=True)
        df_low31.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds3, lb1', marker='x', grid=True)
        df_low32.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds3, lb2', marker='x', grid=True)
        df_mid31.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds3, mid1', marker='x', grid=True)
        df_mid32.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds3, mid2', marker='x', grid=True)
        df_up31.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds3, ub1', marker='x', grid=True)
        df_up32.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds3, ub2', marker='x', grid=True)
        df_low31.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds3, lb1', marker='x', grid=True)
        df_low32.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds3, lb2', marker='x', grid=True)
        df_mid31.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds3, mid1', marker='x', grid=True)
        df_mid32.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds3, mid2', marker='x', grid=True)
        df_up31.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds3, ub1', marker='x', grid=True)
        df_up32.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds3, ub2', marker='x', grid=True)
        ax[0, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='760 nm')
        ax[0, 1].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='780 nm')
        ax[1, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='810 nm')
        ax[1, 1].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='850 nm')
        lines, labels = fig.axes[-1].get_legend_handles_labels()
        ax[0, 0].get_legend().remove()
        ax[0, 1].get_legend().remove()
        ax[1, 0].get_legend().remove()
        ax[1, 1].get_legend().remove()
        fig.legend(lines, labels, bbox_to_anchor=(1.1, 0.5), loc = 'right')
        # ax[0, 0].legend(loc='upper left')
        # ax[0, 1].legend(loc='upper left')
        # ax[1, 0].legend(loc='upper left')
        # ax[1, 1].legend(loc='upper left')
        ax[0, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[0, 1].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[1, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[1, 1].yaxis.set_major_formatter(FuncFormatter(to_percent))
        # ax[0, 0].plot(df_low['780'], df_low['stO2'], marker='o', label='μa_skin lower bound')
        # ax[0, 0].plot(df_up['780'], df_up['stO2'], marker='^', label='μa_skin upper bound')
        # ax[0, 0].plot(df_mid['780'], df_mid['stO2'], marker='x', label='μa_skin middle')
        fig.suptitle(f'μa_muscle {pid}', size=20)
        fig.tight_layout()
        fig.savefig(os.path.join(outpath, f'{pid:05d}.png'), bbox_inches='tight')
        plt.close(fig)
        

# %% bound plot mus_skin
outpath = 'bound_mus_skin_'+ PATH
if os.path.exists(outpath):
        print(f'{outpath} already exists !')
else: os.mkdir(outpath)
iter_param = list(product(mua_skin, mua_fat, mua_muscle, mus_fat, mus_muscle))
column = ['mua_skin', 'mua_fat', 'mua_muscle', 'mus_fat', 'mus_muscle']
df_bound_mua = pd.DataFrame(np.array(iter_param), columns=column)
df_bound_mua.to_csv(os.path.join(outpath, 'param.csv'))
for pid, iter in enumerate(tqdm(iter_param)): 
        mask_low = df_all['mus_skin'] == mus_skin[0]
        mask_mid = df_all['mus_skin'] == mus_skin[1]
        mask_up = df_all['mus_skin'] == mus_skin[2]
        mask_sds1 = df_all['sds'] == 1
        mask_sds2 = df_all['sds'] == 8
        mask_sds3 = df_all['sds'] == 15
        mask2 = df_all['skin'] == iter[0]
        mask3 = df_all['fat'] == iter[1]
        mask4 = df_all['muscle'] == iter[2]
        mask5 = df_all['mus_fat'] == iter[3]
        mask6 = df_all['mus_muscle'] == iter[4]
        df_low1 = df_all[(mask_low & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid1 = df_all[(mask_mid & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up1 = df_all[(mask_up & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low2 = df_all[(mask_low & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid2 = df_all[(mask_mid & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up2 = df_all[(mask_up & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low3 = df_all[(mask_low & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid3 = df_all[(mask_mid & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up3 = df_all[(mask_up & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        df_low1.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds1, ub', marker='o', grid=True)
        
        df_low2.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds2, ub', marker='^', grid=True)
        
        df_low3.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds3, ub', marker='x', grid=True)
        xtick = np.round(np.linspace(0.8, 1.3, 11), 2)
        ax[0, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='760 nm')
        ax[0, 1].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='780 nm')
        ax[1, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='810 nm')
        ax[1, 1].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='850 nm')
        lines, labels = fig.axes[-1].get_legend_handles_labels()
        ax[0, 0].get_legend().remove()
        ax[0, 1].get_legend().remove()
        ax[1, 0].get_legend().remove()
        ax[1, 1].get_legend().remove()
        fig.legend(lines, labels, bbox_to_anchor=(1.1, 0.5), loc = 'right')
        ax[0, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[0, 1].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[1, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[1, 1].yaxis.set_major_formatter(FuncFormatter(to_percent))
        # ax[0, 0].plot(df_low['780'], df_low['stO2'], marker='o', label='μa_skin lower bound')
        # ax[0, 0].plot(df_up['780'], df_up['stO2'], marker='^', label='μa_skin upper bound')
        # ax[0, 0].plot(df_mid['780'], df_mid['stO2'], marker='x', label='μa_skin middle')
        fig.suptitle(f'μs_skin {pid}', size=20)
        fig.tight_layout()
        fig.savefig(os.path.join(outpath, f'{pid:05d}.png'), bbox_inches='tight')
        plt.close(fig)

# %% bound plot mus_fat
outpath = 'bound_mus_fat_'+ PATH
if os.path.exists(outpath):
        print(f'{outpath} already exists !')
else: os.mkdir(outpath)
iter_param = list(product(mua_skin, mua_fat, mua_muscle, mus_skin, mus_muscle))
column = ['mua_skin', 'mua_fat', 'mua_muscle', 'mus_skin', 'mus_muscle']
df_bound_mua = pd.DataFrame(np.array(iter_param), columns=column)
df_bound_mua.to_csv(os.path.join(outpath, 'param.csv'))
for pid, iter in enumerate(tqdm(iter_param)): 
        mask_low = df_all['mus_fat'] == mus_fat[0]
        mask_mid = df_all['mus_fat'] == mus_fat[1]
        mask_up = df_all['mus_fat'] == mus_fat[2]
        mask_sds1 = df_all['sds'] == 1
        mask_sds2 = df_all['sds'] == 8
        mask_sds3 = df_all['sds'] == 15
        mask2 = df_all['skin'] == iter[0]
        mask3 = df_all['fat'] == iter[1]
        mask4 = df_all['muscle'] == iter[2]
        mask5 = df_all['mus_skin'] == iter[3]
        mask6 = df_all['mus_muscle'] == iter[4]
        df_low1 = df_all[(mask_low & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid1 = df_all[(mask_mid & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up1 = df_all[(mask_up & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low2 = df_all[(mask_low & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid2 = df_all[(mask_mid & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up2 = df_all[(mask_up & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low3 = df_all[(mask_low & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid3 = df_all[(mask_mid & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up3 = df_all[(mask_up & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        df_low1.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds1, ub', marker='o', grid=True)
        
        df_low2.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds2, ub', marker='^', grid=True)
        
        df_low3.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds3, ub', marker='x', grid=True)
        xtick = np.round(np.linspace(0.8, 1.3, 11), 2)
        ax[0, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='760 nm')
        ax[0, 1].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='780 nm')
        ax[1, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='810 nm')
        ax[1, 1].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='850 nm')
        lines, labels = fig.axes[-1].get_legend_handles_labels()
        ax[0, 0].get_legend().remove()
        ax[0, 1].get_legend().remove()
        ax[1, 0].get_legend().remove()
        ax[1, 1].get_legend().remove()
        fig.legend(lines, labels, bbox_to_anchor=(1.1, 0.5), loc = 'right')
        ax[0, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[0, 1].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[1, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[1, 1].yaxis.set_major_formatter(FuncFormatter(to_percent))
        # ax[0, 0].plot(df_low['780'], df_low['stO2'], marker='o', label='μa_skin lower bound')
        # ax[0, 0].plot(df_up['780'], df_up['stO2'], marker='^', label='μa_skin upper bound')
        # ax[0, 0].plot(df_mid['780'], df_mid['stO2'], marker='x', label='μa_skin middle')
        fig.suptitle(f'μs_fat {pid}', size=20)
        fig.tight_layout()
        fig.savefig(os.path.join(outpath, f'{pid:05d}.png'), bbox_inches='tight')
        plt.close(fig)

# %% bound plot mus_muscle
outpath = 'bound_mus_muscle_'+ PATH
if os.path.exists(outpath):
        print(f'{outpath} already exists !')
else: os.mkdir(outpath)
iter_param = list(product(mua_skin, mua_fat, mua_muscle, mus_skin, mus_fat))
column = ['mua_skin', 'mua_fat', 'mua_muscle', 'mus_skin', 'mus_fat']
df_bound_mua = pd.DataFrame(np.array(iter_param), columns=column)
df_bound_mua.to_csv(os.path.join(outpath, 'param.csv'))
for pid, iter in enumerate(tqdm(iter_param)): 
        mask_low = df_all['mus_muscle'] == mus_muscle[0]
        mask_mid = df_all['mus_muscle'] == mus_muscle[1]
        mask_up = df_all['mus_muscle'] == mus_muscle[2]
        mask_sds1 = df_all['sds'] == 1
        mask_sds2 = df_all['sds'] == 8
        mask_sds3 = df_all['sds'] == 15
        mask2 = df_all['skin'] == iter[0]
        mask3 = df_all['fat'] == iter[1]
        mask4 = df_all['muscle'] == iter[2]
        mask5 = df_all['mus_skin'] == iter[3]
        mask6 = df_all['mus_fat'] == iter[4]
        df_low1 = df_all[(mask_low & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid1 = df_all[(mask_mid & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up1 = df_all[(mask_up & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low2 = df_all[(mask_low & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid2 = df_all[(mask_mid & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up2 = df_all[(mask_up & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low3 = df_all[(mask_low & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid3 = df_all[(mask_mid & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up3 = df_all[(mask_up & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        df_low1.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds1, ub', marker='o', grid=True)
        
        df_low2.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds2, ub', marker='^', grid=True)
        
        df_low3.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '760', y = 'stO2', ax=ax[0, 0], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '780', y = 'stO2', ax=ax[0, 1], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '810', y = 'stO2', ax=ax[1, 0], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '850', y = 'stO2', ax=ax[1, 1], label='sds3, ub', marker='x', grid=True)
        xtick = np.round(np.linspace(0.8, 1.3, 11), 2)
        ax[0, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='760 nm')
        ax[0, 1].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='780 nm')
        ax[1, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='810 nm')
        ax[1, 1].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='850 nm')
        lines, labels = fig.axes[-1].get_legend_handles_labels()
        ax[0, 0].get_legend().remove()
        ax[0, 1].get_legend().remove()
        ax[1, 0].get_legend().remove()
        ax[1, 1].get_legend().remove()
        fig.legend(lines, labels, bbox_to_anchor=(1.1, 0.5), loc = 'right')
        ax[0, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[0, 1].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[1, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[1, 1].yaxis.set_major_formatter(FuncFormatter(to_percent))
        # ax[0, 0].plot(df_low['780'], df_low['stO2'], marker='o', label='μa_skin lower bound')
        # ax[0, 0].plot(df_up['780'], df_up['stO2'], marker='^', label='μa_skin upper bound')
        # ax[0, 0].plot(df_mid['780'], df_mid['stO2'], marker='x', label='μa_skin middle')
        fig.suptitle(f'μs_muscle {pid}', size=20)
        fig.tight_layout()
        fig.savefig(os.path.join(outpath, f'{pid:05d}.png'), bbox_inches='tight')
        plt.close(fig)
        
# %% new bound plot mua_skin
outpath = f'newbound_mua_skin_'+ PATH
if os.path.exists(outpath):
        print(f'{outpath} already exists !')
else: os.mkdir(outpath)
iter_param = list(product(mua_fat, mua_muscle, mus_skin, mus_fat, mus_muscle))
column = ['mua_fat', 'mua_muscle', 'mus_skin', 'mus_fat', 'mus_muscle']
df_bound_mua = pd.DataFrame(np.array(iter_param), columns=column)
df_bound_mua.to_csv(os.path.join(outpath, 'param.csv'))
for pid, iter in enumerate(tqdm(iter_param)): 
        mask_low = df_new_all['skin'] == mua_skin[0]
        mask_mid = df_new_all['skin'] == mua_skin[1]
        mask_up = df_new_all['skin'] == mua_skin[2]
        mask_sds1 = df_new_all['sds'] == 1
        mask_sds2 = df_new_all['sds'] == 8
        mask_sds3 = df_new_all['sds'] == 15
        mask2 = df_new_all['fat'] == iter[0]
        mask3 = df_new_all['muscle'] == iter[1]
        mask4 = df_new_all['mus_skin'] == iter[2]
        mask5 = df_new_all['mus_fat'] == iter[3]
        mask6 = df_new_all['mus_muscle'] == iter[4]
        df_low1 = df_new_all[(mask_low & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid1 = df_new_all[(mask_mid & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up1 = df_new_all[(mask_up & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low2 = df_new_all[(mask_low & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid2 = df_new_all[(mask_mid & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up2 = df_new_all[(mask_up & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low3 = df_new_all[(mask_low & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid3 = df_new_all[(mask_mid & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up3 = df_new_all[(mask_up & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        fig, ax = plt.subplots(3, 2, figsize=(12, 12))
        df_low1.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds1, ub', marker='o', grid=True)
        
        df_low2.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds2, ub', marker='^', grid=True)
        
        df_low3.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds3, ub', marker='x', grid=True)
        xtick = np.round(np.linspace(0.25, 0.55, 11), 2)
        ax[0, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='730 nm')
        ax[0, 1].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='760 nm')
        ax[1, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='780 nm')
        ax[1, 1].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='810 nm')
        ax[2, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='850 nm')
        lines, labels = fig.axes[0].get_legend_handles_labels()
        ax[0, 0].get_legend().remove()
        ax[0, 1].get_legend().remove()
        ax[1, 0].get_legend().remove()
        ax[1, 1].get_legend().remove()
        ax[2, 0].get_legend().remove()
        fig.legend(lines, labels, bbox_to_anchor=(1.1, 0.5), loc = 'right')
        ax[0, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[0, 1].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[1, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[1, 1].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[2, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        # ax[0, 0].plot(df_low['780'], df_low['stO2'], marker='o', label='μa_skin lower bound')
        # ax[0, 0].plot(df_up['780'], df_up['stO2'], marker='^', label='μa_skin upper bound')
        # ax[0, 0].plot(df_mid['780'], df_mid['stO2'], marker='x', label='μa_skin middle')
        fig.suptitle(f'μa_skin {pid}', size=20)
        fig.tight_layout()
        fig.savefig(os.path.join(outpath, f'{pid:05d}.png'), bbox_inches='tight')
        plt.close(fig)

# %% new bound plot mua_fat
outpath = 'newbound_mua_fat_'+ PATH
if os.path.exists(outpath):
        print(f'{outpath} already exists !')
else: os.mkdir(outpath)
iter_param = list(product(mua_skin, mua_muscle, mus_skin, mus_fat, mus_muscle))
column = ['mua_skin', 'mua_muscle', 'mus_skin', 'mus_fat', 'mus_muscle']
df_bound_mua = pd.DataFrame(np.array(iter_param), columns=column)
df_bound_mua.to_csv(os.path.join(outpath, 'param.csv'))
for pid, iter in enumerate(tqdm(iter_param)): 
        mask_low = df_new_all['fat'] == mua_fat[0]
        mask_mid = df_new_all['fat'] ==  mua_fat[1]
        mask_up = df_new_all['fat'] == mua_fat[2]
        mask_sds1 = df_new_all['sds'] == 1
        mask_sds2 = df_new_all['sds'] == 8
        mask_sds3 = df_new_all['sds'] == 15
        mask2 = df_new_all['skin'] == iter[0]
        mask3 = df_new_all['muscle'] == iter[1]
        mask4 = df_new_all['mus_skin'] == iter[2]
        mask5 = df_new_all['mus_fat'] == iter[3]
        mask6 = df_new_all['mus_muscle'] == iter[4]
        df_low1 = df_new_all[(mask_low & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid1 = df_new_all[(mask_mid & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up1 = df_new_all[(mask_up & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low2 = df_new_all[(mask_low & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid2 = df_new_all[(mask_mid & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up2 = df_new_all[(mask_up & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low3 = df_new_all[(mask_low & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid3 = df_new_all[(mask_mid & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up3 = df_new_all[(mask_up & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        fig, ax = plt.subplots(3, 2, figsize=(12, 12))
        df_low1.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds1, ub', marker='o', grid=True)
        
        df_low2.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds2, ub', marker='^', grid=True)
        
        df_low3.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds3, ub', marker='x', grid=True)
        xtick = np.round(np.linspace(0.25, 0.55, 11), 2)
        ax[0, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='730 nm')
        ax[0, 1].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='760 nm')
        ax[1, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='780 nm')
        ax[1, 1].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='810 nm')
        ax[2, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='850 nm')
        lines, labels = fig.axes[0].get_legend_handles_labels()
        ax[0, 0].get_legend().remove()
        ax[0, 1].get_legend().remove()
        ax[1, 0].get_legend().remove()
        ax[1, 1].get_legend().remove()
        ax[2, 0].get_legend().remove()
        fig.legend(lines, labels, bbox_to_anchor=(1.1, 0.5), loc = 'right')
        ax[0, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[0, 1].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[1, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[1, 1].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[2, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        # ax[0, 0].plot(df_low['780'], df_low['stO2'], marker='o', label='μa_skin lower bound')
        # ax[0, 0].plot(df_up['780'], df_up['stO2'], marker='^', label='μa_skin upper bound')
        # ax[0, 0].plot(df_mid['780'], df_mid['stO2'], marker='x', label='μa_skin middle')
        fig.suptitle(f'μa_fat {pid}', size=20)
        fig.tight_layout()
        fig.savefig(os.path.join(outpath, f'{pid:05d}.png'), bbox_inches='tight')
        plt.close(fig)

# %% new bound plot mua_muscle
outpath = 'newbound_mua_muscle_'+ PATH
if os.path.exists(outpath):
        print(f'{outpath} already exists !')
else: os.mkdir(outpath)
iter_param = list(product(mua_skin, mua_fat, mus_skin, mus_fat, mus_muscle))
column = ['mua_skin', 'mua_fat', 'mus_skin', 'mus_fat', 'mus_muscle']
df_bound_mua = pd.DataFrame(np.array(iter_param), columns=column)
df_bound_mua.to_csv(os.path.join(outpath, 'param.csv'))
for pid, iter in enumerate(tqdm(iter_param)): 
        mask_low1 = df_new_all['muscle'] == mua_muscle[0]
        mask_low2 = df_new_all['muscle'] == mua_muscle[1]
        mask_mid1 = df_new_all['muscle'] == mua_muscle[2]
        mask_mid2 = df_new_all['muscle'] == mua_muscle[3]
        mask_up1 = df_new_all['muscle'] == mua_muscle[4]
        mask_up2 = df_new_all['muscle'] == mua_muscle[5]
        mask_sds1 = df_new_all['sds'] == 1
        mask_sds2 = df_new_all['sds'] == 8
        mask_sds3 = df_new_all['sds'] == 15
        mask2 = df_new_all['skin'] == iter[0]
        mask3 = df_new_all['fat'] == iter[1]
        mask4 = df_new_all['mus_skin'] == iter[2]
        mask5 = df_new_all['mus_fat'] == iter[3]
        mask6 = df_new_all['mus_muscle'] == iter[4]
        df_low11 = df_new_all[(mask_low1 & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low12 = df_new_all[(mask_low2 & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid11 = df_new_all[(mask_mid1 & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid12 = df_new_all[(mask_mid2 & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up11 = df_new_all[(mask_up1 & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up12 = df_new_all[(mask_up2 & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low21 = df_new_all[(mask_low1 & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low22 = df_new_all[(mask_low2 & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid21 = df_new_all[(mask_mid1 & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid22 = df_new_all[(mask_mid2 & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up21 = df_new_all[(mask_up1 & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up22 = df_new_all[(mask_up2 & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low31 = df_new_all[(mask_low1 & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low32 = df_new_all[(mask_low2 & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid31 = df_new_all[(mask_mid1 & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid32 = df_new_all[(mask_mid2 & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up31 = df_new_all[(mask_up1 & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up32 = df_new_all[(mask_up2 & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        fig, ax = plt.subplots(3, 2, figsize=(12, 12))
        
        df_low11.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds1, lb1', marker='o', grid=True)
        df_low12.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds1, lb2', marker='o', grid=True)
        df_mid11.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds1, mid1', marker='o', grid=True)
        df_mid12.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds1, mid2', marker='o', grid=True)
        df_up11.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds1, ub1', marker='o', grid=True)
        df_up12.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds1, ub2', marker='o', grid=True)
        df_low11.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds1, lb1', marker='o', grid=True)
        df_low12.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds1, lb2', marker='o', grid=True)
        df_mid11.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds1, mid1', marker='o', grid=True)
        df_mid12.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds1, mid2', marker='o', grid=True)
        df_up11.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds1, ub1', marker='o', grid=True)
        df_up12.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds1, ub2', marker='o', grid=True)
        df_low11.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds1, lb1', marker='o', grid=True)
        df_low12.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds1, lb2', marker='o', grid=True)
        df_mid11.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds1, mid1', marker='o', grid=True)
        df_mid12.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds1, mid2', marker='o', grid=True)
        df_up11.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds1, ub1', marker='o', grid=True)
        df_up12.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds1, ub2', marker='o', grid=True)
        df_low11.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds1, lb1', marker='o', grid=True)
        df_low12.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds1, lb2', marker='o', grid=True)
        df_mid11.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds1, mid1', marker='o', grid=True)
        df_mid12.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds1, mid2', marker='o', grid=True)
        df_up11.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds1, ub1', marker='o', grid=True)
        df_up12.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds1, ub2', marker='o', grid=True)
        df_low11.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds1, lb1', marker='o', grid=True)
        df_low12.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds1, lb2', marker='o', grid=True)
        df_mid11.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds1, mid1', marker='o', grid=True)
        df_mid12.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds1, mid2', marker='o', grid=True)
        df_up11.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds1, ub1', marker='o', grid=True)
        df_up12.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds1, ub2', marker='o', grid=True)
        
        df_low21.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds2, lb1', marker='^', grid=True)
        df_low22.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds2, lb2', marker='^', grid=True)
        df_mid21.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds2, mid1', marker='^', grid=True)
        df_mid22.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds2, mid2', marker='^', grid=True)
        df_up21.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds2, ub1', marker='^', grid=True)
        df_up22.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds2, ub2', marker='^', grid=True)
        df_low21.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds2, lb1', marker='^', grid=True)
        df_low22.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds2, lb2', marker='^', grid=True)
        df_mid21.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds2, mid1', marker='^', grid=True)
        df_mid22.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds2, mid2', marker='^', grid=True)
        df_up21.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds2, ub1', marker='^', grid=True)
        df_up22.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds2, ub2', marker='^', grid=True)
        df_low21.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds2, lb1', marker='^', grid=True)
        df_low22.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds2, lb2', marker='^', grid=True)
        df_mid21.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds2, mid1', marker='^', grid=True)
        df_mid22.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds2, mid2', marker='^', grid=True)
        df_up21.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds2, ub1', marker='^', grid=True)
        df_up22.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds2, ub2', marker='^', grid=True)
        df_low21.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds2, lb1', marker='^', grid=True)
        df_low22.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds2, lb2', marker='^', grid=True)
        df_mid21.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds2, mid1', marker='^', grid=True)
        df_mid22.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds2, mid2', marker='^', grid=True)
        df_up21.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds2, ub1', marker='^', grid=True)
        df_up22.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds2, ub2', marker='^', grid=True)
        df_low21.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds2, lb1', marker='^', grid=True)
        df_low22.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds2, lb2', marker='^', grid=True)
        df_mid21.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds2, mid1', marker='^', grid=True)
        df_mid22.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds2, mid2', marker='^', grid=True)
        df_up21.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds2, ub1', marker='^', grid=True)
        df_up22.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds2, ub2', marker='^', grid=True)
        
        df_low31.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds3, lb1', marker='x', grid=True)
        df_low32.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds3, lb2', marker='x', grid=True)
        df_mid31.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds3, mid1', marker='x', grid=True)
        df_mid32.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds3, mid2', marker='x', grid=True)
        df_up31.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds3, ub1', marker='x', grid=True)
        df_up32.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds3, ub2', marker='x', grid=True)
        df_low31.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds3, lb1', marker='x', grid=True)
        df_low32.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds3, lb2', marker='x', grid=True)
        df_mid31.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds3, mid1', marker='x', grid=True)
        df_mid32.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds3, mid2', marker='x', grid=True)
        df_up31.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds3, ub1', marker='x', grid=True)
        df_up32.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds3, ub2', marker='x', grid=True)
        df_low31.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds3, lb1', marker='x', grid=True)
        df_low32.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds3, lb2', marker='x', grid=True)
        df_mid31.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds3, mid1', marker='x', grid=True)
        df_mid32.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds3, mid2', marker='x', grid=True)
        df_up31.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds3, ub1', marker='x', grid=True)
        df_up32.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds3, ub2', marker='x', grid=True)
        df_low31.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds3, lb1', marker='x', grid=True)
        df_low32.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds3, lb2', marker='x', grid=True)
        df_mid31.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds3, mid1', marker='x', grid=True)
        df_mid32.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds3, mid2', marker='x', grid=True)
        df_up31.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds3, ub1', marker='x', grid=True)
        df_up32.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds3, ub2', marker='x', grid=True)
        df_low31.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds3, lb1', marker='x', grid=True)
        df_low32.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds3, lb2', marker='x', grid=True)
        df_mid31.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds3, mid1', marker='x', grid=True)
        df_mid32.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds3, mid2', marker='x', grid=True)
        df_up31.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds3, ub1', marker='x', grid=True)
        df_up32.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds3, ub2', marker='x', grid=True)
        xtick = np.round(np.linspace(0.25, 0.55, 11), 2)
        ax[0, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='730 nm')
        ax[0, 1].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='760 nm')
        ax[1, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='780 nm')
        ax[1, 1].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='810 nm')
        ax[2, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='850 nm')
        lines, labels = fig.axes[0].get_legend_handles_labels()
        ax[0, 0].get_legend().remove()
        ax[0, 1].get_legend().remove()
        ax[1, 0].get_legend().remove()
        ax[1, 1].get_legend().remove()
        ax[2, 0].get_legend().remove()
        fig.legend(lines, labels, bbox_to_anchor=(1.1, 0.5), loc = 'right')
        ax[0, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[0, 1].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[1, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[1, 1].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[2, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        fig.suptitle(f'μa_muscle {pid}', size=20)
        fig.tight_layout()
        fig.savefig(os.path.join(outpath, f'{pid:05d}.png'), bbox_inches='tight')
        plt.close(fig)
        

# %% new bound plot mus_skin
outpath = 'newbound_mus_skin_'+ PATH
if os.path.exists(outpath):
        print(f'{outpath} already exists !')
else: os.mkdir(outpath)
iter_param = list(product(mua_skin, mua_fat, mua_muscle, mus_fat, mus_muscle))
column = ['mua_skin', 'mua_fat', 'mua_muscle', 'mus_fat', 'mus_muscle']
df_bound_mua = pd.DataFrame(np.array(iter_param), columns=column)
df_bound_mua.to_csv(os.path.join(outpath, 'param.csv'))
for pid, iter in enumerate(tqdm(iter_param)): 
        mask_low = df_new_all['mus_skin'] == mus_skin[0]
        mask_mid = df_new_all['mus_skin'] == mus_skin[1]
        mask_up = df_new_all['mus_skin'] == mus_skin[2]
        mask_sds1 = df_new_all['sds'] == 1
        mask_sds2 = df_new_all['sds'] == 8
        mask_sds3 = df_new_all['sds'] == 15
        mask2 = df_new_all['skin'] == iter[0]
        mask3 = df_new_all['fat'] == iter[1]
        mask4 = df_new_all['muscle'] == iter[2]
        mask5 = df_new_all['mus_fat'] == iter[3]
        mask6 = df_new_all['mus_muscle'] == iter[4]
        df_low1 = df_new_all[(mask_low & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid1 = df_new_all[(mask_mid & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up1 = df_new_all[(mask_up & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low2 = df_new_all[(mask_low & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid2 = df_new_all[(mask_mid & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up2 = df_new_all[(mask_up & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low3 = df_new_all[(mask_low & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid3 = df_new_all[(mask_mid & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up3 = df_new_all[(mask_up & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        fig, ax = plt.subplots(3, 2, figsize=(12, 12))
        df_low1.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds1, ub', marker='o', grid=True)
        
        df_low2.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds2, ub', marker='^', grid=True)
        
        df_low3.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds3, ub', marker='x', grid=True)
        xtick = np.round(np.linspace(0.25, 0.55, 11), 2)
        ax[0, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='730 nm')
        ax[0, 1].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='760 nm')
        ax[1, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='780 nm')
        ax[1, 1].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='810 nm')
        ax[2, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='850 nm')
        lines, labels = fig.axes[0].get_legend_handles_labels()
        ax[0, 0].get_legend().remove()
        ax[0, 1].get_legend().remove()
        ax[1, 0].get_legend().remove()
        ax[1, 1].get_legend().remove()
        ax[2, 0].get_legend().remove()
        fig.legend(lines, labels, bbox_to_anchor=(1.1, 0.5), loc = 'right')
        ax[0, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[0, 1].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[1, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[1, 1].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[2, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        # ax[0, 0].plot(df_low['780'], df_low['stO2'], marker='o', label='μa_skin lower bound')
        # ax[0, 0].plot(df_up['780'], df_up['stO2'], marker='^', label='μa_skin upper bound')
        # ax[0, 0].plot(df_mid['780'], df_mid['stO2'], marker='x', label='μa_skin middle')
        fig.suptitle(f'μs_skin {pid}', size=20)
        fig.tight_layout()
        fig.savefig(os.path.join(outpath, f'{pid:05d}.png'), bbox_inches='tight')
        plt.close(fig)

# %% new bound plot mus_fat
outpath = 'newbound_mus_fat_'+ PATH
if os.path.exists(outpath):
        print(f'{outpath} already exists !')
else: os.mkdir(outpath)
iter_param = list(product(mua_skin, mua_fat, mua_muscle, mus_skin, mus_muscle))
column = ['mua_skin', 'mua_fat', 'mua_muscle', 'mus_skin', 'mus_muscle']
df_bound_mua = pd.DataFrame(np.array(iter_param), columns=column)
df_bound_mua.to_csv(os.path.join(outpath, 'param.csv'))
for pid, iter in enumerate(tqdm(iter_param)): 
        mask_low = df_new_all['mus_fat'] == mus_fat[0]
        mask_mid = df_new_all['mus_fat'] == mus_fat[1]
        mask_up = df_new_all['mus_fat'] == mus_fat[2]
        mask_sds1 = df_new_all['sds'] == 1
        mask_sds2 = df_new_all['sds'] == 8
        mask_sds3 = df_new_all['sds'] == 15
        mask2 = df_new_all['skin'] == iter[0]
        mask3 = df_new_all['fat'] == iter[1]
        mask4 = df_new_all['muscle'] == iter[2]
        mask5 = df_new_all['mus_skin'] == iter[3]
        mask6 = df_new_all['mus_muscle'] == iter[4]
        df_low1 = df_new_all[(mask_low & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid1 = df_new_all[(mask_mid & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up1 = df_new_all[(mask_up & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low2 = df_new_all[(mask_low & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid2 = df_new_all[(mask_mid & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up2 = df_new_all[(mask_up & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low3 = df_new_all[(mask_low & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid3 = df_new_all[(mask_mid & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up3 = df_new_all[(mask_up & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        fig, ax = plt.subplots(3, 2, figsize=(12, 12))
        df_low1.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds1, ub', marker='o', grid=True)
        
        df_low2.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds2, ub', marker='^', grid=True)
        
        df_low3.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds3, ub', marker='x', grid=True)
        xtick = np.round(np.linspace(0.25, 0.55, 11), 2)
        ax[0, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='730 nm')
        ax[0, 1].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='760 nm')
        ax[1, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='780 nm')
        ax[1, 1].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='810 nm')
        ax[2, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='850 nm')
        lines, labels = fig.axes[0].get_legend_handles_labels()
        ax[0, 0].get_legend().remove()
        ax[0, 1].get_legend().remove()
        ax[1, 0].get_legend().remove()
        ax[1, 1].get_legend().remove()
        ax[2, 0].get_legend().remove()
        fig.legend(lines, labels, bbox_to_anchor=(1.1, 0.5), loc = 'right')
        ax[0, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[0, 1].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[1, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[1, 1].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[2, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        # ax[0, 0].plot(df_low['780'], df_low['stO2'], marker='o', label='μa_skin lower bound')
        # ax[0, 0].plot(df_up['780'], df_up['stO2'], marker='^', label='μa_skin upper bound')
        # ax[0, 0].plot(df_mid['780'], df_mid['stO2'], marker='x', label='μa_skin middle')
        fig.suptitle(f'μs_fat {pid}', size=20)
        fig.tight_layout()
        fig.savefig(os.path.join(outpath, f'{pid:05d}.png'), bbox_inches='tight')
        plt.close(fig)

# %% new bound plot mus_muscle
outpath = 'newbound_mus_muscle_'+ PATH
if os.path.exists(outpath):
        print(f'{outpath} already exists !')
else: os.mkdir(outpath)
iter_param = list(product(mua_skin, mua_fat, mua_muscle, mus_skin, mus_fat))
column = ['mua_skin', 'mua_fat', 'mua_muscle', 'mus_skin', 'mus_fat']
df_bound_mua = pd.DataFrame(np.array(iter_param), columns=column)
df_bound_mua.to_csv(os.path.join(outpath, 'param.csv'))
for pid, iter in enumerate(tqdm(iter_param)): 
        mask_low = df_new_all['mus_muscle'] == mus_muscle[0]
        mask_mid = df_new_all['mus_muscle'] == mus_muscle[1]
        mask_up = df_new_all['mus_muscle'] == mus_muscle[2]
        mask_sds1 = df_new_all['sds'] == 1
        mask_sds2 = df_new_all['sds'] == 8
        mask_sds3 = df_new_all['sds'] == 15
        mask2 = df_new_all['skin'] == iter[0]
        mask3 = df_new_all['fat'] == iter[1]
        mask4 = df_new_all['muscle'] == iter[2]
        mask5 = df_new_all['mus_skin'] == iter[3]
        mask6 = df_new_all['mus_fat'] == iter[4]
        df_low1 = df_new_all[(mask_low & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid1 = df_new_all[(mask_mid & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up1 = df_new_all[(mask_up & mask_sds1 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low2 = df_new_all[(mask_low & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid2 = df_new_all[(mask_mid & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up2 = df_new_all[(mask_up & mask_sds2 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_low3 = df_new_all[(mask_low & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_mid3 = df_new_all[(mask_mid & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        df_up3 = df_new_all[(mask_up & mask_sds3 & mask2 & mask3 & mask4 & mask5 & mask6)]
        fig, ax = plt.subplots(3, 2, figsize=(12, 12))
        df_low1.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds1, ub', marker='o', grid=True)
        df_low1.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds1, lb', marker='o', grid=True)
        df_mid1.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds1, mid', marker='o', grid=True)
        df_up1.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds1, ub', marker='o', grid=True)
        
        df_low2.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds2, ub', marker='^', grid=True)
        df_low2.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds2, lb', marker='^', grid=True)
        df_mid2.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds2, mid', marker='^', grid=True)
        df_up2.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds2, ub', marker='^', grid=True)
        
        df_low3.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '730', y = 'stO2', ax=ax[0, 0], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '760', y = 'stO2', ax=ax[0, 1], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '780', y = 'stO2', ax=ax[1, 0], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '810', y = 'stO2', ax=ax[1, 1], label='sds3, ub', marker='x', grid=True)
        df_low3.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds3, lb', marker='x', grid=True)
        df_mid3.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds3, mid', marker='x', grid=True)
        df_up3.plot(x = '850', y = 'stO2', ax=ax[2, 0], label='sds3, ub', marker='x', grid=True)
        xtick = np.round(np.linspace(0.25, 0.55, 11), 2)
        ax[0, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='730 nm')
        ax[0, 1].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='760 nm')
        ax[1, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='780 nm')
        ax[1, 1].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='810 nm')
        ax[2, 0].set(xlabel='R', ylabel='SijvO2', xticks=xtick, title='850 nm')
        lines, labels = fig.axes[0].get_legend_handles_labels()
        ax[0, 0].get_legend().remove()
        ax[0, 1].get_legend().remove()
        ax[1, 0].get_legend().remove()
        ax[1, 1].get_legend().remove()
        ax[2, 0].get_legend().remove()
        fig.legend(lines, labels, bbox_to_anchor=(1.1, 0.5), loc = 'right')
        ax[0, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[0, 1].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[1, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[1, 1].yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[2, 0].yaxis.set_major_formatter(FuncFormatter(to_percent))
        # ax[0, 0].plot(df_low['780'], df_low['stO2'], marker='o', label='μa_skin lower bound')
        # ax[0, 0].plot(df_up['780'], df_up['stO2'], marker='^', label='μa_skin upper bound')
        # ax[0, 0].plot(df_mid['780'], df_mid['stO2'], marker='x', label='μa_skin middle')
        fig.suptitle(f'μs_muscle {pid}', size=20)
        fig.tight_layout()
        fig.savefig(os.path.join(outpath, f'{pid:05d}.png'), bbox_inches='tight')
        plt.close(fig)
# %% write video
# must check outpath
result_name = os.path.join(outpath, 'output.mp4')

frame_list = sorted(glob(os.path.join(outpath, "*.png")))
print("frame count: ",len(frame_list))

fps = 2
shape = cv2.imread(frame_list[0]).shape # delete dimension 3
size = (shape[1], shape[0])
print("frame size: ",size)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(result_name, fourcc, fps, size)

for idx, path in enumerate(frame_list):
    frame = cv2.imread(path)
    # print("\rMaking videos: {}/{}".format(idx+1, len(frame_list)), end = "")
    current_frame = idx+1
    total_frame_count = len(frame_list)
    percentage = int(current_frame*30 / (total_frame_count+1))
    print("\rProcess: [{}{}] {:06d} / {:06d}".format("#"*percentage, "."*(30-1-percentage), current_frame, total_frame_count), end ='')
    out.write(frame)

out.release()
print("\nFinish making video !!!")
# %%
fig, ax = plt.subplots(1, 1)
ax.plot(test_low['780'], test_low['stO2'], marker='o', label='μa_skin lower bound')
ax.plot(test_up['780'], test_up['stO2'], marker='^', label='μa_skin upper bound')
ax.plot(test_mid['780'], test_mid['stO2'], marker='x', label='μa_skin middle')
ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
ax.legend()
ax.set_xlabel('μa_skin')
ax.set_ylabel('SijvO2')
l1 = (test_mid['780'].values-test_low['780'].values).mean()
l2 = (test_up['780'].values-test_mid['780'].values).mean()
print(f'up/down = {l2/l1}')
print(f'skin up/down = {(0.2269-0.1367)/(0.1367-0.0465)}')
# ax.set_title('μa of skin')
fig.tight_layout()
fig.show()
# %%    linear
# df_mid = df_sds1.iloc[[i for i in range(len(df_sds1)) if i % 10 == 5]]
# df13_sds1_mid = df13_sds1.iloc[[i for i in np.arange(180, 360, 1)]]
# df13_sds1_mid = df13_sds1_mid.iloc[[i for i in np.arange(240, 300, 1)]]
df13_sds2_mid = df13_sds2.iloc[[i for i in np.arange(260, 270, 1)]]
x_train = df13_sds2_mid[['760', '780', '810', '850']]
y_train = df13_sds2_mid[['stO2']]
regr = linear_model.LinearRegression(normalize=True)
regr.fit(x_train, y_train)
y_pred = regr.predict(x_train)
# The coefficients
print("Coefficients: \n", regr.coef_, regr.intercept_)
# The mean squared error
print("Root mean squared error: %.2f" % (np.sqrt(mean_squared_error(y_train, y_pred))))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_train, y_pred))
df_sds2 = df_sds2.reset_index()
for id_sample in range(round(len(df_sds2) / 10)):
        sample_x = df_sds2.loc[[i + id_sample*10 for i in np.arange(10)], ['760', '780', '810', '850']]
        sample_y = df_sds2.loc[[i + id_sample*10 for i in np.arange(10)], ['stO2']]
        sample_pred = regr.predict(sample_x)
        if id_sample == 0:
                sample_rmse = [np.sqrt(mean_squared_error(sample_y, sample_pred))]
                sample_r2score = [r2_score(sample_y, sample_pred)]
        else:
                sample_rmse.append(np.sqrt(mean_squared_error(sample_y, sample_pred)))
                sample_r2score.append(r2_score(sample_y, sample_pred))
df_sample = pd.DataFrame({'rmse' : sample_rmse,
                          'r2score' : sample_r2score})
df_sample = df_sample.sort_values(by='rmse', ascending=True)

# plot rmse
plt.figure()
# n = plt.hist(df_sample['rmse'], bins=50)
n = plt.hist(df_sample['rmse'], bins=[i/10 for i in range(11)])
ticks = [i for i in np.arange(0., 1.01, 0.2)]
labels = ['0%', '20%', '40%', '60%', '80%', '100%']
plt.xticks(ticks, labels, fontsize=10)
plt.title('RMSE of StO2 Prediction Model with 1458 samples')
plt.xlabel('Root Mean Square Error')
plt.ylabel('Count')
plt.show()

# plot r2_score
# plt.figure()
# plt.hist(df_sample['r2score'], bins=100)
# plt.title('R Square of stO2 Prediction Model with all μa and μs Combinations')
# plt.xlabel('R Square')
# plt.ylabel('count')
# plt.show()
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0, 0].set_title('760 nm, SDS = 20.38 mm', fontsize=10)
ax[0, 0].set_xlabel('SijvO2', fontsize=10)
ax[0, 0].set_ylabel('R', fontsize=10)
for j in range(len(x2)):
        ax[0, 0].plot(y_train['stO2'], x_train['760'], marker='o', color='red', alpha=1)
        ax[0, 0].xaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[0, 0].plot(df_sds2['stO2'], df_sds2['760'], marker='.', color='blue', alpha=0.4) 
ax[0, 1].set_title('780 nm, SDS = 20.38 mm', fontsize=10)
ax[0, 1].set_xlabel('SijvO2', fontsize=10)
ax[0, 1].set_ylabel('R', fontsize=10)
for j in range(len(x2)):
        ax[0, 1].plot(y_train['stO2'], x_train['780'], marker='o', color='red', alpha=1)
        ax[0, 1].xaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[0, 1].plot(df_sds2['stO2'], df_sds2['780'], marker='.', color='blue', alpha=0.4) 
ax[1, 0].set_title('810 nm, SDS = 20.38 mm', fontsize=10)
ax[1, 0].set_xlabel('SijvO2', fontsize=10)
ax[1, 0].set_ylabel('R', fontsize=10)
for j in range(len(x2)):
        ax[1, 0].plot(y_train['stO2'], x_train['810'], marker='o', color='red', alpha=1)
        ax[1, 0].xaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[1, 0].plot(df_sds2['stO2'], df_sds2['810'], marker='.', color='blue', alpha=0.4)    
ax[1, 1].set_title('850 nm, SDS = 20.38 mm', fontsize=10)
ax[1, 1].set_xlabel('SijvO2', fontsize=10)
ax[1, 1].set_ylabel('R', fontsize=10)
for j in range(len(x2)):
        ax[1, 1].plot(y_train['stO2'], x_train['850'], marker='o', color='red', alpha=1)
        ax[1, 1].xaxis.set_major_formatter(FuncFormatter(to_percent))
        ax[1, 1].plot(df_sds2['stO2'], df_sds2['850'], marker='.', color='blue', alpha=0.4) 
fig.tight_layout()                        

# %% three sds linear
df13_sds1_mid = df13_sds1.iloc[[i for i in np.arange(260, 270, 1)]]
df13_sds2_mid = df13_sds2.iloc[[i for i in np.arange(260, 270, 1)]]
df13_sds3_mid = df13_sds3.iloc[[i for i in np.arange(260, 270, 1)]]
x_train1 = df13_sds1_mid[['760', '780', '810', '850']]
x_train2 = df13_sds2_mid[['760', '780', '810', '850']]
x_train3 = df13_sds3_mid[['760', '780', '810', '850']]
y_train = df13_sds1_mid[['stO2']]
x_train = np.concatenate([x_train1.values, x_train2.values, x_train3.values], axis=1)
regr = linear_model.LinearRegression(normalize=True)
regr.fit(x_train, y_train)
y_pred = regr.predict(x_train)
# The coefficients
print("Coefficients: \n", regr.coef_, regr.intercept_)
# The mean squared error
print("Root mean squared error: %.2f" % (np.sqrt(mean_squared_error(y_train, y_pred))))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_train, y_pred))
df_sds1 = df_sds1.reset_index()
df_sds2 = df_sds2.reset_index()
df_sds3 = df_sds3.reset_index()
for id_sample in range(len(df_sds2) // 10):
        sample_x1 = df_sds1.loc[[i + id_sample*10 for i in np.arange(10)], ['760', '780', '810', '850']]
        sample_x2 = df_sds2.loc[[i + id_sample*10 for i in np.arange(10)], ['760', '780', '810', '850']]
        sample_x3 = df_sds3.loc[[i + id_sample*10 for i in np.arange(10)], ['760', '780', '810', '850']]
        sample_x = np.concatenate([sample_x1.values,sample_x2.values,sample_x3.values], axis=1)
        sample_y = df_sds2.loc[[i + id_sample*10 for i in np.arange(10)], ['stO2']]
        sample_pred = regr.predict(sample_x)
        if id_sample == 0:
                sample_rmse = [np.sqrt(mean_squared_error(sample_y, sample_pred))]
                sample_r2score = [r2_score(sample_y, sample_pred)]
        else:
                sample_rmse.append(np.sqrt(mean_squared_error(sample_y, sample_pred)))
                sample_r2score.append(r2_score(sample_y, sample_pred))
df_sample = pd.DataFrame({'rmse' : sample_rmse,
                          'r2score' : sample_r2score})
df_sample = df_sample.sort_values(by='rmse', ascending=True)

# plot rmse
plt.figure()
# n = plt.hist(df_sample['rmse'], bins=50)
n = plt.hist(df_sample['rmse'], bins=[i/10 for i in range(11)])
ticks = [i for i in np.arange(0., 1.01, 0.2)]
labels = ['0%', '20%', '40%', '60%', '80%', '100%']
plt.xticks(ticks, labels, fontsize=10)
plt.title('RMSE of StO2 Prediction Model with 1458 samples')
plt.xlabel('Root Mean Square Error')
plt.ylabel('Count')
plt.show()

#  plot R vs. SDS lambda combionation
plt.figure(figsize=(12, 8))
N = 10
cmap = plt.get_cmap('jet',N)

for i in range(10):
        y_r = x_train[i]
        plt.plot(np.arange(0, 12), y_r, c = cmap(i), marker='o', linewidth=0.1)
norm = mpl.colors.Normalize(vmin=30,vmax=75)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ticks=np.linspace(30,75,N), 
             boundaries=np.arange(29.95,75.1,.1))
plt.xlabel('(SDS, λ)')
plt.ylabel('R')
ticks = [i for i in range(12)]
labels = ['(1, 760)','(1, 780)','(1, 810)','(1, 850)','(2, 760)','(2, 780)','(2, 810)','(2, 850)','(3, 760)','(3, 780)','(3, 810)','(3, 850)']
plt.xticks(ticks, labels, fontsize=5)
plt.title('Compare R value to StO2 in different SDS and wavelength')

# %% regression add mua, mus
df_all1 = df_all[df_all['sds'] == 1]
df_all2 = df_all[df_all['sds'] == 8]
df_all3 = df_all[df_all['sds'] == 15]
x_train1 = df_all1[['760', '780', '810', '850', 'skin', 'fat', 'muscle', 'mus_skin', 'mus_fat', 'mus_muscle']]
x_train2 = df_all2[['760', '780', '810', '850', 'skin', 'fat', 'muscle', 'mus_skin', 'mus_fat', 'mus_muscle']]
x_train3 = df_all3[['760', '780', '810', '850', 'skin', 'fat', 'muscle', 'mus_skin', 'mus_fat', 'mus_muscle']]
y_train = df_all1[['stO2']]
x_train = np.concatenate([x_train1.values, x_train2.values, x_train3.values], axis=1)
regr = linear_model.LinearRegression(normalize=True)
regr.fit(x_train, y_train)
# regr = MLPRegressor(random_state=1, max_iter=500).fit(x_train, y_train)
y_pred = regr.predict(x_train)
# The coefficients
print("Coefficients: \n", regr.coef_, regr.intercept_)
# The mean squared error
print("Root mean squared error: %.2f" % (np.sqrt(mean_squared_error(y_train, y_pred))))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_train, y_pred))

# %% Interpolation
df_error = pd.read_csv('0.csv')
df_error1 = df_error[df_error['sds'] == 1]
df_error2 = df_error[df_error['sds'] == 8]
df_error3 = df_error[df_error['sds'] == 15]
x_error1 = df_error1[['760', '780', '810', '850']]
x_error2 = df_error2[['760', '780', '810', '850']]
x_error3 = df_error3[['760', '780', '810', '850']]
x_error = np.concatenate([x_error1.values, x_error2.values, x_error3.values], axis=1)
df_all1 = df_all[df_all['sds'] == 1]
df_all2 = df_all[df_all['sds'] == 8]
df_all3 = df_all[df_all['sds'] == 15]
x_train1 = df_all1[['760', '780', '810', '850']]
x_train2 = df_all2[['760', '780', '810', '850']]
x_train3 = df_all3[['760', '780', '810', '850']]
y_train = df_all1[['stO2']]
x_train = np.concatenate([x_train1.values, x_train2.values, x_train3.values], axis=1)
y_train = y_train.values
err_tissue = pd.DataFrame({'skin': df_error['skin'], 
                        'fat': df_error['fat'],
                        'muscle': df_error['muscle'],
                        'mus_skin': df_error['mus_skin'],
                        'mus_fat': df_error['mus_fat'],
                        'mus_muscle': df_error['mus_muscle']})
err_tissueX = err_tissue.iloc[0].values
sample_tissue = df_all1[['skin', 'fat', 'muscle', 'mus_skin', 'mus_fat', 'mus_muscle']]
sample_tissue = sample_tissue.iloc[[i for i in range(len(sample_tissue)) if  i % 10 == 0]]
middle_tissue = np.array([0.038, 0.107, 0.0148, 20.6, 17.025, 6.455])
# x_pred = x_train[0:10, :]
regr_list = []
for i in np.arange(0, len(x_train), 10):
        regr = linear_model.LinearRegression(normalize=True)
        regr_list.append(regr.fit(x_train[i:i+10, :], y_train[i:i+10, :]))
sample_tissue2 = sample_tissue.copy()
tree = spatial.KDTree(sample_tissue2)
distance1, index1 = tree.query(err_tissueX)
sample_tissue2.iloc[index1] = [value for value in range(10000,10006)]
tree = spatial.KDTree(sample_tissue2)
distance2, index2 = tree.query(err_tissueX) 
y_pred = regr_list[index1].predict(x_error) * (distance2 / (distance1+distance2)) + regr_list[index2].predict(x_error) * (distance1 / (distance1+distance2))
# y_pred = regr.predict(x_train)
# # The coefficients
# print("Coefficients: \n", regr.coef_, regr.intercept_)
# # The mean squared error
# print("Root mean squared error: %.2f" % (np.sqrt(mean_squared_error(y_train, y_pred))))
# # The coefficient of determination: 1 is perfect prediction
# print("Coefficient of determination: %.2f" % r2_score(y_train, y_pred))
# rmse
print(f'rmse:  {rmse(y_train[0:10, :], y_pred)}')

# plot tissue mua mus 
plt.figure(figsize=(8, 6))
y_sample1 = sample_tissue.iloc[index1]
y_sample2 = sample_tissue.iloc[index2]
y_err = err_tissueX
y_middle = middle_tissue
y_sample1[[i for i in range(3)]] = y_sample1[[i for i in range(3)]]*100
y_sample2[[i for i in range(3)]] = y_sample2[[i for i in range(3)]]*100
y_err[[i for i in range(3)]] = y_err[[i for i in range(3)]]*100
y_middle[[i for i in range(3)]] = y_middle[[i for i in range(3)]]*100
plt.plot(np.arange(0, 6), y_sample1, marker='d', linewidth=0.1, alpha=0.3, label='model 1')
plt.plot(np.arange(0, 6), y_sample2, marker='s', linewidth=0.1, alpha=0.3, label='model 2')
plt.plot(np.arange(0, 6), y_err, marker='o', linewidth=0.1, alpha=1., label='bias 15%')
# plt.plot(np.arange(0, 6), y_middle, marker='*', linewidth=0.1, alpha=1., label='without bias')
plt.ylabel('Coefficient (1/mm)')
ticks = [i for i in range(6)]
labels = ['100*μa_skin','100*μa_fat','100*μa_muscle','μs_skin','μs_fat','μs_muscle']
plt.xticks(ticks, labels, fontsize=10)
plt.title('μa, μs of Interpolate Point and Two Prediction Model')
plt.legend()

#  plot R vs. SDS lambda combionation
plt.figure(figsize=(12, 8))
N = 10
cmap = plt.get_cmap('jet',N)

for i in range(10):
        y_r = x_train[i+index1*10, :]
        y_r2 = x_train[i+index2*10, :]
        y_err = x_error[i, :]
        plt.plot(np.arange(0, 12), y_r, c = cmap(i), marker='d', linewidth=0.1, alpha=0.3)
        plt.plot(np.arange(0, 12), y_r2, c = cmap(i), marker='x', linewidth=0.1, alpha=0.3)
        plt.plot(np.arange(0, 12), y_err, c = cmap(i), marker='o', linewidth=0.1, alpha=1.)
norm = mpl.colors.Normalize(vmin=30,vmax=75)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ticks=np.linspace(30,75,N), 
             boundaries=np.arange(29.95,75.1,.1))
plt.xlabel('(SDS, λ)')
plt.ylabel('R')
ticks = [i for i in range(12)]
labels = ['(1, 760)','(1, 780)','(1, 810)','(1, 850)','(2, 760)','(2, 780)','(2, 810)','(2, 850)','(3, 760)','(3, 780)','(3, 810)','(3, 850)']
plt.xticks(ticks, labels, fontsize=7)
plt.title('Interpolate Point and Two Prediction Model')
legend_elements = [Line2D([0], [0], marker='d', color='black', label='model 1'),
                   Line2D([0], [0], marker='x', color='black', label='model 2'),
                   Line2D([0], [0], marker='o', color='black', label='bias 15%')]
plt.legend(handles=legend_elements, loc='upper right')

# %% Interpolation new
df_error = pd.read_csv('0.csv')
df_err = pd.DataFrame({'730': df_error['small_730'] / df_error['large_730'],
        '760': df_error['small_760'] / df_error['large_760'],
        '780': df_error['small_780'] / df_error['large_780'],
        '810': df_error['small_810'] / df_error['large_810'],
        '850': df_error['small_850'] / df_error['large_850'],
        'stO2': df_error['stO2'],
        'sds': df_error['sds']})
df_err1 = df_err[df_err['sds'] == 1]
df_err2 = df_err[df_err['sds'] == 8]
df_err3 = df_err[df_err['sds'] == 15]
err_tissue = pd.DataFrame({'skin': df_error['skin'], 
                        'fat': df_error['fat'],
                        'muscle': df_error['muscle'],
                        'mus_skin': df_error['mus_skin'],
                        'mus_fat': df_error['mus_fat'],
                        'mus_muscle': df_error['mus_muscle']})
err_tissueX = err_tissue.iloc[0].values
df_all1 = df_all[df_all['sds'] == 1]
df_all2 = df_all[df_all['sds'] == 8]
df_all3 = df_all[df_all['sds'] == 15]
sample_tissue = df_all1[['skin', 'fat', 'muscle', 'mus_skin', 'mus_fat', 'mus_muscle']]
sample_tissue = sample_tissue.iloc[[i for i in range(len(sample_tissue)) if  i % 10 == 0]]
x_error1 = df_err1[['730', '760', '780', '810', '850']]
x_error2 = df_err2[['730', '760', '780', '810', '850']]
x_error3 = df_err3[['730', '760', '780', '810', '850']]
x_error = np.concatenate([x_error1.values, x_error2.values, x_error3.values], axis=1)
# x_error /= x_error.mean(axis=1)[:,None]
x_train1 = df_new1[['730', '760', '780', '810', '850']]
x_train2 = df_new2[['730', '760', '780', '810', '850']]
x_train3 = df_new3[['730', '760', '780', '810', '850']]
y_train = df_new1[['stO2']]
x_train = np.concatenate([x_train1.values, x_train2.values, x_train3.values], axis=1)
# x_train /= x_train.mean(axis=1)[:,None]
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_error = scaler.transform(x_error)
y_train = y_train.values
# x_pred = x_train[0:10, :]

regr_list = []
for i in np.arange(0, len(x_train), 10):
        regr = linear_model.LinearRegression(normalize=True)
        regr_list.append(regr.fit(x_train[i:i+10, :], y_train[i:i+10, :]))
# regr = linear_model.LinearRegression(normalize=False)
# regr_list = [regr.fit(x_train[i:i+10, :], y_train[i:i+10, :]) for i in np.arange(0, len(x_train), 10)]
# interp_values = [regr_list[i].predict(x_error)[0] for i in np.arange(0, len(regr_list))]
interp_values = [regr_list[i].predict(x_train)[2] for i in np.arange(0, len(regr_list))]
interp_values = np.array(interp_values).reshape(3, 3, 3, 3, 3, 6)
grid_x = np.array([0.011, 0.0148, 0.0186, 0.0224, 0.0262, 0.03 ])
grid_y = np.array([0.1015 , 0.10425, 0.107 ])
grid_z = np.array([0.038, 0.1115, 0.185])
grid_u = np.array([5.01, 6.455, 7.9])
grid_v = np.array([11.89, 17.025, 22.16])
grid_w = np.array([15.89, 20.6, 25.31])
# interp_points = (grid_x, grid_y, grid_z, grid_u, grid_v, grid_w)
interp_points = (grid_w, grid_v, grid_u, grid_z, grid_y, grid_x)
# err_point = np.array([err_tissueX[2], err_tissueX[1], err_tissueX[0], err_tissueX[5], err_tissueX[4], err_tissueX[3]])
# err_point = np.array([err_tissueX[3], err_tissueX[4], err_tissueX[5], err_tissueX[0],err_tissueX[1], err_tissueX[2]])
# print(err_point)
# err_point = np.array([0.011, 0.107, 0.038, 5.01, 11.89, 15.89])
err_point = np.array([15.89, 11.89, 5.01, 0.038, 0.107, 0.011])
print(err_point)
# err_point[1] = 0.11
print(to_percent(interpn(interp_points, interp_values, err_point, bounds_error=True), None))
# %%

# muscle -> fat -> skin -> mus_muscle -> mus_fat -> mus_skin
# 'skin': array([0.0465, 0.1367, 0.2269]),
#  'fat': array([0.1098 , 0.11555, 0.1213 ]),
#  'muscle': array([0.013  , 0.01916, 0.02532, 0.03148, 0.03764, 0.0438 ])
#  mus_skin	15.89,	20.6,	25.31,
# mus_fat	11.89,	17.025,	22.16,
# mus_muscle	5.01	6.455	7.9
# 'skin': array([0.038, 0.1115, 0.185]),
# ?'fat': array([0.1015 , 0.10425, 0.107 ]),
# ?'muscle': array([0.011 ?, 0.0148, 0.0186, 0.0224, 0.0262, 0.03 ])

# for i in np.arange(0, len(x_train), 10):
#         regr = linear_model.LinearRegression(normalize=False)
#         regr_list.append(regr.fit(x_train[i:i+10, :], y_train[i:i+10, :]))
# sample_tissue2 = sample_tissue.copy()
# tree = spatial.KDTree(sample_tissue2)
# distance1, index1 = tree.query(err_tissueX)
# sample_tissue2.iloc[index1] = [value for value in range(10000,10006)]
# tree = spatial.KDTree(sample_tissue2)
# distance2, index2 = tree.query(err_tissueX) 
# y_pred = regr_list[index1].predict(x_error) * (distance2 / (distance1+distance2)) + regr_list[index2].predict(x_error) * (distance1 / (distance1+distance2))

# y_pred = regr.predict(x_train)
# # The coefficients
# print("Coefficients: \n", regr.coef_, regr.intercept_)
# # The mean squared error
# print("Root mean squared error: %.2f" % (np.sqrt(mean_squared_error(y_train, y_pred))))
# # The coefficient of determination: 1 is perfect prediction
# print("Coefficient of determination: %.2f" % r2_score(y_train, y_pred))
# rmse
# print(f'rmse:  {rmse(y_train[0:10, :], y_pred)}')

# plot tissue mua mus 
plt.figure(figsize=(8, 6))
# y_sample1 = sample_tissue.iloc[index1]
# y_sample2 = sample_tissue.iloc[index2]
y_err = err_tissueX
# y_sample1[[i for i in range(3)]] = y_sample1[[i for i in range(3)]]*100
# y_sample2[[i for i in range(3)]] = y_sample2[[i for i in range(3)]]*100
y_err[[i for i in range(3)]] = y_err[[i for i in range(3)]]*100
# plt.plot(np.arange(0, 6), y_sample1, marker='d', linewidth=0.1, alpha=0.3, label='model 1')
# plt.plot(np.arange(0, 6), y_sample2, marker='s', linewidth=0.1, alpha=0.3, label='model 2')
plt.scatter(np.arange(0, 6), y_err, marker='o', linewidth=0.1, alpha=1., label='bias 15%')
plt.ylabel('Coefficient (1/mm)')
ticks = [i for i in range(6)]
labels = ['μa_skin','μa_fat','μa_muscle','μs_skin','μs_fat','μs_muscle']
plt.xticks(ticks, labels, fontsize=10)
plt.title('μa, μs of Interpolate Point and Two Prediction Model')
plt.legend()

#  plot R vs. SDS lambda combionation
plt.figure(figsize=(12, 8))
N = 10
cmap = plt.get_cmap('jet',N)

for i in range(10):
        # y_r = x_train[i+index1*10, :]
        # y_r2 = x_train[i+index2*10, :]
        y_upper = x_train[i+0, :]
        y_lower = x_train[i+14570, :]
        y_err = x_error[i, :]
        plt.plot(np.arange(0, 15), y_upper, c = cmap(i), marker='d', linewidth=0.1, alpha=0.3)
        plt.plot(np.arange(0, 15), y_lower, c = cmap(i), marker='x', linewidth=0.1, alpha=0.3)
        plt.plot(np.arange(0, 15), y_err, c = cmap(i), marker='o', linewidth=0.1, alpha=1.)
norm = mpl.colors.Normalize(vmin=30,vmax=75)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ticks=np.linspace(30,75,N), 
             boundaries=np.arange(29.95,75.1,.1))
plt.xlabel('(SDS, λ)')
plt.ylabel('R')
ticks = [i for i in range(15)]
labels = ['(1, 730)','(1, 760)','(1, 780)','(1, 810)','(1, 850)','(2, 730)','(2, 760)','(2, 780)','(2, 810)','(2, 850)','(3, 730)','(3, 760)','(3, 780)','(3, 810)','(3, 850)']
plt.xticks(ticks, labels, fontsize=7)
plt.title('Interpolate Point and Two Prediction Model')
legend_elements = [Line2D([0], [0], marker='d', color='black', label='upper bound'),
                   Line2D([0], [0], marker='x', color='black', label='lower bound'),
                   Line2D([0], [0], marker='o', color='black', label='bias 15%')]
plt.legend(handles=legend_elements, loc='upper right')

# %% polynomial
df13_sds2_mid = df13_sds2.iloc[[i for i in np.arange(260, 270, 1)]]
x_train = df13_sds2_mid[['760', '780', '810', '850']]
y_train = df13_sds2_mid[['stO2']]
model = Pipeline([('poly', PolynomialFeatures(degree=4)),
                ('linear', linear_model.LinearRegression(normalize=True))])
model = model.fit(x_train, y_train)
y_pred = model.predict(x_train)
# The coefficients
print("Coefficients: \n", model.named_steps['linear'].coef_)
# The mean squared error
print("Root mean squared error: %.2f" % (np.sqrt(mean_squared_error(y_train, y_pred))))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_train, y_pred))


df_sds2 = df_sds2.reset_index()
for id_sample in range(round(len(df_sds2) / 10)):
        sample_x = df_sds2.loc[[i + id_sample*10 for i in np.arange(10)], ['760', '780', '810', '850']]
        sample_y = df_sds2.loc[[i + id_sample*10 for i in np.arange(10)], ['stO2']]
        sample_pred = model.predict(sample_x)
        if id_sample == 0:
                sample_rmse = [np.sqrt(mean_squared_error(sample_y, sample_pred))]
                sample_r2score = [r2_score(sample_y, sample_pred)]
        else:
                sample_rmse.append(np.sqrt(mean_squared_error(sample_y, sample_pred)))
                sample_r2score.append(r2_score(sample_y, sample_pred))
df_sample = pd.DataFrame({'rmse' : sample_rmse,
                          'r2score' : sample_r2score})
df_sample = df_sample.sort_values(by='rmse', ascending=True)

# %% MLP regression
df13_sds2_mid = df13_sds2.iloc[[i for i in np.arange(260, 270, 1)]]
x_train = df13_sds2_mid[['760', '780', '810', '850']]
y_train = df13_sds2_mid[['stO2']]
regr = MLPRegressor(random_state=1, max_iter=500).fit(x_train, y_train)
regr.predict(x_train)
y_pred = regr.predict(x_train)
# The mean squared error
print("Root mean squared error: %.2f" % (np.sqrt(mean_squared_error(y_train, y_pred))))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_train, y_pred))


df_sds2 = df_sds2.reset_index()
for id_sample in range(round(len(df_sds2) / 10)):
        sample_x = df_sds2.loc[[i + id_sample*10 for i in np.arange(10)], ['760', '780', '810', '850']]
        sample_y = df_sds2.loc[[i + id_sample*10 for i in np.arange(10)], ['stO2']]
        sample_pred = regr.predict(sample_x)
        if id_sample == 0:
                sample_rmse = [np.sqrt(mean_squared_error(sample_y, sample_pred))]
                sample_r2score = [r2_score(sample_y, sample_pred)]
        else:
                sample_rmse.append(np.sqrt(mean_squared_error(sample_y, sample_pred)))
                sample_r2score.append(r2_score(sample_y, sample_pred))
df_sample = pd.DataFrame({'rmse' : sample_rmse,
                          'r2score' : sample_r2score})
df_sample = df_sample.sort_values(by='rmse', ascending=True)


# plot rmse
plt.figure()
n = plt.hist(df_sample['rmse'], bins=[i/10 for i in range(11)])
plt.title('RMSE of stO2 Prediction Model with all μa and μs Combinations')
plt.xlabel('Root Mean Square Error')
plt.ylabel('count')
plt.show()

# %% plot reflectance vs. SDS lambda combionation
df13_ref1 = df13_sds1.iloc[[i for i in np.arange(260, 270, 1)]]
df13_ref2 = df13_sds2.iloc[[i for i in np.arange(260, 270, 1)]]
df13_ref3 = df13_sds3.iloc[[i for i in np.arange(260, 270, 1)]]
df13_ref1_small = df13_ref1[['small_730', 'small_760', 'small_780', 'small_810','small_850', 'stO2']]
df13_ref2_small = df13_ref2[['small_730', 'small_760', 'small_780', 'small_810','small_850', 'stO2']]
df13_ref3_small = df13_ref3[['small_730', 'small_760', 'small_780', 'small_810','small_850', 'stO2']]
df13_ref1_big = df13_ref1[['large_730', 'large_760', 'large_780', 'large_810','large_850', 'stO2']]
df13_ref2_big = df13_ref2[['large_730', 'large_760', 'large_780', 'large_810','large_850', 'stO2']]
df13_ref3_big = df13_ref3[['large_730', 'large_760', 'large_780', 'large_810','large_850', 'stO2']]
plt.figure(figsize=(12, 8))
N = 10
cmap = plt.get_cmap('jet',N)

for i , j in zip(range(260, 270), range(10)):
        y_small = np.array([df13_ref1_small.loc[[i],['small_730', 'small_760', 'small_780', 'small_810','small_850']].values[0], 
                      df13_ref2_small.loc[[i+540],['small_730', 'small_760', 'small_780', 'small_810','small_850']].values[0], 
                      df13_ref3_small.loc[[i+540*2],['small_730', 'small_760', 'small_780', 'small_810','small_850']].values[0]]).reshape(15)
        y_big = np.array([df13_ref1_big.loc[[i],['large_730', 'large_760', 'large_780', 'large_810','large_850']].values[0], 
                      df13_ref2_big.loc[[i+540],['large_730', 'large_760', 'large_780', 'large_810','large_850']].values[0], 
                      df13_ref3_big.loc[[i+540*2],['large_730', 'large_760', 'large_780', 'large_810','large_850']].values[0]]).reshape(15)
        plt.plot(np.arange(0, 15), y_small, c = cmap(j), marker='o', linewidth=0.1)
        plt.plot(np.arange(0, 15), y_big, c = cmap(j), marker='^', linewidth=0.1)
norm = mpl.colors.Normalize(vmin=30,vmax=75)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ticks=np.linspace(30,75,N), 
             boundaries=np.arange(29.95,75.1,.1))
plt.xlabel('(SDS, λ)')
plt.ylabel('Reflectance')
ticks = [i for i in range(15)]
labels = ['(1, 730)', '(1, 760)','(1, 780)','(1, 810)','(1, 850)','(2, 730)','(2, 760)','(2, 780)','(2, 810)','(2, 850)','(3, 730)','(3, 760)','(3, 780)','(3, 810)','(3, 850)']
plt.xticks(ticks, labels, fontsize=5)
legend_elements = [Line2D([0], [0], marker='o', color='black', label='small diameter IJV'),
                   Line2D([0], [0], marker='^', color='black', label='large diameter IJV')]
plt.legend(handles=legend_elements, loc='upper right')


# %%
# plot R vs. stO2
# sds 1
fig = plt.figure(figsize=(12, 8))
fig.subplots_adjust(wspace=0.2, hspace=0.3)
for i in range(4):
        ax = fig.add_subplot(2, 2, i+1)
        if i == 0:
                ax.set_title('760 nm, SDS = 15.235 mm', fontsize=10)
                ax.set_xlabel('StO2', fontsize=10)
                ax.set_ylabel('R', fontsize=10)
                for j in range(len(x1)):
                        ax.scatter(x1[j]['stO2'], x1[j]['760'], marker='.', color='red') 
        elif i == 1:
                ax.set_title('780 nm, SDS = 15.235 mm', fontsize=10)
                ax.set_xlabel('StO2', fontsize=10)
                ax.set_ylabel('R', fontsize=10)
                for j in range(len(x1)):
                        ax.scatter(x1[j]['stO2'], x1[j]['780'], marker='.', color='red')
        elif i == 2:
                ax.set_title('810 nm, SDS = 15.235 mm', fontsize=10)
                ax.set_xlabel('StO2', fontsize=10)
                ax.set_ylabel('R', fontsize=10)
                for j in range(len(x1)):
                        ax.scatter(x1[j]['stO2'], x1[j]['810'], marker='.', color='red')    
        else:
                ax.set_title('850 nm, SDS = 15.235 mm', fontsize=10)
                ax.set_xlabel('StO2', fontsize=10)
                ax.set_ylabel('R', fontsize=10)
                for j in range(len(x1)):
                        ax.scatter(x1[j]['stO2'], x1[j]['850'], marker='.', color='red')

# sds 2
fig = plt.figure(figsize=(12, 8))
fig.subplots_adjust(wspace=0.2, hspace=0.3)

for i in range(4):
        ax = fig.add_subplot(2, 2, i+1)
        if i == 0:
                ax.set_title('760 nm, SDS = 20.38 mm', fontsize=10)
                ax.set_xlabel('StO2', fontsize=10)
                ax.set_ylabel('R', fontsize=10)
                for j in range(len(x2)):
                        ax.scatter(x2[j]['stO2'], x2[j]['760'], marker='.', color='red') 
        elif i == 1:
                ax.set_title('780 nm, SDS = 20.38 mm', fontsize=10)
                ax.set_xlabel('StO2', fontsize=10)
                ax.set_ylabel('R', fontsize=10)
                for j in range(len(x2)):
                        ax.scatter(x2[j]['stO2'], x2[j]['780'], marker='.', color='red')
        elif i == 2:
                ax.set_title('810 nm, SDS = 20.38 mm', fontsize=10)
                ax.set_xlabel('StO2', fontsize=10)
                ax.set_ylabel('R', fontsize=10)
                for j in range(len(x2)):
                        ax.scatter(x2[j]['stO2'], x2[j]['810'], marker='.', color='red')    
        else:
                ax.set_title('850 nm, SDS = 20.38 mm', fontsize=10)
                ax.set_xlabel('StO2', fontsize=10)
                ax.set_ylabel('R', fontsize=10)
                for j in range(len(x2)):
                        ax.scatter(x2[j]['stO2'], x2[j]['850'], marker='.', color='red')

# sds 3
fig = plt.figure(figsize=(12, 8))
fig.subplots_adjust(wspace=0.2, hspace=0.3)
for i in range(4):
        ax = fig.add_subplot(2, 2, i+1)
        if i == 0:
                ax.set_title('760 nm, SDS = 25.525 mm', fontsize=10)
                ax.set_xlabel('StO2', fontsize=10)
                ax.set_ylabel('R', fontsize=10)
                for j in range(len(x3)):
                        ax.scatter(x3[j]['stO2'], x3[j]['760'], marker='.', color='red') 
        elif i == 1:
                ax.set_title('780 nm, SDS = 25.525 mm', fontsize=10)
                ax.set_xlabel('StO2', fontsize=10)
                ax.set_ylabel('R', fontsize=10)
                for j in range(len(x3)):
                        ax.scatter(x3[j]['stO2'], x3[j]['780'], marker='.', color='red')
        elif i == 2:
                ax.set_title('810 nm, SDS = 25.525 mm', fontsize=10)
                ax.set_xlabel('StO2', fontsize=10)
                ax.set_ylabel('R', fontsize=10)
                for j in range(len(x3)):
                        ax.scatter(x3[j]['stO2'], x3[j]['810'], marker='.', color='red')    
        else:
                ax.set_title('850 nm, SDS = 25.525 mm', fontsize=10)
                ax.set_xlabel('StO2', fontsize=10)
                ax.set_ylabel('R', fontsize=10)
                for j in range(len(x3)):
                        ax.scatter(x3[j]['stO2'], x3[j]['850'], marker='.', color='red')

# %%                
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
        
# %%PCA
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

