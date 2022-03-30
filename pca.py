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
        'sds': df_all['sds']})
df_new1 = df_new[df_new['sds'] == 1]
df_new2 = df_new[df_new['sds'] == 8]
df_new3 = df_new[df_new['sds'] == 15]

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
df_sds1_ = pd.concat([df_sds1, df_sds1.describe()], axis=0)
df_sds2_ = pd.concat([df_sds2, df_sds2.describe()], axis=0)
df_sds3_ = pd.concat([df_sds3, df_sds3.describe()], axis=0)
df_sds1_.to_csv(f'{PATH}_sds1.csv')
df_sds2_.to_csv(f'{PATH}_sds2.csv')
df_sds3_.to_csv(f'{PATH}_sds3.csv')


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
fig = plt.figure(figsize=(12, 8))
fig.subplots_adjust(wspace=0.2, hspace=0.3)
for i in range(4):
        ax = fig.add_subplot(2, 2, i+1)
        if i == 0:
                ax.set_title('760 nm, SDS = 20.38 mm', fontsize=10)
                ax.set_xlabel('SijvO2', fontsize=10)
                ax.set_ylabel('R', fontsize=10)
                for j in range(len(x2)):
                        ax.plot(y_train['stO2'], x_train['760'], marker='o', color='red')
                        ax.xaxis.set_major_formatter(FuncFormatter(to_percent))
                        ax.scatter(df_sds2['stO2'], df_sds2['760'], marker='.', color='blue') 
        elif i == 1:
                ax.set_title('780 nm, SDS = 20.38 mm', fontsize=10)
                ax.set_xlabel('SijvO2', fontsize=10)
                ax.set_ylabel('R', fontsize=10)
                for j in range(len(x2)):
                        ax.plot(y_train['stO2'], x_train['780'], marker='o', color='red')
                        ax.xaxis.set_major_formatter(FuncFormatter(to_percent))
                        ax.scatter(df_sds2['stO2'], df_sds2['780'], marker='.', color='blue') 
        elif i == 2:
                ax.set_title('810 nm, SDS = 20.38 mm', fontsize=10)
                ax.set_xlabel('SijvO2', fontsize=10)
                ax.set_ylabel('R', fontsize=10)
                for j in range(len(x2)):
                        ax.plot(y_train['stO2'], x_train['810'], marker='o', color='red')
                        ax.xaxis.set_major_formatter(FuncFormatter(to_percent))
                        ax.scatter(df_sds2['stO2'], df_sds2['810'], marker='.', color='blue')    
        else:
                ax.set_title('850 nm, SDS = 20.38 mm', fontsize=10)
                ax.set_xlabel('SijvO2', fontsize=10)
                ax.set_ylabel('R', fontsize=10)
                for j in range(len(x2)):
                        ax.plot(y_train['stO2'], x_train['850'], marker='o', color='red')
                        ax.xaxis.set_major_formatter(FuncFormatter(to_percent))
                        ax.scatter(df_sds2['stO2'], df_sds2['850'], marker='.', color='blue') 

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
for id_sample in range(round(len(df_sds2) / 10)):
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
plt.plot(np.arange(0, 6), y_sample1, marker='d', linewidth=0.1, alpha=0.3, label='model 1')
plt.plot(np.arange(0, 6), y_sample2, marker='s', linewidth=0.1, alpha=0.3, label='model 2')
plt.plot(np.arange(0, 6), y_err, marker='o', linewidth=0.1, alpha=1., label='bias 15%')
plt.plot(np.arange(0, 6), y_middle, marker='*', linewidth=0.1, alpha=1., label='without bias')
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
x_error /= x_error.mean(axis=1)[:,None]
x_train1 = df_new1[['730', '760', '780', '810', '850']]
x_train2 = df_new2[['730', '760', '780', '810', '850']]
x_train3 = df_new3[['730', '760', '780', '810', '850']]
y_train = df_new1[['stO2']]
x_train = np.concatenate([x_train1.values, x_train2.values, x_train3.values], axis=1)
x_train /= x_train.mean(axis=1)[:,None]
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_error = scaler.transform(x_error)
y_train = y_train.values
# x_pred = x_train[0:10, :]
regr_list = []
for i in np.arange(0, len(x_train), 10):
        regr = linear_model.LinearRegression(normalize=False)
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
plt.plot(np.arange(0, 6), y_sample1, marker='d', linewidth=0.1, alpha=0.3, label='model 1')
plt.plot(np.arange(0, 6), y_sample2, marker='s', linewidth=0.1, alpha=0.3, label='model 2')
plt.plot(np.arange(0, 6), y_err, marker='o', linewidth=0.1, alpha=1., label='bias 15%')
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
        y_r = x_train[i+index1*10, :]
        y_r2 = x_train[i+index2*10, :]
        y_err = x_error[i, :]
        plt.plot(np.arange(0, 15), y_r, c = cmap(i), marker='d', linewidth=0.1, alpha=0.3)
        plt.plot(np.arange(0, 15), y_r2, c = cmap(i), marker='x', linewidth=0.1, alpha=0.3)
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
legend_elements = [Line2D([0], [0], marker='d', color='black', label='model 1'),
                   Line2D([0], [0], marker='x', color='black', label='model 2'),
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

