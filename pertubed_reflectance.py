#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 8 21:18:25 2022

@author: leewei
"""
# %%
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    print("can't open ipython !")
import numpy as np
import matplotlib.pyplot as plt
import jdata as jd
import postprocess
import os
import pandas as pd
from glob import glob
from itertools import product

# try:
#     jd_path = "mcx/example/quicktest/small_ijv_730_0_0_detp.jdat"
#     jdata = jd.load(jd_path)
# except:
#     print('jdata not founded !')
data_path = 'training_data'
wl_folder = glob(os.path.join(data_path, '*nm'))
mus_set = 27

# load mua set
mua_skin = pd.read_csv('skin.csv')
mua_fat = pd.read_csv('fat.csv')
mua_muscle = pd.read_csv('muscle.csv')
wl_list = [730, 760, 780, 810, 850]

# process skin
arr_temp0 = np.interp(wl_list, mua_skin['wavelength'], mua_skin['mua1'])
arr_temp1 = np.interp(wl_list, mua_skin['wavelength'], mua_skin['mua2'])
data = {'wavelength' : wl_list, 
        'mua1' : arr_temp0,
        'mua2' : arr_temp1}
mua_skin2 = pd.DataFrame(data)

# process fat
arr_temp0 = np.interp(wl_list, mua_fat['wavelength'], mua_fat['mua1'])
arr_temp1 = np.interp(wl_list, mua_fat['wavelength'], mua_fat['mua2'])
data = {'wavelength' : wl_list, 
        'mua1' : arr_temp0,
        'mua2' : arr_temp1}
mua_fat2 = pd.DataFrame(data)

# process muscle
arr_temp0 = np.interp(wl_list, mua_muscle['wavelength'], mua_muscle['mua1'])
arr_temp1 = np.interp(wl_list, mua_muscle['wavelength'], mua_muscle['mua2'])
data = {'wavelength' : wl_list, 
        'mua1' : arr_temp0,
        'mua2' : arr_temp1}
mua_muscle2 = pd.DataFrame(data)

# %%
for num_mus in range(mus_set):
    for id, wl_path in enumerate(wl_folder):
        wl = wl_path.split('\\')[1]
        wl = wl.split('nm')[0]
        sessionID = 'small_ijv_' + wl + f'_{num_mus}'

        # define mua change
        mua_change = [[mua_skin2.at[id, 'mua1'], (mua_skin2.at[id, 'mua2'] + mua_skin2.at[id, 'mua1'])/2, mua_skin2.at[id, 'mua2']],
                    [mua_fat2.at[id, 'mua1'], (mua_fat2.at[id, 'mua2'] + mua_fat2.at[id, 'mua1'])/2, mua_fat2.at[id, 'mua2']],
                    [mua_muscle2.at[id, 'mua1'], (mua_muscle2.at[id, 'mua2'] + mua_muscle2.at[id, 'mua1'])/2, mua_muscle2.at[id, 'mua2']]]
        mua_change2 = np.array( product(mua_change))
# %%
        
        mua_set = 
        # choose R ratio baseline
        if id == 0:
            
        else:

sessionID = 'small_ijv_730_0'
mua =   [0,         #1: Air
        1e4,        #2: PLA
        0,          #3: Prism
        0.1232,     #4: Skin
        0.06515,    #5: Fat
        0.0293,     #6: Muscle
        0.0293,     #7: Muscle or IJV (Perturbed Region)
        0.49335,    #8: IJV
        0.44465]    #9: CCA

# %%

# %%
movingAverageFinalReflectanceMean = postprocess.getMovingAverageReflectance(sessionID, mua)
small_reflectance = movingAverageFinalReflectanceMean[8]    # sds9 20.38mm
# %%
movingAverageMeanPathlength = postprocess.getMeanPathlength(sessionID, mua)[1].mean(axis=0)
purturbed_pathlength = movingAverageMeanPathlength[8, 6]   # sds9 20.38mm
# %%
movingAverageNumofScatter = postprocess.getNumofScatter(sessionID, mua)[1].mean(axis=0)
purturbed_num_scatter = movingAverageNumofScatter[8, 6]    # sds9 20.38mm
# %%
model_param = jd.load(os.path.join(sessionID, 'model_parameters.json'))
perturbed_region_mus = model_param['OptParam']['IJV']['mus']
perturbed_region_mua = mua[6]
perturbed_region_mut = perturbed_region_mua + perturbed_region_mus
ijv_mus = model_param['OptParam']['IJV']['mus']
ijv_mua = mua[7]
ijv_mut = ijv_mua + ijv_mus
# #########WAIT CONFIRM###########
# perturbed_coef = ((perturbed_region_mus / perturbed_region_mut) / (ijv_mus / ijv_mut))**purturbed_num_scatter\
#                     * (perturbed_region_mut / ijv_mut)**purturbed_num_scatter * np.exp(-(perturbed_region_mut - ijv_mut) * purturbed_pathlength)
# big_reflectance = small_reflectance * perturbed_coef
# ################################
big_reflectance = small_reflectance * 0.9

R_ratio = big_reflectance / small_reflectance

# %%
print(list(product([1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 1, 1, 1])))