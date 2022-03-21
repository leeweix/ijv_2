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
import sys
import numpy as np
import matplotlib.pyplot as plt
import jdata as jd
import json
import postprocess
import os
from tqdm import tqdm
import pandas as pd
from glob import glob
from itertools import product
from datetime import datetime

def make_ijv_mua(id_wl, epsilon, stO2):
        return list(2.303 * (epsilon.at[id_wl, 'HbO2']/64532*stO2 + epsilon.at[id_wl, 'Hb']/64500*(1.-stO2)) * 150)


# user setting
MODE_PERTURB = 1
data_path = 'training_data'
now = datetime.now()
timestr =  now.strftime('%Y-%m%d-%H-%M-%S')
output_path = f'R_ratio_{timestr}'
if os.path.exists(output_path):
        print('output folder already exist!')
else:
        os.mkdir(output_path)
wl_folder = glob(os.path.join(data_path, '*nm'))
NUM_MUS = 1
NUM_MUA = 1
wl_list = [730, 760, 780, 810, 850]
sds_choose = [7, 9, 11]
sds_choose = np.array(sds_choose)-1
stO2 = np.array([0.3, 0.4, 0.5, 0.6, 0.7])

# load mua set
mua_skin = pd.read_csv('skin.csv')
mua_fat = pd.read_csv('fat.csv')
mua_muscle = pd.read_csv('muscle.csv')
epsilon = pd.read_csv('epsilon.csv')

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
for id_mus in range(NUM_MUS):
        small_reflectance = np.zeros((len(wl_list), NUM_MUA * len(stO2), len(sds_choose)))
        big_reflectance = np.zeros((len(wl_list), NUM_MUA * len(stO2), len(sds_choose)))
        for id_wl, wl_path in enumerate(wl_folder):
                wl = wl_path.split('\\')[1]
                wl = wl.split('nm')[0]
                id_mus = 13
                sessionID = 'small_ijv_' + wl + f'_{id_mus}'
                sessionID2 = 'large_ijv_' + wl + f'_{id_mus}'
                mua_ijv = make_ijv_mua(id_wl, epsilon, stO2)
                
                # define mua change
                # mua_change = {'skin' : [mua_skin2.at[id_wl, 'mua1'], (mua_skin2.at[id_wl, 'mua2'] + mua_skin2.at[id_wl, 'mua1'])/2, mua_skin2.at[id_wl, 'mua2']],
                #         'fat' : [mua_fat2.at[id_wl, 'mua1'], (mua_fat2.at[id_wl, 'mua2'] + mua_fat2.at[id_wl, 'mua1'])/2, mua_fat2.at[id_wl, 'mua2']],
                #         'muscle' : [mua_muscle2.at[id_wl, 'mua1'], (mua_muscle2.at[id_wl, 'mua2'] + mua_muscle2.at[id_wl, 'mua1'])/2, mua_muscle2.at[id_wl, 'mua2']],
                #         'ijv' : mua_ijv}
                mua_change = {'skin' : [mua_skin2.at[id_wl, 'mua1']],
                        'fat' : [mua_fat2.at[id_wl, 'mua1']],
                        'muscle' : [mua_muscle2.at[id_wl, 'mua1']],
                        'ijv' : mua_ijv}
                mua_change2 = np.array(list(product(mua_change['skin'], mua_change['fat'], mua_change['muscle'], mua_change['ijv'])))
                mua_fix = {'air' : 0,
                        'pla' : 1e4,
                        'prism' : 0,
                        'cca' : 0.44465}    
                mua_fix2 = np.array(list(mua_fix.values()) * mua_change2.shape[0]).reshape(mua_change2.shape[0], len(mua_fix))
                mua_all = np.concatenate((mua_fix2[:, 0:3], mua_change2[:, 0:3], mua_change2[:, 2:4], mua_fix2[:, 3:4]), axis=1)
        # ##############coding###################
                for id_mua, mua in enumerate(tqdm(mua_all)):
                        # get WMC reflectance, pathlength, collision times, wait to improve!
                        if MODE_PERTURB == 1:
                                if id_mua == 0:
                                        # movingAverageFinalReflectanceMean
                                        with open(os.path.join(wl_path, sessionID, "config.json")) as f:
                                                config = json.load(f)  # about detector na, & photon number
                                        with open(os.path.join(wl_path, sessionID, "model_parameters.json")) as f:
                                                modelParameters = json.load(f)  # about index of materials & fiber number
                                        fiberSet = modelParameters["HardwareParam"]["Detector"]["Fiber"]
                                        detOutputPathSet = glob(os.path.join(wl_path, sessionID, "mcx_output", "*.jdat"))  # about paths of detected photon data
                                        innerIndex = modelParameters["OptParam"]["Prism"]["n"]
                                        outerIndex = modelParameters["OptParam"]["Prism"]["n"]
                                        detectorNA = config["DetectorNA"]
                                        detectorNum = len(fiberSet)*3*2
                                        photonNum = config["PhotonNum"]
                                        # getReflectance
                                        reflectance = np.empty((len(detOutputPathSet), detectorNum))
                                        meanPathlength = np.empty((len(detOutputPathSet), detectorNum, len(mua)))
                                        NumofScatter = np.empty((len(detOutputPathSet), detectorNum, len(mua)))
                                        for detOutputIdx, detOutputPath in enumerate(detOutputPathSet):
                                                # read detected data
                                                detOutput = jd.load(detOutputPath)
                                                info = detOutput["MCXData"]["Info"]
                                                photonData = detOutput["MCXData"]["PhotonData"]
                                                
                                                # unit conversion for photon pathlength
                                                photonData["ppath"] = photonData["ppath"] * info["LengthUnit"]
                                                
                                                # retrieve valid detector ID and valid ppath
                                                critAng = np.arcsin(detectorNA/innerIndex)
                                                afterRefractAng = np.arccos(abs(photonData["v"][:, 2]))
                                                beforeRefractAng = np.arcsin(outerIndex*np.sin(afterRefractAng)/innerIndex)
                                                validPhotonBool = beforeRefractAng <= critAng
                                                validDetID = photonData["detid"][validPhotonBool]
                                                validDetID = validDetID - 1  # make detid start from 0
                                                validPPath = photonData["ppath"][validPhotonBool]
                                                validNScat = photonData["nscat"][validPhotonBool]
                                                
                                                # calculate reflectance        
                                                for detectorIdx in range(info["DetNum"]):
                                                        usedValidPPath = validPPath[validDetID[:, 0]==detectorIdx]
                                                        arrmua = np.array(mua)
                                                        weight = np.exp(-(usedValidPPath @ arrmua))
                                                        reflectance[detOutputIdx][detectorIdx] = weight.sum() / photonNum
                                                        
                                                        # mean pathlength
                                                        eachPhotonWeight = weight
                                                        if eachPhotonWeight.sum() == 0:
                                                                meanPathlength[detOutputIdx][detectorIdx] = 0
                                                                continue
                                                        eachPhotonPercent = eachPhotonWeight / eachPhotonWeight.sum()
                                                        eachPhotonPercent = eachPhotonPercent.reshape(-1, 1)
                                                        meanPathlength[detOutputIdx][detectorIdx] = np.sum(eachPhotonPercent*usedValidPPath, axis=0)
                                                        
                                                        # num of scatter
                                                        usedValidPPath2 = validNScat[validDetID[:, 0]==detectorIdx]
                                                        eachPhotonWeight2 = np.exp(-(usedValidPPath2 @ arrmua))
                                                        if eachPhotonWeight2.sum() == 0:
                                                                NumofScatter[detOutputIdx][detectorIdx] = 0
                                                                continue
                                                        eachPhotonPercent2 = eachPhotonWeight2 / eachPhotonWeight2.sum()
                                                        eachPhotonPercent2 = eachPhotonPercent2.reshape(-1, 1)
                                                        NumofScatter[detOutputIdx][detectorIdx] = np.sum(eachPhotonPercent2*usedValidPPath2, axis=0)
                                else:
                                        reflectance = np.empty((len(detOutputPathSet), detectorNum))
                                        meanPathlength = np.empty((len(detOutputPathSet), detectorNum, len(mua)))
                                        NumofScatter = np.empty((len(detOutputPathSet), detectorNum, len(mua)))
                                        for detOutputIdx, detOutputPath in enumerate(detOutputPathSet):
                                                # calculate reflectance        
                                                for detectorIdx in range(info["DetNum"]):
                                                        usedValidPPath = validPPath[validDetID[:, 0]==detectorIdx]
                                                        arrmua = np.array(mua)
                                                        weight = np.exp(-(usedValidPPath @ arrmua))
                                                        reflectance[detOutputIdx][detectorIdx] = weight.sum() / photonNum
                                                        
                                                        # mean pathlength
                                                        eachPhotonWeight = weight
                                                        if eachPhotonWeight.sum() == 0:
                                                                meanPathlength[detOutputIdx][detectorIdx] = 0
                                                                continue
                                                        eachPhotonPercent = eachPhotonWeight / eachPhotonWeight.sum()
                                                        eachPhotonPercent = eachPhotonPercent.reshape(-1, 1)
                                                        meanPathlength[detOutputIdx][detectorIdx] = np.sum(eachPhotonPercent*usedValidPPath, axis=0)
                                                        
                                                        # num of scatter
                                                        usedValidPPath2 = validNScat[validDetID[:, 0]==detectorIdx]
                                                        eachPhotonWeight2 = np.exp(-(usedValidPPath2 @ arrmua))
                                                        if eachPhotonWeight2.sum() == 0:
                                                                NumofScatter[detOutputIdx][detectorIdx] = 0
                                                                continue
                                                        eachPhotonPercent2 = eachPhotonWeight2 / eachPhotonWeight2.sum()
                                                        eachPhotonPercent2 = eachPhotonPercent2.reshape(-1, 1)
                                                        NumofScatter[detOutputIdx][detectorIdx] = np.sum(eachPhotonPercent2*usedValidPPath2, axis=0)
                                movingAverageFinalReflectance = reflectance.reshape(reflectance.shape[0], -1, 3, 2).mean(axis=-1)
                                movingAverageFinalReflectance = postprocess.movingAverage2D(movingAverageFinalReflectance, width=3).reshape(movingAverageFinalReflectance.shape[0], -1)
                                movingAverageFinalReflectanceMean = movingAverageFinalReflectance.mean(axis=0)
                                
                                # getMeanPathlength
                                cvSampleNum = 10
                                meanPathlength = meanPathlength.reshape(-1, cvSampleNum, meanPathlength.shape[-2], meanPathlength.shape[-1]).mean(axis=0)
                                movingAverageMeanPathlength = meanPathlength.reshape(meanPathlength.shape[0], -1, 3, 2, meanPathlength.shape[-1]).mean(axis=-2)
                                movingAverageMeanPathlength = postprocess.movingAverage2D(movingAverageMeanPathlength, width=3).reshape(movingAverageMeanPathlength.shape[0], -1, movingAverageMeanPathlength.shape[-1])
                                movingAverageMeanPathlength = movingAverageMeanPathlength.mean(axis=0)
                                
                                # get num of scatter
                                NumofScatter = NumofScatter.reshape(-1, cvSampleNum, NumofScatter.shape[-2], NumofScatter.shape[-1]).mean(axis=0)
                                movingAverageNumofScatter = NumofScatter.reshape(NumofScatter.shape[0], -1, 3, 2, NumofScatter.shape[-1]).mean(axis=-2)
                                movingAverageNumofScatter = postprocess.movingAverage2D(movingAverageNumofScatter, width=3).reshape(movingAverageNumofScatter.shape[0], -1, movingAverageNumofScatter.shape[-1])
                                movingAverageNumofScatter = movingAverageNumofScatter.mean(axis=0)
                                small_reflectance[id_wl, id_mua, :] = movingAverageFinalReflectanceMean[sds_choose]    
                                purturbed_pathlength = movingAverageMeanPathlength[sds_choose, 6]   
                                purturbed_num_scatter = movingAverageNumofScatter[sds_choose, 6] 
                                perturbed_region_mus = modelParameters['OptParam']['IJV']['mus']*0.1
                                perturbed_region_mua = mua[7]
                                perturbed_region_mut = perturbed_region_mua + perturbed_region_mus
                                muscle_mus = modelParameters['OptParam']['Muscle']['mus']*0.1
                                muscle_mua = mua[6]
                                muscle_mut = muscle_mua + muscle_mus
                                #########WAIT CONFIRM###########
                                perturbed_coef = ((perturbed_region_mus / perturbed_region_mut) / (muscle_mus / muscle_mut))**purturbed_num_scatter\
                                                * (perturbed_region_mut / muscle_mut)**purturbed_num_scatter * np.exp(-(perturbed_region_mut - muscle_mut) * purturbed_pathlength)
                                big_reflectance[id_wl, id_mua, :] = small_reflectance[id_wl, id_mua, :] * perturbed_coef
                                ################################
                                # big_reflectance[id_wl, id_mua, :] = small_reflectance[id_wl, id_mua, :] *0.9
                        else:
                                movingAverageFinalReflectanceMean = postprocess.getMovingAverageReflectance(os.path.join(wl_path, sessionID), mua)
                                small_reflectance[id_wl, id_mua, :] = movingAverageFinalReflectanceMean[sds_choose]
                                movingAverageFinalReflectanceMean = postprocess.getMovingAverageReflectance(os.path.join(wl_path, sessionID2), mua)
                                big_reflectance[id_wl, id_mua, :] = movingAverageFinalReflectanceMean[sds_choose]
                                
                                
        ac_div_dc = (small_reflectance - big_reflectance) / big_reflectance
        R_ratio = ac_div_dc[1:ac_div_dc.shape[0], :, :] / ac_div_dc[0, :, :]
        # data processing
        R_ratio = R_ratio.transpose(2, 1, 0)
        wl_list2 = wl_list[1 : len(wl_list)]
        for i in range(len(sds_choose)):
                if i == 0:
                        df = pd.DataFrame(R_ratio[0, :, :], columns=wl_list2)
                        df['sds'] = [sds_choose[0]+1] * df.shape[0]
                        df['stO2'] = list(stO2) * int(df.shape[0] / len(stO2))
                else:
                        df_temp = pd.DataFrame(R_ratio[i, :, :], columns=wl_list2)
                        df_temp['sds'] = [sds_choose[i]+1] * df_temp.shape[0]
                        df_temp['stO2'] = list(stO2) * int(df_temp.shape[0] / len(stO2))
                        df = pd.concat([df, df_temp], axis=0)
        df.fillna(0)
        df.to_csv(os.path.join(output_path, f'{id_mus}.csv'))   
        print(f'Output save to {output_path} !')

        
        
# %%
       
        # mua_set = 
        # # choose R ratio baseline
        # if id == 0:
            
        # else:
# %%           

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



# %% for change file name
for id_mus in range(27):
    for id_wl, wl_path in enumerate(wl_folder):
        wl = wl_path.split('\\')[1]
        wl = wl.split('nm')[0]
        sessionID = 'small_ijv_' + wl + f'_{id_mus}'
        config = jd.load(os.path.join(wl_path, sessionID, 'config.json'))
        config["OutputPath"] = ""
        jd.save(config, os.path.join(wl_path, sessionID, 'config.json'))
        for i in range(100):
                filename = os.path.join(wl_path, sessionID, 'mcx_output', sessionID + f'_{i+10}_detp.jdat')
                filename_af = filename.replace('.jdat', '')
                try:
                        os.rename(filename, filename_af)
                except:
                        continue

# %%


