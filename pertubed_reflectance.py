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

# try:
#     jd_path = "mcx/example/quicktest/small_ijv_730_0_0_detp.jdat"
#     jdata = jd.load(jd_path)
# except:
#     print('jdata not founded !')

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
movingAverageFinalReflectanceMean = postprocess.getMovingAverageReflectance(sessionID, mua)
# %%
movingAverageMeanPathlength = postprocess.getMeanPathlength(sessionID, mua)[1].mean(axis=0)
# %%
movingAverageNumofScatter = postprocess.getNumofScatter(sessionID, mua)[1].mean(axis=0)
print(movingAverageNumofScatter.shape)
# %%
