# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 14:50:03 2018

@author: vissim
"""

import os
import numpy as np
from scipy.io import wavfile
from kmeans import kmeans
from scipy.stats import multivariate_normal
import pickle
from GMM_implementation import prepare_data_applying_trian_window
from GMM_implementation import apply_fft_take_log
from GMM_implementation import mean_sigma_initialization
from GMM_implementation import mean_sigma_of_all
from GMM_implementation import EM_algo_GMM
from GMM_implementation import cal_pdf
from GMM_implementation import test_gmm,cal_accuracy
import matplotlib.pyplot as plt

train_datapath=".\\Data\\speech_music_classification\\train\\"
test_datapath=".\\Data\\speech_music_classification\\test\\"
###getting training parameters
num_gaussian =5
max_itr=10
kind="diag"
files = os.listdir(train_datapath+"speech\\")
alldata=[]
for file in files:
    fs, data = wavfile.read(train_datapath+"speech\\" + file)
    data = prepare_data_applying_trian_window(data)
    data = apply_fft_take_log(data)
    alldata.append(data)
data = alldata[0]
for i in range(1,len(alldata)):
    data = np.vstack((data,alldata[i]))
em = EM_algo_GMM(data,num_gaussian,kind)
log_likelihood1,parametrs1,mix_coeff1 = em.EM_iteration(max_itr,kind)

files = os.listdir(train_datapath+"music\\")
alldata=[]
for file in files:
    fs, data = wavfile.read(train_datapath+"music\\" + file)
    data = prepare_data_applying_trian_window(data)
    data = apply_fft_take_log(data)
    alldata.append(data)
data = alldata[0]
for i in range(1,len(alldata)):
    data = np.vstack((data,alldata[i]))
em = EM_algo_GMM(data,num_gaussian,kind)
log_likelihood2,parametrs2,mix_coeff2 = em.EM_iteration(max_itr,kind)

###testing
alldata=[]
act_label=[]
plt.plot(log_likelihood1)
plt.show()
plt.plot(log_likelihood2)
plt.show()

files = os.listdir(test_datapath)
act_label=[]
pred =[]
for file in files:
    if "speech" in file:
        act_label.append(0)
    else:
        act_label.append(1)
    fs, data = wavfile.read(test_datapath+file)
    data = prepare_data_applying_trian_window(data)
    data = apply_fft_take_log(data)
    pred.append(test_gmm(data,parametrs1,mix_coeff1,parametrs2,mix_coeff2,act_label))
print("accuracy is",cal_accuracy(pred,act_label))