# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 10:55:37 2018

@author: vissim
"""
import os
import numpy as np
from scipy.io import wavfile
from kmeans import kmeans
from scipy.stats import multivariate_normal
import pickle
import matplotlib.pyplot as plt

def prepare_data_applying_trian_window(data_1d,window_length=400,shift=160):
    data_1d = np.asarray(data_1d)
#    print(data_1d.shape[0])
    total_data_points = 1 + int((data_1d.shape[0]-window_length) /shift)
#    print(total_data_points)
    data = np.zeros(shape=(total_data_points,window_length))
    window_to_mult = np.bartlett(window_length+2).tolist()
    window_to_mult.remove(0)
    window_to_mult.remove(0)
    window_to_mult = np.asarray(window_to_mult)
#    print(window_to_mult)
    for i in range(total_data_points):
        data[i,:] = data_1d[i*shift : i*shift + window_length] * window_to_mult
    return data

def apply_fft_take_log(data,output_len=32):
    fft_data = np.zeros(shape=(data.shape[0],32))
    for i in range(data.shape[0]):
        a = np.fft.rfft(data[i,:],n=64)
        fft_data[i] = np.delete(a,-1)
    return np.log(np.abs(fft_data))

def mean_sigma_initialization(data,kind="diag"):
    """data is single labelled..means all data belongs to same class
    data is assumed to be in num_examples X num_features
    kind should be diag/full
    """
#    print("data is",data)
    mean = np.mean(data,axis=0)
#    print(mean)
    if kind=="diag":
        std = np.std(data,axis=0)
        # print("std is",std)
        # print("shape of std is ",std.shape)
        sigma = np.diag(std)
        return [mean,sigma]
    else:
        if kind == "full":
            sigma = np.matmul((data - mean).T,data-mean) / data.shape[0]
            return [mean,sigma]
        else:
            print("kind should be either diag or full..please call properly")
def mean_sigma_of_all(data,label,k_of_kmeans,kind="diag"):
    """
    data is num_exam X dimension
    label should be a one_dimensional array
    label contains 0,1,2...k-1
    kind should be diag or full
    params is a list of lists, each smaller list contains mean and sigma matrix
    """
#    print("label is",label)
    # print("incoming data is",data)
    label = np.asarray(label)
    parameters =[]
    for i in range(k_of_kmeans):
        temp_data = data[np.where(label == i)]
#        print(np.where(label == i))
#        print("temp_data is",temp_data)
        parameters.append(mean_sigma_initialization(temp_data,kind="diag"))
#    print("length is",len(parameters))
#    print(len(parameters[0]))
    return parameters

class EM_algo_GMM:
    def __init__(self,data,num_gaussian=2,kind="diag"):
        self.data = np.matrix(data)
        self.num_gaussian = num_gaussian
        self.labels = kmeans(num_gaussian,data)
        ##calculate mix_coeff..
        self.mix_coeff = [0 for i in range(num_gaussian)]
        self.z_hidden_rv = [-1 for i in range(data.shape[0])]
        count =[0 for i in range(num_gaussian)]
        count = np.asarray(count)
        for i in range(data.shape[0]):
            count[self.labels[i]] += 1
        self.mix_coeff = count / data.shape[0]
        self.parameters = mean_sigma_of_all(data,self.labels,num_gaussian,kind)
#        self.parameters = np.asarray(self.parameters)
#        print(self.parameters.shape)
        self.pdf = np.zeros(shape=(data.shape[0],num_gaussian))
        for i in range(data.shape[0]):
            for j in range(num_gaussian):
#                print("mean is",self.parameters[j][0])
#                print("cov is",self.parameters[j][1])
                self.pdf[i][j] = multivariate_normal.pdf(data[i], mean= self.parameters[j][0],
                        cov = self.parameters[j][1])
        self.z_hidden_rv = np.argmax(self.pdf,axis=1)
        ###calculate self.gammas
        self.gammas = np.zeros(shape=(data.shape[0],num_gaussian))
        for i in range(data.shape[0]):
            denominator = np.sum(self.mix_coeff * self.pdf[i])
            for j in range(num_gaussian):
                self.gammas[i][j] = self.mix_coeff[j] * self.pdf[i][j] / denominator
        
    def cal_log_likelihood_complete_data(self):
        log_like=0
        for i in range(self.data.shape[0]):
            log_like += np.log(self.pdf[i][self.z_hidden_rv[i]])
        print("log_like is",log_like)
        return log_like
    
    def EM_iteration(self,max_itr=100,kind="diag"):
        log_likelihood =[]
        for i in range(max_itr):
            print("iteration started")
            log_likelihood.append(self.cal_log_likelihood_complete_data())
            ###write a breaking condition here
            if(i>2 and log_likelihood[len(log_likelihood)-1] - log_likelihood[len(log_likelihood)-2] < 1):
                return log_likelihood,self.parameters,self.mix_coeff
            ####update parameters
            sum_rows_of_gamma = np.sum(self.gammas,axis=0)
            #update mean and cov_mat
            for j in range(self.num_gaussian):
                # print("before  means is",self.parameters[j][0])
                # print("before cov is",self.parameters[j][1])
                weighted_sum_data = np.zeros(shape=(1,self.data.shape[1]))
                # print(weighted_sum_data.shape)
                weighted_sum_cov = np.zeros(shape=(self.data.shape[1],self.data.shape[1]))
                for k in range(self.data.shape[0]):
                    # print(self.gammas[k][j])
                    weighted_sum_data += self.gammas[k][j] * self.data[k]
                    weighted_sum_cov += self.gammas[k][j] * np.matmul((self.data[k] - self.parameters[j][0]).T,(self.data[k] - self.parameters[j][0]))
                    """update this step for keeping the covariance matrix as diagonal"""
                if kind == "diag":
                    weighted_sum_cov = np.diag(np.diagonal(weighted_sum_cov))
                self.parameters[j][0] = weighted_sum_data / sum_rows_of_gamma[j]
                self.parameters[j][1] = weighted_sum_cov / sum_rows_of_gamma[j]
            self.mix_coeff = sum_rows_of_gamma/self.data.shape[0]
            ###update gammas
            self.pdf = np.zeros(shape=(self.data.shape[0],self.num_gaussian))
            for i in range(self.data.shape[0]):
                for j in range(self.num_gaussian):
                    # print(self.parameters[j][0].shape)
                    self.pdf[i][j] = multivariate_normal.pdf(self.data[i], mean= np.reshape(self.parameters[j][0],(32,)),
                            cov = self.parameters[j][1])
            self.z_hidden_rv = np.argmax(self.pdf,axis=1)
            ###calculate self.gammas
            self.gammas = np.zeros(shape=(self.data.shape[0],self.num_gaussian))
            for i in range(self.data.shape[0]):
                denominator = np.sum(self.mix_coeff * self.pdf[i])
                for j in range(self.num_gaussian):
                    self.gammas[i][j] = self.mix_coeff[j] * self.pdf[i][j] / denominator
                
        return log_likelihood,self.parameters,self.mix_coeff

def cal_pdf(data,mu,cov,mix_coeff):
    """
    mix_coeff is assumed to be a list
    mu is assumed to be a array
    """
    pdf=0
    for i in range(len(mix_coeff)):
        pdf += mix_coeff[i] * multivariate_normal.pdf(data, mean= np.reshape(mu[i],newshape=(32,)),cov = cov[i])
    return pdf

def test_gmm(test_data,mu_cov1,mix_coeff1,mu_cov2,mix_coeff2,actual_label):
    #test_data is frames of one file.
    labels = np.zeros(shape=(test_data.shape[0],2)) ###assumed 2 class only
    mu_s=[]
    cov_s = []
    mu_s_2=[]
    cov_s_2 = []
    for index in range(len(mu_cov1)):
        mu_s.append(mu_cov1[index][0])
        cov_s.append(mu_cov1[index][1])
        mu_s_2.append(mu_cov2[index][0])
        cov_s_2.append(mu_cov2[index][1])
        
    for i in range(test_data.shape[0]):
        labels[i][0] = np.log(cal_pdf(test_data[i],mu_s,cov_s,mix_coeff1))
        labels[i][1] = np.log(cal_pdf(test_data[i],mu_s_2,cov_s_2,mix_coeff2))
#        labels[i][0] = cal_pdf(test_data[i],[mu_cov1[0][0],mu_cov1[1][0]],[mu_cov1[0][1],mu_cov1[1][1]],mix_coeff1)
#        labels[i][1] = cal_pdf(test_data[i],[mu_cov2[0][0],mu_cov2[1][0]],[mu_cov2[0][1],mu_cov2[1][1]],mix_coeff2)
    tot_log_likelihood_file = np.sum(labels,axis=0)
#    actual_label = np.asarray(actual_label)
    return np.argmax(tot_log_likelihood_file)

def cal_accuracy(pred,actual):
    correct = 0
    print(len(pred) == len(actual))
    for i in range(len(pred)):
        if pred[i] == actual[i]:
            correct+=1
    return 100 *correct/len(pred)
    
if __name__ == "__main__":
    train_datapath=".\\Data\\speech_music_classification\\train\\"
    test_datapath=".\\Data\\speech_music_classification\\test\\"

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
    em = EM_algo_GMM(data)
    em_max_itr=60
    log_likelihood1,parametrs1,mix_coeff1 = em.EM_iteration(em_max_itr,kind="diag")
    with open("speech_class_"+str(em_max_itr)+".pickle","wb") as f:
        pickle.dump([parametrs1,mix_coeff1],f,protocol = pickle.HIGHEST_PROTOCOL)
    
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
    em = EM_algo_GMM(data)
    log_likelihood2,parametrs2,mix_coeff2 = em.EM_iteration(em_max_itr,kind="diag")
    with open("music_class_"+str(em_max_itr)+".pickle","wb") as f:
        pickle.dump([parametrs2,mix_coeff2],f,protocol = pickle.HIGHEST_PROTOCOL)
    ###plotting the likelihoods
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