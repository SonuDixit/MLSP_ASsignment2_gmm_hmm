# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 18:27:45 2018

@author: vissim
"""
import numpy as np
from scipy.stats import multivariate_normal
from GMM_implementation import mean_sigma_of_all
from kmeans import kmeans
from scipy.io import wavfile
import os
from GMM_implementation import prepare_data_applying_trian_window,apply_fft_take_log
class HMM:
    def __init__():
        print("HMM")

def viterbiAlgorithm(observation,A,B,pie):
    """
    observation is a 1d array assumed it contains 0,1,2,3,4....starting with 0
    A is transition prob matrix num_states X num_states
    B is emission prob matrix num_states X num_observation_symbols
    pie is initial distribution 1d array
    """
    best_state = np.zeros(shape=[A.shape[0],observation.shape[0]],dtype = np.int32)
    best_state[:,:] = -1
    best_prob1 = pie * B[:,observation[0]]
    print("initially best_prob1 is",best_prob1)
    for time in range(1,observation.shape[0]):
        best_prob2 = np.zeros(best_prob1.shape)
#        print("best_prob1 is",best_prob1)
        for i in range(A.shape[0]):
            p =best_prob1 * A[:,i]
            print(p.shape)
            best_prob2[i] = np.max(p)
            best_state[i,time] = np.argmax(p)
        best_prob2 = best_prob2 * B[:,observation[time]]
        best_prob1 = best_prob2
        print("best_prob1 is",best_prob1)
    return best_state

def viterbiAlgorithm_logscale(observation,A,B,pie):
    """
    observation is a 1d array assumed it contains 0,1,2,3,4....starting with 0
    A is transition prob matrix num_states X num_states
    B is emission prob matrix num_states X num_observation_symbols
    pie is initial distribution 1d array
    """
    A = np.log(A)
    B = np.log(B)
    pie = np.log(pie)
    best_state = np.zeros(shape=[A.shape[0],observation.shape[0]],dtype = np.int32)
    best_state[:,:] = -1
    best_prob1 = pie + B[:,observation[0]]
    print("initially best_prob1 is",best_prob1)
    for time in range(1,observation.shape[0]):
        best_prob2 = np.zeros(best_prob1.shape)
#        print("best_prob1 is",best_prob1)
        for i in range(A.shape[0]):
            p =best_prob1 + A[:,i]
            print(p.shape)
            best_prob2[i] = np.max(p)
            best_state[i,time] = np.argmax(p)
        best_prob2 = best_prob2 + B[:,observation[time]]
        best_prob1 = best_prob2
        print("best_prob1 is",best_prob1)
    state_sequence = [np.argmax(best_prob1)]
    i=observation.shape[0]-1
    while i>0:
        state_sequence.append(best_state[state_sequence[-1]][i])
        i -= 1
        print(state_sequence)
    state_sequence.reverse()
    print(state_sequence)
    return best_state,state_sequence

def forward_calculation(A,B,pie,Observation):
    """
    observation is a 1d array assumed it contains 0,1,2,3,4....starting with 0
    A is transition prob matrix num_states X num_states
    B is emission prob matrix num_states X num_observation_symbols
    pie is initial distribution 1d array
    """
    alpha = np.zeros(shape=(Observation.shape[0],A.shape[0]))
#    print(B[:,Observation[0]])
#    print(pie)
#    print(alpha[0])
    alpha[0] = pie * B[:,Observation[0]]
#    print(alpha[0])
#    return alpha[0],5
    scale_factor=[np.sum(alpha[0])]
    print(scale_factor)
    for i in range(1,Observation.shape[0]):
#        print(alpha[i-1])
#        print(A)
        alpha[i-1] /= scale_factor[i-1]
        temp = np.matmul(alpha[i-1],A)
#        print("temp is",temp)
        alpha[i] = temp * B[:,Observation[i]]
        scale_factor.append(np.sum(alpha[i]))
#        print("sc fac is",scale_factor[i])

    return alpha,scale_factor
def backward_calculation(A,B,pie,Observation,scale_factor):
    beta = np.zeros(shape=(Observation.shape[0],A.shape[0]))
    beta[Observation.shape[0] -1] =1
    time = Observation.shape[0] -2
    while time >=0:
        for i in range(A.shape[0]):
#            beta[time+1] /= scale_factor[time+1]
            beta[time][i] = np.sum(beta[time+1] * A[i] * B[:,Observation[time+1]])
            beta[time] /= scale_factor[time]
        time -= 1
    return beta

def prepare_gamma_table(alpha,beta):
    """ alpha and beta are T X N array
    """
    gamma = alpha * beta
    for i in range(gamma.shape[0]):
        gamma[i] = gamma[i] / np.dot(alpha[i],beta[i])
    return gamma
def pie_initialise(num_states):
    pie = np.zeros(shape=(num_states,))
    pie[0] = 1
    return pie
def A_initialise(num_states,data_shapes):
    """data shapes is a 1d array containing shape(number of frames) of all the files
    """
    data_shapes = np.asarray(data_shapes)
    A = np.zeros(shape=(num_states,num_states))
    for i in range(data_shapes.shape[0]):
        t = int(data_shapes[i]/num_states)
        A[0] += t-1
        A[1] += t
    p=A[0]/A[1]
    A = np.diag(p)
    A[A.shape[0]-1,A.shape[1]-1] = 1
    for i in range(0,A.shape[0]-1):
        A[i][i+1] = 1-A[i][i]
    return A
def B_initialisation_parameters(data,k=2):
    """
    data is assumed to be num_examples X num_features
    its input should be data assigned to one hidden state
    use k_means initialization for GMM mixture
    should return mu,sigma and mixing coefficient for the state.
    """
    centroids = kmeans(k=k,data=data)
    parameters = mean_sigma_of_all(data=data,label=centroids,k_of_kmeans = k,kind="diag")
    #parameters contain mu and sigma only
    # need to calculate mixing coefficients
    mix_coeff = [0 for i in range(k)]
    count =[0 for i in range(k)]
    count = np.asarray(count)
    for i in range(data.shape[0]):
        count[centroids[i]] += 1
    mix_coeff = count / data.shape[0]
    return mix_coeff,parameters
def cal_pdf(data,mu,cov,mix_coeff):
    """
    mix_coeff is assumed to be a list
    mu is assumed to be a array
    """
    pdf=0
    for i in range(len(mix_coeff)):
        pdf += mix_coeff[i] * multivariate_normal.pdf(data, mean= np.reshape(mu[i],newshape=(32,)),cov = cov[i])
    return pdf
def B_initialise(all_observation_segregated,observation_all,num_states,k=2):
    """all_observation_segregated is assumed to be a list where
    all_obser_segregated[0] = data_asociated with state 0
    all_obser_segregated[1] = data_associated with sttae 1
    ...

    observation_All is a list
    observation_all[0] = observation_sequence_1 = frames of file1
    observation_all[1] = observation_Sequence_2 = frames of file2
    returns B matrix in a list and gaussian parameters( mu and sigma) and mixing coefficients 
    """
    temp=observation_all[0]
    for i in range(1,len(observation_all)):
        temp = np.vstack((temp,observation_all[i]))
    B = np.zeros((num_states,temp.shape[0]))
    mix_coeff_list=[]
    mu_and_sigma_list =[]
    for i in range(num_states):
#        print("shape is",all_observation_segregated[i])
        mix_coeff,mu_and_sigmas =B_initialisation_parameters(all_observation_segregated[i],k=k)
        mix_coeff_list.append(mix_coeff)
        mu_and_sigma_list.append(mu_and_sigmas)
        mu_s=[]
        cov_s = []
        for index in range(len(mu_and_sigmas)):
            mu_s.append(mu_and_sigmas[index][0])
            cov_s.append(mu_and_sigmas[index][1])
        for j in range(temp.shape[0]):
            B[i][j] = cal_pdf(temp[j],mu_s,cov_s,mix_coeff)
    ###changing B matrix to a list corrsponding to every observation
    #(since there are more than one observation sequences)
    B_all =[]
    col_count=0
    for i in range(len(observation_all)):
        
        B_all.append(B[:,col_count:col_count +observation_all[i].shape[0]])
        col_count += observation_all[i].shape[0]
    return B_all,mix_coeff_list,mu_and_sigma_list
    
    
def calculate_zeta_t(alpha,beta,A,B,t,Observation):
    # returns a 2d array of num_state X num_state for the given time t
    zeta_table_for_t = np.zeros(A.shape[0],A.shape[0])
    temp = beta[t+1] * B[:,Observation[t+1]]
    temp = np.reshape(temp,newshape=(1,temp.shape[0]))
    temp2 = np.matrix(alpha[t]).T
    temp3 = np.matmul(temp2,temp)
    zeta_table_for_t = temp3 * A
    zeta_table_for_t /= np.sum(zeta_table_for_t)
    return zeta_table_for_t
def gamma_t_table_givenTand_Obs(obs,t,gamma_of_obs,previos_C,B,mu_list,cov_list):
    """gamma_of_obs is T X Num_state
    previous_C is NUm_state X num_mixture_coeff
    B is the B matrix corresponding to this obs sequence
    mu_list contains mu_s for every state similarly for cov_list
    	"""
    #calculate pdf matrix of obs[t] corresponding to each state and each component
    pdf_matrix = np.zeros((gamma_of_obs.shape[1],len(mu_list[0])))
    for i in range(pdf_matrix.shape[0]):
        for j in range(pdf_matrix.shape[1]):
            pdf_matrix[i][j] = multivariate_normal.pdf(obs[t], mean= np.reshape(mu_list[i][j],newshape=(32,)),cov = cov_list[i][j])
    temp = pdf_matrix * previos_C
    temp2 = B[:,t] / gamma_of_obs[t,:]
    temp2 = np.asmatrix(temp2)
    return temp/temp2.T

def update_pi(gamma_all):
    """gamma_all is a list containing gamma for all observation sequences.
    	each gamma in gamma_all is of shape time X num_states 
    """
    pie = gamma_all[0][0]
    for i in range(1,len(gamma_all)):
    	pie += gamma_all[i][0]
    pie /= len(gamma_all)
    return pie
def update_A(gamma_all,alpha_All,beta_All,observation_all):
    """alpha_All = list of aslphas
    beta_all = list of betas
    	Observation_All = list of Observations
    	"""
	  ## prepare denominator
    denominator = np.sum(gamma_all[0],axis =0)
    denominator -= gamma_all[0][gamma_all[0].shape[0]-1] 
    for i in range(1,len(gamma_all)-1):
    		denominator += np.sum(gamma_all[i],axis =0)
    		denominator -= gamma_all[i][gamma_all[i].shape[0]-1]
    	##prepare numerator
    numerator = np.zeros((gamma_all[0].shape[1],gamma_all[0].shape[1])) ###num_states X num_states
    for i in range(len(observation_all)) :
    		o = observation_all[i]
    		for time in range(o.shape[0] - 1):
    			 numerator += calculate_zeta_t(alpha_All[i],beta_All[i],time,o)
    return numerator/denominator
def update_C(gamma_all,previous_mix_coeff,B_all,mu_list,cov_list,observation_all): ##C are mixing coefficients
	##prepare denominator
    denominator = np.sum(gamma_all[0],axis =0)
    for i in range(1,len(gamma_all)):
        denominator += np.sum(gamma_all[i],axis =0)
    ##prepare numerator
    numerator = np.zeros((gamma_all[0].shape[1],len(mu_list[0])))
    for i in range(len(gamma_all)):
        for t in range(gamma_all[i].shape[0]):
            numerator += gamma_t_table_givenTand_Obs(observation_all[i],t,gamma_all[i],previous_mix_coeff,B_all[i],mu_list,cov_list)
    mix_coeff_mat = numerator/denominator
    return mix_coeff_mat
def update_mu(gamma_all,previous_mix_coeff,B_all,mu_list,cov_list,observation_all): ##C are mixing coefficients
    ##prepare denominator
    """nume is 3d matrix, rows = states,col = number of mixture components,
    3rd dimension = mean which is 32 dimension
    """
    deno = np.zeros((gamma_all[0].shape[1],len(mu_list[0])))
    nume = np.zeros((gamma_all[0].shape[1],len(mu_list[0]),observation_all[0][0].shape[0]))
    for i in range(len(gamma_all)):
        for t in range(gamma_all[i].shape[0]):
            temp=gamma_t_table_givenTand_Obs(observation_all[i],t,gamma_all[i],previous_mix_coeff,B_all[i],mu_list,cov_list)
            deno += temp
            temp2 = np.zeros((gamma_all[0].shape[1],len(mu_list[0]),observation_all[0][0].shape[0]))
            for j in range(temp.shape[0]):
                for k in range(temp.shape[1]):
                    temp2[j][k] = temp[j][k] * observation_all[i][t]
            nume += temp2

    for i in range(nume.shape[0]):
        for t in range(nume.shape[1]):
            nume[i][t] /= deno[i][t]
    return nume
def update_covs(gamma_all,previous_mix_coeff,B_all,mu_list,cov_list,observation_all): ##C are mixing coefficients
    ##prepare denominator
    """nume is 4d matrix, rows = states,col = number of mixture components,
    3rd and 4th dimension = covariance which is 32*32 dimension
    """
    deno = np.zeros((gamma_all[0].shape[1],len(mu_list[0])))
    nume = np.zeros((gamma_all[0].shape[1],len(mu_list[0]),observation_all[0][0].shape[0],observation_all[0][0].shape[0]))
    for i in range(len(gamma_all)):
        for t in range(gamma_all[i].shape[0]):
            temp=gamma_t_table_givenTand_Obs(observation_all[i],t,gamma_all[i],previous_mix_coeff,B_all[i],mu_list,cov_list)
            deno += temp
            temp2 = np.zeros((gamma_all[0].shape[1],len(mu_list[0]),observation_all[0][0].shape[0],observation_all[0][0].shape[0]))
            for j in range(temp.shape[0]):
                for k in range(temp.shape[1]):
                    temp2[j][k] = temp[j][k] * np.matmul(np.matrix(observation_all[i][t]-mu_list[i][k]).T,observation_all[i][t]-mu_list[i][k])
            nume += temp2

    for i in range(nume.shape[0]):
        for t in range(nume.shape[1]):
            nume[i][t] /= deno[i][t]
    return nume

def read_data():
    train_datapath=".\\Data\\speech_music_classification\\train\\"
    test_datapath=".\\Data\\speech_music_classification\\test\\"
    files = os.listdir(train_datapath+"speech\\")
    observation_all=[]
    num_obsernations_in_sequence =[]
    for file in files:
        fs, data = wavfile.read(train_datapath+"speech\\" + file)
        data = prepare_data_applying_trian_window(data)
        data = apply_fft_take_log(data)
        observation_all.append(data)
        num_obsernations_in_sequence.append(data.shape[0])
    return observation_all,num_obsernations_in_sequence
def segregate_observations(num_states,observation_all,num_obsernations_in_sequence):
    seg_observations =[]
    every_state_responsible = [int(num_obsernations_in_sequence[i]/num_states) for i in range(len(num_obsernations_in_sequence))]
    print(every_state_responsible)
    row_bound =0
    for i in range(num_states-1):
        d = observation_all[0][row_bound:row_bound + (i+1)*every_state_responsible[0]]
        seg_observations.append(d)
        row_bound += (i+1)*every_state_responsible[0]
    seg_observations.append(observation_all[0][row_bound:])
    
    for j in range(1,len(observation_all)):
        row_bound = 0
        for k in range(num_states-1):
            seg_observations[k] = np.vstack((seg_observations[k],observation_all[j][row_bound: (k+1) * every_state_responsible[j] ]))
            row_bound += (k+1)*every_state_responsible[j]    
        seg_observations[num_states-1] = np.vstack((seg_observations[num_states-1],observation_all[j][row_bound: row_bound * every_state_responsible[j] ]))
    return seg_observations

num_states = 3
k=2
observation_all,num_obsernations_in_sequence = read_data()

seg_obs = segregate_observations(num_states,observation_all,num_obsernations_in_sequence)
pie = pie_initialise(num_states)

A=A_initialise(num_states,num_obsernations_in_sequence)
B_all,mix_coeff,mu_and_sigmas = B_initialise(seg_obs,observation_all,num_states,k)
alphas=[]
betas =[]
gamma_all=[]
obs_int = [i for i in range(observation_all[0].shape[0])]
obs_int = np.asarray(obs_int)
alpha,scale_factor = forward_calculation(A,B_all[0],pie,obs_int)
for i in range(len(observation_all)):
    obs_int = [i for i in range(observation_all[i].shape[0])]
    obs_int = np.asarray(obs_int)
    alpha,scale_factor = forward_calculation(A,B_all[i],pie,obs_int)
    beta = backward_calculation(A,B_all[i],pie,obs_int,scale_factor)
    alphas.append(alpha)
    betas.append(beta)
    gamma_all.append(prepare_gamma_table(alpha,beta))

pie = update_pi(gamma_all)

# observation =np.asarray([2,2,1,0,1,3,2,0,0])
# A = np.asarray([[.5,.5],[.4,.6]])
# B = np.asarray([[.2,.3,.3,.2],[.3,.2,.2,.3]])
# pie = np.asarray([.5,.5])
# best_state2,state_sequence= viterbiAlgorithm_logscale(observation,A,B,pie)
# print(state_sequence)
# alpha = forward_calculation(A,B,pie,observation)
# print(alpha)