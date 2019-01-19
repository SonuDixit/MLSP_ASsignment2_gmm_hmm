# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def initialize_centroids(k,data,method="random"):
    n = data.shape[0]
    ##using random may give duplicates
    indexes = np.random.choice(range(n),k,replace=False)
    cents = np.zeros(shape=(k,data.shape[1]))
    for i in range(k):
        cents[i,:] = data[indexes[i],:]
    return cents

def find_new_cent(data_point,centroids):
    #it is for every dtat_point
    dist = np.linalg.norm(data_point - centroids[0])
    cent =0
    for i in range(1,centroids.shape[0]):
        d = np.linalg.norm(data_point - centroids[i])
        if d<dist:
            dist = d
            cent = i
    return cent

def update_centroids(new_cents,data,k):
    updated_cents = np.zeros(shape=(k,data.shape[1]))
    count =[0 for i in range(k)]
    for i in range(data.shape[0]):
        updated_cents[new_cents[i]] += data[i]
        count[new_cents[i]] += 1
    for i in range(k):
        updated_cents[i] /= count[i]
    return updated_cents

def check_break_condition(prev,now):
    a = True
    for i in range(prev.shape[0]):
        for j in range(now.shape[1]):
            a = a and prev[i][j] == now[i][j]
    return a

def kmeans(k,data):
    # data is num_exam * dimension array
    #k is an integer
    initials = initialize_centroids(k,data)
    itr= 0
    while True:
        prev_centroids = []
        for i in range(data.shape[0]):
            prev_centroids.append(find_new_cent(data[i],initials))
        updated_centroids = update_centroids(prev_centroids,data,k)
        print(itr)
        if check_break_condition(initials,updated_centroids) or itr >25:
            break;
        else:
            initials = updated_centroids
        itr += 1
    return prev_centroids


datapath = ".//Data//DatasetsLA3//"
#if __name__ == "__main__":
def check_with_blobs():
    data = np.loadtxt(datapath+"blobs.txt", delimiter=" ", unpack=False)
    print("input data is:")
    plt.scatter(data[:,0],data[:,1])
    labels = kmeans(2,data[:,0:2])
    print("after k means:")
    plt.scatter(data[:,0],data[:,1],c=labels)
    #def check_with_circles2()
    data = np.loadtxt(datapath+"circles2.txt", delimiter=" ", unpack=False)
    plt.scatter(data[:,0],data[:,1])    
    labels = kmeans(2,data[:,0:2])
    plt.scatter(data[:,0],data[:,1],c=labels)
    ##checking with circles3.txt
    data = np.loadtxt(datapath+"circles3.txt", delimiter=" ", unpack=False)
    plt.scatter(data[:,0],data[:,1])    
    labels = kmeans(3,data[:,0:2])
    plt.scatter(data[:,0],data[:,1],c=labels)
