import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from scipy import interpolate
import scipy.io as sio
from numpy import *
# import torch
import pickle
import random

def censored_target(df, pc):
    l = random.sample(list(df.unit_number.unique()), pc)
    df_censored = df
    df_censored['censored'] = 0
    for i in l:
        dtemp = df.loc[df.unit_number == i]
        k = random.sample(list(dtemp.time_cycle[0:-1]), 1) # on tire au hasard un cycle avant le dernier cycle pour censurer les données
        k = k[0]
        df_censored = df_censored.loc[((df_censored.unit_number == i) & (df_censored.time_cycle <= k)) | (df_censored.unit_number != i)]
        df_censored = df_censored.reset_index(drop=True)
        df_censored.loc[df_censored.unit_number == i, 'censored'] = 1
    for j in range(1,14):
        df_censored['target_'+str(j*10)] = 0
        df_censored.loc[(df_censored.rul <= (10 * j)) & (df_censored.censored == 0), 'target_'+str(j*10)] = 1
        df_censored.loc[(df_censored.rul <= (10 * j)) & (df_censored.censored == 1), 'target_'+str(j*10)] = np.nan
    return(df_censored)


def sliding_window(df, quali, quanti, target, window_Size): # quanti include target
    mu, sigma = df[quanti].mean(), df[quanti].std() # mean & std
    # Normalisation trainset
    df[quanti] = (df[quanti] - mu)/(sigma)
    df = pd.concat([df[quanti], df[quali], df[target]], axis = 1)
    #
    X = []
    Y = []
    # Set sliding time window processing for trainset
    for un in df.unit_number.unique():
        data_temp = df.loc[df.unit_number == un]
        for j in range(len(data_temp) - window_Size + 1):
            X.append(data_temp[j:j + window_Size][quanti + quali]) #
            Y.append(data_temp[j + window_Size - 1: j + window_Size][target])
    # Convert to np.array
    X = np.array(X)
    Y = np.array(Y)
    return(X, Y)

def sliding_window_2(df, quali, quanti, target, window_Size, mu, sigma): # quanti include target
    #mu, sigma = df[quanti].mean(), df[quanti].std() # mean & std
    # Normalisation trainset
    df[quanti] = (df[quanti] - mu)/(sigma)
    df = pd.concat([df[quanti], df[quali], df[target]], axis = 1)
    #
    X = []
    Y = []
    # Set sliding time window processing for trainset
    for un in df.unit_number.unique():
        data_temp = df.loc[df.unit_number == un]
        for j in range(len(data_temp) - window_Size + 1):
            X.append(data_temp[j:j + window_Size][quanti + quali]) #
            Y.append(data_temp[j + window_Size - 1: j + window_Size][target])
    # Convert to np.array
    X = np.array(X)
    Y = np.array(Y)
    return(X, Y)

def fea_extract1(data):  # feature 1
    regr = linear_model.LinearRegression()  # feature of linear coefficient
    fea = []
    x = np.array(range(data.shape[0]))
    for i in range(data.shape[1]):
        regr.fit(x.reshape(-1, 1), np.ravel(data[:, i]))
        fea = fea + list(regr.coef_)
    return fea

def fea_extract2(data):  # feature 2
    fea = []
    for i in range(data.shape[1]):
        fea.append(np.mean(data[:, i]))
    return fea

def statistical_feature(X, window_Size, Feasize):
    n_quali = X.shape[2] - Feasize #### num of qualitative features
    X_fea1 = []
    X_fea2 = []
    X = np.reshape(X[:][:, :, 0:Feasize + n_quali], [X.shape[0], window_Size, Feasize + n_quali, 1])
    
    for i in range(len(X)): 
        data_temp = X[:][:, :, 0:Feasize][i]
        X_fea1.append(fea_extract1(data_temp))
        X_fea2.append(fea_extract2(data_temp))
    
    scale1 = preprocessing.MinMaxScaler().fit(X_fea1)
    X_fea1 = scale1.transform(X_fea1)

    scale2 = preprocessing.MinMaxScaler().fit(X_fea2)
    X_fea2 = scale2.transform(X_fea2)
    
    X_new = []
    for i in range(len(X)):
        data_temp0 = X[:][:, :, 0:Feasize][i]
        data_temp1 = np.reshape(X_fea1[i], [1, Feasize, 1])  # regr.coef_
        data_temp2 = np.reshape(X_fea2[i], [1, Feasize, 1])  # mean_value
        data_temp = np.vstack((data_temp0, data_temp1, data_temp2))
#         print(dat_temp.shape)
        quali_temp0 = test0 = X[:][:, :, Feasize:Feasize+n_quali][i]
        quali_temp1 = test1 = X[:][:, :, Feasize:Feasize+n_quali][i][0:2]
        quali_temp = np.vstack((quali_temp0, quali_temp1))
#         print(quali_temp)
        data_temp_final = np.hstack((data_temp, quali_temp))
#         print(data_temp.shape)
        X_new.append(data_temp_final)
        
    X_new = np.array(X_new)
    X_new = X_new.squeeze()
    return(X_new)