# data_preprocess.py

import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from data_utils import trim_intervals, get_data
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
import GAN
import torch

class data_loader():
    def __init__(self, filename, features = None):
        self.filename = filename
        self.features = features
        self.load_data()


    def PCA(self, data, k=70):
    # preprocess the data
        diff = np.diff(data)
        X = torch.from_numpy(diff)
        X_mean = torch.mean(X,0)
        X = X - X_mean.expand_as(X)

    # svd
        U,S,V = torch.svd(torch.t(X))
        return torch.mm(X,U[:,:k])

    def load_data(self):                                              
        keep_channels=['C3']                                                           
        trial_len = 1.5                                                                
                                                                                   
        # X, y = get_data("../data/CLASubjectA1601083StLRHand.mat", trial_len, keep_channels)
        # "../data/CLASubjectB1512153StLRHand.mat"
        X, y = get_data(self.filename, trial_len, keep_channels)
                                                                                   
        X = X[y != 3]                                                                  
        y = y[y != 3]                                                                  
        # 0 is left hand                                                               
        y[y == 1] = 0                                                                  
        # 1 is right hand                                                              
        y[y == 2] = 1                                                                  
        interval_len = .45                                                             
        X = trim_intervals(X, .15, interval_len)                                       
                                                                                   
        num_channels= len(keep_channels)                                               
        d2 = np.ceil(num_channels * interval_len / 0.005).astype(int)                  
        X = X.reshape(642, d2)                                                         
                             
        
        if 'pca' is self.features:
            X = self.PCA(X).numpy()

        if 'nn' is self.features:
            self.get_nn_features()

        self.X = X
        self.y = y
        

    def test_train_split(self, train_size, test_size):   
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size=train_size, test_size=test_size)
                                                                                   
        return X_train, X_test, y_train, y_test
    def normalize(self):
        self.X = preprocessing.normalize(self.X, norm='l2')



d = data_loader("../data/CLASubjectB1512153StLRHand.mat")
d.test_train_split(0.8, 0.2)