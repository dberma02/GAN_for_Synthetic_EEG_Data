import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from data_utils import trim_intervals, get_data
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from GAN import *


# load data

# keep_channels=['F3', 'C3', 'P3', 'T3']
# keep_channels=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'A1', 'A2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
def load_data():
	keep_channels=['C3']
	trial_len = 1.5

	# X, y = get_data("../data/CLASubjectA1601083StLRHand.mat", trial_len, keep_channels)
	X, y = get_data("../data/CLASubjectB1512153StLRHand.mat", trial_len, keep_channels)

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

	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

	return X_train, X_test, y_train, y_test


def main():

	X_train, X_test, y_train, y_test = load_data()

	print(X_train.shape)

	generator = gen(in_size = X_train.shape[0], out_size = X_train.shape[1], hid_size = 10, activ_func = None)
	discriminator = disc(in_size = X_train.shape[0], out_size = X_train.shape[1], hid_size = 10, activ_func = None)
	gan = GAN(D = discriminator, G = generator, d_learning_rate = 0.01, g_learning_rate = 0.01)

main()