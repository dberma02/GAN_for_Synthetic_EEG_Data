import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from data_utils import trim_intervals, get_data
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
import GAN


# load data

# keep_channels=['F3', 'C3', 'P3', 'T3']
# keep_channels=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'A1', 'A2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
def load_data(train_size, test_size):
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

	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size)

	return X_train, X_test, y_train, y_test

def classify(X_train, y_train, X_test, y_test):

	svc = SVC()
	svc.fit(X_train, y_train)
	print(svc.score(X_test, y_test))

def split_labeled_data(X, y):
	zero = []
	zero_y = []
	one = []
	one_y = []

	for i in range(len(y)):
		if y[i] == 1:
			one.append(X[i])
			one_y.append(y[i])
		else:
			zero.append(X[i])
			zero_y.append(y[i])

	ray_list = array_util([zero, zero_y, one, one_y])


	return ray_list[0], ray_list[1], ray_list[2], ray_list[3]

def array_util(list):
	rays = []
	for l in list:
		rays.append(np.asarray(l))

	return rays

def shuffle(X, y):
	shape = X.shape[0]
	data = np.c_[X, y]

	np.random.shuffle(data)

	return data[:shape], data[-1]

def main():

	#Experimenting with different training sizes

	sizes = [.9 , .8, .7, .6, .5, .4, .3]
	for s in sizes:

		X_train, X_test, y_train, y_test = load_data(s, 1.0 - s)

		zero, zero_y, one, one_y = split_labeled_data(X_train, y_train)

		gan_1 = GAN.GAN((zero, zero_y), g_in=X_train.shape[1], g_hid=X_train.shape[1], g_out=X_train.shape[1],
		                d_in=X_train.shape[1], d_hid=X_train.shape[1], d_out=1)
		gan_1.train(1000)
		one_samples = gan_1.generate_data(100).reshape((100,90))

		gan_0 = GAN.GAN((one, one_y), g_in=X_train.shape[1], g_hid=X_train.shape[1], g_out=X_train.shape[1],
		                d_in=X_train.shape[1], d_hid=X_train.shape[1], d_out=1)
		gan_0.train(1000)
		zero_samples = gan_0.generate_data(100).reshape((100,90))

		# debug note: y_train has more instances, curious

		X_train = np.concatenate([X_train, one_samples, zero_samples])
		y_train = np.concatenate([y_train, one_y, zero_y])

		X_train, y_train = shuffle(X_train, y_train)

		classify(X_train, y_train, X_test, y_test)


	a = 0

main()
