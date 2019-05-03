"""
In this experiment, we compare accuracy achieved between the baseline (the real dataset)
and an augmented dataset. The independent variable of this experiment is different training sizes.
We have two linear classifiers, one to classify the baseline dataset and the other to classify
the augmented data set. We run model selection for both models at every training size.
We have 3 trials and plot the average of the results over 3 trials.

"""



from warnings import simplefilter
import warnings

simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import sklearn
from data_utils import trim_intervals, get_data
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
import GAN
from sklearn.neural_network import MLPClassifier


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

def shuffle(X, y):                                                              
	shape = X.shape[1]
	data = np.c_[X, y]

	np.random.shuffle(data)

	return data[:, :shape], data[:, shape]


def train_GAN(X_train, y_train, sample_num):
		gan = GAN.GAN((X_train, y_train), g_in=X_train.shape[1], g_hid=100, g_out=X_train.shape[1],
					   d_in=X_train.shape[1], d_hid=10, d_out=1)
		gan.train(10000)
		return gan.generate_data(sample_num).reshape((100,90))

def classify(estim,X_test, y_test):
	score = estim.score(X_test, y_test)
	#print(score)
	return score

def sample_mean_diff(fake_left, true_left, fake_right, true_right):
	print("Left sample mean difference:", np.mean(np.mean(fake_left, axis=0) - np.mean(true_left, axis=0)))
	print("Right sample mean difference:", np.mean(np.mean(fake_right, axis=0) - np.mean(true_right, axis=0)))

def find_best_clf(X_train, y_train, X_test, y_test):
	#svc = SVC(C=.001, kernel='linear').fit(X_train, y_train)

	svc = SVC()
	parameters = {'kernel': ('linear', 'rbf'), 'C': [.001, .01, .1, 1, 10]}
	clf = GridSearchCV(svc, parameters, cv=5)

	clf.fit(X_train, y_train)
	bestsvc = clf.best_estimator_
	svc_score = bestsvc.score(X_test, y_test)
	#print("svc", svc_score)
	#########


	##########

	nn = MLPClassifier()
	params = {'activation': ("logistic",  "relu"), 'solver': ('sgd', 'adam'), 'alpha': [.001, .01, .1, 1, 10]}
	clfnn = GridSearchCV(nn, params, cv=5)
	clfnn.fit(X_train, y_train)
	bestnn = clfnn.best_estimator_
	nn_score = bestnn.score(X_test, y_test)
	#print("nn", bestnn.score(X_test, y_test))

	if nn_score > svc_score: return bestnn
	else: return bestsvc

# take the average performance over 5 iterations
sizes = [.9, .8, .7, .6, .5]
real_mean = []
aug_mean = []

for i in range(3):
	real_scores = []
	aug_scores = []
	for s in sizes:
		print(s)
		X_train, X_test, y_train, y_test = load_data(s, 1 - s)

		# condition 0: baseline accuracy with real data
		clf = find_best_clf(X_train, y_train, X_test, y_test)
		real_score = classify(clf, X_test, y_test)
		left_X, left_y = X_train[y_train == 0], y_train[y_train == 0]
		right_X, right_y = X_train[y_train == 1], y_train[y_train == 1]

		#print("left")
		left_GAN = train_GAN(left_X, left_y, 100)

		#print("right")
		right_GAN = train_GAN(right_X, right_y, 100)

		GAN_data = np.append(right_GAN, left_GAN, axis=0)
		GAN_labels = np.append(np.ones(100), np.zeros(100))

		# measure difference between left and right sample means from real data
		#sample_mean_diff(left_GAN, left_X, right_GAN, right_X)

		#print(GAN_data.shape)
		GANX = np.append(GAN_data, X_train, axis=0)
		GANY = np.append(GAN_labels, y_train)
		GANX, GANY = shuffle(GANX, GANY)

		# condition 1: train linear classifier on augmented data
		clf_aug = find_best_clf(GANX, GANY, X_test, y_test)
		aug_score = classify(clf_aug, X_test, y_test)

		print("Augmented data:", aug_score)
		print("Real data:", real_score)
		aug_scores.append(aug_score)
		real_scores.append(real_score)
	aug_mean.append(aug_scores)
	real_mean.append(real_scores)
	print(aug_mean)
	print(real_mean)

r_m = np.mean(real_mean, axis=0)
a_m = np.mean(aug_mean, axis=0)

plt.title('Accuracy on Test of Real and Augmented training datasets')
plt.plot(a_m, 'r--', label='Augmented dataset')
plt.plot(r_m,'b--',label='Real dataset')
plt.legend()
plt.show()





