"""
In this experiment, we will generate sample data at different points of convergence throughout
training, and test their efficacy in improving test accuracy.

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


def train_GAN(X_train, y_train):
		gan = GAN.GAN((X_train, y_train), g_in=X_train.shape[1], g_hid=100, g_out=X_train.shape[1],
					   d_in=X_train.shape[1], d_hid=10, d_out=1)
		return gan.train(10000)


def classify(estim,X_test, y_test):
	score = estim.score(X_test, y_test)
	print(score)
	return score

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

scores = []

X_train, X_test, y_train, y_test = load_data(.8, .2)

# condition 0: baseline accuracy with real data
clf = find_best_clf(X_train, y_train, X_test, y_test)
real_score = classify(clf, X_test, y_test)
scores.append(real_score)
left_X, left_y = X_train[y_train == 0], y_train[y_train == 0]
right_X, right_y = X_train[y_train == 1], y_train[y_train == 1]

print("left")
left_GAN = train_GAN(left_X, left_y)

print("right")
right_GAN = train_GAN(right_X, right_y)

for set in range(len(left_GAN)):

	GAN_data = np.append(right_GAN[set].reshape((100,90)), left_GAN[set].reshape((100,90)), axis=0)
	GAN_labels = np.append(np.ones(100), np.zeros(100))

	#print(GAN_data.shape)
	GANX = np.append(GAN_data, X_train, axis=0)
	GANY = np.append(GAN_labels, y_train)
	GANX, GANY = shuffle(GANX, GANY)

	# condition 1: train linear classifier on augmented data
	clf_aug = find_best_clf(GANX, GANY, X_test, y_test)
	aug_score = classify(clf_aug, X_test, y_test)
	print("set:" + str(set))
	print("Augmented data:", aug_score)
	print("Real data:", real_score)
	scores.append(aug_score)


print(scores)
plt.title('Accuracy at different points of convergence throughout training')
plt.bar(['Real', "set 1", "set 2", "set 3", "set 4", "set 5"], scores)
plt.ylim((.7, .85))
plt.show()
