import numpy as np
import matplotlib.pyplot as plt
from data_utils import trim_intervals, get_data
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
import GAN
from sklearn.neural_network import MLPClassifier
from warnings import simplefilter
import warnings

simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")



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

def shuffle(X, y):
	shape = X.shape[1]
	data = np.c_[X, y]

	np.random.shuffle(data)

	return data[:, :shape], data[:, shape]

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

def main():

	#Experimenting with different training sizes
	X_train, X_test, y_train, y_test = load_data(.8, .2)
	clf = find_best_clf(X_train, y_train, X_test, y_test)
	real_score = classify(clf, X_test, y_test)
	print(real_score)
	left_X, left_y = X_train[y_train == 0], y_train[y_train == 0]
	right_X, right_y = X_train[y_train == 1], y_train[y_train == 1]
	sizes = [200, 150, 100, 50]
	left_GAN = train_GAN(left_X, left_y)
	right_GAN = train_GAN(right_X, right_y)
	scores = []

	for s in sizes:

		left_samples = left_GAN.generate_data(s).reshape((s,90))
		right_samples = right_GAN.generate_data(s).reshape((s,90))
		GAN_data = np.append(right_samples, left_samples, axis=0)
		GAN_labels = np.append(np.ones(s), np.zeros(s))

		GANX = np.append(GAN_data, X_train, axis=0)
		GANY = np.append(GAN_labels, y_train)
		GANX, GANY = shuffle(GANX, GANY)

		clf_aug = find_best_clf(GANX, GANY, X_test, y_test)
		score = classify(clf_aug, X_test, y_test)

		print("Augmented data:", score)
		scores.append(score)

	scores.append(real_score)
	sizes.append("No synthetic data")
	plt.title('Comparing Different ratios of real vs synthetic data during training')
	plt.plot(sizes, scores, 'r--')
	plt.show()




def train_GAN(X_train, y_train):
		gan = GAN.GAN((X_train, y_train), g_in=X_train.shape[1], g_hid=X_train.shape[1], g_out=X_train.shape[1],
					   d_in=X_train.shape[1], d_hid=X_train.shape[1], d_out=1)
		gan.train(1000)
		return gan

  


main()
