import data_utils
import numpy as np
from sklearn.model_selection import train_test_split
import GAN

keep_channels=['C3']
trial_len = 1.5
X, y = data_utils.get_data("../data/CLASubjectB1512153StLRHand.mat", trial_len, keep_channels)

X = X[y != 3]
y = y[y != 3]
# 0 is left hand
y[y == 1] = 0
# 1 is right hand
y[y == 2] = 1

interval_len = .45
X = data_utils.trim_intervals(X, .15, interval_len)

num_channels= len(keep_channels)
d2 = np.ceil(num_channels * interval_len / 0.005).astype(int)
X = X.reshape(642, d2)


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

gan = GAN.GAN((X,y), g_in = X.shape[1], g_hid = X.shape[1], g_out = X.shape[1], d_in = X.shape[1], d_hid = X.shape[1], d_out = 1)
gan.train(1000)
