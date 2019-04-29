# nn.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data_preprocess import *

class TwoLayerNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.layer1 = nn.Linear(D_in, H)
        self.layer2 = nn.Linear(H, H)
        self.layer3 = nn.Linear(H, D_out)
        self.make_network()
    def make_network(self):
        self.net = nn.Sequential(self.layer1,
                                    nn.LeakyReLU(),
                                    self.layer2, 
                                    nn.LeakyReLU(),
                                    self.layer3)

    def forward(self, x):
        y_pred = self.net(x)
        y_pred = torch.sigmoid(y_pred)
        return y_pred
    def train(self, X, y):
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        loss_curve = []
        for t in range(1000):
            optimizer.zero_grad()
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = self.forward(X)

            # Compute and print loss
            loss = criterion(y_pred, y)
            if t % 100 == 0:
                #print(t, loss.item())
                loss_curve.append(loss.item())

            # Zero gradients, perform a backward pass, and update the weights.
            
            loss.backward()
            optimizer.step()
        plt.plot(loss_curve)
        plt.show()
    def decision_boundary(self, y_pred):
        y_pred[y_pred>0.5] = 1
        y_pred[y_pred<0.5] = 0
        return y_pred
    def predict(self, X, y):
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
        y_pred = self.forward(X)
        y_pred = self.decision_boundary(y_pred)
        print(y_pred)
        print(y)

        print(np.float32(y_pred == y).mean())


d = data_loader("../data/CLASubjectB1512153StLRHand.mat")
#d.normalize()
X_train, X_test, y_train, y_test = d.test_train_split(0.8, 0.2)

'''
X1 = torch.randn(1000, 50)
X2 = torch.randn(1000, 50) + 10
X = torch.cat([X1, X2], dim=0)
Y1 = torch.zeros(1000, 1)
Y2 = torch.ones(1000, 1)
Y = torch.cat([Y1, Y2], dim=0)
'''
nn = TwoLayerNet(X_train.shape[1], 90, 1)
nn.train(X_train, y_train)
nn.predict(X_train, y_train)