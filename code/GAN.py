"""
Vanilla GAN
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class gen(nn.Module):

	def __init__(self, in_size, out_size, hid_size, activ_func):
		super(gen,self).__init__()
		self.in_size = in_size
		self.out_size = out_size
		self.hid_size = hid_size
		self.layer1 = nn.Linear(in_size, hid_size)

		self.layer2 = nn.Linear(hid_size, hid_size)
		self.layer3 = nn.Linear(hid_size, out_size)
		self.activ_func = activ_func
		self.make_network()

	def make_network(self):
		"""
		Here we can play with network architecture
		ex:

		self.net = nn.Sequential(
						self.layer1,
		                nn.BatchNorm1d(self.hid_size),
						nn.ReLU(),
						self.layer2,
						nn.BatchNorm1d(self.hid_size)
						nn.ReLU(),
						self.layer3
		                         )
		"""

		self.net = nn.Sequential(self.layer1,
		                         self.layer2,
		                         self.layer3)

	def forward(self, x):
		x = self.net(x)
		return x

class disc(nn.Module):
	def __init__(self, in_size, out_size, hid_size, activ_func):
		super(disc,self).__init__()
		self.in_size = in_size
		self.out_size = out_size
		self.hid_size = hid_size
		self.layer1 = nn.Linear(in_size, hid_size)

		self.layer2 = nn.Linear(hid_size, hid_size)
		self.layer3 = nn.Linear(hid_size, out_size)
		self.activ_func = activ_func
		self.make_network()

	def make_network(self):
		"""
		Here we can play with network architecture
		ex:

		self.net = nn.Sequential(
						self.layer1,
		                nn.BatchNorm1d(self.hid_size),
						nn.ReLU(),
						self.layer2,
						nn.BatchNorm1d(self.hid_size)
						nn.ReLU(),
						self.layer3
		                         )
		"""

		self.net = nn.Sequential(self.layer1,
		                         self.layer2,
		                         self.layer3)


	def forward(self, x):
		x = self.net(x)
		return x

class GAN(object):
	def __init__(self, D, G, d_learning_rate, g_learning_rate):
		self.D = D
		self.G = G
		self.D_optim = optim.SGD(self.D.parameters(), lr=d_learning_rate)
		self.G_optim = optim.SGD(self.G.parameters(), lr=g_learning_rate)
		self.crit = nn.BCELoss()

	def train(self, data, epochs, d_step, g_step):

		g_input_size = 1  # Random noise dimension coming into generator, per output vector
		g_hidden_size = 5  # Generator complexity
		g_output_size = 1  # Size of generated output vector
		d_input_size = 500  # Minibatch size - cardinality of distributions
		d_hidden_size = 10  # Discriminator complexity
		d_output_size = 1  # Single dimension for 'real' vs. 'fake' classification
		# https://github.com/devnag/pytorch-generative-adversarial-networks/blob/master/gan_pytorch.py

		Gen = gen(g_input_size,g_hidden_size,g_output_size,None)
		Disc = disc(d_input_size,d_hidden_size,d_output_size,None)


		for e in range(epochs):
			self.G.train()
			for d in d_step:
				self.D_optim.zero_grad()
				self.D.train()




		#https://github.com/znxlwm/pytorch-generative-model-collections/blob/master/GAN.py
