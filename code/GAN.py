"""
Vanilla GAN
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

class gen(nn.Module):

	def __init__(self, in_size, out_size, hid_size):
		super(gen,self).__init__()
		self.in_size = in_size
		self.out_size = out_size
		self.hid_size = hid_size
		self.layer1 = nn.Linear(in_size, hid_size)

		self.layer2 = nn.Linear(hid_size, hid_size)
		self.layer3 = nn.Linear(hid_size, out_size)

		self.make_network()

	def make_network(self):
		"""
		Here we can play with network architecture
		ex:

		self.net = nn.Sequential(
						self.layer1,
		                nn.BatchNorm1d(self.hid_size),
						nn.LeakyReLU(),
						self.layer2,
						nn.BatchNorm1d(self.hid_size)
						nn.LeakyReLU(),
						self.layer3
		                         )
		"""

		self.net = nn.Sequential(self.layer1,
		                         self.layer2,
		                         self.layer3)

	def forward(self, x):
		x = self.net(x)
		return x

class discriminator(nn.Module):
	def __init__(self, in_size, out_size, hid_size):
		super(discriminator,self).__init__()
		self.in_size = in_size
		self.out_size = out_size
		self.hid_size = hid_size
		self.layer1 = nn.Linear(in_size, hid_size)

		self.layer2 = nn.Linear(hid_size, hid_size)
		self.layer3 = nn.Linear(hid_size, out_size)

		self.make_network()

	def make_network(self):
		"""
		Here we can play with network architecture
		ex:

		self.net = nn.Sequential(
						self.layer1,
		                nn.BatchNorm1d(self.hid_size),
						nn.LeakyReLU(),
						self.layer2,
						nn.BatchNorm1d(self.hid_size)
						nn.LeakyReLU(),
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
	def __init__(self, data):

		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data[0], data[1], train_size=0.8, test_size=0.2)
		self.full_synth = True

	def noise(self, size):
		"""
		Generators noise vector
		"""
		return Variable(torch.randn(size, 100))

	def ones_and_zeros(self, size):
		"""
		Tensors containing ones and zeros
		"""
		ones = Variable(torch.ones(size, 1))
		zeros = Variable(torch.zeros(size, 1))

		return ones,zeros


	def train_disc(self, real, fake):
		N = real.size(0)
		# Reset gradients
		self.D_optim.zero_grad()
		ones, zeros = self.ones_and_zeros(N)

		# 1.1 Train on Real Data
		real_pred = self.D.train(real)

		# Calculate error and backpropagate
		error_real = self.loss(real_pred, ones)
		error_real.backward()

		# 1.2 Train on Fake Data
		fake_pred = self.D.train(fake)

		# Calculate error and backpropagate
		fake_err = self.loss(fake_pred, zeros)
		fake_err.backward()

		# 1.3 Update weights with gradients
		self.D_optim.step()

		# Return error and predictions for real and fake inputs
		return error_real + fake_err, real_pred, fake_pred

	def train_generator(self,fake):
		N = fake.size(0)
		# Reset gradients
		self.G_optim.zero_grad()
		ones, zeros = self.ones_and_zeros(N)

		# Sample noise and generate fake data
		prediction = self.D.train(fake)

		# Calculate error and backpropagate
		error = self.loss(prediction, ones)
		error.backward()

		# Update weights with gradients

		self.G_optim.step()

		# Return error
		return error

	def train(self, epochs):

		if self.full_synth == True:
			g_input_size = 1  # Random noise dimension coming into generator, per output vector
			g_hidden_size = 5  # Generator complexity
			g_output_size = 1  # Size of generated output vector
			d_input_size = 100  # Minibatch size - cardinality of distributions
			d_hidden_size = 10  # Discriminator complexity
			d_output_size = 1  # Single dimension for 'real' vs. 'fake' classification
			# https://github.com/devnag/pytorch-generative-adversarial-networks/blob/master/gan_pytorch.py

			d_learning_rate = 1e-3
			g_learning_rate = 1e-3
			batch_size = 100
			start = 0

			noise = self.noise(self.X_train.shape[1])

			self.G = gen(g_input_size, g_hidden_size, g_output_size)
			self.D = discriminator(d_input_size, d_hidden_size, d_output_size)


			self.D_optim = optim.SGD(self.D.parameters(), lr=d_learning_rate)
			self.G_optim = optim.SGD(self.G.parameters(), lr=g_learning_rate)
			self.loss = nn.BCELoss()

			for epoch in range(epochs):
				for n in range(0, len(self.X_train), batch_size):
					batch = torch.from_numpy(self.X_train[n:n + batch_size, :])
					N = batch.shape[0]

					# 1. Train Discriminator
					real = Variable(batch)
					# Generate fake data and detach
					# (so gradients are not calculated for generator)
					fake = self.G.forward(self.noise(N)).detach()
					# Train D
					d_error, d_pred_real, d_pred_fake = self.train_disc(real, fake)

					# 2. Train Generator
					# Generate fake data
					fake = self.G.forward(self.noise(N))
					# Train G
					g_error = self.train_generator(fake)
					# Log batch error

					# Display Progress every few batches



			#https://github.com/znxlwm/pytorch-generative-model-collections/blob/master/GAN.py
