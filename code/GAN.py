"""
Vanilla GAN
"""
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class gen(nn.Module):

	def __init__(self, in_size, hid_size, out_size):
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
		x = x.float()
		x = self.net(x)
		x = torch.sigmoid(x)
		return x

class discriminator(nn.Module):
	def __init__(self, in_size, hid_size, out_size):
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
		x = x.float()
		x = self.net(x)
		x = torch.sigmoid(x)
		return x

class GAN(object):
	def __init__(self, data, g_in, g_hid, g_out, d_in, d_hid, d_out):

		plt.plot()
		#self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data[0], data[1], train_size=0.8, test_size=0.2)
		self.X = data[0]
		self.y = data[1]
		self.num_features = self.X.shape[1]
		self.full_synth = True
		self.g_input_size = g_in
		self.g_hidden_size = g_hid
		self.g_output_size = g_out
		self.d_input_size = d_in
		self.d_hidden_size = d_hid
		self.d_output_size = d_out
		self.d_learning_rate = 1e-3
		self.g_learning_rate = 1e-3
		self.batch_size = 100


	def noise(self, size):
		"""
		Generators noise vector
		"""
		return Variable(torch.randn(size, self.num_features))

	def ones_and_zeros(self, size):
		"""
		Tensors containing ones and zeros
		"""
		ones = Variable(torch.ones(size, 1))
		zeros = Variable(torch.zeros(size, 1))

		return ones,zeros


	def train_disc(self, real, fake):
		N = real.size(0)

		self.D_optim.zero_grad()
		ones, zeros = self.ones_and_zeros(N)

		real_pred = self.D.forward(real)

		# changed error_real to loss_real and fake_err to loss_fake
		# for clarity
		loss_real = self.loss(real_pred, ones)
		loss_real.backward()	    
		fake_pred = self.D.forward(fake)

		loss_fake = self.loss(fake_pred, zeros)
		loss_fake.backward()

		self.D_optim.step()

		return loss_real + loss_fake, real_pred, fake_pred

	def train_generator(self,fake):
		N = fake.size(0)

		self.G_optim.zero_grad()
		ones, zeros = self.ones_and_zeros(N)
		prediction = self.D.forward(fake)
		loss = self.loss(prediction, ones)
		loss.backward()
		self.G_optim.step()

		return loss

	def plot(self, g, d):
		plt.title('Loss')
		plt.plot(g,'b--',label='Generative')
		plt.plot( d, 'r--', label='Discriminative')
		plt.legend()
		plt.show()
	
	def train(self, epochs):
		if self.full_synth == True:

			self.G = gen(self.g_input_size, self.g_hidden_size, self.g_output_size)
			self.D = discriminator(self.d_input_size, self.d_hidden_size, self.d_output_size)

			self.D_optim = optim.SGD(self.D.parameters(), lr=self.d_learning_rate)
			self.G_optim = optim.SGD(self.G.parameters(), lr=self.g_learning_rate)
			self.loss = nn.BCELoss()

			g_err = []
			d_err = []

			for epoch in range(epochs):
# 				print("epoch {} / {}".format(epoch, epochs))
				for n in range(0, len(self.X), self.batch_size):

					batch = torch.from_numpy(self.X[n:n + self.batch_size, :])
					N = batch.shape[0]

					# Train Discriminator
					real = Variable(batch)

					noise = self.noise(N)
					fake = self.G.forward(noise).detach()

					d_error, d_pred_r, d_pred_f = self.train_disc(real, fake)

					# Train Generator
					fake = self.G.forward(self.noise(N))
					g_error = self.train_generator(fake)


				g_err.append(g_error.item())
				d_err.append(d_error.item())
			self.plot(g_err,d_err)

	def generate_data(self, num_samples):
		# working on getting this into the right form (currently outputing a list of tensors)
		samples = []

		for n in range(num_samples):
			samples.append(self.G.forward(self.noise(1)).detach().numpy())

		return np.asarray(samples)
