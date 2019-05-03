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
		self.layer3 = nn.Linear(hid_size, hid_size)
		self.layer4 = nn.Linear(hid_size, hid_size)
		self.layer5 = nn.Linear(hid_size, out_size)
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
                                         nn.LeakyReLU(),
					 self.layer2,
                                         nn.LeakyReLU(),
					 self.layer3,
                                         nn.LeakyReLU(),
					 self.layer4,
                                         nn.LeakyReLU(),
		                         self.layer5)

	def forward(self, x):
		x = x.float()
		x = self.net(x)
# 		x = torch.sigmoid(x)
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

# 		self.net = nn.Sequential(self.layer1,
# 		                         self.layer2,
# 		                         self.layer3)
# 		self.net = nn.Sequential(self.layer1,
#                                          nn.LeakyReLU(),
# 		                         self.layer2,
#                                          nn.LeakyReLU(),
# 		                         self.layer3)
		self.net = nn.Sequential(self.layer1,
                                         nn.LeakyReLU(),
					 nn.Dropout(p=0.6),
					 self.layer2,
                                         nn.LeakyReLU(),
					 nn.Dropout(p=0.6),
		                         self.layer3)
# 		self.net = nn.Sequential(self.layer1,
#                                          nn.LeakyReLU(),
# 					 nn.Dropout(p=0.6),
# 					 self.layer2,
#                                          nn.LeakyReLU(),
# 					 nn.Dropout(p=0.6),
# 					 self.layer3,
#                                          nn.LeakyReLU(),
# 					 nn.Dropout(p=0.6),
# 		                         self.layer4)


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
		self.d_learning_rate = 1e-2
		self.g_learning_rate = 1e-2
		self.batch_size = 100
		self.converge_exp = False
		self.samples = []


	def noise(self, size):
		"""
		Generators noise vector
		"""

		return Variable(torch.randn(size, self.g_input_size))

	def ones_and_zeros(self, size):
		"""
		Tensors containing ones and zeros
		"""
		ones = Variable(torch.ones(size, 1))
		zeros = Variable(torch.zeros(size, 1))

		return ones,zeros
	
	def soft_labels(self, size):
		ones = Variable(torch.rand(size, 1) * 0.1 + 0.9)
		zeros = Variable(torch.rand(size, 1) * 0.1)
		return ones, zeros

	def soft_labels_2(self, size):
		ones = Variable(torch.ones(size, 1) * 0.9)
		zeros = Variable(torch.ones(size, 1) * 0.1)
		return ones, zeros

	def plot_window(self, x, y, epoch):
		fig, ax = plt.subplots()
		ax.plot(x, y, color="red")
		plt.title("Mean Synthetic Data at Epoch {}".format(epoch)) 
		plt.xlabel("Time (sec)")
		plt.ylabel("ERP")
		plt.show()

	def progress(self, syn_data, epoch):
		avg_syn = np.mean(syn_data, axis=0)
		seconds = np.linspace(.15, .15 + .45, np.ceil(.45 / 0.005).astype(int))
		self.plot_window(seconds, avg_syn, epoch)

	def train_disc(self, real, fake):
		N = real.size(0)
		self.D_optim.zero_grad()
# 		ones, zeros = self.ones_and_zeros(N)
# 		ones, zeros = self.soft_labels(N)
		ones, zeros = self.soft_labels_2(N)
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
		plt.title('Loss: Training size' + str(self.X.shape[0]))
		plt.plot(d, 'r--', label='Discriminative')
		plt.plot(g,'b--',label='Generative')
		plt.legend()
		plt.show()
	
	def train(self, epochs, display_progress=False, d_learning_rate=1e-2, g_learning_rate=1e-2):
		if self.full_synth == True:
			self.d_learning_rate = d_learning_rate
			self.g_learning_rate = g_learning_rate
			
			self.G = gen(self.g_input_size, self.g_hidden_size, self.g_output_size)
			self.D = discriminator(self.d_input_size, self.d_hidden_size, self.d_output_size)

			self.D_optim = optim.Adam(self.D.parameters(), lr=self.d_learning_rate)
			self.G_optim = optim.SGD(self.G.parameters(), lr=self.g_learning_rate)
			self.loss = nn.BCELoss()

			static_noise = self.noise(300)

			g_err = []
			d_err = []

			# epoch is one time seeing all data
			for epoch in range(epochs):

				# print("epoch {} / {}".format(epoch, epochs))
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

				if display_progress and (epoch % 1000 == 0):
					test_samples = self.G.forward(static_noise).detach().numpy()
					self.progress(test_samples, epoch)
					#self.plot(g_err, d_err)

				if self.converge_exp == True and np.abs(g_error.item() - d_error.item()) < 1e-3 and len(self.samples) < 5 and epoch > 2000:
					self.samples.append(self.generate_data(100))

			#self.plot(g_err, d_err)

			if self.converge_exp == True: return np.asarray(self.samples)


	def generate_data(self, num_samples):
		# working on getting this into the right form (currently outputing a list of tensors)
		samples = []

		for n in range(num_samples):
			samples.append(self.G.forward(self.noise(1)).detach().numpy())

		return np.asarray(samples)
