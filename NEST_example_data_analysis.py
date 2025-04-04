import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

from NEST.Dropout_Neural_Network import Dropout_Neural_Network


def compute_mu_sigma(train_in):
	mu = torch.mean(train_in, dim=0)
	sigma = torch.std(train_in, dim=0, unbiased=True)
	
	return mu, sigma


def analyse_results(data_dir):
	# Load training data
	train_in = torch.from_numpy(np.load(f"{data_dir}/train_in.npy"))
	
	# Define input space for plots
	num_grid_samples = 200
	x_range = np.linspace(-10, 10, num_grid_samples)
	y_range = np.linspace(-10, 10, num_grid_samples)
	(X, Y) = np.meshgrid(x_range, y_range)
	
	G = torch.from_numpy(np.stack((X, Y), axis=-1))
	
	# Compute standardisation values
	mu, sigma = compute_mu_sigma(train_in)
	
	# Load neural network
	num_dims = 2
	asymptote = 0.0
	lapse = 0.0
	w = [256, 128, 32]
	p = 0.1
	
	layers = [num_dims] +  w + [1]
	
	nn_params = {"dropout_probability": p, "lapse": lapse, \
		"asymptote": asymptote}\
		
	neural_net = Dropout_Neural_Network(layers, **nn_params)
	nn_filename = f"{data_dir}/neural_net.pth"
	neural_net.load_state_dict(torch.load(nn_filename))
	
	# Evaluate on grid
	neural_net.set_double()
	neural_net.set_predict()
	with torch.no_grad():
		G_out = neural_net((G - mu) / sigma)	
	
	# Visualise result
	fig, ax = plt.subplots()
	ax.imshow(G_out, extent=[-10, 10, -10, 10])
	
	plt.show()
	
if __name__ == "__main__":
	data_dir = os.path.expanduser(sys.argv[1])
	analyse_results(data_dir)
