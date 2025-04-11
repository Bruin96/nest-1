import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

from NEST.Dropout_Neural_Network import Dropout_Neural_Network
from NEST.example_functions.Barten_CSF import Barten_CSF



def evaluate_threshold(x_range):
	# Define the CSF
	param_dict = {'e_g': 3.3, 'sigma_0_base': 1.5, 'k': 2.3, 'eta': 0.04, \
		'self.M_factor': 0.05, 'r_e': 7.633}
	
	csf = Barten_CSF(param_dict)
	
	# Determine where the threshold lies along the second dimension
	thres = np.log10(csf.evaluate_CSF(10**x_range, e=5.0, L=500, \
										p=1.22e6, N_eyes=2, X_0=2.0))

	return thres


def compute_mu_sigma(train_in):
	mu = torch.mean(train_in, dim=0)
	sigma = torch.std(train_in, dim=0, unbiased=True)
	
	return mu, sigma


def analyse_results(data_dir):
	# Load training data
	train_in = torch.from_numpy(np.load(f"{data_dir}/train_in.npy"))
	
	# Define input space for plots
	num_grid_samples = 200
	x_range = np.linspace(-1, 2, num_grid_samples)
	y_range = np.linspace(0, 2.5, num_grid_samples)
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
		
		
	# Evaluate the reference CSF function
	thres = evaluate_threshold(x_range)
	
	# Visualise result:
	# - Red line: threshold of the underlying function
	# - colormap: probability of detection (blue = 0.0, yellow = 1.0)
	fig, ax = plt.subplots()
	ax.imshow(G_out, origin="lower", extent=[-1, 2, 0, 2.5])
	
	ax.plot(x_range, thres, color="tomato")
	ax.set_ylim(bottom=0, top=2.5)
	
	plt.show()
	
if __name__ == "__main__":
	data_dir = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else "./Results"
	analyse_results(data_dir)
