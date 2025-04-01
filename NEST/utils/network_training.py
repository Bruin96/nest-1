'''
	Author: Sjoerd Bruin
	Project: NeuralQUEST
	
	File Description:
	This file contains the training loop used in NeuralQUEST to update
	the neural network used in the algorithm.
'''

import copy
import math
import time
import numpy as np
import torch
from scipy.ndimage.filters import gaussian_filter
import scipy.optimize as sp_opt

from Neural_Network import Neural_Network
from utils.Fisher_information import compute_Fisher_information
from utils.uncertainty_functions import compute_std
from utils.blue_noise import generate_blue_noise


# Trains the network for the given number of epochs
def train_network_no_test(neural_net, optimiser, scheduler, train_in, train_out,\
		num_epochs, lbs, ubs, return_fisher=True, gamma=1e-3, \
		use_GPU=False, eta_noise=0.03, weights=None):
	train_proportion = 1.0
	
	num_dims = len(lbs)
	
	if weights is None:
		weights = [1.0 for i in range(train_in.size(0))]
		
	weights_tensor = torch.tensor(weights)
	if use_GPU: 
		weights_tensor = weights_tensor.to("cuda")
		
	weights_sum = torch.sum(weights_tensor)

	# Define the loss functions
	kl_loss = torch.nn.KLDivLoss()
	mse_loss = torch.nn.MSELoss()
	mae_loss = torch.nn.L1Loss()
	bce_loss = torch.nn.BCELoss()
	bce_loss_ind = torch.nn.BCELoss(reduction="none")
	huber_loss = torch.nn.HuberLoss()
		
	# Cast to double
	neural_net.set_double()
	train_in = train_in.double()
	train_out = train_out.double()
		
	if train_out.dim() == 1:
		train_out = train_out.unsqueeze(1)
		
	mse_losses = []
	mae_losses = []
	
	L_k = 0.0
	
	lr_0 = scheduler.get_last_lr()[0]
	
	if use_GPU:
		neural_net = neural_net.to('cuda')
	
	# Execute training loop
	num_pos = torch.count_nonzero(train_out)
		
	for i in range(num_epochs):				
		train_loss = 0.0
		test_loss = 0.0
		
		lr = scheduler.get_last_lr()[0]

		# Train network
		neural_net.set_train()
					
		# Inject noise into training data
		train_in_curr = train_in.clone() + compute_noise(lbs, ubs, \
												eta_noise, train_in)

		# Shuffle training data			
		perm = torch.randperm(train_in_curr.size()[0])
		train_in_curr = train_in_curr[perm]
		train_out_curr = train_out[perm]
		weights_curr = weights_tensor[perm]
		
		if use_GPU:
			train_in_curr = train_in_curr.to('cuda')
			train_out_curr = train_out_curr.to('cuda')
	
		# Predict results
		prediction = neural_net(train_in_curr[0:math.ceil(train_proportion*train_in_curr.size(0))])
		
		if prediction.dim() == 1:
			prediction = prediction.unsqueeze(0)
	
		# Compute loss
		if train_out.dim() == 1:
			train_out = train_out.unsqueeze(1)

		loss_error = (bce_loss_ind(prediction.squeeze(), \
			train_out_curr.squeeze()) * weights_curr/weights_sum).sum()
		
		# Compute Fisher information
		def unit_loss(input_y, target_y):
			return torch.sum(torch.log(input_y))
			
		step_size = 4	
			
		if return_fisher and i%step_size == 0:
			Fisher_energy_time = time.time()
			neural_net.set_train_without_dropout()
			trace_I = compute_Fisher_information(copy.deepcopy(neural_net), train_in_curr.detach().clone(), \
				train_out_curr.detach().clone(), bce_loss, use_GPU=False)
								
			neural_net.set_train()
				
			# Compute Fisher energy
			l_k = step_size*lr/(lr_0*train_out_curr.size(0)) * torch.sqrt(trace_I)
			L_k += l_k
			
			Fisher_energy_time = time.time() - Fisher_energy_time

		loss = loss_error

		train_loss = loss.detach().item()
	
		# Backpropagation		
		optimiser.zero_grad()
		loss.backward()
		optimiser.step()
		
		# Update learning rate					
		scheduler.step()
		
	if use_GPU:
		neural_net = neural_net.to('cpu')
						
	if return_fisher:	
		return (train_loss, L_k)
	else:
		return train_loss


# Trains the network for the given number of epochs
def train_network(neural_net, optimiser, scheduler, train_in, train_out,\
		test_in, test_out, num_epochs, lbs, ubs, s, mu=None, sigma=None, \
		return_fisher=True, gamma=1e-3, data_dir='../Data', \
		learn_lapse=False, eta_noise=0.1, use_GPU=False, print_test_pred=True, \
		weights=None):
			
	device = 'cuda' if use_GPU else 'cpu'
	
	num_dims = len(lbs)
	
	train_proportion = 1.0
	
	if weights is None:
		weights = [1.0**(train_in.size(0) - i - 1) for i in range(train_in.size(0))]

	# Define the loss functions
	kl_loss = torch.nn.KLDivLoss()
	mse_loss = torch.nn.MSELoss()
	mae_loss = torch.nn.L1Loss()
	bce_loss = torch.nn.BCELoss(weight=torch.tensor(weights))#.double()
	bce_loss_ind = torch.nn.BCELoss(reduction='none')
	huber_loss = torch.nn.HuberLoss()
	nll_gauss_loss = torch.nn.GaussianNLLLoss()
	
	# Cast to double
	neural_net.set_double()
	train_in = train_in.double()
	train_out = train_out.double()
	test_in = test_in.double()
	test_out = test_out.double()
			
	if train_out.dim() == 1:
		train_out = train_out.unsqueeze(1)
		
	mse_losses = []
	mae_losses = []
	train_losses = []
	
	L_k = 0.0
	
	lr_0 = scheduler.get_last_lr()[0]
	
	if shrink_net:
		neural_net.reset_mask()
		
	# Derive lapse from heatmap
	if learn_lapse:
		eta_lapse = 0.15*torch.max(ubs - lbs).item()
		max_value = find_expected_max_value(train_in, train_out, \
			lbs, ubs, eta_lapse)
			
		# Adjust lapse rate to reflect the max value
		neural_net.lapse.data = torch.tensor([1.0-max_value])
	
	rmse_test_losses = torch.zeros(num_epochs+1)
	
	if use_GPU:
		neural_net = neural_net.to('cuda')
		test_in = test_in.to('cuda')
		test_out = test_out.to('cuda')
		train_in = train_in.to('cuda')
		train_out = train_out.to('cuda')
	
	num_pos = torch.count_nonzero(train_out)
	
	train_loss = 0.0
		
	with torch.no_grad():
		neural_net.set_predict()
		rmse_test_losses[0] = torch.sqrt(mse_loss(neural_net(test_in).squeeze(), test_out.squeeze()))
		neural_net.set_train()
	
	for i in range(num_epochs):		
		train_loss = 0.0
		test_loss = 0.0
		
		lr = scheduler.get_last_lr()[0]

		# Train network
		neural_net.set_train()
					
		# Inject noise into training data
		train_in_curr = train_in.clone() + compute_noise(lbs, ubs, \
												eta_noise, train_in)

		# Shuffle training data			
		perm = torch.randperm(train_in_curr.size()[0])
		train_in_curr = train_in_curr[perm, :]
		train_out_curr = train_out[perm]
		weights_curr = weights[perm]
		
		if train_out_curr.dim() == 1:
			train_out_curr = train_out_curr.unsqueeze(1)	
		
		prediction = neural_net(train_in_curr)
		
		if prediction.dim() == 1:
			prediction = prediction.unsqueeze(0)
		
		step_size = 4	

		loss = bce_loss(prediction.squeeze(), train_out_curr.squeeze())
		
		train_loss += loss.detach().item()
		
		# Backpropagation
		optimiser.zero_grad()
		loss.backward()
		optimiser.step()
			
		train_losses.append(train_loss)
			
		if return_fisher and i%step_size == 0:
			neural_net.set_train_without_dropout()
			(trace_I, det_I) = compute_Fisher_information(copy.deepcopy(neural_net), train_in_curr.detach().clone(), \
				train_out_curr.detach().clone(), bce_loss, use_GPU=use_GPU)
				
			neural_net.set_train()
								
			# Compute Fisher energy
			l_k = step_size*lr/(lr_0*train_out_curr.size(0)) * torch.sqrt(trace_I)
			L_k += l_k
		
		# Compute current error
		with torch.no_grad():
			neural_net.set_predict()
			rmse_test_losses[i+1] = torch.sqrt(mse_loss(neural_net(test_in).squeeze(), test_out.squeeze()))
			neural_net.set_train()
		
		# Clamp lapse to range [0, asymp]
		neural_net.lapse.data.clamp_(min=0.0, max=1-neural_net.asymp.detach().item())
		
		# Update learning rate					
		scheduler.step()
		
	if use_GPU:
		neural_net = neural_net.to('cpu')
		test_in = test_in.to('cpu')
		test_out = test_out.to('cpu')
		train_in = train_in.to('cpu')
		train_out = train_out.to('cpu')
	
	if learn_lapse:
		# Unfreeze lapse
		neural_net.lapse.requires_grad = True
		
		print(f"Lapse after regular training loop: {neural_net.lapse}")
	
	with torch.no_grad():
		neural_net.set_predict()
		prediction = neural_net(test_in).squeeze()
		
		std, mean = compute_std(test_in.to('cpu').detach().numpy(), \
			neural_net, num_trials=500, seed=42, use_GPU=False, \
			use_float=False, kernel_size=1, new_p=0.0, all_layers=True)
			
		prediction = torch.tensor(mean)
		
		# Compute MSE and RMSE losses
		loss_mse = mse_loss(prediction.squeeze(), test_out.squeeze())
		mse_loss_out = loss_mse.detach().item()	
		rmse_loss_out = np.sqrt(mse_loss_out)
	
		# Compute MAE loss
		loss_mae = mae_loss(prediction, test_out.squeeze())
		mae_loss_out = loss_mae.detach().item()	
		
		# Compute BCE loss
		loss_bce = bce_loss(prediction, test_out.squeeze())
		bce_loss_out = loss_bce.detach().item()
		
		mae_losses.append(mae_loss_out)
		mse_losses.append(mse_loss_out)
		
	if return_fisher:	
		return (train_loss, mae_loss_out, mse_loss_out, bce_loss_out, rmse_loss_out, L_k)
	else:
		return (train_loss, mae_loss_out, mse_loss_out, bce_loss_out, rmse_loss_out, 0.0)
	

def compute_noise(lbs, ubs, std, means):
	device = 'cuda' if means.is_cuda else 'cpu'
	train_new = torch.zeros(means.size()).to(device)
	num_elems = train_new.size(0)
	
	for i in range(train_new.size(1)):
		train_new[:, i] = torch.normal(torch.zeros(num_elems), std).to(device)
			
	return train_new
	

def find_expected_max_value(train_in, train_out, lbs, ubs, eta_noise):
	train_out_numpy = train_out.detach().numpy()
	train_in_numpy = train_in.detach().numpy()
	train_in_numpy = np.atleast_2d(train_in_numpy)
	
	num_dims = train_in_numpy.shape[1]
	
	# Compute weights based on prevalence (# of +, # of -)
	train_weights = np.ones(train_in_numpy.shape[0])
	num_positive = np.count_nonzero(train_out_numpy)
	num_negative = train_out_numpy.shape[0] - num_positive
	if num_negative == train_out_numpy.shape[0]:
		return 1.0 # Cannot determine a lapse other than 1.0, so block it
	
	elif num_positive == train_out_numpy.shape[0]:
		ratio = 1.0
	else:
		ratio = (num_negative / num_positive)**(1.0/num_dims)
	
	train_weights[train_out_numpy.squeeze() == 0] = 1.0
	train_weights[train_out_numpy.squeeze() == 1] = ratio
	
	train_weights *= train_out_numpy.shape[0] / (np.sum(train_weights))
	
	# Find maximum heat
	def heat_negative(x):
		x = np.atleast_2d(x)
		heat_per_sample = np.exp(-0.5/eta_noise**2 * np.sum(\
			(train_in_numpy - x)**2, axis=1))
		normalisation_value = np.sum(train_weights*heat_per_sample)
		weighted_heat = np.sum(train_weights*heat_per_sample*train_out_numpy.squeeze())
		heat = weighted_heat / normalisation_value
		
		return -heat
	
	# Compute max heat
	x0 = torch.tensor([0.5*(lbs[i]+ubs[i]) for i in range(len(lbs))])
	
	def process(x):
		(x_opt, y_opt, _) = sp_opt.fmin_l_bfgs_b(heat_negative, \
				x0=x, bounds=[(lbs[m], \
				ubs[m]) for m in range(num_dims)], \
				m=50, approx_grad=True, pgtol=1e-10, factr=1e2, \
				maxiter=100, disp=False, maxls=200)
				
		return -y_opt
	
	heat_time = time.time()
	num_samples = 12
	x0 = generate_blue_noise(num_samples, num_dims, lbs.tolist(), \
		ubs.tolist(), m=10)
		
	results = [process(x0[i, :]) for i in range(num_samples)]
	y_opt = -np.amax(np.array(results))
				
	return -y_opt
