

import torch
import numpy as np
import scipy.optimize as sp_opt
import random
import copy
import multiprocessing as mp
import time
from joblib import Parallel, delayed
import os

import psutil

from utils.blue_noise import generate_blue_noise
from utils.heatmap import compute_heat_parzen
from utils.uncertainty_functions import compute_std_single

from utils.multiprocessing_functions.MinimizeStopper import MinimizeStopper, TimeoutReturnError


class Maximisation_Computer:
	def __init__(self, use_GPU=False):
		self.use_GPU = use_GPU
		
		
	def run(self, mu, sigma, train_in, neural_net, lbs, ubs, num_dims, \
									num_starts, max_time, a=1.0, b=1.0, \
									eps_in=1e-8, use_UCB=True, step=0):
		min_loss = 1e20
						
		seed = int(int(time.time()*1e9)%1e9)
		random_state = np.random.default_rng()
		
		neural_net.set_single()
		neural_net.set_predict()
		
		start_time = time.time()
		
		transformed_lbs = ((torch.tensor(lbs)-mu)/sigma).tolist()
		transformed_ubs = ((torch.tensor(ubs)-mu)/sigma).tolist()
		
		h = 0.02**(2.0/num_dims)
		
		# Define optimisation loss function
		if use_UCB:
			def loss_func(x_in):
				x_in = x_in.astype(np.float32)
				kappa = 1.0*0.97**step
				neural_net.set_inference()
				std, mean = compute_std_single(x_in, neural_net, \
									num_trials=100, use_float=True, \
									new_p=neural_net.p)
				
				with torch.no_grad():
					val = neural_net(x_in)
				
				return -(mean + std*kappa)
		else:
			def loss_func(x_in):
				x_in = x_in.astype(np.float32)

				with torch.no_grad():
					val = neural_net(x_in)
					
				heat_val = compute_heat_parzen(x_in, train_in[-5:, :], h, \
					transformed_lbs, transformed_ubs)
				return -a*val + b*heat_val
				
		# Compute how long one evaluation takes
		x_res = torch.tensor([transformed_lbs[m] + np.random.uniform()*\
			(transformed_ubs[m] - transformed_lbs[m]) for m in range(num_dims)])
			
		eval_time = time.time()
				
		_ = loss_func(x_res.detach().numpy())
		
		eval_time = time.time() - eval_time
		n_evals = int(round(max_time/(eval_time+1e-9)))	
		
		def process(x):				
			# Optimise x based on loss function
			try:
				x_opt, y_opt, _ = sp_opt.fmin_l_bfgs_b(loss_func, \
					x0=x.detach().float().numpy(), bounds=[(transformed_lbs[m], \
					transformed_ubs[m]) for m in range(num_dims)], \
					m=50, approx_grad=True, pgtol=1e-12, factr=1e2, \
					maxfun=200000, maxiter=50000, disp=False, maxls=200, 
					epsilon=eps_in, callback=MinimizeStopper(max_time))
			except TimeoutReturnError as err:
				x_opt = err.value
				y_opt = loss_func(x_opt)
								
			return (x_opt, y_opt)
		
		tau = random_state.uniform(0, 1, size=(num_starts, num_dims))	

		x_res = torch.from_numpy(generate_blue_noise(num_starts, num_dims, \
			transformed_lbs, transformed_ubs, m=10)).float()

		num_procs = min(num_starts, psutil.cpu_count(logical = False))
				
		max_time /= num_procs
		results = [process(x_res[i, :]) for i in range(num_procs)]
		
		for (x, y) in results:
			if y < min_loss:
				min_loss = y
				sample = torch.tensor(x)
				
		return sample
