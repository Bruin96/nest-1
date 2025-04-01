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

from utils.heatmap import compute_heat_parzen, compute_heat_parzen_single
from utils.blue_noise import generate_blue_noise
from utils.uncertainty_functions import compute_std, compute_std_single, compute_std_torch, compute_grad

from utils.multiprocessing_functions.MinimizeStopper import MinimizeStopper, TimeoutReturnError



class Optimisation_Computer:
	def __init__(self, use_GPU=False):
		self.queue = mp.Manager().Queue()
		self.use_GPU = use_GPU
				
		
	def run(self, mu, sigma, train_in, h, max_grad, max_heat, max_std, \
			a, b, c, neural_net, lbs, ubs, num_trials, num_dims, num_starts, max_time, p_acqf):
		min_loss = 1e20
				
		seed = int(int(time.time()*1e9)%1e9)
		random_state = np.random.default_rng()
		
		neural_net.set_single()
		
		start_time = time.time()
		
		transformed_lbs = ((torch.tensor(lbs)-mu)/sigma).tolist()
		transformed_ubs = ((torch.tensor(ubs)-mu)/sigma).tolist()
		
		# Initialise target probability randomly
		val = neural_net.asymp + (0.5 + 0.3*random_state.uniform(-1, 1))\
											* (1 - neural_net.asymp)
		z = torch.tensor([val], dtype=torch.float64)
		z_np = z.detach().numpy()
		
		h_arr = np.array([h*(ubs[i]-lbs[i]) for i in range(len(lbs))])
		
		# Define MSE position loss function
		def mse_func(x_in):
			y = neural_net(torch.tensor(x_in).float())
			return np.mean((y.to('cpu').detach().numpy() - z_np)**2)
		
		# Define optimisation loss function
		def loss_func(x_in):	
			# Determine uncertainty
			std_time = time.time()
			if c != 0.0:
				(std, _) = compute_std_single(x_in, neural_net, \
					num_trials=int(num_trials), seed=42, use_GPU = False, \
					use_float=True, kernel_size=1, new_p=p_acqf)
			else:
				std = 1.0

			# Determine gradient
			neural_net.set_train_without_dropout()
			grad = compute_grad(x_in, neural_net, use_GPU=False)
				
			# Determine heat
			heat = np.nan_to_num(compute_heat_parzen_single(x_in, \
					train_in.detach(), h, transformed_lbs, transformed_ubs), nan=0.0, posinf=1e20, neginf=0.0)
			
			val = np.nan_to_num(1 - ( (max(1e-30, grad/max_grad))**a * (max(1e-30, 1.0-heat/max_heat))**b \
				* (max(1e-30, std/max_std))**c)**(1/(a+b+c+1e-30)), nan=1e20, posinf=1e20, neginf=-1e20)	
									
			return val
				
		# Compute how long one evaluation takes
		x_res = torch.tensor([transformed_lbs[m] + np.random.uniform()*\
			(transformed_ubs[m] - transformed_lbs[m]) for m in range(num_dims)])
			
		eval_time = time.time()
		
		_ = loss_func(x_res.detach().numpy())

		eval_time = time.time() - eval_time
		n_evals = int(round(max_time/(eval_time+1e-9)))	
				
		def process(x):			
			proc_time = time.time()
			
			x_time = torch.tensor([transformed_lbs[m] + np.random.uniform()*\
				(transformed_ubs[m] - transformed_lbs[m]) for m in range(num_dims)])
			
			eval_time = time.time()
			
			_ = loss_func(x_time.detach().numpy())
	
			eval_time = time.time() - eval_time
			n_evals_process = int(round(max_time/(eval_time+1e-9)))	
			
			# Optimise x towards position
			x_opt, y_opt, _ = sp_opt.fmin_l_bfgs_b(mse_func, \
				x0=x.detach().numpy(), bounds=[(transformed_lbs[m], \
				transformed_ubs[m]) for m in range(num_dims)], \
				m=50, approx_grad=True, pgtol=1e-12, factr=1e2, \
				maxiter=100, disp=False, maxls=200, epsilon=1e-6, \
				callback=self.MinimizePrinter())
								
			x = torch.tensor(x_opt)
							
			# Optimise x based on loss function
			try:
				x_opt, y_opt, _ = sp_opt.fmin_l_bfgs_b(loss_func, \
					x0=x.detach().numpy(), bounds=[(transformed_lbs[m], \
					transformed_ubs[m]) for m in range(num_dims)], \
					m=10, approx_grad=True, pgtol=1e-12, factr=1e2, \
					maxfun=200000, maxiter=50, disp=False, maxls=50, 
					epsilon=1e-4, callback=MinimizeStopper(max_time))
			except TimeoutReturnError as err:
				x_opt = err.value
				y_opt = loss_func(x_opt)
				
			return (x_opt, y_opt)
		
		tau = random_state.uniform(0, 1, size=(num_starts, num_dims))	

		x_res = torch.from_numpy(generate_blue_noise(num_starts, num_dims, \
			transformed_lbs, transformed_ubs, m=10))
			
		
		# Initialise half of the samples around the previous selected trial		
		num_procs = min(num_starts, psutil.cpu_count(logical = False))
		
		max_time /= num_procs
		results = [process(x_res[i, :]) for i in range(num_procs)]
				
		for (x, y) in results:
			if y < min_loss:
				min_loss = y
				sample = torch.tensor(x)
				
		return sample
		
