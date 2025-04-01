import torch
import numpy as np
import scipy.optimize as sp_opt
import random
import copy
import multiprocessing as mp
import time
import os

import psutil

from utils.uncertainty_functions import compute_std, compute_std_single, compute_std_torch
from utils.blue_noise import generate_blue_noise

from utils.multiprocessing_functions.MinimizeStopper import MinimizeStopper, TimeoutReturnError

from joblib import Parallel, delayed


class Max_Stddev_Computer:	
	def __init__(self, use_GPU=False):
		self.queue = mp.Manager().Queue()
		self.use_GPU = use_GPU
		
	
	def set_GPU(self, use_GPU):
		self.use_GPU = use_GPU
		
		
	def run(self, mu, sigma, neural_net, lbs, ubs, num_dims, num_trials, \
							num_starts, max_time, p_acqf):
		max_std = -1e20
		
		std_seed = 42
		
		neural_net.set_single()
		
		stds_res = np.zeros(num_starts)
		
		seed = int(int(time.time()*1e9)%1e9)
		random_state = np.random.default_rng()
	
		start_time = time.time()
				
		transformed_lbs = ((torch.tensor(lbs)-mu)/sigma).tolist()
		transformed_ubs = ((torch.tensor(ubs)-mu)/sigma).tolist()
		
		# Compute how long one run takes
		x_res = torch.tensor([transformed_lbs[m] + random_state.uniform()*\
			(transformed_ubs[m] - transformed_lbs[m]) for m in range(num_dims)])
			
		# Define function
		def stddev_negative(x_in):
			stddev_negative_time = time.time()
			(std_value, _) = compute_std_single(x_in, neural_net, \
				num_trials=num_trials, seed=std_seed, use_GPU=False, \
				use_float=True, kernel_size=1, new_p=p_acqf)

			return -std_value

		tau = random_state.uniform(0, 1, size=(num_starts, num_dims))
		
		def process(x):
			proc_time = time.time()
			
			func_start_time = time.time()
			
			_ = stddev_negative(x.detach().numpy())

			func_time = time.time() - func_start_time
			n_evals = int(round(max_time / (func_time+1e-9)))

			try:
				x_opt, final_stds, _ = sp_opt.fmin_l_bfgs_b(stddev_negative, \
					x0=x.detach().numpy(), bounds=[(transformed_lbs[m], \
					transformed_ubs[m]) for m in range(num_dims)], \
					m=10, approx_grad=True, pgtol=1e-7, factr=1e7, \
					maxfun=n_evals, maxiter=100, disp=False, maxls=2, \
	                epsilon=1e-6, callback=MinimizeStopper(max_time))
			except TimeoutReturnError as err:
				x_opt = err.value
				final_stds = stddev_negative(x_opt)
								
			return -float(final_stds)
				
		x_res = torch.from_numpy(generate_blue_noise(num_starts, num_dims, \
			transformed_lbs, transformed_ubs, m=10))
				
		num_procs = min(num_starts, psutil.cpu_count(logical = False))	
		
		max_time /= num_procs
		stds_res = [process(x_res[i, :]) for i in range(num_procs)]
		
		max_std = np.amax(stds_res)
		
		return max(1e-12, max_std)
	
