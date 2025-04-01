'''
	Author: Sjoerd Bruin
	
	File Description:
	
'''

import torch
import numpy as np
import scipy.optimize as sp_opt
import random
import copy
import multiprocessing as mp
import time
from joblib import Parallel, delayed
import psutil
import os

from utils.uncertainty_functions import compute_grad
from utils.blue_noise import generate_blue_noise

from utils.multiprocessing_functions.MinimizeStopper import MinimizeStopper, TimeoutReturnError


class Max_Grad_Computer:
	def __init__(self, use_GPU=False):
		self.use_GPU = use_GPU
		
	
	def set_GPU(self, use_GPU):
		self.use_GPU = use_GPU


	def run(self, mu, sigma, neural_net, lbs, ubs, num_dims, num_starts, max_time=1.0):
		max_grad = -1e20
		
		neural_net.set_single()
		
		seed = int(int(time.time()*1e9)%1e9)
		random_state = np.random.default_rng()
		
		start_time = time.time()
		
		transformed_lbs = ((torch.tensor(lbs)-mu)/sigma).tolist()
		transformed_ubs = ((torch.tensor(ubs)-mu)/sigma).tolist()
		
		def grad_negative(x_in):
			return -compute_grad(x_in, neural_net)
			
		x_res = torch.tensor([transformed_lbs[m] + np.random.uniform()*\
			(transformed_ubs[m] - transformed_lbs[m]) for m in range(num_dims)])
			
		eval_time = time.time()
		_ = grad_negative(x_res.detach().numpy())
		eval_time = time.time() - eval_time
		
		n_evals = int(round(max_time/(eval_time+1e-9)))	
		
		def process(x):				
			# Optimise x based on loss function
			try:
				x_opt, y_opt, _ = sp_opt.fmin_l_bfgs_b(grad_negative, \
					x0=x.detach().numpy(), bounds=[(transformed_lbs[m], \
					transformed_ubs[m]) for m in range(num_dims)], \
					m=10, approx_grad=True, pgtol=1e-7, factr=1e7, \
					maxfun=n_evals, maxiter=50, disp=False, maxls=20, \
	                epsilon=1e-6, callback=MinimizeStopper(max_time))
				
			except TimeoutReturnError as err:
				x_opt = err.value
				y_opt = grad_negative(x_opt)
								
			return (x_opt, -y_opt)
				
		x_res = torch.from_numpy(generate_blue_noise(num_starts, num_dims, \
			transformed_lbs, transformed_ubs, m=10))
				
		num_procs = min(num_starts, psutil.cpu_count(logical = False))
		max_time /= num_procs
		
		results = [process(x_res[i, :]) for i in range(num_procs)]
					
		# Compute best result out of the runs
		for _, y in results:
			if y > max_grad:
				max_grad = y
				
		return max(1e-12, max_grad)
