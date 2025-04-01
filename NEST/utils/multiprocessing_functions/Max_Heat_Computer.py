'''

'''

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

from utils.multiprocessing_functions.MinimizeStopper import MinimizeStopper, TimeoutReturnError


class Max_Heat_Computer:	
	def __init__(self, use_GPU=False):
		self.use_GPU = use_GPU
		
		
	def run(self, mu, sigma, train_in, h, lbs, ubs, num_dims, num_starts, max_time=1.0):
		max_heat = -1e20
		
		transformed_lbs = ((torch.tensor(lbs)-mu)/sigma).tolist()
		transformed_ubs = ((torch.tensor(ubs)-mu)/sigma).tolist()
		
		seed = int(int(time.time()*1e9)%1e9)
		random_state = np.random.default_rng()
				
		heats = np.zeros(num_starts)
		
		def heat_negative(x_in):
			heat = np.nan_to_num(compute_heat_parzen_single(x_in, train_in, h, transformed_lbs, transformed_ubs), \
				nan=0.0, posinf=0.0, neginf=0.0)
			return -heat
			
		# Determine n_evals
		tau = random_state.uniform(0, 1, size=num_dims)
		x0 = torch.tensor([transformed_lbs[m] + tau[m]*\
			(transformed_ubs[m] - transformed_lbs[m]) for m in range(num_dims)])
			
		eval_time = time.time()
		_ = heat_negative(x0.detach().numpy())
		eval_time = time.time() - eval_time
		
		n_evals = n_evals = int(round(max_time / (eval_time+1e-9)))
		
		def process(x):
			try:
				(x_opt, y_opt, _) = sp_opt.fmin_l_bfgs_b(heat_negative, \
					x0=x.detach().numpy(), bounds=[(transformed_lbs[m], \
					transformed_ubs[m]) for m in range(num_dims)], \
					m=50, approx_grad=True, pgtol=1e-7, factr=1e7, \
					maxiter=20, disp=False, maxls=50, epsilon=1e-6,
					callback = MinimizeStopper(max_time))
			except TimeoutReturnError as err:
				x_opt = err.value
				y_opt = heat_negative(x_opt)
				
			return -y_opt
		
		# Launch processes
		tau = random_state.uniform(0, 1, size=(num_starts, num_dims))
				
		x_res = torch.from_numpy(generate_blue_noise(num_starts, num_dims, \
			transformed_lbs, transformed_ubs, m=10))
				
		num_procs = min(num_starts, psutil.cpu_count(logical = False))	
		
		max_time /= num_procs
		heats = [process(x_res[i, :]) for i in range(num_procs)]
		
		max_heat = np.amax(heats)
		
		return max(1e-12, max_heat)		
