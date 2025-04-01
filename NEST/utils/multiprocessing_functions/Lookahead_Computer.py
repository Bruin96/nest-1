import torch
import numpy as np
import scipy.optimize as sp_opt
import random
import copy
import multiprocessing as mp
import time
from joblib import Parallel, delayed
import os
import functorch as ft
import scipy
import math

import psutil

from utils.heatmap import compute_heat_parzen, compute_heat_parzen_single
from utils.blue_noise import generate_blue_noise, generate_next_sample
from utils.uncertainty_functions import compute_std, compute_std_single, compute_std_torch, compute_grad
from utils.NTK import compute_NTK
from utils.uncertainty_functions import compute_std_single

from utils.multiprocessing_functions.MinimizeStopper import MinimizeStopper, TimeoutReturnError


class Lookahead_Computer:
	def __init__(self, use_GPU=False):
			self.use_GPU = use_GPU		
			self.bce_loss = torch.nn.L1Loss()
			
			
	def run(self, mu, sigma, X, Y, neural_net, neural_net_zero, lbs, ubs, \
			num_dims, num_samples, num_starts, max_heat, \
			max_grad, max_std, a=0.0, b=0.0, c=0.0, d=1.0, \
			max_time=3.0, p_acqf=0.1, num_trials=100):
																		
		# Set neural net to single
		neural_net.set_single()
		neural_net.set_predict()
				
		X = X.float()
		Y = Y.float()

				
		# Transform lbs and ubs
		transformed_lbs = ((torch.tensor(lbs)-mu)/sigma).tolist()
		transformed_ubs = ((torch.tensor(ubs)-mu)/sigma).tolist()	
		
		lbs_np = np.array(transformed_lbs)	
		ubs_np = np.array(transformed_ubs)
		
		
		# Scale train_in to [0, 1] range
		eps=1e-30
		scaled_X = np.clip((X.numpy()-lbs_np) / (ubs_np-lbs_np), eps, 1-eps)
	
		# Generate num_samples potential samples
		sobol = torch.quasirandom.SobolEngine(num_dims, seed=int((time.time()*1e9)%1e9), scramble=True)
		U = (ubs_np-lbs_np)*sobol.draw(num_samples).numpy() + lbs_np
	
		
		# Pre-compute NTKs
		jac_X = neural_net.jacobian_of_parameters(X)
		jac_X = [ten.view(ten.size(0), -1) for ten in jac_X]
		jac_U = neural_net.jacobian_of_parameters(torch.from_numpy(U).float())
		jac_U = [ten.view(ten.size(0), -1) for ten in jac_U]
		
		NTK_XX = compute_NTK(neural_net, X, X, None, None, None, jac_A=jac_X, jac_B=jac_X)
		NTK_XU = compute_NTK(neural_net, X, torch.from_numpy(U).float(), \
			None, None, None, jac_A=jac_X, jac_B=jac_U)
		NTK_XX_inv = torch.linalg.pinv(NTK_XX)
		
		# Compute predictions for X
		with torch.no_grad():
			predictions_U = neural_net(torch.from_numpy(U).float())
			predictions_X = neural_net(X.float())
			
		# Initialise rng
		rng = np.random.default_rng()	
		
		val = neural_net.asymp + (0.5 + 0.3*rng.uniform(-1, 1))\
											* (1 - neural_net.asymp)
		z = torch.tensor([val], dtype=torch.float32)
		z_np = z.detach().numpy()
				
		# Define MSE position loss function
		def mse_func(x_in):
			y = neural_net(x_in.astype(np.float32))
			return (y - z_np)**2
			
		# Define loss function
		def loss_func(x):
			# Compute heat
			if b != 0.0:
				heat = compute_heat_parzen_single(x, X.numpy(), 0.25, transformed_lbs, transformed_ubs)
			else:
				heat = 1.0
			
			# Compute grad
			if a != 0.0:
				grad = compute_grad(x, neural_net)
			else:
				grad = 1.0
			
			# Compute std
			std_time = time.time()
			if c != 0.0:
				(std, _) = compute_std_single(x, neural_net, \
					num_trials=int(num_trials), seed=42, use_GPU=False, \
						use_float=True, kernel_size=1, new_p=p_acqf)
			else:
				std = 1.0
						
			# Compute lookahead
			eps=1e-30
			
			if np.isnan(x).any():
				print(f"input value is NaN. Aborting.")
				exit(1)
			
			x_prime = torch.from_numpy(x).float()
			
			# Compute neural tangent kernel segments
			jac_xprime = neural_net.jacobian_of_parameters(x_prime.unsqueeze(0))
			jac_xprime = [ten.view(ten.size(0), -1) for ten in jac_xprime]
			
			NTK_Xxprime = compute_NTK(neural_net, X, \
				x_prime.unsqueeze(0), \
				None, None, None, jac_A=jac_X, jac_B=jac_xprime)
				
			NTK_xprimeU = compute_NTK(neural_net, torch.from_numpy(U).float(), \
				x_prime.unsqueeze(0), None, None, None, jac_A=jac_U, jac_B=jac_xprime)
			
			NTK_xx = compute_NTK(neural_net, x_prime.unsqueeze(0), \
				x_prime.unsqueeze(0), None, None, None, jac_A=jac_xprime, jac_B=jac_xprime)
							
			# Compute (x_prime, y_prime)
			with torch.no_grad():
				y_prime = neural_net(x_prime).squeeze()
					
			# Assess loss function	
			val = 0.0
			loop_time = time.time()
			v = NTK_XX @ NTK_Xxprime
			u = NTK_xx - NTK_Xxprime.T @ v + eps
			
			asymp_mod_val = (y_prime - neural_net.asymp)/(1-neural_net.asymp)
						
			f_plus = 1/u * (NTK_XU.T @ v - NTK_xprimeU) * \
				(v.T @ (Y-predictions_X) - (1.0 - asymp_mod_val))

			f_min = 1/u * (NTK_XU.T @ v - NTK_xprimeU) * \
				(v.T @ (Y-predictions_X) - (0.0 - asymp_mod_val))
			
			val_plus = torch.mean(torch.abs(f_plus))
			val_min = torch.mean(torch.abs(f_min))			
			val = min(val_plus, val_min)
									
			return np.nan_to_num(- ( (max(1e-30, val.detach().numpy()))**d * \
				(max(1e-30, grad/max_grad))**a * (max(1e-30, 1.0-heat/max_heat))**b * \
				(max(1e-30, std/max_std))**c)**(1/(a+b+c+d+1e-30)), nan=1e20, \
				posinf=1e20, neginf=-1e20)
		
		# Define multiple-run process
		def process(x):
			x, y_opt, _ = sp_opt.fmin_l_bfgs_b(mse_func, \
				x0=x.detach().numpy(), bounds=[(transformed_lbs[m], \
				transformed_ubs[m]) for m in range(num_dims)], \
				m=10, approx_grad=True, pgtol=1e-12, factr=1e2, \
				maxiter=1000, disp=False, maxls=20, epsilon=1e-4)
						
			try:			
				x_opt, y_opt, _ = sp_opt.fmin_l_bfgs_b(loss_func, \
					x0=x, bounds=[(transformed_lbs[m], \
					transformed_ubs[m]) for m in range(num_dims)], \
					m=20, approx_grad=True, pgtol=1e-12, factr=1e2, \
					maxfun=200000, maxiter=50, disp=False, maxls=50, 
					epsilon=1e-4, callback=MinimizeStopper(max_time))
			except TimeoutReturnError as err:
				x_opt = err.value
				y_opt = loss_func(x_opt)
								
			return (x_opt, y_opt)
		
		# Compute initial starting points
		opt_time = time.time()
		
		x_res = torch.from_numpy(generate_blue_noise(num_starts, num_dims, \
			transformed_lbs, transformed_ubs, m=10)).float()
			
		# Run processes
		num_procs = int(min(num_starts, psutil.cpu_count(logical = False)))
		max_time /= num_procs
		results = [process(x_res[i, :]) for i in range(num_procs)]
		
		min_loss = 1e20
		x_max = None
		for (x, y) in results:
			if y < min_loss:
				min_loss = y
				x_max = torch.tensor(x)
		
		# Set neural_net to double again
		neural_net.set_double()
		neural_net_zero.set_double()
						
		return x_max
