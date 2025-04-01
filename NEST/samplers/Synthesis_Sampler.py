


from samplers.Random_Heat_One_Zero_Sampler import Random_Heat_One_Zero_Sampler
from samplers.Random_One_Zero_Sampler import Random_One_Zero_Sampler

from utils.heatmap import compute_heat_parzen
from utils.uncertainty_functions import compute_std_torch, compute_grad

from utils.multiprocessing_functions.Max_Grad_Computer import Max_Grad_Computer
from utils.multiprocessing_functions.Optimisation_Computer import Optimisation_Computer
from utils.multiprocessing_functions.Max_Heat_Computer import Max_Heat_Computer
from utils.multiprocessing_functions.Max_Stddev_Computer import Max_Stddev_Computer
from utils.multiprocessing_functions.Lookahead_Computer import Lookahead_Computer

import numpy as np
import time
import torch
import random
import scipy.optimize as sp_opt
import threading
import multiprocessing
import queue
import copy


class Synthesis_Sampler(Random_Heat_One_Zero_Sampler):    
	def __init__(self, dist, **params):
		super().__init__(dist, **params)
		
		self.queue = queue.Queue()
		self.num_starts = 12
		
		self.max_grad_computer = Max_Grad_Computer()
		self.optimisation_computer = Optimisation_Computer()
		self.max_heat_computer = Max_Heat_Computer()
		self.max_stddev_computer = Max_Stddev_Computer()
		self.lookahead_computer = Lookahead_Computer()
		
		self.neural_net = None
		self.random_chance = 0.8
		self.a = 1.0
		self.b = 1.0
		self.c = 1.0
		self.d = 0.0
		self.p_acqf = 0.1
		self.num_trials = 20
		self.max_grad = 1.0
		self.h = 0.25
		
		self.set_params(**params)
		
		
	def set_params(self, **params):
		super().set_params(**params)
		
		for (key, value) in params.items():
			if key == 'neural_net':
				self.neural_net = value
				self.neural_net_zero = copy.deepcopy(value)
			if key == 'random_chance':
				self.random_chance = value
			if key == 'num_trials':
				self.num_trials = value
			if key == 'a':
				self.a = value
			if key == 'b':
				self.b = value
			if key == 'c':
				self.c = value
			if key == 'd':
				self.d = value
			if key == 'h':
				self.h = value
			if key == 'p_acqf':
				self.p_acqf = value
				
	
	# Generates a sample at the boundary of the decision space for the
	# current neural network
	def sample(self, random_chance=0.5, train_in=None, train_out=None, \
			mu=None, sigma=None, skip=False, from_super=False):
				
		is_random = False
		seed = int(int(time.time()*1e9)%1e9)
		random_state = np.random.default_rng()
		tau = random_state.uniform()
				
		if not from_super and (self.neural_net is None or skip or \
				tau <= random_chance):
			sample = super().sample(train_in, mu, sigma)
			is_random = True		
		else:
			if train_out is not None and self.d > 0.0:
				sample_tensor = self.synthesise_sample_lookahead(\
												train_in, train_out, mu, sigma)
			else:
				sample_tensor = self.synthesise_sample(train_in, \
											mu, sigma, train_out=train_out)
			
			# Reformat sample to list
			sample = []
			for i in range(self.num_dims):
				sample.append(sample_tensor[i].detach().item())
	
			if self.distribution is not None:
				out = self.distribution.compute(sample)
				
				sample.append(out)
		
		return (sample, is_random)
	
	# Find the location of maximum gradient by using backpropagation to search the space
	def compute_max_grad(self, mu, sigma):	
		max_time = 0.25
						
		max_grad = self.max_grad_computer.run(mu, sigma, copy.deepcopy(self.neural_net), \
				self.lbs, self.ubs, self.num_dims, self.num_starts, \
				max_time=max_time)

		return max_grad
		
		
	def compute_max_heat(self, mu, sigma, train_in, h):
		max_time = 0.25
		
		max_heat = self.max_heat_computer.run(mu, sigma, train_in, self.h, \
			self.lbs, self.ubs, self.num_dims, self.num_starts, max_time)
		
		return max_heat
		
		
	def compute_max_stddev(self, mu, sigma):
		max_time = 0.25
						
		max_std = self.max_stddev_computer.run(mu, sigma, copy.deepcopy(self.neural_net), \
				self.lbs, self.ubs, self.num_dims, self.num_trials, self.num_starts, max_time, \
				self.p_acqf)

		return max_std
		
		
	def synthesise_sample_lookahead(self, train_in, train_out, mu, sigma):
		num_samples = 500
		
		# Compute maximum gradient for later normalisation of the gradient value
		if self.a != 0.0:
			max_grad = self.compute_max_grad(mu, sigma)
		else:
			max_grad = 1.0	
		
		# Compute the maximum heat for later normalisation of the heat value
		if self.b != 0.0:
			max_heat = self.compute_max_heat(mu, sigma, train_in, self.h)
		else:
			max_heat = 1.0
		
		# Compute the maximum stddev for later normalisation of the stddev value
		if self.c != 0.0:
			max_std = self.compute_max_stddev(mu, sigma)
		else:
			max_std = 1.0
		
		# Compute the lookahead acquisition result		
		sample = self.lookahead_computer.run(mu, sigma, train_in, train_out, \
			self.neural_net, self.neural_net_zero, self.lbs, self.ubs, \
			self.num_dims, num_samples, self.num_starts, max_heat, max_grad, \
			max_std, a=self.a, b=self.b, c=self.c, d=self.d, \
			p_acqf=self.p_acqf, num_trials=self.num_trials, max_time=2.0)
			
		return sample

	
	def synthesise_sample(self, train_in=None, mu=None, sigma=None, train_out=None):
		sample = None
				
		# Define transformed upper and lower bounds
		transformed_lbs = ((torch.tensor(self.lbs)-mu)/sigma).tolist()
		transformed_ubs = ((torch.tensor(self.ubs)-mu)/sigma).tolist()
		
		# Define seed
		t = int(int(time.time()*1e9)%(1e9))
		
		# Compute maximum gradient for later normalisation of the gradient value
		max_grad = self.compute_max_grad(mu, sigma)	
		
		# Compute the maximum heat for later normalisation of the heat value
		max_heat = self.compute_max_heat(mu, sigma, train_in, self.h)
		
		# Compute the maximum stddev for later normalisation of the stddev value
		if self.c != 0.0:
			max_std = self.compute_max_stddev(mu, sigma)
		else:
			max_std = 1.0
				
		# Launch optimisation worker threads
		max_time = 2.0
		sample = self.optimisation_computer.run(mu, sigma, train_in, self.h, \
			max_grad, max_heat, max_std, self.a, self.b, self.c, copy.deepcopy(self.neural_net), \
			self.lbs, self.ubs, self.num_trials, self.num_dims, self.num_starts, max_time, \
			self.p_acqf)
		
		return sample
