'''
	Author: Sjoerd Bruin
	Date of last modification: 19-09-2022
	
	Class description: This class creates a concrete realisation of the
	Abstract_Sampler class by implementing random sampling on the input
	distribution with a probability distribution based on the normalised
	heat distribution.
'''

from samplers.Random_Sampler import Random_Sampler
import numpy as np
from numpy.random import default_rng
import time

from utils.blue_noise import generate_next_sample


class Random_Heat_Sampler(Random_Sampler):
	def __init__(self, dist, **params):
		super().__init__(dist, **params)
		self.h = 0.25
				
		self.set_params(**params)
		
	def set_params(self, **params):
		for (key, value) in params.items():
			if key == 'h':
				self.h = value
			if key == 'lower_bounds':
				self.lbs = value
			if key == 'upper_bounds':
				self.ubs = value
				
		super().set_params(**params)
		
		
	def sample(self, train_in=None, mu=None, sigma=None):
		if train_in is None or mu is None or sigma is None:
			return super().sample() # Do random sampling
			
		seed = int(int(time.time()*1e9)%1e9)
		random_state = np.random.default_rng()
		
		lbs_np = np.array(self.lbs)
		ubs_np = np.array(self.ubs)
		
		train_in_curr = (train_in.detach() * sigma + mu).numpy()
		train_in_curr = (train_in_curr - lbs_np) / (ubs_np -lbs_np)
		
		eps = 1e-12
		train_in_curr = np.clip(train_in_curr, 0.0 + eps, 1.0 - eps)
		
		inputs = generate_next_sample(train_in_curr, lbs_np, ubs_np)
		inputs = inputs.tolist()
		
		# Compute and return corresponding output sample
		outputs = [a for a in inputs]
		if self.distribution is not None:
			outputs.append(self.distribution.compute(inputs))
				
		return outputs
	
