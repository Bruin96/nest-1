'''
	Author: Sjoerd Bruin
	Date of last modification: 19-09-2022
	
	Class description: This class creates a concrete realisation of the
	Abstract_Sampler class by implementing random sampling on the input
	distribution.
'''

from samplers.Abstract_Sampler import Abstract_Sampler
import numpy as np
from numpy.random import default_rng
import time

class Random_Sampler(Abstract_Sampler):
	
	def __init__(self, dist, **params):
		super().__init__(dist, **params)
		
		self.seed = None
		self.sigma = 0.1
		self.num_dims = 1
		self.lbs = [0.0]
		self.ubs=[1.0]
		self.set_params(**params)
		
		self.rng = default_rng(self.seed)
		
	
	def set_params(self, **params):
		
		for (key, value) in params.items():
			if key == 'seed':
				self.seed = value
			if key == 'std':
				self.sigma = value
			if key == 'lower_bounds':
				self.lbs = value
				self.num_dims = len(self.lbs)
			if key == 'upper_bounds':
				self.ubs = value
				self.num_dims = len(self.ubs)
			
	
	def sample(self):
		# Generate a random input sample
		inputs = []
		
		random_state = np.random.default_rng()
		tau = random_state.uniform(0, 1, size=self.num_dims)
		
		for i in range(self.num_dims):
			lb = self.lbs[i]
			ub = self.ubs[i]
			
			inputs.append(tau[i] * (ub-lb) + lb)
			
		# Compute and return corresponding output sample
		outputs = [a for a in inputs]
		if self.distribution is not None:
			outputs.append(self.distribution.compute(inputs))
			
		return outputs
