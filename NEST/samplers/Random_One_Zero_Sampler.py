'''
	Author: Sjoerd Bruin
	Date of last modification: 19-09-2022
	
	Class description: This class creates a concrete realisation of the
	Abstract_Sampler class by implementing random sampling on the input
	distribution.
'''

from samplers.Random_Sampler import Random_Sampler
import numpy as np
from numpy.random import default_rng
import time


class Random_One_Zero_Sampler(Random_Sampler):
	
	def __init__(self, dist, **params):
		super().__init__(dist, **params)
					
	
	def sample(self):
		seed = int(int(time.time()*1e9)%1e9)
		random_state = np.random.default_rng()
		
		# Generate a random input sample
		inputs = []
		for i in range(self.num_dims):
			lb = self.lbs[i]
			ub = self.ubs[i]
			
			inputs.append(self.rng.random() * (ub-lb) + lb)
			
		# Compute probability of a 1.0, and simulate outcome
		outputs = inputs
		
		if self.distribution is not None:
			probability = self.distribution.compute(inputs)
			tau = random_state.uniform()
			
			if probability >= tau:
				outputs.append(1.0)
			else:
				outputs.append(0.0)
				
		return outputs
