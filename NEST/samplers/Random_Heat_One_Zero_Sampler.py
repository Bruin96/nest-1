'''
	Author: Sjoerd Bruin
	Date of last modification: 19-09-2022
	
	Class description: This class creates a concrete realisation of the
	Abstract_Sampler class by implementing random sampling on the input
	distribution.
'''

from samplers.Random_Heat_Sampler import Random_Heat_Sampler
import numpy as np
from numpy.random import default_rng
import time


class Random_Heat_One_Zero_Sampler(Random_Heat_Sampler):
	
	def __init__(self, dist, **params):
		super().__init__(dist, **params)
		
		self.sigma = 0.0 # Override any attempts to inject noise
		self.random_state = np.random.default_rng()
			
	
	def sample(self, train_in=None, mu=None, sigma=None):
		outputs = super().sample(train_in, mu, sigma)
		if self.distribution is not None:
			outputs = outputs[0:len(outputs)-1]
			
		
		if self.distribution is not None:
			probability = self.distribution.compute(outputs)
			tau = self.random_state.uniform()
		
			if probability >= tau:
				outputs.append(1.0)
			else:
				outputs.append(0.0)
		
		return outputs
