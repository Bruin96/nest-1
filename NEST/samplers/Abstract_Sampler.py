'''
	Author: Sjoerd Bruin
	Date of last modification: 19-09-2022
	
	Class description:
	This class implements an abstract class for sampling from a given
	function. The function is taken as a parameter, and the sampler
	subsequently allows for sampling by calling the method 'sample()',
	which is implemented in the concrete realisations of this class.
'''

from abc import ABC, abstractmethod
import numpy as np

class Abstract_Sampler(ABC):
	
	def __init__(self, dist, **params):
		self.distribution = dist
		self.neural_net = None
		
		self.set_params(**params)
			
			
	def set_distribution(self, dist):
		self.distribution = dist
		
		
	def set_neural_net(self, neural_net):
		self.neural_net = neural_net
			
	@abstractmethod
	def set_params(self, **params):
		pass
		
		
	@abstractmethod
	def sample(self):
		pass
