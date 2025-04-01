'''
	Author: Sjoerd Bruin
	Project: NeuralQUEST
	
	File Description:
	This file contains functions to initialise the data, depending on the
	input method indicated by the caller.
'''

import numpy as np
import torch
import scipy.optimize as sp_opt
import itertools

from samplers.Random_Sampler import Random_Sampler
from samplers.Random_One_Zero_Sampler import Random_One_Zero_Sampler
from samplers.Synthesis_Sampler import Synthesis_Sampler


def initialise_samplers(sample_method, func, sampler_params):
	true_sampler = Random_Sampler(func, **sampler_params)

	if sample_method == 'random':
		sampler = Random_One_Zero_Sampler(func, **sampler_params)
	elif sample_method == 'synthesis':
		sampler = Synthesis_Sampler(func, **sampler_params)
		
	return (sampler, true_sampler)
