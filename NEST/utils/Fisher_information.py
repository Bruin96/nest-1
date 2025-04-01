'''
	Author: Sjoerd Bruin
	Project: NeuralQUEST
	
	File Description:
	This file contains the necessary functions to compute the Fisher
	information of the input neural net based on the given training data
	and with respect to the given loss function.
'''

import numpy as np
import torch
import functorch as ft
import time
from torch.nn.utils._per_sample_grad import call_for_per_sample_grads


# Computes the Jacobian with respect to the neural network parameters
def compute_jacobian(neural_net, train_in, train_out, loss_func):	
	if train_out.dim() == 1:
		train_out = train_out.unsqueeze(1)
	
	with torch.no_grad():
		loss = loss_func(neural_net(train_in).squeeze(), train_out.squeeze())
		
		jac = neural_net.jacobian_of_parameters(train_in, loss_value=loss)
		
		# Zip jacobian into a single tensor
		out = None
		for mat in jac:
			mat = mat.reshape((mat.size(0), -1))
			if out is None:
				out = mat
			else:
				out = torch.cat((out, mat), dim=1)
			
		jac = out
			
	return jac
	
	
def compute_Fisher_information(neural_net, train_in, train_out, loss_func, use_GPU=False):
	neural_net.set_train_without_dropout()
	
	if use_GPU:
		train_in = train_in.to('cuda')
	
	# Compute Jacobian
	jac = compute_jacobian(neural_net, train_in, train_out, loss_func)
		
	# Compute Fisher information
	I = 1.0/train_out.size(0) * torch.sum(jac**2)
	
	# Reset network to regular training mode
	neural_net.set_train()
		
	return I.to('cpu')
