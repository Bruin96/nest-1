import numpy as np
import torch
import functorch as ft
import time
import scipy
import math

def d_ReLU(x):
	out = torch.ones(x.size())
	out[x < 0] = 0.0
	
	return out


def compute_NTK_single(neural_net, A, B, fnet=None, params=None):
	NTK_time = time.time()
	
	if fnet is None or params is None:
		detached_params = {k: v.detach() for k, v in \
			neural_net.named_parameters()}
		fnet, params = torch.func.functional_call(neural_net, \
			detached_params, A)
		
	def fnet_single(params, x):
		return torch.func.functional_call(neural_net, params, \
			x.unsqueeze(0)).squeeze(0)
		
	jacrev = torch.func.jacrev(fnet_single, argnums=0)
		
	jac_A = jacrev(params, A)
	jac_B = jacrev(params, B)
	
	jac_A = [ten.reshape(ten.size(0), -1) for ten in jac_A.values()]
	jac_B = [ten.reshape(ten.size(0), -1) for ten in jac_B.values()]
	
	NTK = torch.stack([torch.einsum('Nf,Mf->NM', j1, j2) \
		for j1, j2 in zip(jac_A, jac_B)])
	NTK = NTK.sum(0)
			
	return NTK


def compute_NTK(neural_net, A, B, A_out, B_out, loss_func, fnet=None, params=None, jac_A=None, jac_B=None):	
	NTK = torch.stack([torch.einsum('ij, kj -> ik', j1, j2) / j1.size(1) for j1, j2 in zip(jac_A, jac_B)])
	
	num_params = sum([torch.numel(jac) for jac in jac_A]) / jac_A[0].size(0)
	
	NTK = 1.0/np.sqrt(NTK.size(1)*NTK.size(2)) * NTK.sum(0)# + 1e-30

	return NTK
