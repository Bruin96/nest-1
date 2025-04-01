
import torch
import numpy as np

def compute_heat_parzen_single(x, train_samples, h, lbs, ubs):
	train_samples_2d = np.atleast_2d(train_samples)
	x_2d = np.atleast_2d(x)
	
	K = train_samples_2d.shape[1]
	const = 1.0/np.sqrt(2*np.pi)**K
	
	diff = np.abs(train_samples_2d - x_2d)
	heat = const * np.sum(np.exp(-0.5/h**2 * np.sum((diff)**2, axis=1)))
	
	return heat / (train_samples_2d.shape[0]*h**K)


# Compute the heat based on Gaussian Parzen windows	
def compute_heat_parzen(x, train_samples, h, lbs, ubs):
	heat = 0.0
	train_samples_2d = np.atleast_2d(train_samples)
	K = train_samples_2d.shape[1]
	
	for i in range(train_samples_2d.shape[0]):
		heat = heat + kernel(x, train_samples_2d[i, :], h, lbs, ubs)
	
	
	if np.isscalar(h):
		heat = heat / (train_samples_2d.shape[0]*h**K)
	else:
		heat = heat / (train_samples_2d.shape[0]*np.sum(h))

	return heat
	
# Compute the kernel function value in response to a position x based on
# the current sample under consideration
def kernel(x, sample, h, lbs, ubs):
	x_2d = np.atleast_2d(x)
	sample = np.atleast_2d(sample)
	
	diff = np.abs(sample - x_2d)
	
	if np.isscalar(h):
		return 1.0 / (np.sqrt(2*np.pi))**(x_2d.shape[1]) * np.exp(-0.5 / h**2 * np.sum(np.square(diff), axis=1))
	else:
		return 1.0 / (np.sqrt(2*np.pi))**(x_2d.shape[1]) * np.exp(-0.5 * np.sum((diff) * (1/h**2) * (diff), axis=1))
