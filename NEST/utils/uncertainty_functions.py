import torch
import numpy as np
import random
import itertools

def compute_std_single(x_input, neural_net, num_trials=500, seed=0, \
		use_GPU=False, use_float=False, new_p=0.1, kernel_size=1):
	# Set random seeds
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
		
	# Cast to float if required
	if use_float:
		if type(x_input) is np.ndarray:
			x_input = x_input.astype(np.float32)
		else:
			x_input = x_input.float()			
			
	# Define last layer evaluation
	weight = neural_net.last_layer.weight.data.detach().numpy()
	bias = neural_net.last_layer.bias.data.detach().numpy()
	lapse = neural_net.lapse.item()
	asymp = neural_net.asymp.item()	
	p = neural_net.p
	
	
	def forward_parallel(x, weight, bias, lapse, asymp):
		y = np.einsum('ab, b -> a', x, weight, optimize=True) + bias

		return 0.5*(1 + asymp - (1-asymp)*lapse) + \
			0.5*(1 - asymp - lapse) * (1 - 2*np.exp(-10**(y)))
	
		
	def forward_dropout_local(x, weight, bias, lapse, asymp, p):	
		dropout_mask = np.random.rand(*x.shape)
		x[dropout_mask < p] = 0.0
		new_bias = np.empty(x.shape[0])
		
		for i in range(x.shape[0]):
			new_bias[i] = bias[0]
			
		y = forward_parallel(x, weight[0, :], new_bias, lapse, asymp)

		return y
	
			
	if kernel_size == 1:
		with torch.no_grad():
			if use_GPU:
				x_input = torch.tensor(x_input).to('cuda')
			
			neural_net.set_predict()
			pred_inter = neural_net.forward_non_dropout(x_input)
			
			p_original = neural_net.p
			neural_net.set_dropout_probability(new_p)
			
			neural_net.set_inference()
						
			vals = np.zeros(num_trials)
						
			pred_inter = np.tile(pred_inter, (num_trials, 1))
			
			prediction = forward_dropout_local(pred_inter, weight, bias, lapse, asymp, p)
			vals = prediction
			
			neural_net.set_dropout_probability(p_original)
							
		mean = np.mean(vals)		
		std = np.std(vals, ddof=1)
		std = np.clip(std, 1e-12, None)
				
	else: # Kernel size is not unity, so average over a kernel
		input_dims = x_input.shape[-1]
		kernel_width = int((kernel_size-1)/2)
		eps = 0.005
		num_random_samples = 100
		
		# Fill x_aug with random samples
		rng_object = np.random.default_rng()
		x_aug = rng_object.uniform(0.0, eps, size=(num_random_samples, \
			x_input.shape[-1])) + x_input
			
		if use_float:
			x_aug = x_aug.astype(np.float32)
			
		vals = np.zeros((num_random_samples, num_trials))
						
		
		with torch.no_grad():	
			neural_net.set_predict()	
			pred_inter = neural_net.forward_non_dropout(x_aug)
						
			p_original = neural_net.p
			neural_net.set_dropout_probability(new_p)
			
			neural_net.set_inference()
									
			val_idx = tuple([slice(0, vals.shape[k]) for k in range(len(vals.shape) \
					- 1)])
				
			for i in range(int(num_trials)):
				# Compute prediction
				if not torch.is_tensor(pred_inter):
					prediction = forward_dropout_local(pred_inter, weight, bias, lapse, asymp, p)
				else:
					prediction = neural_net.forward_dropout(pred_inter, weight, bias, lapse, asymp)

				# Store results
				if use_GPU:
					vals[:, i] = prediction.squeeze().to('cpu').detach().numpy()
				else:
					vals[:, i] = prediction.squeeze()
					
			neural_net.set_dropout_probability(p_original)
		
		# Define Gaussian kernel function	
		def gauss(x, mu=None, sigma=0.5):
			K = x.shape[0]
			if mu is None:
				mu = np.zeros(x.shape[0])

			return np.exp( -np.dot(x-mu, x-mu) / (2*sigma**2) )				
		
		# Compute mean and standard deviation as normal
		mean = np.mean(vals, axis=1)		
		std = np.std(vals, ddof=1, axis=1)
		std = np.clip(std, 1e-12, None)
		
		reduce_dims = tuple([i for i in range(-input_dims, 0)])
		std = np.mean(std, axis=0)
		mean = np.mean(mean, axis=0)
					
	return(std, mean)
	
	
def compute_std_no_kernel(x_input, neural_net, num_trials=500, seed=0, \
		use_GPU=False, use_float=False, new_p=0.3, all_layers=False):
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	
	vals = np.zeros(x_input.shape + (num_trials,))
	
	if type(x_input) is np.ndarray:
		x_input = np.atleast_2d(x_input)
	else:
		if x_input.dim() == 1:
			x_input.unsqueeze(0)
	
	if use_float:
		if type(x_input) is np.ndarray:
			x_input = x_input.astype(np.float32)
		else:
			x_input = x_input.float()
			
	vals = np.zeros(x_input.shape[0:-1] + (num_trials,))
	
	with torch.no_grad():
		if use_GPU:
			x_input = torch.tensor(x_input).to('cuda')
		
		neural_net.set_predict()
		if not all_layers:
			pred_inter = neural_net.forward_non_dropout(x_input)
		
		neural_net.set_inference(all_layers=all_layers)
		
		p_original = neural_net.p
		neural_net.set_dropout_probability(new_p)
		
		for i in range(int(num_trials)):
			neural_net.set_inference(all_layers=all_layers)

			if not all_layers:
				prediction = neural_net.forward_dropout(pred_inter)
			else:
				pred_inter = neural_net.forward_non_dropout(x_input)
				prediction = neural_net.forward_dropout(pred_inter)
			
			# Store results
			if use_GPU:
				vals[tuple([slice(0, vals.shape[k]) for k in range(len(vals.shape) \
					- 1)]) + (i,)] = prediction.squeeze(len(x_input.size()) -1).to('cpu').detach().numpy()
			else:
				vals[tuple([slice(0, vals.shape[k]) for k in range(len(vals.shape) \
					- 1)]) + (i,)] = prediction.squeeze(-1)
						
		neural_net.set_dropout_probability(p_original)
		
	mean = np.mean(vals, axis=-1)		
	std = torch.std(torch.from_numpy(vals), dim=-1, unbiased=True).numpy()
	
	return (std, mean)


def compute_std(x_input, neural_net, num_trials=50, seed=0, use_GPU=False, \
        use_float=False, kernel_size=5, new_p=0.1, all_layers=False):
	if kernel_size == 1:
		return compute_std_no_kernel(x_input, neural_net, num_trials, \
			seed, use_GPU, use_float, new_p=new_p, all_layers=all_layers)
		
	eps = 0.005
	kernel_width = int(np.floor(kernel_size/2))
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	mean = 0.0
	
	input_dims = x_input.shape[-1]
	
	dims = tuple([x_input.shape[i] for i in range(0, len(x_input.shape) - 1)]) + (num_trials,)
		
	# Augment data with 1-ring of elements around input
	aug_dims = tuple([x_input.shape[i] for i in range(len(x_input.shape) - 1)]) + \
		tuple([kernel_size for i in range(x_input.shape[-1])]) + (x_input.shape[-1],)
	if use_float:
		x_aug = np.zeros(aug_dims, dtype=np.float32)
	else:
		x_aug = np.zeros(aug_dims)
	
	dims = tuple([x_aug.shape[i] for i in range(0, \
		len(x_aug.shape) - 1)]) + (num_trials,)
	vals = np.zeros(dims)
	accum_axis = len(vals.shape) - 1
	
	ranges = [range(0, kernel_size) for i in range(x_input.shape[-1])]
	
	for idxs in itertools.product(*ranges):
		epses = eps*(np.array(list(idxs)) - kernel_width)
		x_aug[tuple([slice(0, x_input.shape[k]) for k in \
				range(len(x_input.shape) - 1)]) + idxs + \
				(slice(0, x_input.shape[-1]),)] = x_input + epses 
	
	if use_float:
		if type(x_aug) is np.ndarray:
			x_aug = x_aug.astype(np.float32)
		else:
			x_aug = x_aug.float()
	
	with torch.no_grad():
		if use_GPU:
			x_aug = torch.tensor(x_aug).to('cuda')
		
		if not all_layers:
			pred_inter = neural_net.forward_non_dropout(x_aug)
		
		p_original = neural_net.p
		neural_net.set_dropout_probability(new_p)
		
		neural_net.set_inference(neural_net.set_inference(all_layers=all_layers))
			
		for i in range(int(num_trials)):
			# Compute prediction
			if not all_layers:
				prediction = neural_net.forward_dropout(pred_inter)
			else:
				prediction = neural_net(x_aug)
			
			# Store results
			if use_GPU:
				vals[tuple([slice(0, vals.shape[k]) for k in range(len(vals.shape) \
					- 1)]) + (i,)] = prediction.squeeze(len(x_aug.shape) -1).to('cpu').detach().numpy()
			else:
				vals[tuple([slice(0, vals.shape[k]) for k in range(len(vals.shape) \
					- 1)]) + (i,)] = prediction.squeeze(len(x_aug.shape) -1)
			
		neural_net.set_dropout_probability(p_original)
					
	mean = np.mean(vals, axis=accum_axis)		
	std = np.std(vals, axis=accum_axis, ddof=1)
		
	# Precompute gauss values
	def gauss(x, mu=None, sigma=0.5):
		K = x.shape[0]
		if mu is None:
			mu = np.zeros(x.shape[0])
		return np.exp( -np.dot(x-mu, x-mu) / (2*sigma**2) )
		
	filter_sigma = 2*eps
	
	gauss_vals = np.zeros(tuple([kernel_size for i in range(x_input.shape[-1])]))
	
	for idxs in itertools.product(*ranges):
		epses = eps*(np.array(list(idxs)) - kernel_width)
		gauss_vals[idxs] = gauss(epses, sigma=filter_sigma)
		
	gauss_vals /= np.sum(gauss_vals)
	
	reduce_dims = tuple([i for i in range(-input_dims, 0)])
	std_filtered = np.sum(std * gauss_vals, axis=reduce_dims) #+ 1e-100
				
	return (std_filtered, mean)
	
	
def compute_std_torch(x_input, neural_net, num_trials=50, seed=0, use_GPU=False, use_float=False, kernel_size=5):
	eps = 0.005
	kernel_width = max(1, int(np.floor(kernel_size/2)))
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	mean = 0.0
	
	if kernel_size == 1:
		return compute_std_no_kernel(x_input, neural_net, num_trials, \
			seed, use_GPU, use_float)
	
	input_dims = x_input.size(-1)
	
	if x_input.dim() == 1:
		x_input.unsqueeze(0)
	
	dims = tuple([x_input.shape[i] for i in range(0, len(x_input.size()) - 1)]) \
		+ (num_trials,)
	
	# Augment data with ring of elements around input
	aug_dims = tuple([x_input.size(i) for i in range(len(x_input.size()) - 1)]) + \
		tuple([kernel_size for i in range(x_input.size(-1))]) + (x_input.size(-1),)
	x_aug = torch.zeros(aug_dims)
	
	dims = tuple([x_aug.size(i) for i in range(0, \
		len(x_aug.size()) - 1)]) + (num_trials,)
	vals = torch.zeros(dims)
	accum_axis = len(vals.size()) - 1
	
	ranges = [range(0, kernel_size) for i in range(x_input.size(-1))]
	
	for idxs in itertools.product(*ranges):
		epses = eps*(torch.tensor(list(idxs)) - kernel_width)
		x_aug[tuple([slice(0, x_input.size(k)) for k in \
				range(len(x_input.size()) - 1)]) + idxs + \
				(slice(0, x_input.size(-1)),)] = x_aug[tuple([slice(0, \
				x_input.size(k)) for k in range(len(x_input.size()) - 1)]) \
				+ idxs + (slice(0, x_input.size(-1)),)] + x_input + epses 
	
	neural_net.set_inference()
	
	if use_float:
		x_aug = x_aug.float()
	else: 
		x_aug = x_aug.double()
		
	(x_aug, aug_dims) = linearise_batch(x_aug)
	
	if use_GPU:
		x_aug = x_aug.to('cuda')

	pred_inter = neural_net.forward_non_dropout(x_aug)

	for i in range(int(num_trials)):
		# Compute prediction
		prediction = neural_net.forward_dropout(pred_inter)
		prediction = delinearise_batch(prediction, aug_dims)
		
		# Store results
		if use_GPU:
			vals[tuple([slice(0, vals.shape[k]) for k in range(len(vals.shape) \
				- 1)]) + (i,)] = prediction.squeeze(len(x_aug.size()) -1).to('cpu').detach().numpy()
		else:
			vals[tuple([slice(0, vals.shape[k]) for k in range(len(vals.shape) \
				- 1)]) + (i,)] = prediction
					
	#vals_np = np.array(vals)
	mean = torch.mean(vals, axis=accum_axis)		
	std = torch.std(vals, dim=accum_axis, unbiased=True) 
			
	# Precompute gauss values
	def gauss(x, mu=None, sigma=0.5):
		K = x.size(0)
		if mu is None:
			mu = torch.zeros(x.size(0))

		return torch.exp( -torch.dot(x-mu, x-mu) / (2*sigma**2) )
		
	filter_sigma = 2*eps
	
	gauss_vals = torch.zeros(tuple([kernel_size for i in range(x_input.size(-1))]))
	
	for idxs in itertools.product(*ranges):
		epses = eps*(torch.tensor(list(idxs)) - kernel_width)
		gauss_vals[idxs] = gauss(epses, sigma=filter_sigma)
		
	gauss_vals /= torch.sum(gauss_vals)
		
	# Apply filter to std results
	std_filtered = torch.zeros(tuple([x_input.size(i) for i in range(0, \
					len(x_input.size()) - 1)]), requires_grad=True)
					
	reduce_dims = tuple([i for i in range(-input_dims, 0)])
						
	std_filtered = torch.sum(std * gauss_vals, dim=reduce_dims)
				
	return (std_filtered, mean)

def linearise_batch(x):
	num_samples = 1
	dims = x.size()
	for i in range(len(dims) - 1):
		num_samples *= dims[i]
		
	return (torch.reshape(x, (num_samples, -1)), dims[0:-1])
	
def delinearise_batch(y, dims):
	return torch.reshape(y, tuple(dims))
	

def compute_grad(x, neural_net, use_GPU=False):
	if torch.is_tensor(x):
		x_tensor = x.float()
	else:
		x_tensor = torch.from_numpy(x).float()
	if use_GPU:
		neural_net.to('cuda')
		x_tensor = x_tensor.to('cuda')
	x_tensor.requires_grad = True
	
	single_dim = False
	if x_tensor.dim() == 1:
		single_dim = True
	
	if single_dim:
		y = neural_net(x_tensor)
		grads = torch.autograd.grad(y, x_tensor, allow_unused=True)[0]
	else:
		grads = []
		neural_net.zero_grad()
		y = neural_net(x_tensor)
		y.backward(torch.ones_like(y))

	if use_GPU:
		neural_net.to('cpu')
		return max(1e-12, torch.sqrt(torch.sum(grads**2)).to('cpu').detach().numpy())
	
	if single_dim:
		return max(1e-12, torch.sqrt(torch.sum(grads**2)))
	else:
		res = max(1e-12, torch.sqrt(torch.sum(x_tensor.grad**2), axis=1))
		return res
	
