'''

'''

import torch
import numpy as np
import time

from NEST.Neural_Network import Neural_Network


class Dropout_Neural_Network(Neural_Network):

	def __init__(self, layers, learn_lapse=False, **params):
		self.num_dropout_layers = 0
		self.p = 0.2
		self.logistic_layer = None
		super().__init__(layers, learn_lapse=learn_lapse, **params)
		self.learn_lapse = learn_lapse
		
		#self.input_params = input_params
		self.set_params(**params)
		self.layers = layers
				
		
	def set_params(self, **params):
		super().set_params(**params)
		for (key, value) in params.items():
			if key == 'dropout_probability':
				self.p = value
				
				
	def set_dropout_probability(self, new_p):
		for module in self.first_part.modules():
			if isinstance(module, torch.nn.Dropout):
				module.p = new_p
				
		for module in self.second_part.modules():
			if isinstance(module, torch.nn.Dropout):
				module.p = new_p
				
	
	def set_predict(self):
		self.eval()
		self.first_part.eval()
		self.second_part.eval()

		for module in self.first_part.modules():
			module.eval()	
			
		for module in self.second_part.modules():
			module.eval()	
		
    			
	def set_inference(self, all_layers=False):
		self.eval()
		self.first_part.eval()
		self.second_part.eval()
				
		i = 0
		for module in self.first_part.modules():
			if all_layers and i == 0 and isinstance(module, torch.nn.Dropout):
				i += 1
				module.train()
			else:
				module.eval()
		
		for module in self.second_part.modules():
			if not all_layers and isinstance(module, torch.nn.Dropout):
				module.train()
			else:
				module.eval()
	
    	
	def set_double(self):
		self.double()
		self.first_part.double()
		self.second_part.double()
		
		for module in self.first_part.modules():
			module = module.double()
			
		for module in self.second_part.modules():
			module = module.double()
			
			
	def set_single(self):
		self.float()
		self.first_part.float()
		self.second_part.float()
		
		for module in self.first_part.modules():
			module = module.float()
			
		for module in self.second_part.modules():
			module = module.float()
	
    			
	def set_train(self):
		self.train()
		self.first_part.train()
		self.second_part.train()
		for module in self.first_part.modules():
			module.train()
			
		for module in self.second_part.modules():
			module.train()

	
	
	def set_train_without_dropout(self):
		self.train()
		self.first_part.train()
		self.second_part.train()
		for module in self.first_part.modules():
			if not isinstance(module, torch.nn.Dropout):
				module.train()
			else:
				module.eval()
				
		for module in self.second_part.modules():
			if not isinstance(module, torch.nn.Dropout):
				module.train()
			else:
				module.eval()
				
				
	# This function shrinks the current network weights and perturbs them randomly
	def shrink_and_perturb_weights(self, shrink_factor=0.5, std=0.02):
		
		for (name, param) in self.named_parameters():
			if '.gamma' in name or '.T' in name:
				random_vals = np.random.normal(0.0, std)
				param.data = shrink_factor*param.data + (1-shrink_factor)*\
					self.logistic_layer.gamma_base + torch.tensor(\
					random_vals).to('cuda' if param.data.is_cuda else 'cpu')
			if 'weight' in name or 'bias' in name:
				sizes = tuple(list(param.size()))
				random_vals = np.random.normal(0.0, std, sizes)
				param.data = shrink_factor*param.data + torch.tensor(\
				random_vals).to('cuda' if param.data.is_cuda else 'cpu')
			
		
	def construct_layers(self, inputs):
		layers = []
		self.num_layers = len(inputs) - 1
		
		if len(inputs) == 3: # Non-dropout part of MC dropout evaluation
			self.first_part = torch.nn.Identity()
		
		for i in range(len(inputs)-2):
			curr_layer = torch.nn.Linear(inputs[i], inputs[i+1])
			torch.nn.init.xavier_uniform_(curr_layer.weight, \
                            gain=torch.nn.init.calculate_gain('relu'))
			layers.append(curr_layer)
			layers.append(torch.nn.ReLU())

			if i == len(inputs) - 3: # Non-dropout part of MC dropout evaluation
				self.first_part = torch.nn.Sequential(*layers)
				layers = []
			
			layers.append(torch.nn.Dropout(self.p))
		
		# Add final layer with sigmoid-like activation function
		self.last_layer = torch.nn.Linear(inputs[len(inputs)-2], inputs[len(inputs)-1])
		torch.nn.init.xavier_uniform_(self.last_layer.weight, \
			gain=torch.nn.init.calculate_gain('tanh'))
		layers.append(self.last_layer)

		self.logistic_layer = self.Weibull_Layer()
		layers.append(self.logistic_layer)

		self.second_part = torch.nn.Sequential(*layers)
		
		
	def forward_non_dropout(self, x):
		scipy_flag = False
		
		if not torch.is_tensor(x):
			scipy_flag = True
			x = torch.from_numpy(x)
		
		y = self.first_part(x)

		if scipy_flag:
			return y.detach().numpy()
		
		return y
		
	def forward_dropout(self, x):
		scipy_flag = False
		
		if not torch.is_tensor(x):
			scipy_flag = True
			x = torch.from_numpy(x)
		
		y = 0.5*(1 + self.asymp - (1-self.asymp)*self.lapse) + 0.5*(1 - \
			self.asymp - self.lapse) * self.second_part(x)
		
		if scipy_flag:
			return y.detach().numpy()
		
		return y
		
	def forward(self, x):	
		scipy_flag = False
		if not torch.is_tensor(x):
			scipy_flag = True
			x = torch.from_numpy(x)
        
		y = self.forward_dropout(self.forward_non_dropout(x))     
			
		if scipy_flag:
			return y.detach().numpy()
		
		return y
		
	
	def apply_unscaled(self, x):
		scipy_flag = False
		if not torch.is_tensor(x):
			scipy_flag = True
			x = torch.from_numpy(x)
			
		y = self.second_part(self.forward_non_dropout(x))
		
		if scipy_flag:
			return y.detach().numpy()
		
		return y
		
	
	'''
		Compute Jacobian of output with respect to parameters.
	'''	
	def jacobian_of_parameters(self, x, loss_value=1.0):
		if x.is_cuda:
			device = 'cuda'
		else:
			device = 'cpu'
			
		
		# Define grad of ReLU
		def d_ReLU(x):
			out = torch.zeros_like(x)
			out[x > 0.0] = 1.0
			return out
			
		# Define inverse operator for tanh layers
		def inv_Tanh(x):
			a = 0.5*(1 + self.asymp - (1-self.asymp)*self.lapse)
			b = 0.5*(1 - self.asymp - self.lapse)
			x = (x-a)/b
			return 0.5 * (torch.log(1+x) - torch.log(1-x))
			
		# define grad of tanh layer
		def d_Tanh(x):
			return 0.5*(1 - torch.nn.functional.tanh(x)**2)
			
		# Define inverse operator for Weibull layer	
		def inv_Weibull(x):
			eps = 1e-20
			a = 0.5*(1 + self.asymp - (1-self.asymp)*self.lapse)
			b = 0.5*(1 - self.asymp - self.lapse)
			return torch.log10(-torch.log(0.5*(1 - (x-a)/b) + eps) + eps)
			
		# Define grad of Weibull layer
		def d_Weibull(x):
			return torch.log(10.0*torch.ones_like(x)) * 10.0**x * \
				torch.exp(-10.0**x)
				
		d_func = d_Tanh if isinstance(self.logistic_layer, \
							torch.nn.Tanh) else d_Weibull
		inv_func = inv_Tanh if isinstance(self.logistic_layer, \
							torch.nn.Tanh) else inv_Weibull
		
		with torch.no_grad():	
			# Apply forward pass and collect intermediate results
			forward_time = time.time()
			X_inter = [x]
			Bs = []
			Ws = []
			y = x
			for m in self.first_part.modules():
				if isinstance(m, torch.nn.Sequential):
					continue
				
				y = m(y)
		
				if isinstance(m, torch.nn.Linear):
					Ws.append(m.weight)	
		
				if isinstance(m, torch.nn.ReLU) or \
						isinstance(m, self.Weibull_Layer) or \
						isinstance(m, torch.nn.Tanh):
					X_inter.append(y)		
					
			for m in self.second_part.modules():
				if isinstance(m, torch.nn.Sequential):
					continue
					
				y = m(y)
				
				if y.dim() == 1:
					y = y.unsqueeze(0)
					
				if isinstance(m, torch.nn.Linear):
					Ws.append(m.weight)	
		
				if isinstance(m, torch.nn.ReLU):
					X_inter.append(y)		
					
				if isinstance(m, self.Weibull_Layer) or \
						isinstance(m, torch.nn.Tanh):
					y = 0.5*(1 + self.asymp - (1-self.asymp)*self.lapse) + \
						0.5*(1 - self.asymp - self.lapse) * y
					X_inter.append(y)
			
			# Loop over layers in backward fashion and collect intermediate
			# Jacobians for weight and bias
			jacs = []
			num_layers = len(self.layers)
			delta = loss_value
			for l in range(num_layers-1, 0, -1):
				if l == num_layers-1:
					delta = d_func(inv_func(X_inter[l].unsqueeze(2))) * \
						torch.diag(torch.ones(1, 1, dtype=x.dtype)).expand(\
						x.size(0), 1, 1).to(device)

					jacs.append(delta.clone())
					jacs.append(torch.einsum('bi, bj -> bij', \
						delta.squeeze(1), X_inter[l-1]))
				else:
					delta = delta @ (Ws[l].unsqueeze(0) * d_ReLU(X_inter[l]).unsqueeze(1)) 
									
					# Compute outer product of delta and X_inter[l] and 
					# store as partial jacobian
					jacs.append(delta.clone())
					jacs.append(torch.einsum('bi, bj -> bij', delta.squeeze(1), X_inter[l-1]))
		
		jacs.reverse()
		return jacs
		
	'''
		Computes the result of the neural network before the sigmoid 
		output layer.
	'''	
	def hidden_space(self, x):
		y = self.forward_non_dropout(x)

		for module in self.second_part.modules():
			if not module == self.logistic_layer and not isinstance(module, torch.nn.Sequential):
				y = module(y)
		
		return y
