'''

'''

import time
import torch
import numpy as np


class Neural_Network (torch.nn.Module):
	def __init__(self, input_params, learn_lapse=False, **params):
		super().__init__()
		
		self.lapse = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)
		self.asymp = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)
		self.learn_lapse = learn_lapse
		if self.learn_lapse:
			self.lapse = torch.nn.Parameter(torch.tensor([1e-3])) # Initialise with small non-zero value
		
		self.layer_sizes = input_params
		self.set_params(**params)
				
		self.layers = self.construct_layers(input_params)
		
	def set_params(self, **params):
		for (key, value) in params.items():
			if key == 'lapse':
				if not self.learn_lapse:
					self.lapse = torch.nn.Parameter(torch.tensor([value]), \
													requires_grad=False)
			if key == 'asymptote':
				self.asymp = torch.nn.Parameter(torch.tensor([value]), \
													requires_grad=False)
		
		
	def update_lapse(self, new_lapse):
		if not self.learn_lapse: # Only update if it is not being learned
			self.lapse = torch.nn.Parameter(torch.tensor([new_lapse]), \
													requires_grad=False)
		
		
	def set_predict(self):
		self.eval()
		for module in self.modules():
			print(module)
			module.eval()			
	
				
	def set_inference(self):
		self.eval()
		
		for module in self.modules():
			if isinstance(module, torch.nn.Dropout):
				module.train()
			else:
				module.eval()
	
		
	def set_double(self):
		self.double()
		
		for module in self.modules():
			module = module.double()
			
			
	def set_single(self):
		self.float()
		for module in self.modules():
			module = module.float()
	
				
	def set_train(self):
		self.train()
		for module in self.modules():
			module.train()
			
			
	def get_list_of_weights(self):
		weights = []
		for param in self.model.parameters():
			weights.append(param.data.detach().clone())
			
		return weights
	
	
	def weights_init(self, m):
		if isinstance(m, torch.nn.Sequential):
			for sub_m in m.children():
				if isinstance(sub_m, self.Weibull_Layer):
					prev_sub_m.reset_parameters()
					torch.nn.init.xavier_uniform_(prev_sub_m.weight, \
						gain=torch.nn.init.calculate_gain('tanh'))
						
				elif isinstance(sub_m, torch.nn.Tanh):
					prev_sub_m.reset_parameters()
					torch.nn.init.xavier_uniform_(prev_sub_m.weight, \
						mode='fan_in', nonlinearity='tanh')
						
				elif not isinstance(sub_m, torch.nn.Dropout) and \
					not isinstance(sub_m, torch.nn.ReLU):
					sub_m.reset_parameters()
					torch.nn.init.kaiming_uniform_(sub_m.weight, \
						mode='fan_in', nonlinearity='relu')
					
				prev_sub_m = sub_m
							
			
	def reset_parameters(self):
		self.first_part.apply(self.weights_init)
		self.second_part.apply(self.weights_init)
	
	# This function shrinks the current network weights and perturbs them randomly
	def shrink_and_perturb_weights(self, shrink_factor=0.5, std=0.02):
		
		for (name, param) in self.named_parameters():
			rng = default_rng()
			if 'weight' in name or 'bias' in name:
				sizes = tuple(list(param.size()))
				random_vals = rng.normal(0.0, std, sizes)
				param.data = shrink_factor*param.data + random_vals
		
	def reset_warmstart(self):
		for sub_m in self.modules():
			if isinstance(sub_m, self.Logistic_Layer) or \
					isinstance(sub_m, self.Average_Layer) or \
					isinstance(sub_m, self.Weibull_Layer):
				sub_m.reset_parameters()
				
			
	# Define Weibull layer
	class Weibull_Layer(torch.nn.Module):
		def __init__(self, gamma=1.0):
			super().__init__()
			self.gamma_base = gamma
			self.T_base = 0.0
			self.T = 0.0
			self.gamma = 0.01
			self.zero_tensor = torch.tensor([0.0], requires_grad=False)
			
		def reset_parameters(self):
			return
			
		def forward(self, x):
			y = torch.tanh(x)
			
			return y
			
		
	def construct_layers(self, inputs):
		layers = []
		
		for i in range(len(inputs)-2):
			curr_layer = torch.nn.Linear(inputs[i], inputs[i+1])
			layers.append(curr_layer)
			torch.nn.init.xavier_uniform_(curr_layer.weight)
			layers.append(torch.nn.ReLU())
			
		curr_layer = torch.nn.Linear(inputs[len(inputs)-2], inputs[len(inputs)-1])
		torch.nn.init.xavier_uniform_(curr_layer.weight)
		layers.append(curr_layer)
		layers.append(self.Weibull_Layer())
			
		self.model = torch.nn.Sequential(*layers)
		
		
	def forward(self, x):
		return 0.5*(1 + self.asymp - self.lapse) + \
			0.5*(1 - self.asymp - self.lapse)*self.model(x)
