'''
	Author: Sjoerd Bruin
	Project: Neural Network-Based Estimation by Sequential Testing (NEST)
	
	Class Description:
	This class represents a server instance, which can be launched in order
	to wait for inputs of a connecting client. The input data should be:
	-	An open request
	-	A response value for the last delivered sample
	-	A close request
	
	This server is used as a server for communicating with a NEST instance.
	The server closes when the client disconnects. If new NEST
	servers need to be spun up, a new instance of this class can simply
	be created.
'''

import json
import socket
import os
import sys
import time
import random
import numpy as np
import torch
import traceback
from enum import Enum

from Dropout_Neural_Network import Dropout_Neural_Network

from utils.data_initialisation import initialise_samplers
from utils.network_training import train_network_no_test
from utils.multiprocessing_functions.Maximisation_Computer import Maximisation_Computer

plot_graphs = False

class NEST_Server:
	def __init__(self, config_filename):
		self.config_filename = config_filename
		self.host = "127.0.0.1"
		self.port = 3000
		self.use_print = True
		self.round_sample = False
		
		self.is_initialised = False
		self.trial_count = 0
		
		# Initialise maximisation computer for finding maximum
		self.max_computer = Maximisation_Computer()
		
		# Set Fisher convergence level
		self.convergence_level = None
		self.num_to_converge = 15
		
		# Initialise training data
		self.Fisher_energy = []
		self.sample_weights = []
		self.train_in = None
		self.train_out = None
		self.next_trial = None
		
		self.save_dir = '.'
		self.save_name = ''
		
		print(f"parsing config...")
		
		self.parse_config(self.config_filename)		
		
		self.open_connection(self.port, self.host)
	
	# Define enumeration for types of messages
	class Message_Type(Enum):
		INITIALISE = 1
		TERMINATE = 2
		NEXT_TRIAL = 3
		RESTART = 4
		SET_CONFIG = 5
		EVALUATE = 6
		FIND_MAX = 7
		
		
	# Read server configuration settings from JSON file	
	def parse_config(self, config_filename):
		if config_filename == '':
			return
		
		# else: Read from the config file
		config_dict = None
		with open(config_filename) as config_file:
			config_dict = json.load(config_file)
		
		self.round_sample = config_dict['round_sample'] if 'round_sample' in \
			config_dict else self.round_sample	
		self.port = config_dict['port'] if 'port' in config_dict else self.port
		self.host = config_dict['host'] if 'host' in config_dict else self.host
		self.use_print = config_dict['verbose'] if 'verbose' in config_dict \
			else self.use_print
		self.save_dir = config_dict['save_dir'] if 'save_dir' in \
			config_dict else self.save_dir
			
			
	# Parse input config_dict and set parameters
	def parse_config_dict(self, config_dict):
		for (key, value) in config_dict.items():
			if key == 'savename':
				self.save_name = value
			if key == 'save_dir':
				self.save_dir = value
			if key == 'verbose':
				self.use_print = value
			if key == 'round_sample':
				self.round_sample = value
			if key == 'convergence_level':
				self.convergence_level = value
		
	
	# Open a connection and serve an incoming client	
	def open_connection(self, port_no, host_name):
		print(f"Opening connection on server side...")
		# Create socket object and listen for incoming connection
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			# Find an available por to use
			for i in range(1024, 40000):
				try:
					s.bind((host_name, int(i)))
					self.port = int(i)
					
					# save port number to file
					filename = os.path.join(self.save_dir, "port_number.txt")
					with open(filename, "w+") as f:
						f.write(str(int(i)))
					break
				except OSError as e:
					pass
			print(f"Listening...")
			s.listen()
			(conn, addr) = s.accept()
			print(f"Accepted connection: {addr}")
			
			try:
				if not self.use_print: # Block printing
					sys.stdout = open(os.devnull, 'w')
				
				# Serve connection
				self.serve(conn)
				
				if not self.use_print: # Allow printing again
					sys.stdout = sys.__stdout__
			except Exception as e:
				sys.stdout = sys.__stdout__
				print(f"an exception occurred: {e}")
			finally:
				conn.close()
				s.shutdown(socket.SHUT_RDWR)
				s.close()
				
				
				
	def serve(self, conn):
		with conn:
			terminate_connection = False
			# Loop until connection is closed
			while not terminate_connection:
				if not self.use_print: # Block printing
					sys.stdout = open(os.devnull, 'w')
				
				seed = int(int(time.time()*1e9)%1e9)
				torch.manual_seed(seed)
				np.random.seed(seed)
				random.seed(seed)
				
				#print(f"Waiting for data...")
				data = conn.recv(2000000)
				serve_time = time.time()
				#print(f"data: {data}")
				(value, message_type) = self.read_input(data)
				try:
					if message_type == self.Message_Type.INITIALISE.value:
						self.initialise(value)
												
						# select initial trial in center of input space
						trial_vals = [0.0 for i in range(self.num_dims)]
						for i in range(self.num_dims):
							trial_vals[i] = round(0.5*(self.lbs[i]+self.ubs[i]))
						self.next_trial = trial_vals
						
						# Send trial and wait for response	
						return_dict = {'next_trial': self.next_trial, 'converged': False}
						return_string = json.dumps(return_dict)				
						conn.send(return_string.encode(encoding='utf-8'))
						
					elif message_type == self.Message_Type.TERMINATE.value:
						terminate_connection = True
						is_finished = value['finished']
						self.save_data(is_finished=is_finished)
						
					elif message_type == self.Message_Type.NEXT_TRIAL.value:
						# Increment trial count
						self.trial_count += 1
											
						# Check for weight of value
						if isinstance(value, float):
							curr_value = value
							curr_weight = 1.0
						else:
							curr_value = value['value']
							curr_weight = value['weight']
							
						self.sample_weights.append(curr_weight)
						
						# Add value to train out
						if self.train_out is None:
							self.train_out = torch.tensor([curr_value])
						else:
							self.train_out = torch.cat((self.train_out, \
								torch.tensor([curr_value])))
								
						# Add trial to train data
						if self.train_in is None:
							self.train_in = torch.tensor(self.next_trial).unsqueeze(0)
						else:
							self.train_in = torch.cat((self.train_in, \
								torch.tensor(self.next_trial).unsqueeze(0)), dim=0)
						
						# Retrain network
						L_k = self.update_network()
						
						# Select next trial
						self.next_trial = self.select_next_trial()
						
						# Check Fisher convergence
						if self.convergence_level is not None:
							has_converged = self.check_convergence()
						else:
							has_converged = False
						
						return_dict = {'Fisher_energy': L_k.detach().item(), \
										'next_trial': self.next_trial,
										'converged': has_converged}
						return_string = json.dumps(return_dict)
						
						# Save data so far
						self.save_data()
						
						# Send next trial and wait for response					
						conn.send(return_string.encode(encoding='utf-8'))
					
					elif message_type == self.Message_Type.RESTART.value:
						data_dir = value['data_dir']
						config_dict = value['config_dict']
						self.restart(data_dir, config_dict)
												
						L_k = self.Fisher_energy[-1]
											
						# Generate a new trial
						self.next_trial = self.select_next_trial()
						
						# Send trial and wait for response	
						return_dict = {'next_trial': self.next_trial, 'converged': False}
						return_string = json.dumps(return_dict)				
						conn.send(return_string.encode(encoding='utf-8'))
						
					elif message_type == self.Message_Type.SET_CONFIG.value:
						print(f"Parsing config_dict...")
						config_dict = value['config_dict']
						self.parse_config_dict(config_dict)
						
						print(f"Parsed config_dict.")
						
						return_dict = {"OK": "OK"}
						return_string = json.dumps(return_dict)	
						print(f"return_string after SET_CONFIG: {return_string}")
						encoded = return_string.encode(encoding="utf-8")
						print(f"encoded return_string after SET_CONFIG: {encoded}")			
						conn.send(return_string.encode(encoding='utf-8'))
						print(f"Sent confirmation of SET_CONFIG.")
					
					elif message_type == self.Message_Type.EVALUATE.value:
						X = torch.tensor(value)
						Y = self.evaluate(X).numpy()
						
						return_dict = {"output": Y.tolist()}
						return_string = json.dumps(return_dict)
						conn.send(return_string.encode(encoding="utf-8"))
					elif message_type == self.Message_Type.FIND_MAX.value:
						X_max = self.find_maximum()
						
						return_dict = {"maximum": X_max.tolist()}
						return_string = json.dumps(return_dict)
						conn.send(return_string.encode(encoding="utf-8"))
						
					# Send message to client
					#conn.send(return_string.encode(encoding="utf-8"))
				except TypeError as typeError:
					traceback_output = traceback.format_exc()
					print(f"An error occurred: {typeError}")
					print(f"Traceback: {traceback_output}")
					# Send a message to client stating that the message
					# type was not valid
					return_dict = {"ERROR": typeError}
					return_string = json.dumps(return_dict)
					conn.send(return_string.encode(encoding="utf-8"))
				except ValueError as valueError:
					traceback_output = traceback.format_exc()
					print(f"An error occurred: {valueError}")
					print(f"Traceback: {traceback_output}")
					# Send a message to client stating that a value error
					# occurred.
					return_dict = {"ERROR": valueError}
					return_string = json.dumps(return_dict)
					conn.send(return_string.encode(encoding="utf-8"))
					
				except Exception as genericError:
					traceback_output = traceback.format_exc()
					print(f"An error occurred: {genericError}")
					print(f"Traceback: {traceback_output}")
					# Send a message to client stating that an internal
					# error occurred.
					return_dict = {"ERROR": genericError, "error_message": \
						f"The following internal error occurred: {genericError} \n" \
						f"Apologies for the inconvenience. If you believe that this bug " \
						f"did not originate on the user end, then please report this bug on Github."}
					return_string = json.dumps(return_dict)
					conn.send(return_string.encode(encoding="utf-8"))
					
					# Close the connection
					terminate_connection = True
								
				serve_time = time.time() - serve_time
				print(f"Time to serve request: {serve_time} seconds.")
				
				if not self.use_print: # Allow printing again
					sys.stdout = sys.__stdout__
						 		
						 		
	def read_input(self, data):
		if data == b'':
			message_type = self.Message_Type.TERMINATE.value
			value = {'finished': False}
			return (value, message_type)
			
		data_string = data.decode(encoding="utf-8")
		data = json.loads(data_string)
		
		message = data['message']
		value = data['value']
		
		if message == 'NEXT_TRIAL':
			message_type = self.Message_Type.NEXT_TRIAL.value
		elif message == 'INITIALISE':
			message_type = self.Message_Type.INITIALISE.value
		elif message == 'TERMINATE':
			message_type = self.Message_Type.TERMINATE.value
		elif message == 'RESTART':
			message_type = self.Message_Type.RESTART.value
		elif message == 'SET_CONFIG':
			message_type = self.Message_Type.SET_CONFIG.value
		elif message == "EVALUATE":
			message_type = self.Message_Type.EVALUATE.value
		elif message == "FIND_MAX":
			message_type = self.Message_Type.FIND_MAX.value
		else:
			raise TypeError("Incorrect message type.")
		
		return (value, message_type)
		
		
	# Save data, with a parameter determining whether to add 'finished'
	# to code base
	def save_data(self, is_finished=False):	
		if len(self.Fisher_energy) == 0:
			return # Nothing to save
			
		if is_finished:
			# Save all relevant data in save_dir with regards to
			# the network, the training data, and the Fisher
			# energy
			Fisher_energy_np = np.array(self.Fisher_energy)
			np.save(self.save_dir + '/' + self.save_name + 'Fisher_energy_finished.npy', Fisher_energy_np)
			sample_weights_np = np.array(self.sample_weights)
			np.save(self.save_dir + '/' + self.save_name + 'sample_weights_finished.npy', sample_weights_np)
			
			np.save(self.save_dir + '/' + self.save_name + 'train_in_finished.npy', self.train_in.numpy())
			np.save(self.save_dir + '/' + self.save_name + 'train_out_finished.npy', self.train_out.numpy())
			
			torch.save(self.neural_net.state_dict(), self.save_dir + \
				'/' + self.save_name + 'neural_net_finished.pth')
				
			# Remove old, unfinished version of results
			try:
				os.remove(self.save_dir + '/' + self.save_name + 'Fisher_energy.npy')
				os.remove(self.save_dir + '/' + self.save_name + 'sample_weights.npy')
				os.remove(self.save_dir + '/' + self.save_name + 'train_in.npy')
				os.remove(self.save_dir + '/' + self.save_name + 'train_out.npy')
				os.remove(self.save_dir + '/' + self.save_name + 'neural_net.pth')
			except: # Files do not exist, so just end the operation
				pass
		else: # Not finished, so save without '_finished' in name
			# Save all relevant data in save_dir with regards to
			# the network, the training data, and the Fisher
			# energy
			Fisher_energy_np = np.array(self.Fisher_energy)
			np.save(self.save_dir + '/' + self.save_name + 'Fisher_energy.npy', Fisher_energy_np)
			sample_weights_np = np.array(self.sample_weights)
			np.save(self.save_dir + '/' + self.save_name + 'sample_weights.npy', sample_weights_np)
			
			np.save(self.save_dir + '/' + self.save_name + 'train_in.npy', self.train_in.numpy())
			np.save(self.save_dir + '/' + self.save_name + 'train_out.npy', self.train_out.numpy())
			
			torch.save(self.neural_net.state_dict(), self.save_dir + \
				'/' + self.save_name + 'neural_net.pth')
				
		
	
	# Initialise neural network and its parameters	
	def initialise(self, config_dict):
		if self.is_initialised: # Do not initialise if already initialised
			return
			
		# Initialise convergence criterion
		self.convergence_level = config_dict['convergence_level'] if \
			'convergence_level' in config_dict else None

		# Check for using print
		self.use_print is config_dict["verbose"] if "verbose" in config_dict \
			else self.use_print

		# Initialise input space
		if 'num_dims' not in config_dict:
			raise ValueError("The parameter num_dims must be given when initialising the NEST procedure.")
		else:
			self.num_dims = config_dict['num_dims']

		self.lbs = config_dict['lbs'] if 'lbs' in config_dict else \
			[-1 for i in range(self.num_dims)]
		self.ubs = config_dict['ubs'] if 'ubs' in config_dict else \
			[-1 for i in range(self.num_dims)]

		# Initialise neural network
		self.p = config_dict['p'] if 'p' in config_dict else 0.1
		self.layers = [self.num_dims] + config_dict['hidden_layers'] + [1] if \
			'hidden_layers' in config_dict else [self.num_dims, 256, 128, 32, 1]
			
		self.lapse = config_dict['lapse'] if 'lapse' in config_dict else 0.0
		self.asymp = config_dict['asymptote'] if 'asymptote' in \
							config_dict else 0.0
		
		self.network_params = {'dropout_probability': self.p, 'lapse': \
			self.lapse, 'asymptote': self.asymp}	
		self.neural_net = Dropout_Neural_Network(self.layers, \
			**self.network_params) 
			
		self.num_epochs = config_dict['num_epochs'] if 'num_epochs' in \
			config_dict else 100
		
		self.lr = config_dict['learning_rate'] if 'learning_rate' in \
			config_dict else 0.0003
			
		# Initialise acquisition function
		self.a = config_dict['a'] if 'a' in config_dict else 0.8
		self.b = config_dict['b'] if 'b' in config_dict else 10.6
		self.c = config_dict['c'] if 'c' in config_dict else 6.0
		self.d = config_dict['d'] if 'd' in config_dict else 4.0
		
		self.random_multiplier = max(0.0, config_dict['random_multiplier'] \
			if 'random_multiplier' in config_dict else 0.97)
		self.random_choice_prob = min(1.0, max(0.0, \
			config_dict['random_chance'] if 'random_chance' in \
			config_dict else 0.5))
			
		self.num_std_trials = config_dict['num_trials'] if 'num_trials' \
			in config_dict else 50
		
		self.sampler_params = {'neural_net': self.neural_net, \
			'lower_bounds': self.lbs, 'upper_bounds': self.ubs, \
			'seed': 42, 'h': 0.25, 'a': self.a, 'b': self.b, \
			'c': self.c, 'd': self.d, 'num_trials': self.num_std_trials}
		(self.sampler, _) = initialise_samplers('synthesis', None, \
								self.sampler_params)
								
		self.is_initialised = True
		
		
	# Load data from given directory and initialise parameters
	# to their proper value
	def restart(self, data_dir, config_dict):		
		# Initialise convergence criterion
		self.convergence_level = config_dict['convergence_level'] if \
			'convergence_level' in config_dict else None
		
		# Initialise input space
		if 'num_dims' not in config_dict:
			raise ValueError("The parameter num_dims must be given when initialising the NEST procedure.")
		else:
			self.num_dims = config_dict['num_dims']

		self.lbs = config_dict['lbs'] if 'lbs' in config_dict else \
			[-1 for i in range(self.num_dims)]
		self.ubs = config_dict['ubs'] if 'ubs' in config_dict else \
			[-1 for i in range(self.num_dims)]

		# Initialise neural network
		self.p = config_dict['p'] if 'p' in config_dict else 0.1
		self.layers = [self.num_dims] + config_dict['hidden_layers'] + [1] if \
			'hidden_layers' in config_dict else [self.num_dims, 256, 128, 32, 1]
			
		self.lapse = config_dict['lapse'] if 'lapse' in config_dict else 0.0
		self.asymp = config_dict['asymptote'] if 'asymptote' in \
							config_dict else 0.0
		
		self.network_params = {'dropout_probability': self.p, 'lapse': \
			self.lapse, 'asymptote': self.asymp}	
		self.neural_net = Dropout_Neural_Network(self.layers, \
			**self.network_params) 
			
		self.num_epochs = config_dict['num_epochs'] if 'num_epochs' in \
			config_dict else 100
		
		self.lr = config_dict['learning_rate'] if 'learning_rate' in \
			config_dict else 0.0003
						
		# Load training data, Fisher energy and neural network
		self.neural_net.load_state_dict(torch.load(os.path.join(data_dir, \
			self.save_name + 'neural_net.pth')), strict=False)
		self.neural_net.asymp = torch.nn.Parameter(torch.tensor([self.asymp]), requires_grad=False)
		self.neural_net.lapse = torch.nn.Parameter(torch.tensor([self.lapse]), requires_grad=False)
		
		
		self.train_in = torch.from_numpy(np.load(os.path.join(data_dir, \
			self.save_name + 'train_in.npy'))).double()
		self.train_out = torch.from_numpy(np.load(os.path.join(data_dir, \
			self.save_name + 'train_out.npy'))).double()
		self.Fisher_energy = np.load(os.path.join(data_dir, \
			self.save_name + 'Fisher_energy.npy')).tolist()
		self.sample_weights = np.load(os.path.join(data_dir, \
			self.save_name + 'sample_weights.npy')).tolist()
		
		self.trial_count = len(self.Fisher_energy)
		self.start_trial_count = len(self.Fisher_energy)
		
		self.neural_net.set_double()
			
		# Initialise acquisition function
		self.a = config_dict['a'] if 'a' in config_dict else 0.8
		self.b = config_dict['b'] if 'b' in config_dict else 10.6
		self.c = config_dict['c'] if 'c' in config_dict else 6.0
		self.d = config_dict['d'] if 'd' in config_dict else 4.0
		
		self.random_multiplier = max(0.0, config_dict['random_multiplier'] \
			if 'random_multiplier' in config_dict else 0.97)
		self.random_choice_prob = min(1.0, max(0.0, \
			config_dict['random_chance'] if 'random_chance' in \
			config_dict else 0.5))
			
		# Fast-forward random_choice_prob
		self.random_choice_prob = min(1.0, self.random_choice_prob * \
			self.random_multiplier**(self.trial_count-2))
		
		self.num_std_trials = config_dict['num_trials'] if 'num_trials' \
			in config_dict else 50
		
		self.sampler_params = {'neural_net': self.neural_net, \
			'lower_bounds': self.lbs, 'upper_bounds': self.ubs, \
			'seed': 42, 'h': 0.25, 'a': self.a, 'b': self.b, \
			'c': self.c, 'd': self.d, 'num_trials': self.num_std_trials, \
			'base_trial_count': self.start_trial_count}
		
		self.sampler_type = config_dict["sampler_type"] if "sampler_type" \
			in config_dict else "synthesis"
			
		(self.sampler, _) = initialise_samplers(self.sampler_type, None, \
								self.sampler_params)
								
		self.is_initialised = True		
	
	# Evaluate input
	def evaluate(self, X):
		(mu, sigma) = self.compute_mu_sigma()
				
		X_normalised = (X.double() - mu) / sigma
				
		self.neural_net.set_double()
		self.neural_net.set_predict()
		
		with torch.no_grad():
			Y = self.neural_net(X_normalised.double()).squeeze().detach()
						
		return Y
		
		
	# Find maximum in neural network
	def find_maximum(self):
		(mu, sigma) = self.compute_mu_sigma()
		X_max = self.max_computer.run(mu, sigma, self.train_in, \
			self.neural_net, self.lbs, self.ubs,  self.num_dims, \
			24, 10.0, b=0.0, eps_in=1e-4, use_UCB=False)
			
		X_max = (X_max*sigma + mu).numpy()
			
		return X_max
	
	
	# Update network
	def update_network(self):
		# Apply shrink-and-perturb
		self.neural_net.shrink_and_perturb_weights(0.9, 0.01)
				
		# Initialise optimiser and scheduler
		opt = torch.optim.AdamW(self.neural_net.parameters(), lr=self.lr, \
			weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False)
		sched = torch.optim.lr_scheduler.LinearLR(opt, \
			start_factor=1.0, end_factor = 0.01, \
			total_iters=self.num_epochs)
					
		# Normalise input data
		(mu, sigma) = self.compute_mu_sigma()
		
		train_in = (self.train_in.detach().clone() - mu) / sigma
			
		# Train network
		use_GPU = True if torch.cuda.is_available() else False
		(_, L_k) = train_network_no_test(self.neural_net, opt, sched, \
			train_in, self.train_out, self.num_epochs, self.lbs, \
			self.ubs, return_fisher=True, gamma=0.0, use_GPU=use_GPU, \
			weights=self.sample_weights)
			
		self.Fisher_energy.append(L_k.detach().item())
		
		return L_k			
					
	# Generate a new sample
	def select_next_trial(self):
		(mu, sigma) = self.compute_mu_sigma()
		
		# Generate a sample
		if self.trial_count < 2:
			(sample, _) = self.sampler.sample(random_chance=1.0, \
				train_in=self.train_in, mu=mu, sigma=sigma)
		else:
			train_in_normalised = (self.train_in.detach().clone() - mu) / sigma
			(sample, is_random) = self.sampler.sample(random_chance=\
				self.random_choice_prob, train_in=train_in_normalised, \
				train_out=self.train_out, mu=mu, sigma=sigma)
		
			if not is_random: # Undo mapping to neural network space
				sample = (torch.tensor(sample) * sigma + mu).tolist()
		
			# Update probability of random selection
			self.random_choice_prob = min(1.0, self.random_choice_prob * \
											self.random_multiplier)

		if self.round_sample:
			sample[0:3] = torch.round(torch.tensor(sample[0:3])).tolist()
					
		return sample
		
		
	# Compute mu and sigma based on training data
	def compute_mu_sigma(self):
		if self.trial_count < 10:
			mu = torch.tensor([0.5*(self.lbs[i] + self.ubs[i]) for i \
				in range(len(self.ubs))])
			sigma = torch.tensor([0.5*(self.ubs[i] - self.lbs[i]) for i \
				in range(len(self.ubs))])
		else:
			mu = torch.mean(self.train_in, dim=0)
			sigma = torch.std(self.train_in, dim=0, unbiased=True)
		
		return (mu, sigma)
		
	def compute_windowed_Fisher_energy(self, energies, window=10):
	    N = energies.shape[0]
	    Fisher_diff = np.zeros(N-1)
	    for i in range(1, N):
	        Fisher_diff[i-1] = np.abs(energies[i]	- energies[i-1])	
	        	        
	    Fisher_window_diff = np.array([np.sum(np.array([1/window*Fisher_diff[k-j] \
			for j in range(window)])) for k in range(window-1, len(Fisher_diff))])
	    
	    return Fisher_window_diff
	    
	    
	def check_convergence(self):
		if self.convergence_level is None:
			return False
						
		num_correct = torch.sum(self.train_out).item()
		if self.trial_count >= 15 + self.num_to_converge + 1 and \
				num_correct != 0 and \
				num_correct != self.train_in.size(0):
			Fisher_window_diff = self.compute_windowed_Fisher_energy(\
											np.array(self.Fisher_energy))
			window_diff_len = Fisher_window_diff.shape[0]

			for i in range(window_diff_len-self.num_to_converge, window_diff_len):
				if Fisher_window_diff[i] > self.convergence_level:
					return False
			
			return True
		
		return False


if __name__ == "__main__":
	config_filename = ''
	if len(sys.argv) > 1:
		config_filename = sys.argv[1]
	NEST_server = NEST_Server(config_filename)
	NEST_server.open_connection(NEST_server.port, NEST_server.host)
