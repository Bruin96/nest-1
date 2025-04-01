import json
import numpy as np
import os
import socket
import subprocess
import sys
import time



class NEST:
	def __init__(self, config_filename=None):
		self.results_dir = "./Results"
		self.host = "127.0.0.1"
		self.port = 3000
		
		self.newest_trial = None
		
		self.register_config_file(config_filename)
		
		
	def get_trial(self):
		return self.newest_trial
		
		
	def register_result(self, Y, weight=1.0):
		# Dispatch result to server
		data_dict = {	
						'message': 'NEXT_TRIAL', \
						'value': {'value': Y, 'weight': weight}
					}
		data_string = json.dumps(data_dict)
		self.sock.send(data_string.encode(encoding='utf-8'))
		
		
		# Receive response and store trial
		return_data = self.sock.recv(2000000)
		return_data = json.loads(return_data.decode(encoding='utf-8')) 

		self.newest_trial = np.array(return_data['next_trial'])
		
		
	
	def register_config_file(self, config_filename):
		if config_filename is not None:
			self._parse_config(config_filename)
				
		
	def start_new(self, server_config_dict={}, experiment_params={}):
		# Construct server config file
		with open(self.results_dir + '/server_config.json', 'w') as f:
			json.dump(server_config_dict, f)
			
		# Initialise experiment params
		self.set_experiment_params(**experiment_params)
		
		# Start server
		subprocess.Popen([sys.executable, \
			'NEST/NEST_Server.py', \
			self.results_dir + '/server_config.json'])
			
		# Wait for server to start up
		print(f"Waiting for server to spin up...")
		time.sleep(5.0)
		
		# Initialise connection
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		
		# Read port number from file
		with open(self.results_dir + "/port_number.txt", "r") as file_obj:
			self.port = int(file_obj.readline())
		
		self.sock.connect(('127.0.0.1', self.port))
		
		print(f"Connected to server.")
		
		try:
			self.update_server_configuration(**server_config_dict)
			self._initialise_connection()
			print(f"Opened connection.")
		except ValueError as valueError:
			print(f"A value error occurred on the server side:\n{valueError}")
		except TypeError as typeError:
			print(f"A type error occurred on the server side:\n{typeError}")
		except Exception as genericError:
			print(f"An internal error occurred on the server side:\n{genericError}" \
				f"Apologies for the inconvenience. If you believe this is " \
				f"not a user error, then please report this bug on Github.")
			
		
		
		
		
	def update_server_configuration(self, **kwargs):
		# Construct config_dict
		config_dict = {}
		for (key, value) in kwargs.items():
			if "save_dir" in key:
				self.results_dir = value
			
				config_dict[key] = value
			
		# Dispatch config_dict to server
		data_dict = {	
						'message': 'SET_CONFIG', \
						'value': {'config_dict': config_dict}
					}
					
		config_string = json.dumps(data_dict)
		self.sock.send(config_string.encode(encoding='utf-8'))
				
		return_data = self.sock.recv(4096)		
		return_data = json.loads(return_data.decode(encoding='utf-8'))
				
		# Check if there was an error
		if "ERROR" in return_data:
			raise config_dict["ERROR"]
		
			
	def set_experiment_params(self, **kwargs):
		self.experiment_dict = {}
		for (key, value) in kwargs.items():
			self.experiment_dict[key] = value
		
			
	def evaluate(self, X):
		is_numpy = isinstance(X, np.ndarray)
		if is_numpy:
			X = X.tolist()
			
		data_dict = {"message": "EVALUATE", "value": X}
		data_string = json.dumps(data_dict)	
		self.sock.send(data_string.encode(encoding='utf-8'))
		
		return_data = self.sock.recv(2000000)
		return_data = json.loads(return_data.decode(encoding='utf-8'))
		Y = return_data["output"]
		
		if is_numpy and isinstance(Y, list):
			Y = np.array(Y)
			
		return Y
				
		
	def find_max(self):
		data_dict = {"message": "FIND_MAX", "value": None}
		data_string = json.dumps(data_dict)	
		self.sock.send(data_string.encode(encoding='utf-8'))
		
		return_data = self.sock.recv(2000000)
		return_data = json.loads(return_data.decode(encoding='utf-8'))
		
		X = return_data["maximum"]
		
		return X
		
		
	def terminate(self):
		data_dict = {"message": "TERMINATE", "value": {"finished": False}}
		data_string = json.dumps(data_dict)
		self.sock.send(data_string.encode(encoding='utf-8'))
		
	
	@staticmethod
	def create_config_file(host="127.0.0.1", port=3000, save_dir="."):
		pass
		
		
	def _parse_config(self, config_filename):
		with open(config_filename, "r") as f:
			self.results_dir = config_filename["save_dir"]
			self.host = config_filename["host"]
			self.port = config_filename["port"]
			
	def _initialise_connection(self):
		# Send 'INITIALISE' message and register experiment parameters
		data_dict = {'message': 'INITIALISE', 'value': self.experiment_dict}
		data_string = json.dumps(data_dict)
				
		self.sock.send(data_string.encode(encoding='utf-8'))
		
		# Receive response and store trial
		return_data = self.sock.recv(2000000)
		return_data = json.loads(return_data.decode(encoding='utf-8')) 

		self.newest_trial = np.array(return_data['next_trial'])
		
	
	def _restart_connection(self):
		pass	
