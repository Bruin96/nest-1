import numpy as np
import time

from NEST.NEST import NEST


def run_example_NEST():
	num_trials = 100
	num_dims = 2
	lbs = [-10 for i in range(num_dims)]
	ubs = [10 for i in range(num_dims)]
	
	NEST_object = NEST()
	
	# Start up connection
	server_config = {"port": 3000, "host": "127.0.0.1", \
		"save_dir": "./Results"}
		
	experiment_params = {"num_dims": num_dims, "lbs": lbs, \
		"ubs": ubs, "p": 0.1, "lapse": 0.0, "asymptote": 0.0, \
		"convergence_level": -1, "hidden_layers": [256, 128, 32], \
		"random_chance": 0.5}
		
	NEST_object.start_new(server_config_dict=server_config, \
						  experiment_params=experiment_params)
						  
						  
	# Random interaction with the NEST server
	for i in range(num_trials):
		print(f"Trial {i+1}/{num_trials}:")
		sample = NEST_object.get_trial()
		print(f"Sample: {sample}")
		
		# Generate random response value
		mean = np.array([0.0 for i in range(num_dims)])
		cov = 2.0 * np.eye(num_dims)
		val = np.random.multivariate_normal(mean, cov)
		
		tau = np.random.uniform(0, 1)
		response = 0.0 if sample[0] < 0.0 else 1.0
		
		NEST_object.register_result(response)
	
	# Now evaluate some points
	sample1 = [0.1, 5.0]
	result1 = NEST_object.evaluate(sample1)		
	print(f"result1: {result1}")
	
	sample2 = [ [0.1, 5.0], [4.5, 6.7], [-1.0, -2.5] ]
	result2 = NEST_object.evaluate(sample2)
	print(f"result2: {result2}")
	
	sample3 = np.array(sample2)
	result3 = NEST_object.evaluate(sample3)
	print(f"result3: {result3}")
	
	sample4 = np.array(sample1)
	result4 = NEST_object.evaluate(sample4)
	print(f"result4: {result4}")
	
	# Test the maximum function
	max_X = NEST_object.find_max()
	print(f"max_X: {max_X}")
				  
						  
	NEST_object.terminate()
	
	print(f"--- END OF TEST SCRIPT ---")	
	
	
if __name__ == "__main__":
	run_example_NEST()
