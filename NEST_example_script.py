import numpy as np
import scipy.stats

from NEST.NEST import NEST
from NEST.example_functions.Barten_CSF import Barten_CSF


''' 
	This function gives an example CSF function.
	It first computes the threshold, and then defines a transition
	over that threshold funcion to produce a probability of detection.
'''
def evaluate_CSF(sample):
	# Define the CSF
	param_dict = {'e_g': 3.3, 'sigma_0_base': 1.5, 'k': 2.3, 'eta': 0.04, \
		'self.M_factor': 0.05, 'r_e': 7.633}
	
	csf = Barten_CSF(param_dict)
	
	# Determine where the threshold lies along the second dimension
	thres = np.log10(csf.evaluate_CSF(10**np.array([sample[0]]), e=5.0, L=500, \
										p=1.22e6, N_eyes=2, X_0=2.0))
											
	# Map distance of current sample from threshold to a probability of detection
	out = float(scipy.stats.norm.cdf(thres - sample[1], scale=0.2))
	
	return out
	


def run_example_NEST():
	# Define parameters for the experiment
	num_trials = 100 
	num_dims = 2 # Number of stimulus dimensions, e.g. spatial frequency, eccentricity, etc.
	lbs = [-1, 0] # Minimum value of the input space for each stimulus dimension
	ubs = [2, 2.5] # Maximum value of the input space for each stimulus dimension
	
	# Initialize the NEST object for using the NEST procedure
	NEST_object = NEST()
	
	# Start up connection
	server_config = {"port": 3000, "host": "127.0.0.1", \
		"save_dir": "./Results"}
		
	experiment_params = {
							"num_dims": num_dims, 				# Number of stimulus dimensions in the experiment
							"lbs": lbs, 						# Minimum values for each stimulus dimension
							"ubs": ubs, 						# Maximum values for each stimulus dimension 
							"p": 0.1, 							# Dropout probability of neural network. Set to default p = 0.1
							"lapse": 0.0, 						# Lapse rate, i.e. the probability of making a mistake even if the stimulus is clearly visible
							"asymptote": 0.0, 					# Lower asymptote, i.e. the minimum probability of a correct response when guessing
							"convergence_level": -1, 			# Convergence level of the convergence criterion. Set to -1 if not used
							"hidden_layers": [256, 128, 32],	# Architecture of the neural network. Set to default w = [256, 128, 32]
							"random_chance": 0.5 				# Initial rate of selecting a random sample instead of one from the acquisition function
						}
		
	NEST_object.start_new(server_config_dict=server_config, \
						  experiment_params=experiment_params)
						  
						  
	# Perform the experiment loop
	for i in range(num_trials):
		print(f"Trial {i+1}/{num_trials}:")
		
		# Get the next sample from the NEST procedure. The sample 
		# represents the values of the stimulus dimensions to test in
		# the current trial
		sample = NEST_object.get_trial()
		print(f"Sample: {sample}")
		
		# Generate random response value		
		
		response = evaluate_CSF(sample)
		
		NEST_object.register_result(response)
	
	# Evaluate the psychometric function for the following example stimuli 
	# by providing values for each stimulus dimension and asking NEST to 
	# evaluate the probability of detection
	sample1 = [0.1, 5.0]
	result1 = NEST_object.evaluate(sample1)		
	print(f"sample1 probability of detection: {result1}")
	
	sample2 = [ [0.1, 5.0], [4.5, 6.7], [-1.0, -2.5] ]
	result2 = NEST_object.evaluate(sample2)
	print(f"sample2 probability of detection: {result2}")
	
	sample3 = np.array(sample2)
	result3 = NEST_object.evaluate(sample3)
	print(f"sample3  probability of detection: {result3}")
	
	sample4 = np.array(sample1)
	result4 = NEST_object.evaluate(sample4)
	print(f"sample4  probability of detection: {result4}")				  
			
	# Send a termination message to the NEST procedure. This closes down
	# the NEST procedure in an appropriate way.			  
	NEST_object.terminate()
	
	print(f"--- END OF TEST SCRIPT ---")	
	
	
if __name__ == "__main__":
	run_example_NEST()
