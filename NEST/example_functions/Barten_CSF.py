import numpy as np


class Barten_CSF:
	def __init__(self, param_dict=None):
		# Set unchangeable parameters
		self.N_c0 = 12000
		self.N_r0 = 12000
		self.N_g0 = 36000
		
		self.sigma_ret_base = 0.40 / 60.0
		self.Phi_0 = 3e-8
		self.u_0 = 7.0
		self.X_max = 12.0
		
		self.T = 0.1
		self.N_max = 15.0
		self.C_ab = 0.08 / 60.0
		
		# Set changeable parameters to default
		self.reset_params()
			
		# Update changeable parameters if param_dict was given
		if param_dict is not None:
			self.set_params(param_dict)
			
			
	def reset_params(self):
		self.e_g = 3.3
		self.sigma_0_base = 0.50 / 60.0
		self.k = 3.0
		self.eta = 0.03
		self.M_factor = 0.05
		self.r_e = 7.633
		
		
	def set_params(self, param_dict):
		for (key, value) in param_dict.items():
			if key == 'e_g':
				self.e_g = value
			if key == 'sigma_0_base':
				self.sigma_0_base = value / 60.0
			if key == 'k':
				self.k = value
			if key == 'eta':
				self.eta = value
			if key == 'M_factor':
				self.M_factor = value
			if key == 'r_e':
				self.r_e = value
			
	'''
		Computes the Barten CSF, which is a function of both spatial frequency u
		and eccentricity e. Other parameters are:
		-	p: photon factor, reflects the ...
		-	N_eyes: The number of eyes looking at the stimulus. Typical values
			are 1 or 2.
		-	e: Eccentricity at which the stimulus is viewed. Defaults to 0.0 (foveal vision).
		- 	is_round: indicates whether the stimulus is circular or not
		- 	X_0: the width or diameter of the stimulus.
		- 	Y_0: The height of the stimulus. If Y_0 is None, then Y_0 = X_0. If
			is_round = True, then Y_0 is ignored.
	'''
	def evaluate_CSF(self, u, L=100, p=1.285e6, N_eyes=2, e=0.0, \
								is_round=False, X_0=10.0, Y_0=None):
		
		if Y_0 is None: # Assume a square field
			Y_0 = X_0
								
		# Define cell densities
		N_g = self.N_g0 * ( 0.85/(1 + (e/0.45)**2) + 0.15/(1 + (e/self.e_g)**2) )
		
		# Compute effective midget cell count
		d_c0 = 1/1.12 * 14804.6
		e_m = 41.03
		a_k = 0.98
		r_2 = 1.071
		d_mf = 2*d_c0 * (1 + e/e_m)**(-1) * (a_k*(1 + e/r_2)**(-2) + (1-a_k)* np.exp(-e/self.r_e))
		
		e_t = 0.0
		N_e = d_mf
		
		# Compute sigma_0
		sigma_ret = 1 / np.sqrt(7.2*np.sqrt(3)*N_e)
		sigma_00 = np.sqrt(self.sigma_0_base**2 - self.sigma_ret_base**2)
		sigma_0 = np.sqrt(sigma_00**2 + sigma_ret**2)
		
		
		# Compute Phi_0
		Phi_0 = self.Phi_0 * self.N_g0/N_g
		
		# Compute u_0
		u_0 = 7.0
		u_0 = self.u_0 * (N_g/self.N_g0)**0.5 * (0.85/(1+(e/4.0)**2) + \
							0.13/(1+(e/20)**2) + 0.02)**(-0.5)
		
		# Compute eta
		eta = self.eta * (0.4/(1+(e/7)**2) + 0.48/(1+(e/20)**2) + 0.12)
		
		
		# Compute X_max and Y_max
		X_max = self.X_max*(0.85/(1+(e/4)**2) + 0.15/(1+(e/12)**2))**(-0.5)
		Y_max = X_max
		
		# Compute expected pupil diameter (mm)
		if is_round:
			area = np.pi/4 * X_0**2
		else:
			area = X_0*Y_0
			
		d = 5 - 3*np.tanh(0.4*np.log(L*area / 1600))
	    		
		# Compute retinal illuminance
		E = L * np.pi * (0.5*d)**2 * (1 - (d/9.7)**2 + (d/12.4)**4)
		
		# Compute sigma
		sigma = np.sqrt(sigma_0**2 + self.C_ab**2 * d**2)
		
		# Compute X and Y
		if is_round:
			area_X = np.pi/4 * X_0**2
			area_Y = np.pi/4 * X_0**2
		else:
			area_X = X_0**2
			area_Y = Y_0**2
			
		X = (1/area_X + 1/X_max**2 + ((0.5*X_0)**2 + 4*e**2)/((0.5*X_0)**2 + \
										e**2) * (u/self.N_max)**2)**(-0.5)
		Y = (1/area_Y + 1/Y_max**2 + ((0.5*X_0)**2)/((0.5*X_0)**2 + e**2) * \
												(u/self.N_max)**2)**(-0.5)
		
		# Compute contrast sensitivity
		M_opt = np.exp(-2*np.pi**2 * sigma**2 * u**2)
		M_lat = 1-np.exp(-(u/u_0)**2)
		Phi_photon = 1/(eta*p*E+1e-20)
				
		S = 1/(2*self.k) * M_opt * np.sqrt(X*Y*self.T/(Phi_photon + Phi_0/M_lat))
		
		if N_eyes == 2:
			S *= np.sqrt(2)
		
		return S
	
