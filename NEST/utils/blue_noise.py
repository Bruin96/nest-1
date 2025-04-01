import numpy as np
import time
from sklearn.neighbors import BallTree
from scipy.spatial import KDTree

def create_KDTree(points):
	return KDTree(points, boxsize = 1.0)


def generate_next_sample(points, lbs, ubs, m=3):
	N = points.shape[0]
	rng = np.random.default_rng()
	num_dims = len(lbs)
	
	num_candidates = m*N
	candidates = rng.uniform(size=(num_candidates, num_dims))
	
	# Find best candidate to use as the next sample
	ball_tree = create_KDTree(points)
	point = find_max_wraparound_dist(ball_tree, candidates)
	point = point * (ubs-lbs) + lbs
	
	return point


def generate_blue_noise(num_samples, num_dims, lbs, ubs, m=3, seed=None):		
	rng = np.random.default_rng(seed=seed)
	point = rng.uniform(size=num_dims)
	
	samples = np.zeros((num_samples, num_dims))
	samples[0, :] = point
	
	ball_tree = create_KDTree(np.atleast_2d(samples[0,:]))
	
	for i in range(1, num_samples):
		num_candidates = m*i
		candidates = rng.uniform(size=(num_candidates, num_dims))
		
		# Find and store best candidate to use as the next sample
		point = find_max_wraparound_dist(ball_tree, candidates)
		samples[i, :] = point
		ball_tree = create_KDTree(np.atleast_2d(samples[0:i+1,:]))
	
	# Scale samples using lbs and ubs
	lbs = np.array(lbs)
	ubs = np.array(ubs)
	
	samples = samples * (ubs-lbs) + lbs
	
	return samples
	
'''
	This function finds the point in the candidates set with the highest 
	minimum distance any point in the points set. This candidate is then
	selected as the next sample in the blue noise sample. We use a kd-tree
	to efficiently implement the nearest neighbour evaluation, such that
	the algorithm is O(k log n + n log n), where k is the size of the 
	candidates set and n is the size of the points set. The brute-force
	approach has a time complexity of O(k*n), in comparison. Since k ~ n,
	this becomes O(n log n) vs. O(n^2), indicating large savings compared
	to the brute-force algorithm.
'''
def find_max_wraparound_dist(ball_tree, candidates):
	# Find nearest neighbours of the points set
	(dists, _) = ball_tree.query(candidates, workers=-1)
	
	# select maximum distance for nearest neighbours
	best_idx = np.argmax(dists)
	
	return candidates[best_idx, :]
