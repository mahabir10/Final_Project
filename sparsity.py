## This is my (mahabir) implementation of code

import numpy as np
from scipy.stats import bernoulli

def add_sparsity( adjacency , prob ):
	## What this function does is that it adds sparsity to the graph
	## And finds out the result
	## This Code assumes that adjacency matrix size = N X N

	print(type(adjacency) , adjacency.shape)

	temp_adjacency = adjacency.toarray()

	print(type(temp_adjacency) , temp_adjacency.shape)

	for i in range(temp_adjacency.shape[0]):
		for j in range(temp_adjacency.shape[1]):
			if(temp_adjacency[i][j] == 1):
				adjacency._set_arrayXarray(i, j, bernoulli.rvs(p=prob)) 

	return adjacency

