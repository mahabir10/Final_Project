## This is my (mahabir) implementation of code
## This piece of code generates the graph using the probabilities p and q

import numpy as np
from scipy.stats import bernoulli
import scipy.sparse
from scipy.sparse import csr_matrix

## I have to create an Graph 

def create_sbm(n_nodes , p , q , n_classes):
	adjacency = [[0 for x in range(n_nodes)] for y in range(n_nodes)]
	labels = []
	features = []
	label_nodes = []

	for i in range(n_nodes):
		labels.append(i%n_classes)

	for i in range(n_nodes):
		for j in range(n_nodes):
			if(labels[i] == labels[j]):
				## Add edge with probability p
				randi = bernoulli.rvs(p=p)
				if(randi == 1):
					adjacency[i][j] = 1
					adjacency[j][i] = 1
			else:
				randi = bernoulli.rvs(p=q)
				if(randi == 1):
					adjacency[i][j] = 1
					adjacency[j][i] = 1

	## For the sbm we will treate feature as the degree of the node
	for i in range(n_nodes):
		c_zero = 0
		c_one = 0
		c_two = 0
		c_three = 0
		for j in range(n_nodes):
			if(adjacency[i][j] == 1 and labels[j] == 0):
				c_zero = c_zero + 1
			elif(adjacency[i][j] == 1 and labels[j] == 1):
				c_one = c_one + 1
			elif(adjacency[i][j] == 1 and labels[j] == 2):
				c_two = c_two + 1
			elif(adjacency[i][j] == 1):
				c_three = c_three + 1
		features.append([c_zero , c_one , c_two , c_three])

	## I have to give more to the features
	## Like number of similar nodes friend with a node

	for i in range(n_nodes):
		label_nodes.append(i)

	## But before returning i have to make adjacency matrix as scipy csr matrix
	features = csr_matrix(features)
	adjacency = csr_matrix(adjacency)
	labels = np.array(labels)
	label_nodes = np.array(label_nodes)

	return adjacency,features,labels,label_nodes

