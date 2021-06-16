

# Below is the Code that i borrowed from the authors of google research. I made some modifications to this 

import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt 
import networkx as nx

def draw_graph(adjacency , labels):
	## This assumes that adjacency has size NXN, N = Number of nodes
	## label0 = RED
	## label1 = BLUE
	## label2 = GREEN
	## label3 = YELLOW
	## So on...
	array = adjacency.toarray()
	G = nx.Graph()
	for i in range(array.shape[0]):
		for j in range(i):
			if(array[i][j] == 1):
				## Then add the edge to the graph
				G.add_edge(i,j)


	color_map = []
	for node in G:
		if labels[node] == 0:
			color_map.append('red')
		elif labels[node] == 1:
			color_map.append('blue')
		elif labels[node] == 2:
			color_map.append('green')
		else:
			color_map.append('yellow')

	nx.draw(G, node_color=color_map, with_labels=True)
	plt.show()


