# Below is the Code that i borrowed from the authors of google research. I made some modifications to this 

import metrics
import sbm
import visualize
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt 
import networkx as nx
import scipy.sparse
from scipy.sparse import csr_matrix

adjacency = [[1,1,1,0,0,0],[1,1,1,0,0,0],[1,1,1,1,0,0],[0,0,1,1,1,1],[0,0,0,1,1,1],[0,0,0,1,1,1]]
clusters = [0,0,0,1,1,1]


adjacency = np.array(adjacency)

data_bern = bernoulli.rvs(p=0.1)

print(data_bern)


## DRAWING GRAPH ##
"""
G = nx.Graph()
p = 0.9
q = 0.1
n_classes = 4
n_nodes = 50
labels = []

for i in range(n_nodes):
	labels.append(i%n_classes)

for i in range(n_nodes):
	for j in range(n_nodes):
		if(labels[i] == labels[j]):
			## Add edge with probability p
			randi = bernoulli.rvs(p=p)
			if(randi == 1):
				G.add_edge(i,j)
		else:
			randi = bernoulli.rvs(p=q)
			if(randi == 1):
				G.add_edge(i,j)


## G = nx.erdos_renyi_graph(20, 0.1)
## print(type(G))

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
"""

adjacency,features,labels,label_nodes = sbm.create_sbm(20 , 0.9 , 0.1 , 4)
print(labels.shape)
print(labels)

visualize.draw_graph(adjacency , labels)

## print(metrics.conductance( adjacency , clusters ))


