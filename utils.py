# coding=utf-8
## This is the code i modified from the code of the paper [Graph Clustering with Graph Neural Networks]
## https://arxiv.org/abs/2006.16904).
## I do not own any licence for it

"""Helper functions for graph processing."""
import numpy as np
import scipy.sparse
from scipy.sparse import base


def normalize_graph(graph,
                    normalized = True,
                    add_self_loops = True):
  """Normalized the graph's adjacency matrix in the scipy sparse matrix format.

  Args:
    graph: A scipy sparse adjacency matrix of the input graph.
    normalized: If True, uses the normalized Laplacian formulation. Otherwise,
      use the unnormalized Laplacian construction.
    add_self_loops: If True, adds a one-diagonal corresponding to self-loops in
      the graph.

  Returns:
    A scipy sparse matrix containing the normalized version of the input graph.
  """
  if add_self_loops:
    graph = graph + scipy.sparse.identity(graph.shape[0])
  degree = np.squeeze(np.asarray(graph.sum(axis=1)))
  if normalized:
    with np.errstate(divide='ignore'):
      inverse_sqrt_degree = 1. / np.sqrt(degree)
    inverse_sqrt_degree[inverse_sqrt_degree == np.inf] = 0
    inverse_sqrt_degree = scipy.sparse.diags(inverse_sqrt_degree)
    return inverse_sqrt_degree @ graph @ inverse_sqrt_degree
  else:
    with np.errstate(divide='ignore'):
      inverse_degree = 1. / degree
    inverse_degree[inverse_degree == np.inf] = 0
    inverse_degree = scipy.sparse.diags(inverse_degree)
    return inverse_degree @ graph
