#!/usr/bin/python3
import math
import json
import re

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))


import time
import numpy as np
from TSPClasses import *
from copy import copy
import heapq as hq
import networkx as nx


class SearchState:
	def __init__(self, m, parentBound, i, ncities, visited):
		# Each state will keep track of its current bound cost, matrix, and which cities it has visited
		self.bound = parentBound
		self.matrix = m
		self.visited = visited

		self.visited.append(i)
		self.ncities = ncities

		# current is the current city it's visiting
		self.current = i


	def __lt__(self, other):
		# Override comparison for the prioirity queue to break ties
		if len(self.visited) != len(other.visited):
			return len(self.visited) < len(other.visited)
		return self.current < other.current

	def reduce(self, m): # Time: O(n^2), Space: O(n)
		# Grab the min values from each row
		# Time: O(n), Space: O(n) since min is already stored in a numpy array
		rows_min = np.min(m, axis=1)[:, None]
		# Replace any occurrences of inf with 0
		# Time: O(n), Space: O(1)
		rows_min[np.isinf(rows_min)] = 0
		# Subtract mins rowwise from matrix
		# Time: O(n^2), Space: O(1)
		m = m - rows_min

		# Grab the min values from each column
		# Time: O(n), Space: O(n) since min is already stored in a numpy array
		cols_min = np.min(m, axis=0)[:, None]
		# Replace any occurrences of inf with 0
		# Time: O(n), Space: O(1)
		cols_min[np.isinf(cols_min)] = 0
		# Subtract mins columnwise from matrix
		# Time: O(n^2), Space: O(1)
		m = m - np.transpose(cols_min)

		bound = sum(cols_min) + sum(rows_min)

		# Return sum of the row and column mins
		return m, bound[0]

	def visit(self, m, i, j, current_bound): # Time: O(n^2), Space: O(n)
		if math.isinf(m[i, j]):
			return None, math.inf

		current_bound += m[i, j]

		# Time: O(n), Space: O(1)
		m[:, j] = math.inf
		m[i, :] = math.inf
		m[j, i] = math.inf

		# Time: O(n^2), Space: O(n)
		m, add_bound = self.reduce(m)
		current_bound += add_bound

		return m, current_bound

	def expand(self): # Time: O(n^3), Space: O(n^3)
		i = self.current

		states = []

		# Time: O(n^3), Space: O(n^3)
		for j in range(self.ncities):
			if j == i:
				continue

			if j == self.visited[0] and len(self.visited) != self.ncities:
				continue

			# Time: O(n^2), Space: O(n)
			new_m, new_bound = self.visit(copy(self.matrix), i, j, self.bound)
			# Space: O(n^2)
			s = SearchState(new_m, new_bound, j, self.ncities, copy(self.visited))
			states.append(s)

		return states


class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None
		self.BSSF = None
		self.bssf_cost = math.inf
		self.lower_bound = math.inf


	def setupWithScenario( self, scenario ):
		self._scenario = scenario

	def reduce(self, m): # Time: O(n^2), Space: O(n)
		# Grab the min values from each row
		# Time: O(n), Space: O(n) since min is already stored in a numpy array
		rows_min = np.min(m, axis=1)[:, None]
		# Replace any occurrences of inf with 0
		# Time: O(n), Space: O(1)
		rows_min[np.isinf(rows_min)] = 0
		# Subtract mins rowwise from matrix
		# Time: O(n^2), Space: O(1)
		m = m - rows_min

		# Grab the min values from each column
		# Time: O(n), Space: O(n) since min is already stored in a numpy array
		cols_min = np.min(m, axis=0)[:, None]
		# Replace any occurrences of inf with 0
		# Time: O(n), Space: O(1)
		cols_min[np.isinf(cols_min)] = 0
		# Subtract mins columnwise from matrix
		# Time: O(n^2), Space: O(1)
		m = m - np.transpose(cols_min)

		bound = sum(cols_min) + sum(rows_min)

		# Return sum of the row and column mins
		return m, bound[0]


	def defaultRandomTour(self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	def init_matrix(self, cities, ncities): # Time: O(n^2), Space: O(n^2)
		m = np.zeros((ncities, ncities))

		# Time: O(n^2), Space: O(n^2)
		for i in range(len(cities)):
			for j in range(len(cities)):
				# We will treat this as constant since it doesn't increase with n cities
				m[i,j] = cities[i].costTo(cities[j])

		# Time: O(n^2), Space: O(n)
		return self.reduce(m)



	def branchAndBound(self, time_allowance=60.0): # Time: O(n^3 * n!), Space: O(n^3 * n!)
		# initialize BSSF
		self.BSSF = None
		self.bssf_cost = math.inf
		self.lower_bound = math.inf

		init_solution = self.greedy()

		self.BSSF = init_solution['soln']
		self.bssf_cost = init_solution['cost']

		results = {}
		results['pruned'] = 0
		results['updates'] = 0
		results['total'] = 1
		results['max'] = 0
		solutions = [self.BSSF] if self.BSSF else []
		heap = []

		# Time: O(n), Space: O(n)
		cities = self._scenario.getCities()
		ncities = len(cities)

		start_time = time.time()

		# Time: O(n^2), Space: O(n^2)
		m, parent_bound = self.init_matrix(cities, ncities)
		self.lower_bound = parent_bound

		# Helps determine searchstate priority in queue
		DEPTH_FACTOR = self.lower_bound / (ncities * .9)

		start = self.BSSF.route[0]._index
		s = SearchState(m, parent_bound, start, ncities, visited=[])
		heap.append((parent_bound, parent_bound, s))

		# Time: O(n^3 * n!), Space: O(n^3 * n!)
		while time.time() - start_time < time_allowance and len(heap) > 0:
			if len(heap) > results['max']:
				results['max'] = len(heap)

			# Time: O(log(n)), Space: O(1)
			s = hq.heappop(heap)[-1]

			# Time: O(n^3), Space: O(n^3)
			states = s.expand()
			results['total'] += len(states)

			# Time: O(n^2), Space: O(n^2)
			for state in states:
				# Skip states that have finished visiting all cities and either can't connect the first and last cities
				# or have a bound of infinity
				if len(state.visited) == ncities and (math.isinf(state.bound) or (state.matrix is not None and math.isinf(state.matrix[state.current, start]))):
					continue

				if len(state.visited) == ncities and state.bound < math.inf:
					solutions.append(state)

					# Time: O(n), Space: O(n)
					cities_visited = [cities[x] for x in state.visited]

					if state.bound < self.bssf_cost:
						# Time: O(n), Space: O(n)
						self.BSSF = TSPSolution(cities_visited)
						self.bssf_cost = self.BSSF.cost

						# Now that cost is updated, prune unneccessary states
						new_heap = []
						for x in heap:
							if x[1] > self.bssf_cost:
								results['pruned'] += 1
								continue
							else:
								new_heap.append(x)

						heap = new_heap
						hq.heapify(heap)

						results['updates'] += 1

					if state.bound == self.lower_bound:
						results['pruned'] += len(heap)
						heap = []
						break
					continue

				if state.bound < self.bssf_cost:
					# Time: O(log(n)), Space: O(1)
					hq.heappush(heap, (state.bound - (DEPTH_FACTOR * len(state.visited)), state.bound, state))
				else:
					results['pruned'] += 1

		if time.time() - start_time > time_allowance:
			print('ran out of time!')
			print(f"{len(heap)} more states on the heap")

		end_time = time.time()
		results['cost'] = self.BSSF.cost if self.BSSF else math.inf
		results['time'] = end_time - start_time
		results['count'] = len(solutions)
		results['soln'] = self.BSSF
		results['pruned'] += len(heap)

		return results
















	####################################################################
	# @ HOAN AND KYLE
	# we only need to submit the code for here onward
	####################################################################

	import networkx as nx
	import numpy as np

	def init_matrix_no_reduce(self, cities, ncities):  # Time: O(n^2), Space: O(n^2)
		m = np.zeros((ncities, ncities))

		# Time: O(n^2), Space: O(n^2)
		for i in range(len(cities)):
			for j in range(len(cities)):
				# We will treat this as constant since it doesn't increase with n cities
				m[i, j] = cities[i].costTo(cities[j])

		# Time: O(n^2), Space: O(n)
		return m

	# This can definitely be improved, I think I was reducing the matrix each step for no reason
	def greedy(self, time_allowance=60.0): # Time: O(n^3), Space: O(n^2)
		results = {}
		cities = self._scenario.getCities()

		# Time: O(n), Space: O(1)
		ncities = len(cities)
		visited = []

		foundTour = False
		start_time = time.time()
		count = 1

		print()

		while time.time()-start_time < time_allowance:
			if foundTour:
				break

			# Time: O(n^3), Space: O(n^2)
			for start in reversed(range(ncities)):
				if foundTour:
					break

				visited = []
				i = start

				# Time: O(n^2), Space: O(n^2)
				m = self.init_matrix_no_reduce(cities, ncities)
				while not foundTour:
					if len(visited) == ncities:
						for v in visited:
							print(v._name, end=" ")
						print()

						foundTour = True
						break

					visited.append(cities[i])

					if sum(m[i, :]==math.inf) == ncities:
						print('fail')
						# FAILURE
						break

					j = np.argmin(m[i])

					# Time: O(n), Space: O(n)
					m[:, j] = math.inf
					m[i, :] = math.inf
					m[j, i] = math.inf

					i = j
			break


		# Time: O(n), Space: O(n)
		bssf = TSPSolution(visited)

		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count if foundTour else 0
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None

		if not foundTour:
			results = self.defaultRandomTour()
			results['time'] = end_time - start_time

		return results

	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
	def fancy(self,time_allowance=60.0):
		results = {}
		start_time = time.time()

		cities = self._scenario.getCities()
		ncities = len(cities)

		# Get cost matrix between all cities
		m = self.init_matrix_no_reduce(cities, ncities)

		hard_mode = False
		if (m == m.T).all():
			# Easy Mode
			# If this block of code is reached it means the graph is undirected (cost of i to j == cost of j to i)

			# G = undirected graph that only has one edge between every two vertices
			G = nx.Graph()

			# This adds the edges to both G and MG
			for i in range(ncities):
				for j in range(ncities):
					G.add_edge(i, j)
					G.edges[i, j]['distance'] = m[i, j]
		else:
			# Hard Mode
			# If this block of code is reached it means the graph is directed (cost of i to j != cost of j to i)
			hard_mode = True

			# Creates an undirected graph from a matrix represented directed graph
			# makes an incoming and outcoming node for each of n nodes
			G = nx.Graph()
			MG = nx.MultiGraph() # Also makes a Multigraph, I was playing around with consolidating it into a Graph
			for i in range(ncities):
				for j in range(ncities):
					i_out = f"{i}_out"
					j_in = f"{j}_in"

					G.add_edge(str(i), i_out)
					G.edges[str(i), i_out]['distance'] = 0

					G.add_edge(i_out, j_in)
					G.edges[i_out, j_in]['distance'] = m[i, j]

					G.add_edge(j_in, str(j))
					G.edges[j_in, str(j)]['distance'] = 0

					MG.add_edge(i, j)

			for i, j, z in MG.edges:
				if z == 0:
					distance = m[i, j]
				else:
					distance = m[j, i]

				MG.edges[i, j, z]['distance'] = distance

		# Find minimum spanning tree
		T = nx.minimum_spanning_tree(G, weight='distance')

		if hard_mode:
			minG = nx.Graph()
			for i, j, z in MG.edges:
				i = str(i)
				j = str(j)
				minG.add_edge(j, i)

			for i, j in minG.edges:
				i = str(i)
				j = str(j)

				to = math.inf
				fro = math.inf
				if (i, j, 0) in MG.edges:
					to = MG.edges[i, j, 0]['distance']
				if (i, j, 1) in MG.edges:
					fro = MG.edges[i, j, 1]['distance']

				min_distance = min(to, fro)
				minG.edges[i, j]['distance'] = min_distance

			# This creates a new minimum spanning tree without the incoming and outgoing dummy nodes from before
			parsedT = nx.Graph()
			for i, j in T.edges:
				i_match = re.search(r"(\d+)(_in)*(_out)*", i)
				if i_match:
					i = i_match.group(1)
				j_match = re.search(r"(\d+)(_in)*(_out)*", j)
				if j_match:
					j = j_match.group(1)

				if i != j:
					parsedT.add_edge(i, j)

			T = parsedT

		# Find nodes in minimum spanning tree T which have an odd degree
		odd_nodes = [v for v in T.nodes() if T.degree(v) % 2 == 1]

		# Have to use the minG Graph for hard mode because it consolidates the paths to and fro into one
		if hard_mode:
			# Create a dummy value for negative distance so that 'max_weight_matching' finds the min
			for i, j in minG.edges:
				minG.edges[i, j]['neg_distance'] = - minG.edges[i, j]['distance']

			# Find the minimum perfect matching for pairs of odd degree nodes
			matching = nx.max_weight_matching(minG.subgraph(odd_nodes), maxcardinality=True, weight='neg_distance')
			print('okay')
		else:
			# Create a dummy value for negative distance so that 'max_weight_matching' finds the min
			for i, j in G.edges:
				G.edges[i, j]['neg_distance'] = - G.edges[i, j]['distance']

		# Find the minimum perfect matching for pairs of odd degree nodes
		matching = nx.max_weight_matching(G.subgraph(odd_nodes), maxcardinality=True, weight='neg_distance')

		# New multigraph H we'll add the edges from T and the matched pairs to
		H = nx.MultiGraph()


		if hard_mode:
			# Have to convert indices back to integers on hard mode
			# (because they turn to strings when we add nodes i_in and i_out to each node i)
			# Add edges from T and matching to multigraph H
			H.add_nodes_from([int(x) for x in T.nodes])
			H.add_edges_from([(int(x[0]), int(x[1])) for x in T.edges()])
			H.add_edges_from([(int(x[0]), int(x[1])) for x in matching])
		else:
			# Add edges from T and matching to multigraph H
			H.add_nodes_from(range(ncities))
			H.add_edges_from(T.edges())
			H.add_edges_from(matching)

		# Find a path that traverses each edge only once (eulerian circuit)
		initial_tour = list(nx.eulerian_circuit(H, source=0))

		# Find a path that traverses each vertex only once, ie it skips the repeat vertices (hamiltonian circuit)
		tour = [0]
		for (i, j) in initial_tour:
			if j not in tour:
				tour.append(j)

		# We were using the city indices before so this converts indices to city objects
		cities_visited = [cities[i] for i in tour]

		# Find the solution from the list of ordered cities
		BSSF = TSPSolution(cities_visited)

		# This was me trying to see if reversing the list would make it work on hard, I don't think it ever helps
		if math.isinf(BSSF.cost):
			cities_visited.reverse()
			BSSF = TSPSolution(cities_visited)

		end_time = time.time()
		results['cost'] = BSSF.cost
		results['time'] = end_time - start_time
		results['count'] = 1
		results['soln'] = BSSF
		results['max'] = None
		results['total'] = None
		results['pruned'] = None

		return results
