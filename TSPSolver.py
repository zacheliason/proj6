#!/usr/bin/python3
import math
import json

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

	def init_matrix_no_reduce(self, cities, ncities):  # Time: O(n^2), Space: O(n^2)
		m = np.zeros((ncities, ncities))

		# Time: O(n^2), Space: O(n^2)
		for i in range(len(cities)):
			for j in range(len(cities)):
				# We will treat this as constant since it doesn't increase with n cities
				m[i, j] = cities[i].costTo(cities[j])

		# Time: O(n^2), Space: O(n)
		return m

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

	# This can definitely be improved, I think I was reducing the matrix each step for no reason
	def greedy(self, time_allowance=60.0, init=False): # Time: O(n^3), Space: O(n^2)
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
				m, bound = self.init_matrix(cities, ncities)
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

					# Time: O(n^2), Space: O(1)
					m, bound = self.reduce(m)

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

		if init and results['cost'] < self.bssf_cost:
			self.BSSF = bssf
			self.bssf_cost = results['cost']

		return results


	def branchAndBound(self, time_allowance=60.0): # Time: O(n^3 * n!), Space: O(n^3 * n!)
		# initialize BSSF
		self.BSSF = None
		self.bssf_cost = math.inf
		self.lower_bound = math.inf
		self.greedy(init=True)

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

		# G = undirected graph that only has one edge between every two vertices
		G = nx.Graph()
		# MG = undirected multigraph that can have multiple edges between vertices. I don't think it's
		# the right application since our case is more of a directed graph but this is all work in progress anyways, can scrap it later
		MG = nx.MultiGraph()

		# This adds the edges to both G and MG
		for i in range(ncities):
			for j in range(ncities):
				MG.add_edge(j, i)
				G.add_edge(j, i)

		# This adds the edge costs to MG using the cost matrix m
		for i, j, z in MG.edges:
			if z == 1:
				distance = m[i, j]
				print(f"{i} -> {j}:  {distance}")
			else:
				distance = m[j, i]
				print(f"{j} -> {i}:  {distance}")

			MG.edges[i, j, z]['distance'] = distance


		# This adds the minimum cost for each edge in G using MG
		for i, j in G.edges:
			to = math.inf
			fro = math.inf
			if (i, j, 0) in MG.edges:
				to = MG.edges[i, j, 0]['distance']
			if (i, j, 1) in MG.edges:
				fro = MG.edges[i, j, 1]['distance']

			min_distance = min(to, fro)
			G.edges[i, j]['distance'] = min_distance

		# I wrote these lines to output the scenario to test in a scratch file
		# save = {}
		# for i in range(ncities):
		# 	for j in range(ncities):
		# 		save[f"{i} to {j}"] = m[i,j]
		# nx.write_edgelist(MG, "ZACHgraph.wb")
		# my_pos = {i: (cities[i]._x, cities[i]._y) for i in range(len(cities))}
		# with open("ZACHmatrix.txt", "w") as pos:
		# 	pos.write(json.dumps(save))

		# Find minimum spanning tree
		T = nx.minimum_spanning_tree(G, weight='distance')

		# Find nodes in minimum spanning tree T which have an odd degree
		odd_nodes = [v for v in T.nodes() if T.degree(v) % 2 == 1]

		# Create a negative distance value so 'max_weight_matching' really finds the minimum
		for i, j in G.edges:
			G.edges[i, j]['neg_distance'] = - G.edges[i, j]['distance']

		# Find the minimum perfect matching for pairs of odd degree nodes
		matching = nx.max_weight_matching(G.subgraph(odd_nodes), maxcardinality=True, weight='neg_distance')

		# New multigraph we'll add the edges from T and the matched pairs to
		H = nx.MultiGraph()

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
