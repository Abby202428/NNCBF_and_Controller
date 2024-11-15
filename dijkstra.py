# Import necessary libraries
import heapq  # For priority queue implementation
import numpy as np  # For numerical operations

# The dijkstra.py file contains an implementation of Dijkstra's algorithm for pathfinding.
# This code can be used to compute the shortest path in a graph, which may represent environments for multi-agent navigation.

# Overview of the Dijkstra's algorithm implementation in this file:
# 1. **Graph Class**: The Graph class represents the environment as a graph, where nodes represent locations, and edges represent paths between locations.
# 2. **add_edge Method**: This method adds edges between nodes in the graph, with a specified weight representing the distance or cost of traveling between those nodes.
# 3. **dijkstra Method**: This method implements Dijkstra's algorithm to find the shortest path from a given starting node to all other nodes in the graph. It maintains a priority queue to always expand the nearest unvisited node.
#    - The `distances` list stores the shortest distance from the start node to every other node.
#    - The algorithm iterates through each node's neighbors, updating distances if a shorter path is found.
#    - This ensures that the final distances list contains the shortest paths from the start node to all reachable nodes.
# 4. **Example Usage**: The `example_usage` function demonstrates how to use the Graph class to create a graph, add edges, and compute the shortest paths using Dijkstra's algorithm.

# Class to represent a graph using an adjacency list representation
class Graph:
    def __init__(self, num_nodes):
        """
        Initializes the Graph object.
        Args:
            num_nodes: The number of nodes in the graph.
        """
        self.num_nodes = num_nodes  # Total number of nodes in the graph
        self.edges = [[] for _ in range(num_nodes)]  # Adjacency list to store edges

    def add_edge(self, u, v, weight):
        """
        Adds an edge to the graph.
        Args:
            u: The starting node of the edge.
            v: The ending node of the edge.
            weight: The weight of the edge between nodes u and v.
        """
        self.edges[u].append((v, weight))  # Add edge from u to v with the given weight
        self.edges[v].append((u, weight))  # Since the graph is undirected, add edge from v to u as well

    def dijkstra(self, start_node):
        """
        Performs Dijkstra's algorithm to find the shortest path from a start node to all other nodes.
        Args:
            start_node: The node from which to start the shortest path search.
        Returns:
            distances: A list of shortest distances from the start node to each other node.
        """
        # Initialize distances to all nodes as infinity, except the start node which is set to 0
        distances = [float('inf')] * self.num_nodes
        distances[start_node] = 0
        # Priority queue to store (distance, node) pairs
        priority_queue = [(0, start_node)]

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)  # Pop the node with the smallest distance

            # If the current distance is greater than the recorded distance, skip processing this node
            if current_distance > distances[current_node]:
                continue

            # Iterate over the neighbors of the current node
            for neighbor, weight in self.edges[current_node]:
                distance = current_distance + weight  # Calculate distance to the neighbor

                # If a shorter path is found, update the distance and push to the priority queue
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))

        return distances  # Return the list of shortest distances

# Example usage of the Graph class and Di
