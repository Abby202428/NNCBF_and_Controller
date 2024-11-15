# Import necessary libraries
import heapq  # For priority queue implementation
import numpy as np  # For numerical operations

# The dijkstra.py file contains an implementation of Dijkstra's algorithm for pathfinding.
# This code can be used to compute the shortest path in a graph, which may represent environments for multi-agent navigation.

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

# Example usage of the Graph class and Dijkstra's algorithm
def example_usage():
    # Create a graph with 5 nodes
    graph = Graph(5)
    # Add edges to the graph with their respective weights
    graph.add_edge(0, 1, 2)
    graph.add_edge(0, 2, 4)
    graph.add_edge(1, 2, 1)
    graph.add_edge(1, 3, 7)
    graph.add_edge(2, 4, 3)
    # Run Dijkstra's algorithm from the start node 0
    distances = graph.dijkstra(0)
    # Print the shortest distances from node 0 to all other nodes
    print("Shortest distances from node 0:", distances)

# Run the example usage if the script is executed directly
if __name__ == "__main__":
    example_usage()
