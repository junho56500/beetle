#include <iostream>
#include <vector>
#include <queue>
#include <limits> // Required for std::numeric_limits


// Input
// The input consists of:
// The total number of vertices, V, in the graph.
// A list of edges, where each edge is represented by a triple: (u,v,w), indicating a directed edge from vertex u to vertex v with weight w.
// The source vertex, s, from which to calculate all shortest paths.

// Output
// The output should be a list of distances, where the i-th element represents the shortest distance from the source vertex s to vertex i.

// Example

// Graph
// Consider a graph with 6 vertices, labeled 0 through 5. The edges and their weights are as follows:
// (0,1,10)
// (0,3,30)
// (0,4,100)
// (1,2,50)
// (2,4,10)
// (3,2,20)
// (3,4,60)
// (4,5,5)

// Expected Output
// Given the source vertex s=0, the algorithm should produce the following shortest distances:
// Distance to vertex 0: 0
// Distance to vertex 1: 10
// Distance to vertex 2: 50
// Path: 0→3→2 (Cost: 30+20=50) is shorter than 0→1→2 (Cost: 10+50=60).
// Distance to vertex 3: 30
// Distance to vertex 4: 60
// Path: 0→3→2→4 (Cost: 30+20+10=60) is shorter than 0→4 (Cost: 100).
// Distance to vertex 5: 65
// Path: 0→3→2→4→5 (Cost: 30+20+10+5=65).

// A pair to represent an edge: {weight, destination_vertex}
using Edge = std::pair<int, int>;
// An adjacency list to represent the graph: {source_vertex: [{weight, destination_vertex}, ...]}
using Graph = std::vector<std::vector<Edge>>;

// Function to find the shortest path from a source node to all other nodes
// using Dijkstra's algorithm.
std::vector<int> dijkstra(const Graph& graph, int source) {
    int num_vertices = graph.size();
    
    // Create a vector to store the shortest distances from the source.
    // Initialize all distances to infinity.
    std::vector<int> distances(num_vertices, std::numeric_limits<int>::max());

    // Create a min-priority queue to store pairs of {distance, vertex}.
    // std::pair sorts by the first element by default, so we store distance first.
    // The std::greater<Edge> makes it a min-heap.
    std::priority_queue<Edge, std::vector<Edge>, std::greater<Edge>> min_heap;

    // The distance to the source node is 0.
    distances[source] = 0;
    min_heap.push({0, source});

    // Loop until the priority queue is empty.
    while (!min_heap.empty()) {
        // Extract the vertex with the minimum distance.
        int current_distance = min_heap.top().first;
        int current_vertex = min_heap.top().second;
        min_heap.pop();

        // If the extracted distance is greater than the already known shortest
        // distance, we can skip this entry as it's a stale one.
        if (current_distance > distances[current_vertex]) {
            continue;
        }

        // Iterate through all neighbors of the current vertex.
        for (const auto& edge : graph[current_vertex]) {
            int neighbor = edge.second;
            int weight = edge.first;

            // Check if a shorter path to the neighbor is found.
            if (distances[current_vertex] + weight < distances[neighbor]) {
                // Update the distance to the neighbor.
                distances[neighbor] = distances[current_vertex] + weight;
                // Push the updated neighbor onto the priority queue.
                min_heap.push({distances[neighbor], neighbor});
            }
        }
    }
    
    return distances;
}

int main() {
    // Example graph represented as an adjacency list.
    // The format is { {weight, destination}, ... }.
    // Vertex indices: 0, 1, 2, 3, 4, 5
    Graph graph(6);
    graph[0].push_back({10, 1});
    graph[0].push_back({30, 3});
    graph[0].push_back({100, 4});
    graph[1].push_back({50, 2});
    graph[2].push_back({10, 4});
    graph[3].push_back({20, 2});
    graph[3].push_back({60, 4});
    graph[4].push_back({5, 5});

    int source_vertex = 0;
    std::vector<int> shortest_distances = dijkstra(graph, source_vertex);

    std::cout << "Shortest distances from source vertex " << source_vertex << ":" << std::endl;
    for (int i = 0; i < shortest_distances.size(); ++i) {
        std::cout << "To vertex " << i << ": ";
        if (shortest_distances[i] == std::numeric_limits<int>::max()) {
            std::cout << "Not reachable" << std::endl;
        } else {
            std::cout << shortest_distances[i] << std::endl;
        }
    }

    return 0;
}
