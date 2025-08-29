#include <iostream>
#include <vector>
#include <stack>
#include <queue>
#include <map>

using namespace std;

// This function performs a traversal using a stack.
// This is actually a Depth-First Search (DFS).
void dfs_with_stack(const map<int, vector<int>>& adj, int start_node) {
    // A map to keep track of visited nodes to avoid cycles
    map<int, bool> visited;
    // A stack to store nodes to be visited
    stack<int> s;

    // Start with the initial node
    s.push(start_node);

    cout << "Traversal using a stack (DFS): ";
    while (!s.empty()) {
        // Get the top node and remove it
        int current_node = s.top();
        s.pop();

        // If we haven't visited this node yet
        if (!visited[current_node]) {
            cout << current_node << " ";
            visited[current_node] = true;

            // Push all unvisited neighbors onto the stack.
            // Pushing in reverse order ensures we visit smaller numbers first.
            // If you iterate from begin() to end(), the order of discovery will be different.
            for (int i = adj.at(current_node).size() - 1; i >= 0; --i) {
                int neighbor = adj.at(current_node)[i];
                if (!visited[neighbor]) {
                    s.push(neighbor);
                }
            }
        }
    }
    cout << endl;
}

// This function performs the correct Breadth-First Search (BFS) using a queue.
void bfs_with_queue(const map<int, vector<int>>& adj, int start_node) {
    // A map to keep track of visited nodes
    map<int, bool> visited;
    // A queue to store nodes to be visited
    queue<int> q;

    // Start with the initial node
    q.push(start_node);
    visited[start_node] = true;

    cout << "Correct Breadth-First Search (BFS) using a queue: ";
    while (!q.empty()) {
        // Get the front node and remove it
        int current_node = q.front();
        q.pop();

        cout << current_node << " ";

        // Enqueue all unvisited neighbors
        for (int neighbor : adj.at(current_node)) {
            if (!visited[neighbor]) {
                q.push(neighbor);
                visited[neighbor] = true; // Mark as visited right when it's added to the queue
            }
        }
    }
    cout << endl;
}

int main() {
    // Represent the graph using an adjacency list
    map<int, vector<int>> adjacency_list = {
        {0, {1, 2}},
        {1, {3, 4}},
        {2, {5}},
        {3, {0}}, // Example of a back edge
        {4, {}},
        {5, {}}
    };

    int start_node = 0;

    // Call the function you requested
    dfs_with_stack(adjacency_list, start_node);

    // Call the correct BFS function
    bfs_with_queue(adjacency_list, start_node);

    return 0;
}