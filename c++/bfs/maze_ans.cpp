// -----------------------------------------------------------------------------
// Coding Test Problem: Maze Solver (Full Search)
// -----------------------------------------------------------------------------
// Problem Description:
// You are given a rectangular grid representing a maze.
// The grid contains:
// - 'S' for the starting point
// - 'E' for the end point
// - '.' for an empty path
// - '#' for a wall
//
// Your task is to write a program that finds the length of the shortest path
// from 'S' to 'E'. You can only move up, down, left, and right.
//
// If no path exists from 'S' to 'E', the program should return -1.
//
// Input Conditions:
// - A 2D vector of characters representing the maze.
// - The maze will have at least one 'S' and one 'E'.
// - Dimensions of the maze will be between 1x1 and 10x10.
//
// Output Conditions:
// - The length of the shortest path, or -1 if no path exists.

#include <iostream>
#include <vector>
#include <queue>
#include <utility>

// Structure to hold coordinates and distance from the start.
struct State {
    int x;
    int y;
    int dist;
};

// This function finds the shortest path using a Breadth-First Search (BFS).
// BFS is a classic full-search algorithm for finding the shortest path in an unweighted graph.
int solveMaze(const std::vector<std::vector<char>>& maze) {
    // Get maze dimensions.
    int rows = maze.size();
    if (rows == 0) return -1;
    int cols = maze[0].size();

    // Find the starting point 'S'.
    int start_x, start_y;
    bool start_found = false;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (maze[i][j] == 'S') {
                start_x = i;
                start_y = j;
                start_found = true;
                break;
            }
        }
        if (start_found) break;
    }

    // A queue for our BFS algorithm. It stores states to visit.
    std::queue<State> q;
    // A 2D vector to keep track of visited cells to prevent cycles and redundant checks.
    std::vector<std::vector<bool>> visited(rows, std::vector<bool>(cols, false));

    // Initialize the queue with the starting state.
    q.push({start_x, start_y, 0});
    visited[start_x][start_y] = true;

    // Define possible moves (up, down, left, right).
    int dx[] = {0, 0, 1, -1};
    int dy[] = {1, -1, 0, 0};

    // The main BFS loop.
    while (!q.empty()) {
        State current = q.front();
        q.pop();

        // If we've reached the end, return the distance.
        if (maze[current.x][current.y] == 'E') {
            return current.dist;
        }

        // Explore all possible neighbors.
        for (int i = 0; i < 4; ++i) {
            int new_x = current.x + dx[i];
            int new_y = current.y + dy[i];

            // Check if the new coordinates are within the maze boundaries.
            if (new_x >= 0 && new_x < rows && new_y >= 0 && new_y < cols) {
                // Check if the cell is not a wall and has not been visited.
                if (maze[new_x][new_y] != '#' && !visited[new_x][new_y]) {
                    // Mark as visited and add to the queue.
                    visited[new_x][new_y] = true;
                    q.push({new_x, new_y, current.dist + 1});
                }
            }
        }
    }

    // If the queue becomes empty and the end was not found, no path exists.
    return -1;
}

int main() {
    // Example 1: A simple path exists.
    std::vector<std::vector<char>> maze1 = {
        {'S', '.', '.'},
        {'#', '#', '.'},
        {'.', '.', 'E'}
    };
    int result1 = solveMaze(maze1);
    std::cout << "Example 1 (Path Exists): ";
    if (result1 != -1) {
        std::cout << "Shortest path length is " << result1 << std::endl;
    } else {
        std::cout << "No path found." << std::endl;
    }
    std::cout << std::endl;

    // Example 2: A more complex path exists.
    std::vector<std::vector<char>> maze2 = {
        {'S', '.', '.', '#'},
        {'#', '#', '.', '#'},
        {'.', '.', '.', 'E'}
    };
    int result2 = solveMaze(maze2);
    std::cout << "Example 2 (Complex Path): ";
    if (result2 != -1) {
        std::cout << "Shortest path length is " << result2 << std::endl;
    } else {
        std::cout << "No path found." << std::endl;
    }
    std::cout << std::endl;

    // Example 3: No path exists.
    std::vector<std::vector<char>> maze3 = {
        {'S', '#', '.'},
        {'.', '#', '#'},
        {'.', '#', 'E'}
    };
    int result3 = solveMaze(maze3);
    std::cout << "Example 3 (No Path): ";
    if (result3 != -1) {
        std::cout << "Shortest path length is " << result3 << std::endl;
    } else {
        std::cout << "No path found." << std::endl;
    }
    std::cout << std::endl;

    return 0;
}