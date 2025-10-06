#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>    // Required for BFS
#include <utility>  // Required for std::pair

using namespace std;

class Solution {
public:
    // Helper function for Iterative Breadth First Search (BFS) 
    // to explore and mark a connected landmass as visited (0).
    int explore_island(vector<vector<int>>& grid) {
        // Define movement directions (Down, Up, Right, Left)
        vector<vector<int>> directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

        int rows = grid.size();
        int cols = grid[0].size();

        // 1. Initialize Queue and Start
        // The queue stores the coordinates (row, col) of land cells to visit.
        queue<pair<int, int>> q;

        vector<vector<bool>> visited(rows, vector<bool>(cols));

        int cnt = 0;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (!visited[i][j] && grid[i][j] != 1)
                {
                    cout << cnt << endl;
                    cnt++;        
                    q.push({i, j});
                    while (!q.empty()) {
                        pair<int, int> cur = q.front();
                        q.pop();

                        int curr_r = cur.first;
                        int curr_c = cur.second;

                        if(!visited[curr_r][curr_c])
                        {
                            visited[curr_r][curr_c] = true;
                            cout << curr_r << " : " << curr_c << endl;

                            // Explore all four neighbors
                            for (const auto& dir : directions) {
                                int next_r = curr_r + dir[0];
                                int next_c = curr_c + dir[1];

                                // Check boundaries and land status:
                                // Check if the neighbor is within the grid AND is land ('1')
                                if (next_r >= 0 && next_r < rows && 
                                    next_c >= 0 && next_c < cols && 
                                    !visited[next_r][next_c] &&
                                    grid[next_r][next_c] != 1) {
                                    q.push({next_r, next_c});
                                }
                            }
                        }
                    }
                }
            }
        }
        return cnt;
    }
};

int main() {
    // The grid provided by the user (using 1 for land, 0 for water)
    vector<vector<int>> grid = {
        {0, 0, 1, 1, 0},
        {0, 0, 0, 1, 1},
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 0}
    };

    Solution solver;
    int islands = solver.explore_island(grid);

    cout << "The grid is:" << endl;
    // The grid is printed after the function runs, showing all islands marked as 0.
    for (const auto& row : grid) {
        for (int cell : row) {
            cout << cell << " ";
        }
        cout << endl;
    }

    cout << "\nNumber of islands found (using iterative BFS): " << islands << endl; // Expected output: 2

    return 0;
}
