#include "stdio.h"
#include <vector>
#include <queue>
#include <map>
#include <utility>

using namespace std;

struct State{
    int x;
    int y;
    int dist;
};

vector<pair<int,int>> findWay(const vector<vector<char>>& maze)
{
    
    vector<int> dx = {0, 0, -1, 1};
    vector<int> dy = {1, -1, 0, 0};

    vector<pair<int,int>> traj;
    //find 'S'
    int sx = 0;
    int sy = 0;

    int row = maze.size();
    int col = maze[0].size();
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            if([i][j] == 'S')
            {
                sx = i;
                sy = j;
            }
        }
    }

    vector<vector<bool>> visited;
    queue<State> q;

    int dist = 0;
    q.push({sx, sy, dist});

    while (!q.empty())
    {
        State i = q.front();
        q.pop();
        
        traj.push_back({i.x, i.y});
        if(maze[i.x][i.y] == 'E')
        {
            break;
        }
        if (!visited[i.x][i.y])
        {
            visited[i.x][i.y] = true;
            dist++;
            for (int j = 0; j < 4; j++)
            {
                int new_x = i.x + dx[j];
                int new_y = i.y + dy[j];

                if((new_x < row) && (new_x > 0) && (new_y < col) && (new_y > 0))
                {
                    if ((maze[new_x][new_y] == '.'))
                    {
                        q.push({new_x, new_y,dist});
                    }
                }
            }
        }

    }
    return traj;
}

int main(){
    std::vector<std::vector<char>> maze1 = {
        {'S', '.', '.'},
        {'#', '#', '.'},
        {'.', '.', 'E'}
    };
    findWay(maze1);
    return 0;
}