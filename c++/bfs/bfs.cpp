#include <queue>
#include <vector>
#include <map>
#include <iostream>

using namespace std;

vector<int> bfs(const map<int, vector<int>>& graph, const int v)
{
    map<int, bool> visited;
    queue<int> q;
    q.push(v);
    vector<int> ret;

    while(!q.empty())
    {
        int i = q.front();
        q.pop();
        ret.push_back(i);

        if (!visited[i])
        {
            visited[i] = true;
            for (const int& j: graph.at(i))
            {
                if(!visited[j])
                {
                    q.push(j);
                }
            }
        }
    }
    return ret;

}


int main()
{
    map<int, vector<int>> graph=
    {
        {0, {1, 2}},
        {1, {3, 4}},
        {2, {5}},
        {3, {0}}, // Example of a back edge
        {4, {}},
        {5, {}}
    };

//    vector<int> bfs = bfs(graph, 0);
    vector<int> d = bfs(graph, 0);
    for (const auto& i: d)
    {
        cout << i;
    }

    return 0;
}




vector<int> bfs(const map<int, vector<int>>& graph, int v)
{
    map<int, bool> visited;
    queue<int> q;
    q.push(v);
    vector<int> ret;

    while(!q.empty())
    {
        int i = q.front();
        q.pop();
        ret.push_back(i);
        if (!visited[i])
        {
            visited[i] = true;
            for (const auto& j : graph.at(i))
            {
                if (!visited[j])
                {
                    q.push(j);
                }
            }
        }
    }
}