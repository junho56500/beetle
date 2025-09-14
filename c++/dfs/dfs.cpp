#include "stdio.h"
#include <iostream>
#include <vector>
#include <map>
#include <stack>
#include <queue>

using namespace std;

vector<int> dfs(map<int, vector<int>> graph, int v)
{
    map<int,bool> visited;
    stack<int> s;
    s.push(v);
    vector<int> ret;

    while(!s.empty())
    {
        int i = s.top();
        s.pop();
        // cout << "i : " << i << '\n';
        ret.push_back(i);
        // stack<int> temp = s;
        // while (!temp.empty())
        // {
        //     cout << "temp : " << temp.top() << '\n';
        //     temp.pop();
        // }

        if(!visited[i])
        {
            visited[i] = true;
            for(const int& j: graph.at(i))
            {
                if(!visited[j])
                {
                    s.push(j);
                    // cout << "push : " << j << '\n';
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
    vector<int> d = dfs(graph, 0);
    for (const auto& i: d)
    {
        cout << i;
    }

    return 0;
}


vector<int> dfs(map<int,vector<int>> graph, int v)
{
    map<int,bool> visited;
    stack<int> s;
    s.push(v);

    while(!s.empty())
    {
        int i = s.top();
        s.pop();
        if(!visited[i])
        {
            visited[i]= true;
            for(auto j:graph.at(i))
            {
                if(!visited[j])
                {
                    s.push(j);
                }
            }
        }
    }
}