from collections import deque

order = []

def bfs(graph, v, visited):
    queue = deque([v])
    visited[v] = True
    order.append(v)
    
    while queue:
        v = queue.popleft()
        for i in graph[v]:
            if not visited[i]:
                queue.append(i)
                visited[i] = True
                order.append(i)

#JUNO 
def bfs2(graph, v, end):
    visited = []
    queue = deque([(v,0)])
    while queue:
        i, cnt = queue.popleft()
        if i not in visited:
            visited.append(i)
            cnt += 1
            print(i,cnt,visited)
            if i == end:
                return(visited, cnt)
            for j in graph[i]:
                if j not in visited:
                    queue += [(j,cnt)]
    return "not matched"
  
    
def main():
    graph = [
        [],
        [2,3,8],
        [1,7],
        [1,4,5],
        [3,5],
        [3,4],
        [7],
        [2,6,8],
        [1,7]
    ]
    visited = [False]*9
    bfs(graph, 1, visited)
    print(order)
    print(bfs2(graph,1, 5))
    
    # a = [2,6,43,8,5,87,3,8]
    # a.sort()
    # print(a)
    # b = sorted(a)
    # print(b)

if __name__ == '__main__':
    main()

            
            