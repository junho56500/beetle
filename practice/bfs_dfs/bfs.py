from collections import deque

order = []

def bfs(graph, v, visited):
    queue = deque([v])
    visited[v] = True
    order.append(v)
    
    while queue:
        print(queue)
        v = queue.popleft()
        for i in graph[v]:
            if not visited[i]:
                queue.append(i)
                visited[i] = True
                order.append(i)

def bfs2(graph, v):
    visited = []
    que = deque([v])
    while que:
        i = que.popleft()
        if i not in visited:
            visited.append(i)
            for x in graph[i]:
                if x not in visited:
                    que.append(x)
    return visited
                   
#JUNO 
# def bfs_cnt(graph, v, end):
#     visited = []
#     queue = deque([(v,0)])
#     while queue:
#         i, cnt = queue.popleft()
#         if i not in visited:
#             visited.append(i)
#             cnt += 1
#             print(i,cnt,visited)
#             if i == end:
#                 return(visited, cnt)
#             for j in graph[i]:
#                 if j not in visited:
#                     queue += [(j,cnt)]
#     return "not matched"

def bfs_cnt(graph, v, end):
    visited = []
    q = deque([(v,0)])
    cnt = 0
    while q:
        i,cnt = q.popleft()
        if i not in visited:
            visited.append(i)
            if i == end:
                return cnt
            for x in graph[i]:
                if x not in visited:
                    q.append((x,cnt+1))
    return "not matched"
    
def main():
    graph = [
        [],
        [2,3,8],
        [1,7],
        [1,4,5],
        [3,5],
        [3,4,9],
        [7],
        [2,6,8],
        [1,7],
        [2,10],
        [1]
    ]
    visited = [False]*len(graph)
    bfs(graph, 1, visited)
    print(order)
    print(bfs_cnt(graph,1, 10))
    
    # print(bfs2(graph,1))
    # a = [2,6,43,8,5,87,3,8]
    # a.sort()
    # print(a)
    # b = sorted(a)
    # print(b)

if __name__ == '__main__':
    main()
