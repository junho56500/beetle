
order = []

def dfs(graph, v, visited):
    visited[v] = True
    order.append(v)
    for i in graph[v]:
        if not visited[i]:
            dfs(graph, i, visited)

def dfs2(graph,v):
    visited = []
    stack = [v]
    while stack:
        s = stack.pop()
        if s not in visited:
            visited.append(s)
            stack += [g for g in graph[s] if g not in visited]
            
#JUNO 
# def dfs_cnt(graph, v, end):
#     visited = []
#     stack = [(v,0)]
#     while stack:
#         i,cnt = stack.pop()
#         if i not in visited:
#             visited.append(i)
#             cnt += 1
#             if i == end:
#                 return visited, cnt
#             for j in graph[i]:
#                 if j not in visited:
#                     stack += [(j,cnt)]
    
#     return "not matched"

def dfs_cnt(graph, v, end):
    visited=[]
    stack = [(v,0)]
    cnt=0
    while stack:
        s,cnt=stack.pop()
        if s not in visited:
            visited.append(s)
            if s == end:
                return cnt
            for i in graph[s]:
                if i not in visited:
                    stack.append((i,cnt+1))
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
    # visited = [False]*9
    # dfs(graph, 1, visited)
    # print(order)
    
    print(dfs_cnt(graph, 1, 6))
    
    # a = [2,6,43,8,5,87,3,8]
    # a.sort()
    # print(a)
    # b = sorted(a)
    # print(b)

if __name__ == '__main__':
    main()