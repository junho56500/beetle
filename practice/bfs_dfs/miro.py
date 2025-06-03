from collections import deque

move = [(1,0), (0,1), (-1,0), (0,-1)] 

#JUNO    
def miro(graph, v):
    visited = []
    q = [v]
    while q:
        x,y = q.pop()
        if x == len(graph)-1 and y == len(graph[0])-1:
            break
        if graph[x][y] == 1:        # and (x,y) not in visited:
            visited.append((x,y))
            for dx,dy in move:
                if 0 <= x+dx < len(graph) and 0 <= y+dy < len(graph[0]) and graph[x+dx][y+dy] == 1 and (x+dx,y+dy) not in visited:
                    q.append((x+dx,y+dy))
    return visited
    
#dfs may have more advantage to find the way
def main():
    graph = [
        [1,0,1,0,1,0],
        [1,1,1,1,1,1],
        [0,0,0,0,0,1],
        [1,1,1,1,1,1],
        [1,1,1,1,1,1],
    ] 
    print(miro(graph, (0,0)))

if __name__ == '__main__':
    main()
