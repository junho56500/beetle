from collections import deque

direction = [(0,1),(0,-1), (1,0), (-1,0)]

#JUNO
def bfs(grid):
    visited = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            queue = deque([(i,j)])
            while queue:
                x,y = queue.popleft()
                if grid[x][y] == 0 and (x,y) not in visited:
                    visited.append((x,y))
                    # check 4 directions and if it's 0 and not in visited, add to queue
                    for dx,dy in direction:
                        if 0 <= x+dx < len(grid) and 0 <= y+dy < len(grid[0]) and grid[x+dx][y+dy] == 0 and grid[x+dx][y+dy] not in visited:
                            queue.append((x+dx, y+dy))
    return visited

  
# Example usage
grid = [
    [0, 0, 1, 1, 0],
    [0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0]
]


def main():
    print(bfs(grid))

if __name__ == '__main__':
    main()

    