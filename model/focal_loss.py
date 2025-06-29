
order = []

def count_islands(grid):
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])
    visited = set()
    island_count = 0

    def dfs_stack(start_row, start_col):
        """Uses an explicit stack to traverse connected land."""
        stack = [(start_row, start_col)]
        while stack:
            row, col = stack.pop()
            if (row, col) in visited:
                continue
            visited.add((row, col))

            # Explore all possible directions (up, down, left, right)
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < rows and 0 <= new_col < cols and grid[new_row][new_col] == 0:
                    stack.append((new_row, new_col))

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and (r, c) not in visited:
                island_count += 1  # Found a new island
                dfs_stack(r, c)  # Perform DFS iteratively

    return island_count

# Example usage
grid = [
    [0, 0, 1, 1, 0],
    [0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0]
]


def main():
    print(count_islands(grid))

if __name__ == '__main__':
    main()
    
#JUNO    
def func(grid, v):
    row = len(grid)
    col = len(grid[0])
    visited = set()
    stack = []
    count = 0
    move = [(0,-1), (0,1), (-1,0), (1,0)]  
    
    for i in range(row):
        for j in range(col):
            if grid[i][j] == 0 and (i, j) not in visited:
                count += 1
                stack = [i,j]
                while(stack):
                    visited.add(i,j)
                    for k,l in move:
                        new_row = i + k
                        new_col = j + l    
                        if grid[new_row][new_col] == 0 not in visited and (new_row < row and new_row >= 0) and (new_col < col and new_col >= 0):
                            stack.append(new_row,new_col)
    
    return count
                  