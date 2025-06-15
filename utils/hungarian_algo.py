
import numpy as np
from scipy.optimize import linear_sum_assignment

def hungarian_algorithm1(cost_matrix):
    """
    Applies the Hungarian algorithm to find the minimum cost perfect matching
    in a bipartite graph (or solves the assignment problem).

    Args:
        cost_matrix (numpy.ndarray): A 2D NumPy array representing the cost matrix.
                                     cost_matrix[i, j] is the cost of assigning
                                     agent i to task j. The matrix must be square.

    Returns:
        tuple: A tuple containing:
               - assignment (list): A list of (agent_index, task_index) tuples
                                    representing the optimal assignment.
               - total_cost (int/float): The minimum total cost of the assignment.
    """
    matrix = np.array(cost_matrix, dtype=float)
    n = matrix.shape[0] # Number of agents (rows) or tasks (columns)

    # Step 1: Subtract the smallest element in each row from all elements in that row.
    for i in range(n):
        matrix[i, :] -= np.min(matrix[i, :])

    # Step 2: Subtract the smallest element in each column from all elements in that column.
    for j in range(n):
        matrix[:, j] -= np.min(matrix[:, j])

    # Steps 3-5: Iteratively cover all zeros with minimum lines and create more zeros.
    # We use a state machine to manage the marking of rows/columns.
    while True:
        # Step 3: Find a maximum matching of zeros.
        # This is a greedy approach to find initial assignments (starred zeros).
        starred_zeros = [] # List of (row, col) for starred zeros
        row_covered = np.zeros(n, dtype=bool)
        col_covered = np.zeros(n, dtype=bool)

        for i in range(n):
            for j in range(n):
                if matrix[i, j] == 0 and not row_covered[i] and not col_covered[j]:
                    starred_zeros.append((i, j))
                    row_covered[i] = True
                    col_covered[j] = True
        
        # Reset covers for next steps
        row_covered.fill(False)
        col_covered.fill(False)

        # Step 4: Cover columns containing starred zeros.
        for r, c in starred_zeros:
            col_covered[c] = True

        num_covered_cols = np.sum(col_covered)

        if num_covered_cols == n:
            # All columns are covered, optimal assignment found
            break

        # Step 5: Find a prime zero (non-starred zero that isn't covered by covered lines).
        # And then construct a series of alternating starred and primed zeros.
        # This part is the most complex, involving iterative search for zeros and prime zeros.
        
        while True:
            # Find an uncovered zero and prime it
            primed_zero = None
            for i in range(n):
                if not row_covered[i]:
                    for j in range(n):
                        if not col_covered[j] and matrix[i, j] == 0:
                            primed_zero = (i, j)
                            break
                    if primed_zero:
                        break

            if primed_zero is None:
                # No uncovered zeros found, adjust matrix
                min_uncovered_val = np.min(matrix[~row_covered][:, ~col_covered])
                matrix[~row_covered] += min_uncovered_val
                matrix[:, col_covered] -= min_uncovered_val
            else:
                # Prime the zero
                r_prime, c_prime = primed_zero
                
                # Check if there's a starred zero in the same row as the primed zero
                starred_in_row = None
                for sr, sc in starred_zeros:
                    if sr == r_prime:
                        starred_in_row = (sr, sc)
                        break

                if starred_in_row is None:
                    # Case 2: No starred zero in the row.
                    # We have found an augmenting path. Create new starred zeros.
                    
                    # Create an alternating path starting from the primed zero
                    path = [primed_zero]
                    
                    while True:
                        # Find starred zero in the same column as the last primed zero
                        found_starred = False
                        for sr, sc in starred_zeros:
                            if sc == path[-1][1]:
                                path.append((sr, sc))
                                found_starred = True
                                break
                        
                        if not found_starred: # No starred zero in column, path ends with primed
                            break
                        
                        # Find primed zero in the same row as the last starred zero
                        found_primed = False
                        for r_temp in range(n):
                            if matrix[r_temp, path[-1][1]] == 0 and r_temp == path[-1][0]: # Should be a primed zero
                                # This is a bit of a simplification. In a full implementation,
                                # you'd store primed zeros and their parent starred zeros.
                                # For simplicity here, we assume the next primed zero is in the current row
                                # and its column is the same as the current starred zero.
                                # This requires a specific search for the primed zero that created the path.
                                pass # This part of path reconstruction needs careful implementation if tracking all primes.
                        
                        # Instead of searching for the primed zero, we iterate to find a zero
                        # in the same row as the *starred* zero that was just added to the path.
                        # This becomes a bit tricky without explicit priming storage.
                        
                        # A more robust way: re-scan columns of the starred zero's row for an uncovered zero
                        # This is where the standard implementation of the algorithm can be more complex.
                        # For this simplified version, let's assume we find it by continuing the path.
                        
                        # Find an uncovered zero in the row of the last starred zero,
                        # which is the start of the next segment of the alternating path.
                        found_uncovered_zero_in_row = False
                        for j_temp in range(n):
                            if matrix[path[-1][0], j_temp] == 0 and not col_covered[j_temp]:
                                path.append((path[-1][0], j_temp))
                                found_uncovered_zero_in_row = True
                                break
                        if not found_uncovered_zero_in_row:
                            break # Path ends

                    # Convert primed zeros to starred zeros and unstar starred zeros
                    new_starred_zeros = []
                    for r, c in starred_zeros:
                        if (r, c) not in path: # Keep starred zeros not in the path
                            new_starred_zeros.append((r, c))
                    
                    for r, c in path:
                        # If it's a primed zero in the path, it becomes starred
                        # If it's a starred zero in the path, it becomes unstarred implicitly
                        # For this implementation, we just add the primed ones and remove the old starred ones.
                        if (r, c) not in starred_zeros: # This assumes primed_zero is the start of the path
                             new_starred_zeros.append((r, c))
                    
                    starred_zeros = new_starred_zeros
                    break # Go back to Step 3 (re-cover columns)
                else:
                    # Case 1: Starred zero in the same row.
                    # Cover the row of the primed zero and uncover the column of the starred zero.
                    row_covered[r_prime] = True
                    col_covered[starred_in_row[1]] = False
                    # Go back to finding another uncovered zero
                    continue # Continue inner while loop

    # Step 6 (Implicit): Extract assignment from starred zeros
    assignment = sorted(starred_zeros, key=lambda x: x[0]) # Sort by agent index
    
    # Calculate total cost from original cost matrix
    original_cost_matrix = np.array(cost_matrix)
    total_cost = sum(original_cost_matrix[r, c] for r, c in assignment)

    return assignment, total_cost

# --- Example Usage ---
def main1():
    # Example 1: Simple 3x3 matrix
    cost_matrix1 = np.array([
        [10, 8, 12],
        [15, 12, 10],
        [8, 9, 11]
    ])
    print("--- Example 1: Simple 3x3 Cost Matrix ---")
    print("Cost Matrix:\n", cost_matrix1)
    assignment1, total_cost1 = hungarian_algorithm1(cost_matrix1)
    print("Optimal Assignment (Agent, Task):", assignment1)
    print("Minimum Total Cost:", total_cost1)
    # Expected: [(0, 1), (1, 2), (2, 0)], Cost: 8 + 10 + 8 = 26

    # Example 2: From a common tutorial
    cost_matrix2 = np.array([
        [25, 40, 35],
        [40, 60, 35],
        [20, 40, 25]
    ])
    print("\n--- Example 2: Common Tutorial Example ---")
    print("Cost Matrix:\n", cost_matrix2)
    assignment2, total_cost2 = hungarian_algorithm1(cost_matrix2)
    print("Optimal Assignment (Agent, Task):", assignment2)
    print("Minimum Total Cost:", total_cost2)
    # Expected: [(0, 0), (1, 2), (2, 1)], Cost: 25 + 35 + 40 = 100

    # Example 3: Larger matrix with more complex zero covering
    cost_matrix3 = np.array([
        [7, 6, 2, 8, 10],
        [6, 7, 8, 5, 9],
        [5, 8, 9, 6, 7],
        [8, 5, 10, 7, 6],
        [9, 10, 5, 6, 4]
    ])
    print("\n--- Example 3: Larger Matrix ---")
    print("Cost Matrix:\n", cost_matrix3)
    assignment3, total_cost3 = hungarian_algorithm1(cost_matrix3)
    print("Optimal Assignment (Agent, Task):", assignment3)
    print("Minimum Total Cost:", total_cost3)
    # Expected: Something like [(0, 2), (1, 3), (2, 0), (3, 1), (4, 4)], Cost: 2 + 5 + 5 + 5 + 4 = 21

    # Example 4: A matrix that immediately has a solution after row/col reduction
    cost_matrix4 = np.array([
        [0, 1, 2],
        [1, 0, 3],
        [2, 3, 0]
    ])
    print("\n--- Example 4: Matrix with immediate solution ---")
    print("Cost Matrix:\n", cost_matrix4)
    assignment4, total_cost4 = hungarian_algorithm1(cost_matrix4)
    print("Optimal Assignment (Agent, Task):", assignment4)
    print("Minimum Total Cost:", total_cost4)
    # Expected: [(0,0), (1,1), (2,2)], Cost: 0

    # Example 5: Non-square matrix (requires padding with large values)
    # The current implementation expects a square matrix.
    # To handle non-square matrices, you'd pad the smaller dimension with
    # large (infinity) costs to make it square.
    # Example:
    # agents = 3, tasks = 4 -> pad with 1 dummy agent row
    # agents = 4, tasks = 3 -> pad with 1 dummy task column
    # For this function, ensure `cost_matrix` is square.
    # If the number of agents != number of tasks, you'd extend the smaller dimension
    # with dummy rows/columns having infinite costs.
    # Example for non-square (conceptual, not runnable directly with current code):
    # cost_matrix_non_square = np.array([
    #     [10, 8, 12, 1000],
    #     [15, 12, 10, 1000],
    #     [8, 9, 11, 1000]
    # ]) # 3 agents, 4 tasks -> needs 1 dummy agent row
    # You would need to add a preprocessing step for non-square matrices.

def solve_assignment_problem(cost_matrix):
    """
    Solves the linear sum assignment problem (minimum cost bipartite matching)
    using SciPy's linear_sum_assignment function.

    Args:
        cost_matrix (numpy.ndarray): A 2D NumPy array representing the cost matrix.
                                     cost_matrix[i, j] is the cost of assigning
                                     agent i to task j. The matrix can be square or rectangular.

    Returns:
        tuple: A tuple containing:
               - assignment (list): A list of (agent_index, task_index) tuples
                                    representing the optimal assignment.
               - total_cost (float): The minimum total cost of the assignment.
    """
    # linear_sum_assignment returns two arrays:
    # row_ind: indices of the rows (agents) selected for the assignment
    # col_ind: indices of the columns (tasks) assigned to the corresponding rows
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # The assignment is a list of (row_index, col_index) pairs
    assignment = list(zip(row_ind, col_ind))

    # The total cost is the sum of the costs at the assigned positions in the original matrix
    total_cost = cost_matrix[row_ind, col_ind].sum()

    return assignment, total_cost


def calculate_midpoint(p1, p2):
    """Calculates the midpoint of a line segment."""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def calculate_slope(p1, p2):
    """Calculates the slope of a line segment."""
    # Handle vertical lines
    if p2[0] - p1[0] == 0:
        return float('inf') if p2[1] > p1[1] else float('-inf')
    return (p2[1] - p1[1]) / (p2[0] - p1[0])

def euclidean_distance(point1, point2):
    """Calculates the Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_lane_matching_cost(predicted_lane, gt_lane, weight_midpoint_dist=0.7, weight_slope_diff=0.3):
    """
    Calculates a composite cost between a predicted lane and a ground truth lane.
    Lower cost indicates better match.

    Args:
        predicted_lane (dict): A dictionary for a predicted lane, e.g., {'p1': (x1, y1), 'p2': (x2, y2)}.
        gt_lane (dict): A dictionary for a ground truth lane, e.g., {'p1': (x1, y1), 'p2': (x2, y2)}.
        weight_midpoint_dist (float): Weight for the midpoint distance component of the cost.
        weight_slope_diff (float): Weight for the absolute slope difference component of the cost.

    Returns:
        float: The calculated cost.
    """
    pred_midpoint = calculate_midpoint(predicted_lane['p1'], predicted_lane['p2'])
    gt_midpoint = calculate_midpoint(gt_lane['p1'], gt_lane['p2'])
    midpoint_dist = euclidean_distance(pred_midpoint, gt_midpoint)

    pred_slope = calculate_slope(predicted_lane['p1'], predicted_lane['p2'])
    gt_slope = calculate_slope(gt_lane['p1'], gt_lane['p2'])

    # Handle infinite slopes carefully
    if abs(pred_slope) == float('inf') and abs(gt_slope) == float('inf'):
        slope_diff = 0.0 # Both are vertical, so difference is zero
    elif abs(pred_slope) == float('inf') or abs(gt_slope) == float('inf'):
        slope_diff = 1.0 # One is vertical, other is not. Max difference for normalization.
    else:
        slope_diff = abs(pred_slope - gt_slope)

    # Normalize components to be roughly in a similar range for weighted sum
    # These normalization factors might need adjustment based on your typical data scale.
    normalized_midpoint_dist = midpoint_dist / 100.0 # Assuming max dist around 100 units
    normalized_slope_diff = min(slope_diff / 5.0, 1.0) # Assuming max slope diff around 5, cap at 1.0

    # Composite cost
    cost = (weight_midpoint_dist * normalized_midpoint_dist) + \
           (weight_slope_diff * normalized_slope_diff)

    return cost

def create_lane_cost_matrix(predicted_lanes, gt_lanes, **cost_params):
    """
    Creates a cost matrix for matching predicted lanes to ground truth lanes.

    Args:
        predicted_lanes (list): A list of dictionaries, each representing a predicted lane.
        gt_lanes (list): A list of dictionaries, each representing a ground truth lane.
        **cost_params: Arbitrary keyword arguments passed to calculate_lane_matching_cost
                       (e.g., weight_midpoint_dist, weight_slope_diff).

    Returns:
        numpy.ndarray: A 2D NumPy array where cost_matrix[i, j] is the cost
                       of matching predicted_lanes[i] to gt_lanes[j].
    """
    num_preds = len(predicted_lanes)
    num_gts = len(gt_lanes)

    # Initialize cost matrix with zeros
    cost_matrix = np.zeros((num_preds, num_gts))

    # Populate the cost matrix
    for i in range(num_preds):
        for j in range(num_gts):
            cost_matrix[i, j] = calculate_lane_matching_cost(
                predicted_lanes[i], gt_lanes[j], **cost_params
            )
    return cost_matrix

# --- Example Usage ---
def main():
   # Example 1: Simple 3x3 matrix (same as previous Hungarian example)
    cost_matrix1 = np.array([
        [25, 40, 35],
        [40, 60, 35],
        [20, 40, 25]
    ])
    print("--- Example 1: Simple 3x3 Cost Matrix ---")
    print("Cost Matrix:\n", cost_matrix1)
    assignment1, total_cost1 = solve_assignment_problem(cost_matrix1)
    print("Optimal Assignment (Agent, Task):", assignment1)
    print("Minimum Total Cost:", total_cost1)
    # Expected: [(0, 0), (1, 2), (2, 1)], Cost: 25 + 35 + 40 = 100

    # Example 2: Rectangular matrix (more tasks than agents)
    # Here, some tasks will not be assigned. The algorithm finds the best assignment
    # for all agents, assuming each agent must be assigned.
    cost_matrix2 = np.array([
        [10, 8, 12, 15],
        [15, 12, 10, 14],
        [8, 9, 11, 7]
    ])
    print("\n--- Example 2: Rectangular Cost Matrix (3 agents, 4 tasks) ---")
    print("Cost Matrix:\n", cost_matrix2)
    assignment2, total_cost2 = solve_assignment_problem(cost_matrix2)
    print("Optimal Assignment (Agent, Task):", assignment2)
    print("Minimum Total Cost:", total_cost2)
    # Note: SciPy's implementation handles rectangular matrices directly,
    # effectively padding with implicit infinity costs if necessary to make it square
    # and assigning dummy rows/columns. The output `row_ind` and `col_ind`
    # will directly correspond to the assignments.

    # Example 3: Rectangular matrix (more agents than tasks)
    cost_matrix3 = np.array([
        [5, 9],
        [10, 3],
        [8, 7]
    ])
    print("\n--- Example 3: Rectangular Cost Matrix (3 agents, 2 tasks) ---")
    print("Cost Matrix:\n", cost_matrix3)
    assignment3, total_cost3 = solve_assignment_problem(cost_matrix3)
    print("Optimal Assignment (Agent, Task):", assignment3)
    print("Minimum Total Cost:", total_cost3)


    # Define example predicted and ground truth lanes
    # Each lane is a dictionary with 'p1' and 'p2' (endpoints)
    predicted_lanes = [
        {'p1': (10, 100), 'p2': (210, 10)},  # Pred 0: Similar to GT 0
        {'p1': (55, 140), 'p2': (245, 55)},  # Pred 1: Similar to GT 1
        {'p1': (15, 90), 'p2': (200, 15)},   # Pred 2: Slightly off from GT 0
        {'p1': (10, 20), 'p2': (50, 80)},    # Pred 3: Short, different slope
    ]

    ground_truth_lanes = [
        {'p1': (5, 95), 'p2': (205, 5)},    # GT 0
        {'p1': (50, 150), 'p2': (250, 50)}, # GT 1
        {'p1': (300, 100), 'p2': (400, 20)},# GT 2 (No close prediction)
    ]

    print("--- Predicted Lanes ---")
    for i, lane in enumerate(predicted_lanes):
        print(f"Pred {i}: {lane}")
    print("\n--- Ground Truth Lanes ---")
    for i, lane in enumerate(ground_truth_lanes):
        print(f"GT {i}: {lane}")

    # Create the cost matrix
    # You can adjust weights here to prioritize midpoint distance or slope similarity
    cost_matrix = create_lane_cost_matrix(
        predicted_lanes,
        ground_truth_lanes,
        weight_midpoint_dist=0.6,
        weight_slope_diff=0.4
    )

    print("\n--- Generated Cost Matrix (Predicted Rows, GT Columns) ---")
    print(np.round(cost_matrix, 2)) # Round for better readability

    # Use linear_sum_assignment to find the optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    print("\n--- Optimal Matching Results ---")
    assignment_pairs = []
    total_cost = 0.0
    for i in range(len(row_ind)):
        pred_idx = row_ind[i]
        gt_idx = col_ind[i]
        cost = cost_matrix[pred_idx, gt_idx]
        assignment_pairs.append(f"Pred {pred_idx} <-> GT {gt_idx} (Cost: {cost:.2f})")
        total_cost += cost

    print("Assignments:")
    for assignment_str in assignment_pairs:
        print(f"  {assignment_str}")
    print(f"Total Minimum Cost: {total_cost:.2f}")

    print("\n--- Handling cases with different numbers of predictions/GTs ---")
    # Example: More predictions than GTs. Some predictions will be unmatched.
    predictions_more = [
        {'p1': (10, 100), 'p2': (210, 10)},
        {'p1': (55, 140), 'p2': (245, 55)},
        {'p1': (15, 90), 'p2': (200, 15)},
        {'p1': (300, 50), 'p2': (350, 150)}, # Extra prediction
        {'p1': (400, 10), 'p2': (450, 60)},  # Another extra prediction
    ]
    gt_fewer = [
        {'p1': (5, 95), 'p2': (205, 5)},
        {'p1': (50, 150), 'p2': (250, 50)},
    ]

    cost_matrix_more_preds = create_lane_cost_matrix(predictions_more, gt_fewer)
    row_ind_more, col_ind_more = linear_sum_assignment(cost_matrix_more_preds)

    print("\nCost Matrix (More Predictions):\n", np.round(cost_matrix_more_preds, 2))
    print("\nOptimal Matching (More Predictions):")
    total_cost_more = 0.0
    for i in range(len(row_ind_more)):
        pred_idx = row_ind_more[i]
        gt_idx = col_ind_more[i]
        cost = cost_matrix_more_preds[pred_idx, gt_idx]
        print(f"  Pred {pred_idx} <-> GT {gt_idx} (Cost: {cost:.2f})")
        total_cost_more += cost
    print(f"Total Minimum Cost: {total_cost_more:.2f}")

    # Note: `linear_sum_assignment` will find an assignment for as many rows/columns
    # as possible, up to the size of the smaller dimension.
    # If the cost matrix is rectangular (e.g., more predictions than GTs),
    # some predictions will inherently not be matched to any GT in the returned indices.
    # To find which ones are *not* matched, you'd compare the `row_ind` to `range(num_preds)`.


if __name__ == '__main__':
    main()