import numpy as np
from scipy.spatial import KDTree

def find_nearest_and_unmatched_points(group_a, group_b):
    """
    Finds nearest neighbors between Group A and Group B using a KD-Tree,
    and identifies unmatched points from Group B.

    Args:
        group_a (np.ndarray): A NumPy array of shape (N, 2) representing N points (x, y).
        group_b (np.ndarray): A NumPy array of shape (M, 2) representing M points (x, y).

    Returns:
        tuple: A tuple containing:
            - matched_pairs (list): List of tuples, where each tuple is
              ((x_a, y_a), (x_b, y_b), distance).
            - unmatched_b_points (list): List of (x, y) points from Group B
              that did not find a unique closest match from Group A.
    """
    if len(group_a) == 0 or len(group_b) == 0:
        return [], list(group_b) # If group_a is empty, all of group_b is unmatched

    # 1. Build KD-Tree from Group B
    tree_b = KDTree(group_b)

    matched_pairs = []
    # Keep track of which points in Group B have been claimed as a match
    # Use a set for efficient lookup
    matched_b_indices = set()

    # To handle potential multiple A points finding the same B point
    # We'll store potential matches and resolve them later
    # Format: (distance, index_a, index_b)
    potential_matches = []

    # 2. For each point in Group A, find its nearest neighbor in Group B
    for i, pt_a in enumerate(group_a):
        # Query the KDTree for the nearest neighbor in group_b
        # k=1 for the single nearest neighbor
        distance, index_b = tree_b.query(pt_a, k=1)
        
        potential_matches.append((distance, i, index_b))

    # Sort potential matches by distance to prioritize closer matches
    potential_matches.sort()

    # Resolve matches: iterate through sorted potential matches and claim B points
    claimed_a_indices = set()
    for dist, idx_a, idx_b in potential_matches:
        if idx_a not in claimed_a_indices and idx_b not in matched_b_indices:
            # If both A and B points are not yet claimed, make a match
            matched_pairs.append((group_a[idx_a], group_b[idx_b], dist))
            claimed_a_indices.add(idx_a)
            matched_b_indices.add(idx_b)

    # 3. Identify unmatched points in Group B
    unmatched_b_points = []
    for i, pt_b in enumerate(group_b):
        if i not in matched_b_indices:
            unmatched_b_points.append(pt_b)

    return matched_pairs, unmatched_b_points

# --- Example Usage ---
print("--- Finding Nearest Points and Unmatched Points ---")

# Define Group A points (3 points)
group_a_points = np.array([
    [1.0, 1.0],  # Point A0
    [5.0, 5.0],  # Point A1
    [1.1, 1.2]   # Point A2 (close to A0)
])
print(f"Group A points:\n{group_a_points}\n")

# Define Group B points (4 points)
group_b_points = np.array([
    [1.05, 1.1], # Point B0 (close to A0/A2)
    [5.1, 4.9],  # Point B1 (close to A1)
    [10.0, 10.0], # Point B2 (far from A)
    [0.9, 0.8]   # Point B3 (another point close to A0/A2)
])
print(f"Group B points:\n{group_b_points}\n")

# Find nearest points and unmatched points
matched_pairs, unmatched_b = find_nearest_and_unmatched_points(group_a_points, group_b_points)

print("--- Results ---")
print("\nMatched Pairs (Group A point, Group B point, Distance):")
for match_a, match_b, dist in matched_pairs:
    print(f"  {match_a} <-> {match_b} (Distance: {dist:.4f})")

print("\nUnmatched points from Group B:")
if unmatched_b:
    for pt in unmatched_b:
        print(f"  {pt}")
else:
    print("  None")

# --- Another Example: More A points than B points ---
print("\n" + "="*50 + "\n--- Example 2: More A points than B points ---")
group_a_ex2 = np.array([
    [0,0], [1,1], [2,2], [3,3] # 4 points
])
group_b_ex2 = np.array([
    [0.1, 0.1], [1.1, 1.1] # 2 points
])

matched_pairs_ex2, unmatched_b_ex2 = find_nearest_and_unmatched_points(group_a_ex2, group_b_ex2)

print("\nMatched Pairs (Ex2):")
for match_a, match_b, dist in matched_pairs_ex2:
    print(f"  {match_a} <-> {match_b} (Distance: {dist:.4f})")

print("\nUnmatched points from Group B (Ex2):")
if unmatched_b_ex2:
    for pt in unmatched_b_ex2:
        print(f"  {pt}")
else:
    print("  None")