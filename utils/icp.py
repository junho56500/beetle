import numpy as np
from scipy.spatial import KDTree # For efficient nearest neighbor search
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting
from scipy.spatial.transform import Rotation

# --- Helper Function to Compute Jacobian and Covariance ---
def compute_jacobian_and_covariance(
    source_points_final_aligned: np.ndarray,
    target_points_corresponding: np.ndarray,
    final_rmse: float
) -> np.ndarray:
    """
    Estimates the 6x6 covariance matrix of the rigid transformation.
    This is a simplified approximation based on the final ICP residuals.

    Args:
        source_points_final_aligned (np.ndarray): The source points after the final transformation.
                                                  Shape (N, 3).
        target_points_corresponding (np.ndarray): The corresponding target points for final_aligned_source_points.
                                                  Shape (N, 3).
        final_rmse (float): The final Root Mean Squared Error of the ICP alignment.

    Returns:
        np.ndarray: The 6x6 covariance matrix of the transformation (rx, ry, rz, tx, ty, tz).
    """
    N = source_points_final_aligned.shape[0]
    if N < 3: # Need at least 3 points for a 3D transform
        return np.full((6,6), np.inf) # Cannot estimate covariance

    # Estimate noise variance from RMSE
    sigma_squared = final_rmse**2

    # Transformation parameters are [rx, ry, rz, tx, ty, tz]
    # Jacobian of residual (p' - q) w.r.t. parameters
    # For a point p' = R*p + t, where R is linearized for small angles (rx, ry, rz)
    # p' = (I + [rx,ry,rz]x) * p + t
    # dp'/drx = [0, -pz, py]
    # dp'/dry = [pz, 0, -px]
    # dp'/drz = [-py, px, 0]
    # dp'/dtx = [1, 0, 0]
    # dp'/dty = [0, 1, 0]
    # dp'/dtz = [0, 0, 1]

    J = np.zeros((N * 3, 6)) # (Number of residuals * 3 dimensions) x 6 parameters

    for i in range(N):
        px, py, pz = source_points_final_aligned[i] # Use the *aligned* source points for Jacobian

        # Jacobian for rotation (linearized approximation)
        J[i*3 + 0, 0] = 0;   J[i*3 + 0, 1] = pz;  J[i*3 + 0, 2] = -py # d(x')/d(roll), d(x')/d(pitch), d(x')/d(yaw)
        J[i*3 + 1, 0] = -pz; J[i*3 + 1, 1] = 0;   J[i*3 + 1, 2] = px  # d(y')/d(roll), d(y')/d(pitch), d(y')/d(yaw)
        J[i*3 + 2, 0] = py;  J[i*3 + 2, 1] = -px; J[i*3 + 2, 2] = 0   # d(z')/d(roll), d(z')/d(pitch), d(z')/d(yaw)

        # Jacobian for translation
        J[i*3 + 0, 3] = 1; J[i*3 + 0, 4] = 0; J[i*3 + 0, 5] = 0 # d(x')/d(tx), d(x')/d(ty), d(x')/d(tz)
        J[i*3 + 1, 3] = 0; J[i*3 + 1, 4] = 1; J[i*3 + 1, 5] = 0 # d(y')/d(tx), d(y')/d(ty), d(y')/d(tz)
        J[i*3 + 2, 3] = 0; J[i*3 + 2, 4] = 0; J[i*3 + 2, 5] = 1 # d(z')/d(tx), d(z')/d(ty), d(z')/d(tz)
    
    # Information Matrix (Hessian approximation)
    H = J.T @ J

    # Add a small regularization for numerical stability (if H is close to singular)
    # This is common in practice to prevent issues with inversion
    H += np.eye(6) * 1e-9 # Add a small diagonal term

    # Covariance Matrix
    covariance_matrix = np.linalg.inv(H) * sigma_squared

    return covariance_matrix

# --- 1. Helper function for rigid transformation estimation using SVD ---
def align_points_svd(source_points, target_points):
    """
    Calculates the optimal rigid transformation (rotation R, translation t)
    that aligns source_points to target_points using SVD.

    Args:
        source_points (np.ndarray): Nx3 array of source points.
        target_points (np.ndarray): Nx3 array of target points,
                                    corresponding to source_points.

    Returns:
        tuple: (R, t) where R is a 3x3 rotation matrix and t is a 3-element
               translation vector.
    """
    if source_points.shape[0] == 0 or target_points.shape[0] == 0:
        return np.eye(3), np.zeros(3)

    # Compute centroids
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)

    # Center the points
    P_centered = source_points - centroid_source
    Q_centered = target_points - centroid_target

    # Compute covariance matrix H
    H = P_centered.T @ Q_centered

    # Perform Singular Value Decomposition (SVD) on H
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation matrix R
    R = Vt.T @ U.T

    # Special reflection case (if det(R) is negative, it's a reflection, not pure rotation)
    # This ensures R is a valid rotation matrix.
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation vector t
    t = centroid_target - R @ centroid_source

    return R, t

# --- 2. Iterative Closest Point (ICP) algorithm implementation ---
def icp(source_lanes, target_lanes, max_iterations=100, tolerance=1e-6):
    """
    Performs ICP to align 3D source lanes to target lanes.

    Args:
        source_lanes (list of np.ndarray): List of 3 Nx3 arrays, each representing a lane.
        target_lanes (list of np.ndarray): List of 3 Mx3 arrays, each representing a lane.
        max_iterations (int): Maximum number of ICP iterations.
        tolerance (float): Convergence threshold for the change in transformation.

    Returns:
        tuple: (transformed_source_cloud, R_final, t_final, mse_history)
               - transformed_source_cloud (np.ndarray): The source point cloud
                 after final alignment.
               - R_final (np.ndarray): The final 3x3 rotation matrix.
               - t_final (np.ndarray): The final 3-element translation vector.
               - mse_history (list): List of Mean Squared Errors at each iteration.
    """
    # Flatten the list of lane arrays into single point clouds
    source_cloud = np.vstack(source_lanes)
    target_cloud = np.vstack(target_lanes)

    # Initialize transformation: Identity rotation and zero translation
    current_R = np.eye(3)
    current_t = np.zeros(3)

    transformed_source_cloud = source_cloud.copy()

    # Build KD-Tree for efficient nearest neighbor search on the target cloud
    kdtree_target = KDTree(target_cloud)

    mse_history = []
    prev_mse = float('inf')
    final_rmse = float('inf') # To store the RMSE at convergence

    print("Starting ICP...")
    for i in range(max_iterations):
        # 1. Find the closest points in the target cloud for each point in the transformed source cloud
        # distances: distances to closest points
        # indices: indices of closest points in target_cloud
        distances, indices = kdtree_target.query(transformed_source_cloud)

        # Get the corresponding points from the target cloud
        corresponding_target_points = target_cloud[indices]

        # Calculate Mean Squared Error (MSE) for current alignment
        current_mse = np.mean(distances**2)
        mse_history.append(current_mse)
        print(f"Iteration {i+1}: MSE = {current_mse:.6f}")

        # 2. Estimate the optimal rigid transformation (R_iter, t_iter)
        # that aligns the current transformed_source_cloud to its closest points
        # in the target_cloud.
        R_iter, t_iter = align_points_svd(transformed_source_cloud, corresponding_target_points)

        # 3. Apply the newly computed transformation to the source cloud
        # Accumulate the transformations:
        # New_transformed_source_cloud = R_iter @ (transformed_source_cloud.T) + t_iter
        transformed_source_cloud = (R_iter @ transformed_source_cloud.T).T + t_iter

        # Accumulate the global transformation
        current_R = R_iter @ current_R
        current_t = R_iter @ current_t + t_iter

        # 4. Check for convergence
        if abs(prev_mse - current_mse) < tolerance:
            print(f"ICP converged after {i+1} iterations.")
            final_rmse = np.sqrt(current_mse) # Store final RMSE
            break
        prev_mse = current_mse
    else:
        print(f"ICP finished after {max_iterations} iterations without full convergence.")
        final_rmse = np.sqrt(current_mse) # Store final RMSE even if not fully converged

    # --- Calculate Covariance after ICP converges ---
    # Re-find the final correspondences and residuals to ensure they match the final R, t
    # (Important if ICP stopped due to max_iterations, not just tolerance)
    final_distances, final_indices = kdtree_target.query(transformed_source_cloud)
    final_corresponding_target_points = target_cloud[final_indices]
    
    # Calculate covariance
    covariance_matrix = compute_jacobian_and_covariance(
        transformed_source_cloud, # The source points in their final aligned position
        final_corresponding_target_points, # Their corresponding target points
        final_rmse # The RMSE at the end of ICP
    )
    
    return transformed_source_cloud, current_R, current_t, mse_history, covariance_matrix

# --- 3. Example Usage: Generating 3 pairs of 3D lanes ---
def generate_lane(start_point, direction, length=10, num_points=20, noise_std=0.1):
    """Generates a synthetic 3D lane (a line with some noise)."""
    t = np.linspace(0, 1, num_points)[:, np.newaxis]
    lane = start_point + t * direction * length
    lane += np.random.normal(0, noise_std, lane.shape)
    return lane

if __name__ == "__main__":
    # --- Generate Original Target Lanes ---
    print("Generating target lanes...")
    target_lane1 = generate_lane(np.array([0, 0, 0]), np.array([1, 0.1, 0.2]), length=10)
    target_lane2 = generate_lane(np.array([2, 5, 1]), np.array([0.5, 1, 0.3]), length=8)
    target_lane3 = generate_lane(np.array([-3, 2, 4]), np.array([0.2, -0.8, 1]), length=12)
    target_lanes = [target_lane1, target_lane2, target_lane3]
    target_full_cloud = np.vstack(target_lanes)

    # --- Create a known rigid transformation for Source Lanes ---
    # Define a rotation matrix (around Z-axis by 30 degrees)
    theta = np.deg2rad(30)
    true_R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta),  0],
        [0,             0,              1]
    ])
    # Define a translation vector
    true_t = np.array([5.0, -2.0, 3.0])

    # Apply the transformation and add some noise to generate source lanes
    print("Generating source lanes by transforming target lanes...")
    source_lanes = []
    for lane in target_lanes:
        # Apply true_R and true_t
        transformed_lane = (true_R @ lane.T).T + true_t
        # Add some extra noise to the source lanes to simulate real-world data
        transformed_lane += np.random.normal(0, 0.1, transformed_lane.shape)
        source_lanes.append(transformed_lane)
    source_initial_cloud = np.vstack(source_lanes)

    # --- Run ICP ---
    transformed_source_cloud_final, R_computed, t_computed, mse_history, covariance_matrix = \
        icp(source_lanes, target_lanes, max_iterations=100, tolerance=1e-7)

    print("\n--- ICP Results ---")
    print("Computed Rotation Matrix (R_computed):\n", R_computed)
    print("\nTrue Rotation Matrix (true_R):\n", true_R)
    print("\nComputed Translation Vector (t_computed):\n", t_computed)
    print("\nTrue Translation Vector (true_t):\n", true_t)
    print(f"\nFinal MSE: {mse_history[-1]:.6f}")

    # Convert final R to degrees for easier interpretation
    r_rot_final = Rotation.from_matrix(R_computed)
    R_final_degrees = np.degrees(r_rot_final.as_euler('xyz', degrees=False))
    print(f"Final Rotation (Euler XYZ degrees): {R_final_degrees}")

    print(f"\n--- Transformation Covariance Matrix (6x6) ---")
    print(covariance_matrix.round(8)) # Round for better display

    # Interpret variances (diagonal elements)
    print("\nVariances (diagonal):")
    print(f"  Var(roll):  {covariance_matrix[0,0]:.8f} (rad^2)")
    print(f"  Var(pitch): {covariance_matrix[1,1]:.8f} (rad^2)")
    print(f"  Var(yaw):   {covariance_matrix[2,2]:.8f} (rad^2)")
    print(f"  Var(tx):    {covariance_matrix[3,3]:.8f} (m^2)")
    print(f"  Var(ty):    {covariance_matrix[4,4]:.8f} (m^2)")
    print(f"  Var(tz):    {covariance_matrix[5,5]:.8f} (m^2)")
    
    # --- Visualization ---
    fig = plt.figure(figsize=(15, 7))

    # Plot initial alignment
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Initial Alignment (Before ICP)')
    ax1.scatter(target_full_cloud[:, 0], target_full_cloud[:, 1], target_full_cloud[:, 2],
                c='blue', marker='o', label='Target Lanes', s=10)
    ax1.scatter(source_initial_cloud[:, 0], source_initial_cloud[:, 1], source_initial_cloud[:, 2],
                c='red', marker='^', label='Source Lanes (Initial)', s=10)
    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_aspect('auto') # Try 'equal' for true proportions

    # Plot final alignment
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Final Alignment (After ICP)')
    ax2.scatter(target_full_cloud[:, 0], target_full_cloud[:, 1], target_full_cloud[:, 2],
                c='blue', marker='o', label='Target Lanes', s=10)
    ax2.scatter(transformed_source_cloud_final[:, 0], transformed_source_cloud_final[:, 1], transformed_source_cloud_final[:, 2],
                c='green', marker='^', label='Source Lanes (Transformed)', s=10)
    ax2.legend()
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_aspect('auto')

    plt.tight_layout()
    plt.show()

    # Plot MSE history
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(mse_history)), mse_history, marker='o')
    plt.title('MSE History During ICP Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.show()