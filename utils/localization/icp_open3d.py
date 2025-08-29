import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation # For converting rotation matrix to Euler angles
from typing import List, Tuple, Dict, Union
import matplotlib.pyplot as plt

def icp_lane_registration(
    source_lanes: List[np.ndarray],  # List of Nx3 arrays (N points, 3D coords)
    target_lanes: List[np.ndarray],  # List of Mx3 arrays (M points, 3D coords)
    initial_transform: np.ndarray = np.eye(4), # Optional 4x4 initial transformation matrix
    max_correspondence_distance: float = 0.5, # Max distance for point correspondence (meters)
    max_iterations: int = 100,       # Max ICP iterations
    tolerance: float = 1e-6,         # Convergence tolerance for fitness/RMSE
    estimation_method_type: str = 'point_to_plane', # 'point_to_point', 'point_to_plane', 'generalized'
    normal_estimation_radius: float = 0.1, # Radius for normal estimation (meters)
    normal_estimation_max_nn: int = 30 # Max neighbors for normal estimation
) -> Dict[str, Union[np.ndarray, float, List[np.ndarray]]]: # Added List[np.ndarray] to return type
    """
    Performs ICP registration between source and target lane point clouds
    and returns the estimated transformation (R, t) and its covariance matrix,
    along with the transformed source lanes.

    Args:
        source_lanes (List[np.ndarray]): List of NumPy arrays, each representing a 3D lane segment.
        target_lanes (List[np.ndarray]): List of NumPy arrays, each representing a 3D target lane segment.
        initial_transform (np.ndarray): A 4x4 homogeneous transformation matrix as an initial guess.
        max_correspondence_distance (float): Maximum distance between corresponding points.
        max_iterations (int): Maximum number of ICP iterations.
        tolerance (float): Convergence threshold for relative fitness and RMSE.
        estimation_method_type (str): Type of ICP estimation method.
                                      'point_to_point', 'point_to_plane', or 'generalized'.
        normal_estimation_radius (float): Search radius for normal estimation.
        normal_estimation_max_nn (int): Max nearest neighbors for normal estimation.

    Returns:
        Dict[str, Union[np.ndarray, float, List[np.ndarray]]]: A dictionary containing:
            'R' (np.ndarray): Final 3x3 rotation matrix.
            't' (np.ndarray): Final 3-element translation vector.
            'covariance' (np.ndarray): 6x6 covariance matrix of the transformation.
            'rmse' (float): Final inlier RMSE of the registration.
            'fitness' (float): Final fitness score (overlap ratio).
            'transformation_matrix' (np.ndarray): The full 4x4 homogeneous transformation matrix.
            'transformed_source_lanes' (List[np.ndarray]): The source lane point clouds
                                                          after final alignment, in original list format.
    """
    
    # Store the number of points per original lane
    points_per_lane = [len(lane) for lane in source_lanes]
    
    # Flatten list of lane arrays into single NumPy arrays for Open3D
    source_cloud_np = np.vstack(source_lanes).astype(np.float64)
    target_cloud_np = np.vstack(target_lanes).astype(np.float64)

    # Convert NumPy arrays to Open3D PointCloud objects
    source_cloud_o3d = o3d.geometry.PointCloud()
    source_cloud_o3d.points = o3d.utility.Vector3dVector(source_cloud_np)

    target_cloud_o3d = o3d.geometry.PointCloud()
    target_cloud_o3d.points = o3d.utility.Vector3dVector(target_cloud_np)

    # Estimate Normals (REQUIRED for Point-to-Plane and Generalized ICP)
    if estimation_method_type in ['point_to_plane', 'generalized']:
        print("Estimating normals for source cloud...")
        source_cloud_o3d.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_estimation_radius, max_nn=normal_estimation_max_nn
            )
        )
        print("Estimating normals for target cloud...")
        target_cloud_o3d.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_estimation_radius, max_nn=normal_estimation_max_nn
            )
        )

    # Define ICP Estimation Method
    if estimation_method_type == 'point_to_point':
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        icp_function = o3d.pipelines.registration.registration_icp
    elif estimation_method_type == 'point_to_plane':
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        icp_function = o3d.pipelines.registration.registration_icp
    elif estimation_method_type == 'generalized':
        estimation_method = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(epsilon=1e-4)
        icp_function = o3d.pipelines.registration.registration_generalized_icp
    else:
        raise ValueError(f"Unsupported estimation_method_type: {estimation_method_type}")

    # Define Convergence Criteria
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=max_iterations,
        relative_fitness=tolerance,
        relative_rmse=tolerance
    )

    # Run ICP
    print(f"\nStarting Open3D {estimation_method_type.upper()} ICP registration...")
    
    # Conditionally call the correct ICP function and pass request_output_covariance
    if estimation_method_type == 'generalized':
        registration_result = icp_function(
            source_cloud_o3d,
            target_cloud_o3d,
            max_correspondence_distance,
            initial_transform,
            estimation_method,
            criteria
        )
    else:
        # For point_to_point or point_to_plane, do NOT pass request_output_covariance
        registration_result = icp_function(
            source_cloud_o3d,
            target_cloud_o3d,
            max_correspondence_distance,
            initial_transform,
            estimation_method,
            criteria
        )
        print("Note: Covariance output is explicitly requested only for 'generalized' ICP.")
        print("       Returning identity covariance for other methods as a placeholder.")
        # Placeholder covariance for non-generalized ICP where it's not directly returned by Open3D
        covariance_matrix = np.eye(6) * np.inf # Set to Inf to indicate high uncertainty/not provided
        # Or, if you have a method for point-to-plane covariance, call it here.


    print('registration_result :', registration_result)
    
    # Extract results
    R_final = registration_result.transformation[:3, :3]
    t_final = registration_result.transformation[:3, 3]
    
    print(o3d.__version__)
    # Get covariance (use actual result if generalized, otherwise the placeholder)
    if estimation_method_type == 'generalized':
        voxel_size = 0.05 # Voxel size is a parameter to consider for the calculation
        information_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source_cloud_o3d, target_cloud_o3d, voxel_size * 1.4, registration_result.transformation)
        # else: covariance_matrix already set above as placeholder
        covariance_matrix = np.linalg.inv(information_matrix)
        print('covariance_matrix :', covariance_matrix)

        final_rmse = registration_result.inlier_rmse
        final_fitness = registration_result.fitness
    
    # --- New Logic: Transform source_cloud and split back into lanes ---
    # Apply the final transformation to the original flattened source cloud (NumPy array)
    # Convert to homogeneous coordinates, apply transform, convert back to Cartesian
    source_cloud_np_hom = np.hstack((source_cloud_np, np.ones((source_cloud_np.shape[0], 1))))
    transformed_source_cloud_hom = registration_result.transformation @ source_cloud_np_hom.T
    transformed_source_cloud_np = transformed_source_cloud_hom[:3, :].T # Back to (N, 3)

    # Split the transformed flattened cloud back into individual lane arrays
    # np.split requires exact divisions, np.array_split is more flexible but gives list
    # Use np.vsplit if you know the number of points per split is fixed (e.g., 3 lanes always)
    # Otherwise, calculate split points:
    split_indices = np.cumsum(points_per_lane[:-1]) # Indices to split at
    transformed_source_lanes = np.array_split(transformed_source_cloud_np, split_indices, axis=0)


    return {
        'R': R_final,
        't': t_final,
        'covariance': covariance_matrix,
        'rmse': final_rmse,
        'fitness': final_fitness,
        'transformation_matrix': registration_result.transformation,
        'transformed_source_lanes': transformed_source_lanes # New return value
    }


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
    # transformed_source_cloud_final, R_computed, t_computed, mse_history = \
    #     icp(source_lanes, target_lanes, max_iterations=100, tolerance=1e-7)

    icp_results = icp_lane_registration(
        source_lanes=source_lanes,
        target_lanes=target_lanes,
        max_correspondence_distance=3.1, # Adjust based on expected point cloud density/overlap
        max_iterations=100,
        tolerance=1e-7,
        estimation_method_type='generalized', # or 'generalized'
        normal_estimation_radius=0.35 # Needs to be appropriate for your point cloud density
    )
    
    transformed_source_cloud_final = np.vstack(icp_results['transformed_source_lanes'])
    R_computed = icp_results['R']
    t_computed = icp_results['t']
    final_mse = icp_results['rmse']
    final_fitness = icp_results['fitness']
    
    print("\n--- ICP Results ---")
    print("Computed Rotation Matrix (R_computed):\n", R_computed)
    print("\nTrue Rotation Matrix (true_R):\n", true_R)
    print("\nComputed Translation Vector (t_computed):\n", t_computed)
    print("\nTrue Translation Vector (true_t):\n", true_t)
    print(f"\nFinal MSE: {final_mse:.6f}")
    print(f"\nFinal Fitness: {final_fitness:.6f}")

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
    # plt.figure(figsize=(8, 5))
    # plt.plot(range(len(mse_history)), mse_history, marker='o')
    # plt.title('MSE History During ICP Iterations')
    # plt.xlabel('Iteration')
    # plt.ylabel('Mean Squared Error')
    # plt.grid(True)
    # plt.show()
    
    
    # Convert R to Euler angles for easier interpretation (if needed for state)
    r_rot_final = Rotation.from_matrix(icp_results['R'].copy())
    R_final_degrees = np.degrees(r_rot_final.as_euler('xyz', degrees=False))
    print(f"Final Rotation (Euler XYZ degrees): {R_final_degrees}")

    # --- How to use these results in a Kalman Filter ---
    # Your Kalman Filter's update step would receive:
    z_icp = np.concatenate((icp_results['t'], np.radians(R_final_degrees))) # If state is [x,y,z,roll,pitch,yaw]
    R_icp = icp_results['covariance']
    # You would then pass z_icp and R_icp to your EKF.update() method.