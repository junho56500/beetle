import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.spatial.transform import Rotation # For converting R to Euler angles
from typing import List, Tuple # For type hints
# Assuming KDTree is imported or defined here (for run_icp)
from icp import run_icp
from scipy.spatial import KDTree # For KDTree

# --- (Your run_icp, align_points_svd, compute_jacobian_and_covariance functions here) ---
# Please ensure these functions are defined or imported in your actual script.

# --- Example of your run_icp (from previous turns) ---
# Paste this entire function definition here if it's not in a separate module
def align_points_svd(source_points: np.ndarray, target_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)
    centered_source = source_points - centroid_source
    centered_target = target_points - centroid_target
    H = centered_source.T @ centered_target
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = centroid_target - R @ centroid_source
    return R, t

def compute_jacobian_and_covariance(source_points_final_aligned: np.ndarray, target_points_corresponding: np.ndarray, final_rmse: float) -> np.ndarray:
    N = source_points_final_aligned.shape[0]
    if N < 3: return np.full((6,6), np.inf)
    sigma_squared = final_rmse**2
    J = np.zeros((N * 3, 6))
    for i in range(N):
        px, py, pz = source_points_final_aligned[i]
        J[i*3 + 0, 0] = 0; J[i*3 + 0, 1] = pz; J[i*3 + 0, 2] = -py
        J[i*3 + 1, 0] = -pz; J[i*3 + 1, 1] = 0; J[i*3 + 1, 2] = px
        J[i*3 + 2, 0] = py; J[i*3 + 2, 1] = -px; J[i*3 + 2, 2] = 0
        J[i*3 + 0, 3] = 1; J[i*3 + 0, 4] = 0; J[i*3 + 0, 5] = 0
        J[i*3 + 1, 3] = 0; J[i*3 + 1, 4] = 1; J[i*3 + 1, 5] = 0
        J[i*3 + 2, 3] = 0; J[i*3 + 2, 4] = 0; J[i*3 + 2, 5] = 1
    H = J.T @ J
    H += np.eye(6) * 1e-9
    covariance_matrix = np.linalg.inv(H) * sigma_squared
    return covariance_matrix

# This is your custom ICP function that returns covariance
# def run_icp(source_lanes: List[np.ndarray], target_lanes: List[np.ndarray], max_iterations: int = 100, tolerance: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float], np.ndarray]:
#     source_cloud = np.vstack(source_lanes).astype(np.float64)
#     target_cloud = np.vstack(target_lanes).astype(np.float64)
#     current_R = np.eye(3)
#     current_t = np.zeros(3)
#     transformed_source_cloud = source_cloud.copy()
#     kdtree_target = KDTree(target_cloud)
#     mse_history = []
#     prev_mse = float('inf')
#     final_rmse = float('inf')
#     # print("Starting ICP...") # Commented to reduce verbose output in main loop
#     for i in range(max_iterations):
#         distances, indices = kdtree_target.query(transformed_source_cloud)
#         corresponding_target_points = target_cloud[indices]
#         current_mse = np.mean(distances**2)
#         mse_history.append(current_mse)
#         # print(f"  Iteration {i+1}: MSE = {current_mse:.6f}")
#         R_iter, t_iter = align_points_svd(transformed_source_cloud, corresponding_target_points)
#         transformed_source_cloud = (R_iter @ transformed_source_cloud.T).T + t_iter
#         current_t = R_iter @ current_t + t_iter
#         current_R = R_iter @ current_R
#         if abs(prev_mse - current_mse) < tolerance:
#             # print(f"ICP converged after {i+1} iterations.")
#             final_rmse = np.sqrt(current_mse)
#             break
#         prev_mse = current_mse
#     else:
#         # print(f"ICP finished after {max_iterations} iterations without full convergence.")
#         final_rmse = np.sqrt(current_mse)
    
#     final_distances, final_indices = kdtree_target.query(transformed_source_cloud)
#     final_corresponding_target_points = target_cloud[final_indices]
    
#     covariance_matrix = compute_jacobian_and_covariance(
#         transformed_source_cloud,
#         final_corresponding_target_points,
#         final_rmse
#     )
#     return transformed_source_cloud, current_R, current_t, mse_history, covariance_matrix

# --- EKF Components (as defined in previous answers) ---
# State: x = [x, y, yaw] (dim_x = 3)
# Control Input: u = [steer_angle, speed] (dim_u = 2)
# Measurement: z = [x_m, y_m, yaw_m] (dim_z = 3)

def f_x_vehicle_kinematics_u(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    x_k = x.copy()
    px, py, yaw = x[0], x[1], x[2]
    steer_angle, speed = u[0], u[1]
    L = 2.0 
    if abs(steer_angle) < 1e-6:
        x_k[0] += speed * np.cos(yaw) * dt
        x_k[1] += speed * np.sin(yaw) * dt
    else:
        radius = L / np.tan(steer_angle)
        angular_vel_yaw = speed / radius
        x_k[2] += angular_vel_yaw * dt
        x_k[0] += radius * (np.sin(yaw + angular_vel_yaw * dt) - np.sin(yaw))
        x_k[1] += radius * (np.cos(yaw) - np.cos(yaw + angular_vel_yaw * dt))
    x_k[2] = np.arctan2(np.sin(x_k[2]), np.cos(x_k[2]))
    return x_k

def F_jacobian_vehicle_kinematics_u(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    F = np.eye(3)
    px, py, yaw = x[0], x[1], x[2]
    steer_angle, speed = u[0], u[1]
    L = 2.0
    if abs(steer_angle) < 1e-6:
        F[0, 2] = -speed * np.sin(yaw) * dt
        F[1, 2] =  speed * np.cos(yaw) * dt
        F[0, 0] = 1.0 # px/px
        F[1, 1] = 1.0 # py/py
        F[2, 2] = 1.0 # yaw/yaw
    else:
        radius = L / np.tan(steer_angle)
        angular_vel_yaw = speed / radius
        F[0, 2] = radius * (np.cos(yaw + angular_vel_yaw * dt) - np.cos(yaw))
        F[1, 2] = radius * (np.sin(yaw + angular_vel_yaw * dt) - np.sin(yaw))
        F[2, 2] = 1.0
        # More precise bicycle model Jacobians would have more non-zero terms
        # depending on which vars are independent
    return F

def h_x_icp_measurement(x: np.ndarray) -> np.ndarray:
    return x[:3].copy() # Measures x, y, yaw from state

def H_jacobian_icp_measurement(x: np.ndarray) -> np.ndarray:
    return np.eye(3) # Jacobian for [x,y,yaw] measurement is identity

# --- Simulation Data Generation (for lanes and control inputs) ---
def simulate_data_for_ekf_with_icp(num_steps: int, dt: float, icp_update_interval: int):
    true_states = [] # [x, y, yaw]
    icp_measurements_data = [] # Store tuple of (source_lanes, target_lanes) or None
    control_inputs = [] # [steer_angle, speed]

    # Initial true state: [x, y, yaw]
    x_true = np.array([0.0, 0.0, 0.0])
    true_states.append(x_true.copy())

    # Initial lane points (for the first 'source_lanes')
    # Assume fixed lane structure for simplicity
    base_lane1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    base_lane2 = np.array([[0.0, 0.5, 0.0], [1.0, 0.5, 0.0], [2.0, 0.5, 0.0]])
    current_source_lanes = [base_lane1, base_lane2]
    
    # Noise parameters for simulation
    sim_process_noise_pos_std = 0.05
    sim_process_noise_yaw_std = np.radians(0.5)
    
    # Noise for generating ICP measurements
    sim_icp_measurement_noise_pos_std = 0.8
    sim_icp_measurement_noise_yaw_std = np.radians(3.0)

    for i in range(num_steps):
        # 1. Determine Control Input (u)
        if i < 50:
            current_steer = np.radians(0)
            current_speed = 5.0
        elif i < 100:
            current_steer = np.radians(10)
            current_speed = 5.0
        elif i < 150:
            current_steer = np.radians(-10)
            current_speed = 5.0
        else:
            current_steer = np.radians(0)
            current_speed = 5.0
        
        u_current = np.array([current_steer, current_speed])
        control_inputs.append(u_current)

        # 2. Update True State (for simulation realism)
        px, py, yaw = x_true[0], x_true[1], x_true[2]
        L = 2.0 
        if abs(current_steer) < 1e-6:
            px += current_speed * np.cos(yaw) * dt
            py += current_speed * np.sin(yaw) * dt
        else:
            radius = L / np.tan(current_steer)
            angular_vel_yaw = current_speed / radius
            yaw += angular_vel_yaw * dt
            px += radius * (np.sin(yaw) - np.sin(yaw - angular_vel_yaw * dt))
            py += radius * (np.cos(yaw - angular_vel_yaw * dt) - np.cos(yaw))
        
        # Add process noise to true state
        px += np.random.normal(0, sim_process_noise_pos_std)
        py += np.random.normal(0, sim_process_noise_pos_std)
        yaw += np.random.normal(0, sim_process_noise_yaw_std)
        
        yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
        x_true = np.array([px, py, yaw])
        
        true_states.append(x_true.copy())

        # 3. Simulate ICP Measurement Data (source_lanes, target_lanes for your run_icp)
        if (i + 1) % icp_update_interval == 0:
            # Simulate transformation of the *base lanes* to the current true pose
            # This forms the current 'target_lanes' for the ICP algorithm
            
            # Rotation matrix from base to current true yaw
            R_base_to_true_yaw = Rotation.from_euler('xyz', [0, 0, x_true[2]], degrees=False).as_matrix()
            
            # Translation vector to current true x,y,z (assuming z=0 for lanes)
            t_base_to_true_xyz = np.array([x_true[0], x_true[1], 0.0]) # Z is 0 for lanes
            
            # Create the 'target_lanes' for ICP by transforming base lanes
            # and adding noise to simulate sensor noise/ICP error
            sim_target_lanes = []
            for lane_segment in [base_lane1, base_lane2]: # Use fixed base lanes
                transformed_segment = (R_base_to_true_yaw @ lane_segment.T).T + t_base_to_true_xyz
                # Add noise to simulate sensor measurement noise in target_lanes (what ICP sees)
                transformed_segment += np.random.normal(0, 0.05, transformed_segment.shape) # Sensor noise
                sim_target_lanes.append(transformed_segment)
            
            # In a real scenario, current_source_lanes would be the *actual* lane detection output
            # from the current LiDAR scan, and sim_target_lanes would be a map or a previous scan.
            # For this simulation, let's keep `current_source_lanes` fixed to simply simulate *a* ICP measurement.
            # In reality, you'd have your current LiDAR features as source_lanes and previous as target.
            
            icp_measurements_data.append((current_source_lanes, sim_target_lanes)) # Store (source, target) for ICP
        else:
            icp_measurements_data.append(None) # No ICP measurement for this step

    return np.array(true_states), control_inputs, icp_measurements_data

# --- Main EKF Loop ---
if __name__ == "__main__":
    dt = 0.1 # Time step
    num_steps = 200
    icp_update_interval = 10 # Perform ICP every 10 steps

    # 1. Initialize EKF
    # State: [x, y, yaw] (dim_x = 3)
    initial_x = np.array([0.0, 0.0, 0.0]) # Initial guess for pose
    initial_P = np.diag([0.5**2, 0.5**2, np.radians(5.0)**2]) # Initial uncertainty

    ekf = ExtendedKalmanFilter(dim_x=3, dim_z=3, dim_u=2) # dim_x: state dim (x,y,yaw), dim_z: measurement dim (x,y,yaw), dim_u: control dim (steer, speed)

    ekf.x = initial_x.copy()
    ekf.P = initial_P.copy()

    # Assign motion and measurement functions
    ekf.f = lambda x_current, u_current: f_x_vehicle_kinematics_u(x_current, u_current, dt)
    ekf.F_jac = lambda x_current, u_current: F_jacobian_vehicle_kinematics_u(x_current, u_current, dt)
    
    ekf.h = h_x_icp_measurement 
    ekf.H_jac = H_jacobian_icp_measurement

    # Assign noise covariances
    ekf.Q = np.diag([0.1**2, 0.1**2, np.radians(1.0)**2]) # Process Noise (how much prediction drifts)

    # R is dynamically set by ICP's covariance, but give a default
    ekf.R = np.diag([0.05**2, 0.05**2, np.radians(0.5)**2]) # Default Measurement Noise (used if no ICP)

    ekf.B = np.zeros((ekf.dim_x, ekf.dim_u))
    
    # Simulate data for EKF with ICP
    true_states_full, control_inputs_full, icp_data_for_steps = simulate_data_for_ekf_with_icp(
        num_steps, dt, icp_update_interval
    )

    # Store results for plotting
    ekf_states_full = []
    
    print("\nStarting 3-DOF Vehicle EKF simulation (with ICP integration)...")
    for i in range(num_steps):
        # --- Prediction Step ---
        # Pass control input 'u' for the current step to predict
        ekf.predict(u=control_inputs_full[i]) 

        # --- Update Step (if ICP measurement is available for this step) ---
        if icp_data_for_steps[i] is not None:
            print(f"Step {i+1}: ICP measurement available. Running ICP...")
            
            # Run YOUR ICP function
            source_lanes_for_icp, target_lanes_for_icp = icp_data_for_steps[i]
            
            # Call your run_icp function here
            # Your run_icp returns: (transformed_source_cloud, R_final, t_final, mse_history, covariance_matrix)
            _, R_icp, t_icp, _, covariance_icp = run_icp(
                source_lanes=source_lanes_for_icp,
                target_lanes=target_lanes_for_icp,
                max_iterations=50, # Set max_iterations for ICP
                tolerance=1e-5     # Set tolerance for ICP
            )

            # 1. Format ICP result into measurement vector 'z_icp' (3x1)
            # Your EKF state is [x, y, yaw]
            # ICP's t is [tx, ty, tz]. ICP's R is 3x3. We need yaw from R.
            r_rot_icp = Rotation.from_matrix(R_icp.copy()) # .copy() to avoid read-only error
            euler_angles_icp_rad = r_rot_icp.as_euler('xyz', degrees=False)
            
            # The ICP measurement (z_icp) needs to be an ABSOLUTE pose measurement
            # to match h_x_icp_measurement(x) which returns x[:3] (absolute x,y,yaw).
            # This requires knowing the previous absolute pose, or integrating incremental ICP poses.
            # For simplicity, let's assume this ICP measurement gives us a direct absolute pose
            # based on how we generated sim_target_lanes.
            
            # This is a critical point: how does your ICP give you an absolute pose?
            # If your ICP matches current scan to a GLOBAL map, it directly gives global pose.
            # If your ICP matches current scan to PREVIOUS scan, it gives a RELATIVE pose.
            # This simulation generates absolute-like targets.
            
            z_icp = np.array([
                t_icp[0],       # Measured X position
                t_icp[1],       # Measured Y position
                euler_angles_icp_rad[2] # Measured Yaw (from Z-rotation of R)
            ])
            
            print(z_icp)
            # 2. Extract Measurement Covariance 'R_icp_filtered' (3x3)
            # ICP covariance is 6x6 (rx,ry,rz,tx,ty,tz). We need 3x3 for (tx,ty,tz) if using global translation,
            # or (rx,ry,rz) if global rotation, or specific parts for [x,y,yaw].
            # Assuming pose covariance order is [rx, ry, rz, tx, ty, tz]
            # We need covariances for [tx, ty, yaw]. Roll and Pitch are ignored in this EKF state.
            
            # The covariance matrix from ICP usually represents uncertainty in (rx, ry, rz, tx, ty, tz).
            # We need the sub-matrix corresponding to (tx, ty, tz) (indices 3,4,5) and (yaw_rate)
            # which is (rx,ry,rz) for yaw.
            # Let's extract the (tx, ty, tz) part and the (rz) part for yaw (index 2 for rz).
            
            # To get [x,y,yaw] covariance from ICP's [rx,ry,rz,tx,ty,tz] covariance:
            # We need index mapping for: (tx,ty,yaw_from_rz)
            # Assuming yaw is directly from rz.
            # Use indices 2 (rz), 3 (tx), 4 (ty)
            
            indices_for_z = np.array([3, 4, 2]) # Assuming ICP covariance is ordered [rx,ry,rz,tx,ty,tz]
            R_icp_filtered = covariance_icp[np.ix_(indices_for_z, indices_for_z)]

            # Ensure R_icp_filtered is positive semi-definite (optional but good practice)
            # Small regularization to avoid singular matrices
            R_icp_filtered += np.eye(3) * 1e-6 
            
            ekf.update(z=z_icp, R=R_icp_filtered, HJacobian=ekf.H_jac, Hx=ekf.h) 

        ekf_states_full.append(ekf.x.copy())

    ekf_states_full = np.array(ekf_states_full)

    # --- Plotting Results ---
    plt.figure(figsize=(12, 8))
    plt.plot(true_states_full[:, 0], true_states_full[:, 1], 'g-', linewidth=2, label='True Path (XY)')
    plt.plot(ekf_states_full[:, 0], ekf_states_full[:, 1], 'b-', linewidth=2, alpha=0.8, label='EKF Estimated Path (XY)')
    
    # Plot ICP measurements that were used (only x,y for plotting)
    icp_meas_x = [m[0] for m in icp_data_for_steps if m is not None] # Extract x from simulated data
    icp_meas_y = [m[1] for m in icp_data_for_steps if m is not None] # Extract y from simulated data
    
    # In this simulation, icp_data_for_steps[i] is (source_lanes, target_lanes)
    # We need to extract the actual measurement that ICP *would* provide.
    # To plot the *raw* ICP measurement, we would need to run ICP on all those points
    # and extract their t_icp.
    # For simplicity, let's plot the true path at ICP measurement points.
    
    # Let's directly plot the raw true positions from the data where ICP would happen
    icp_measurement_steps = np.where([x is not None for x in icp_data_for_steps])[0]
    icp_measurement_true_x = true_states_full[icp_measurement_steps, 0]
    icp_measurement_true_y = true_states_full[icp_measurement_steps, 1]

    plt.scatter(icp_measurement_true_x, icp_measurement_true_y, color='purple', marker='o', s=50, alpha=0.6, label='True Path at ICP Measurement Points')

    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.title('Vehicle Pose Estimation (3-DOF EKF with ICP Integration)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    print(f"\nFinal EKF Estimated Pose (x,y,yaw - m, rad):\n{ekf.x.round(3)}")
    print(f"Final EKF Covariance:\n{ekf.P.round(5)}")
    print(f"Final EKF Pose Yaw (degrees): {np.degrees(ekf.x[2]).round(3)}")