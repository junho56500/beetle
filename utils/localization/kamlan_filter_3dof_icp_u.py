import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.spatial.transform import Rotation # For interpreting ICP R matrix

# --- 1. Define Motion Model and its Jacobian ---
# State: x = [x, y, yaw] (dim_x = 3)
# Control Input: u = [steer_angle, speed] (dim_u = 2)
# The motion model uses u to predict the next state.

def f_x_vehicle_kinematics_u(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """
    Non-linear state transition function for a vehicle kinematic model.
    State: [x, y, yaw]
    Control Input (u): [steer_angle, speed]
    """
    x_k = x.copy()
    
    # Extract current state components
    px, py, yaw = x[0], x[1], x[2]

    # Extract control input components
    steer_angle, speed = u[0], u[1]

    # Update x, y, yaw using bicycle model approximation (simplified)
    # Assume wheelbase is 2.0 for simplicity (L)
    L = 2.0 # Wheelbase (meters)

    if abs(steer_angle) < 1e-6: # Straight motion (steer angle approx 0)
        x_k[0] += speed * np.cos(yaw) * dt
        x_k[1] += speed * np.sin(yaw) * dt
        # yaw doesn't change
    else: # Turning motion
        # Calculate turning radius and angular velocity
        turning_radius = L / np.tan(steer_angle)
        angular_vel_yaw = speed / turning_radius # omega = v / R

        # Update yaw
        x_k[2] += angular_vel_yaw * dt
        
        # Update position based on average yaw during the step
        x_k[0] += turning_radius * (np.sin(yaw + angular_vel_yaw * dt) - np.sin(yaw))
        x_k[1] += turning_radius * (np.cos(yaw) - np.cos(yaw + angular_vel_yaw * dt))

    # Normalize yaw to [-pi, pi]
    x_k[2] = np.arctan2(np.sin(x_k[2]), np.cos(x_k[2]))

    return x_k

def F_jacobian_vehicle_kinematics_u(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """
    Jacobian of the state transition function f_x_vehicle_kinematics_u with respect to the state x.
    State: [x, y, yaw]
    Control Input (u): [steer_angle, speed]
    """
    F = np.eye(3) # 3x3 Jacobian for [x, y, yaw] w.r.t. [x, y, yaw]

    # Extract current state and control input components
    px, py, yaw = x[0], x[1], x[2]
    steer_angle, speed = u[0], u[1]
    L = 2.0 # Wheelbase

    if abs(steer_angle) < 1e-6: # Straight motion
        F[0, 2] = -speed * np.sin(yaw) * dt # dx/dyaw
        F[1, 2] =  speed * np.cos(yaw) * dt # dy/dyaw
    else: # Turning motion
        radius = L / np.tan(steer_angle)
        angular_vel_yaw = speed / radius
        
        F[0, 2] = radius * (np.cos(yaw + angular_vel_yaw * dt) - np.cos(yaw)) # dx/dyaw
        F[1, 2] = radius * (np.sin(yaw + angular_vel_yaw * dt) - np.sin(yaw)) # dy/dyaw
        F[2, 2] = 1.0 # dyaw/dyaw

    return F

# --- 2. Define Measurement Model and its Jacobian (h_x, H_jacobian) ---
# Measurement: [x_m, y_m, yaw_m] (3 dimensions from ICP)
# State: [x, y, yaw, steer_angle, speed]

def h_x_icp_measurement(x: np.ndarray) -> np.ndarray:
    """
    Measurement function: Maps the state [x, y, yaw, steer_angle, speed]
    to the measurement [x, y, yaw] from ICP.
    """
    return x[:3].copy() # Measures the first three elements of the state

def H_jacobian_icp_measurement(x: np.ndarray) -> np.ndarray:
    """
    Jacobian of the measurement function h_x_icp_measurement with respect to the state x.
    This is a 3x3 matrix (3 measurements, 3 relevant states).
    The actual EKF H_jac needs to be 3xdim_x.

    Args:
        x (np.ndarray): The current state vector.

    Returns:
        np.ndarray: The 3xdim_x Jacobian matrix.
    """
    # H_mat = np.zeros((3, 5)) # 3 measurements, 5 states
    # H_mat[0, 0] = 1.0 # dx_m / dx
    # H_mat[1, 1] = 1.0 # dy_m / dy
    # H_mat[2, 2] = 1.0 # dyaw_m / dyaw
    H_mat = np.eye(3) # Creates a 3x3 identity matrix automatically
    
    return H_mat

# --- 3. Simulation Setup ---
def simulate_vehicle_data_u(num_steps: int, dt: float):
    """Generates true path, noisy measurements, and control inputs (u)."""
    true_poses_and_u = [] # [x, y, yaw, steer_angle, speed]
    measurements = [] # [x, y, yaw] from ICP
    control_inputs = [] # [steer_angle, speed]

    # Initial true state
    x_true_state_and_u = np.array([0.0, 0.0, 0.0, np.radians(0.0), 5.0]) # [x, y, yaw, steer_angle, speed]
    true_poses_and_u.append(x_true_state_and_u.copy())

    # Noise parameters for simulation
    sim_process_noise_pos_std = 0.05
    sim_process_noise_yaw_std = np.radians(0.5)
    sim_measurement_noise_pos_std = 0.8
    sim_measurement_noise_yaw_std = np.radians(3.0)

    for i in range(num_steps):
        # Determine control input for this step (simulate driver/planner)
        if i < 50:
            current_steer = np.radians(0)
            current_speed = 5.0
        elif i < 100:
            current_steer = np.radians(10) # Turn left
            current_speed = 5.0
        elif i < 150:
            current_steer = np.radians(-10) # Turn right
            current_speed = 5.0
        else:
            current_steer = np.radians(0)
            current_speed = 5.0
        
        control_input_u = np.array([current_steer, current_speed])
        control_inputs.append(control_input_u)

        # Update true state based on control input (f_x_vehicle_kinematics_u logic)
        px, py, yaw, _, _ = x_true_state_and_u # current true pose
        L = 2.0
        if abs(current_steer) < 1e-6:
            px += current_speed * np.cos(yaw) * dt
            py += current_speed * np.sin(yaw) * dt
        else:
            radius = L / np.tan(current_steer)
            angular_vel_yaw = current_speed / radius
            yaw += angular_vel_yaw * dt
            px += radius * (np.sin(yaw) - np.sin(yaw - angular_vel_yaw * dt)) # Corrected position update
            py += radius * (np.cos(yaw - angular_vel_yaw * dt) - np.cos(yaw)) # Corrected position update
        
        # Add process noise to true state (for simulation realism)
        px += np.random.normal(0, sim_process_noise_pos_std)
        py += np.random.normal(0, sim_process_noise_pos_std)
        yaw += np.random.normal(0, sim_process_noise_yaw_std)
        
        # Normalize yaw
        yaw = np.arctan2(np.sin(yaw), np.cos(yaw))

        # Update true state in array
        x_true_state_and_u[0], x_true_state_and_u[1], x_true_state_and_u[2] = px, py, yaw
        x_true_state_and_u[3], x_true_state_and_u[4] = current_steer, current_speed # Update true u values in array

        true_poses_and_u.append(x_true_state_and_u.copy())

        # Simulate ICP measurement (noisy pose)
        if (i + 1) % 10 == 0: # Measure every 10 steps
            meas_noise_pos = np.random.normal(0, sim_measurement_noise_pos_std, 2)
            meas_noise_yaw = np.random.normal(0, sim_measurement_noise_yaw_std)
            
            measurements.append(np.array([
                px + meas_noise_pos[0],
                py + meas_noise_pos[1],
                yaw + meas_noise_yaw
            ]))
        else:
            measurements.append(None) # No measurement for this step

    return np.array(true_poses_and_u), measurements, np.array(control_inputs)

# --- Main EKF Loop ---
if __name__ == "__main__":
    dt = 0.1 # Time step
    num_steps = 200

    # 1. Initialize EKF
    # State: [x, y, yaw] (dim_x = 3)
    initial_x = np.array([0.0, 0.0, 0.0]) # Initial guess for pose
    initial_P = np.diag([0.5**2, 0.5**2, np.radians(5.0)**2]) # Initial uncertainty

    ekf = ExtendedKalmanFilter(dim_x=3, dim_z=3, dim_u=2) # dim_x: state dim (x,y,yaw), dim_z: measurement dim (x,y,yaw)

    ekf.x = initial_x.copy()
    ekf.P = initial_P.copy()

    # Assign motion and measurement functions
    # f_x_vehicle_kinematics_u needs current state (x), control input (u), and dt
    ekf.f = lambda x_current, u_current: f_x_vehicle_kinematics_u(x_current, u_current, dt)
    ekf.F_jac = lambda x_current, u_current: F_jacobian_vehicle_kinematics_u(x_current, u_current, dt)
    
    ekf.h = h_x_icp_measurement # h_x_icp_measurement needs x
    ekf.H_jac = H_jacobian_icp_measurement # H_jacobian_icp_measurement needs x
    
    # Assign noise covariances
    # Q: Process Noise Covariance (Uncertainty in our vehicle kinematic model)
    # Reflects how much the state [x,y,yaw] drifts if only u is used.
    ekf.Q = np.diag([
        0.05**2, # x noise per dt
        0.05**2, # y noise per dt
        np.radians(0.5)**2 # yaw noise per dt
    ])

    # R: Measurement Noise Covariance (from ICP/LiDAR)
    ekf.R = np.diag([0.2**2, 0.2**2, np.radians(3.0)**2]) # Match sim measurement noise

    ekf.B = np.zeros((ekf.dim_x, ekf.dim_u)) 
    
    # Simulate data
    true_states_full, measurements_full, control_inputs_full = simulate_vehicle_data_u(num_steps, dt)

    # Store results for plotting
    ekf_states_full = []
    
    print("\nStarting 3-DOF Vehicle EKF simulation (x,y,yaw state, steer/speed as u)...")
    for i in range(num_steps):
        # --- Prediction Step ---
        # Pass control input 'u' for the current step to predict
        ekf.predict(u=control_inputs_full[i]) 

        # --- Update Step (if ICP measurement is available) ---
        if measurements_full[i] is not None:
            # Pass the measurement (z) and its covariance (R)
            # R_icp (from your icp_lane_registration) would go here
            ekf.update(z=measurements_full[i], R=ekf.R, HJacobian=ekf.H_jac, Hx=ekf.h) 
        print(measurements_full[i])
        ekf_states_full.append(ekf.x.copy())

    ekf_states_full = np.array(ekf_states_full)

    # --- Plotting Results (XY Plane) ---
    plt.figure(figsize=(12, 8))
    plt.plot(true_states_full[:, 0], true_states_full[:, 1], 'g-', linewidth=2, label='True Path (XY)')
    plt.plot(ekf_states_full[:, 0], ekf_states_full[:, 1], 'b-', linewidth=2, alpha=0.8, label='EKF Estimated Path (XY)')
    
    # Plot measurements
    meas_x = [m[0] for m in measurements_full if m is not None]
    meas_y = [m[1] for m in measurements_full if m is not None]
    plt.scatter(meas_x, meas_y, color='purple', marker='o', s=50, alpha=0.6, label='ICP Measurements (XY)')

    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.title('Vehicle Pose Estimation (3-DOF EKF with Control Input U - XY Plane Projection)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    print(f"\nFinal EKF Estimated Pose (x,y,yaw - m, rad):\n{ekf.x.round(3)}")
    print(f"Final EKF Covariance:\n{ekf.P.round(5)}")
    print(f"Final EKF Pose Yaw (degrees): {np.degrees(ekf.x[2]).round(3)}")