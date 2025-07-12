import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.spatial.transform import Rotation # For converting R to Euler angles

# --- 1. Define Motion Model and its Jacobian (f_x, F_jacobian) ---
# State: [x, y, yaw, steer_angle, speed] (dim_x = 5)
# This model assumes steer_angle and speed are constant over dt
# and updates x, y, yaw based on them.

def f_x_vehicle_kinematics(x: np.ndarray, dt: float) -> np.ndarray:
    """
    Non-linear state transition function for a vehicle kinematic model.
    State: [x, y, yaw, steer_angle, speed]
    Assumes constant steer_angle and speed over dt for prediction.
    """
    x_k = x.copy()
    
    # Extract current state components
    px, py, yaw, steer_angle, speed = x[0], x[1], x[2], x[3], x[4]

    # Update x, y, yaw using bicycle model approximation (simplified)
    # This is a common non-linear motion model for vehicles
    if abs(steer_angle) < 1e-6: # Straight motion
        x_k[0] += speed * np.cos(yaw) * dt
        x_k[1] += speed * np.sin(yaw) * dt
    else: # Turning motion
        # Simplified turn: assumes turning radius is based on steer_angle and wheelbase
        # For simplicity, we'll use a direct angular velocity approximation
        # A more accurate model would use wheelbase and actual turning radius
        angular_vel_yaw = speed * np.tan(steer_angle) / 2.0 # simplified, 2.0 is dummy wheelbase
        
        # Update yaw
        x_k[2] += angular_vel_yaw * dt
        
        # Update position based on new yaw
        # This is a common integration for non-holonomic robots
        x_k[0] += speed * np.cos(yaw + angular_vel_yaw * dt / 2) * dt
        x_k[1] += speed * np.sin(yaw + angular_vel_yaw * dt / 2) * dt

    # Normalize yaw to [-pi, pi]
    x_k[2] = np.arctan2(np.sin(x_k[2]), np.cos(x_k[2]))

    # steer_angle and speed are assumed constant for prediction
    # x_k[3] = steer_angle (already copied)
    # x_k[4] = speed (already copied)

    return x_k

def F_jacobian_vehicle_kinematics(x: np.ndarray, dt: float) -> np.ndarray:
    """
    Jacobian of the state transition function f_x_vehicle_kinematics.
    This is for the prediction step.
    State: [x, y, yaw, steer_angle, speed]
    """
    F = np.eye(5) # 5x5 identity matrix

    # Extract current state components
    px, py, yaw, steer_angle, speed = x[0], x[1], x[2], x[3], x[4]

    # Derivatives of x, y, yaw w.r.t. x, y, yaw, steer_angle, speed
    # dx/dyaw, dx/dsteer, dx/dspeed, etc.
    
    # df_x/dx: [1,0,0,0,0]
    # df_y/dy: [0,1,0,0,0]
    # df_yaw/dyaw: [0,0,1,0,0]
    
    # Non-zero terms from motion model:
    # dx_k / dyaw_{k-1}
    # dy_k / dyaw_{k-1}
    # dx_k / dspeed_{k-1}
    # dy_k / dspeed_{k-1}
    # dyaw_k / dsteer_angle_{k-1}
    # dyaw_k / dspeed_{k-1}

    # This Jacobian is complex and highly non-linear.
    # For simplicity, let's provide a basic approximation for a constant velocity model.
    # A full derivation of the bicycle model Jacobian is extensive.
    # For a constant velocity/angular velocity assumption, it's often sparse.

    # Simplified Jacobian for illustration (assuming small dt and direct integration)
    # This is a rough approximation. For a real bicycle model, it's much more complex.
    F[0, 2] = -speed * np.sin(yaw) * dt # dx/dyaw
    F[1, 2] =  speed * np.cos(yaw) * dt # dy/dyaw
    F[0, 4] = np.cos(yaw) * dt          # dx/dspeed
    F[1, 4] = np.sin(yaw) * dt          # dy/dspeed

    if abs(steer_angle) > 1e-6: # Only if turning
        angular_vel_yaw = speed * np.tan(steer_angle) / 2.0
        # dyaw_k / dsteer_angle_{k-1}
        F[2, 3] = speed / (2.0 * np.cos(steer_angle)**2) * dt # dyaw/dsteer_angle
        # dyaw_k / dspeed_{k-1}
        F[2, 4] = np.tan(steer_angle) / 2.0 * dt # dyaw/dspeed

    # Other diagonal elements are 1 (e.g., steer_angle_k = steer_angle_{k-1})
    # All other off-diagonal elements are 0.

    return F

# --- 2. Define Measurement Model and its Jacobian (h_x, H_jacobian) ---
# Measurement: [x_m, y_m, yaw_m] (3 dimensions)
# This comes from ICP (LiDAR odometry)

def h_x_icp_measurement(x: np.ndarray) -> np.ndarray:
    """
    Measurement function: Maps the state [x, y, yaw, steer_angle, speed]
    to the measurement [x, y, yaw] from ICP.

    Args:
        x (np.ndarray): The current state vector.

    Returns:
        np.ndarray: The predicted measurement [x, y, yaw].
    """
    # ICP measures the first three elements of the state (x, y, yaw)
    return x[:3].copy()

def H_jacobian_icp_measurement(x: np.ndarray) -> np.ndarray:
    """
    Jacobian of the measurement function h_x_icp_measurement with respect to the state x.
    This is a 3x5 matrix (3 measurements, 5 states).

    Args:
        x (np.ndarray): The current state vector (not used in this constant Jacobian).

    Returns:
        np.ndarray: The 3x5 Jacobian matrix.
    """
    H_mat = np.zeros((3, 5))
    H_mat[0, 0] = 1.0 # dx_m / dx
    H_mat[1, 1] = 1.0 # dy_m / dy
    H_mat[2, 2] = 1.0 # dyaw_m / dyaw
    return H_mat

# --- 3. Simulation Setup ---
def simulate_vehicle_data(num_steps: int, dt: float):
    """Generates true path, noisy measurements, and noisy kinematic inputs for vehicle."""
    true_states = [] # [x, y, yaw, steer_angle, speed]
    measurements = [] # [x, y, yaw] from ICP
    
    # Initial true state
    x_true = np.array([0.0, 0.0, 0.0, np.radians(0.0), 5.0]) # [x, y, yaw, steer_angle, speed]
    true_states.append(x_true.copy())

    # Noise parameters for simulation
    sim_process_noise_pos_std = 0.05 # Std dev for position drift per step
    sim_process_noise_yaw_std = np.radians(0.5) # Std dev for yaw drift per step
    sim_measurement_noise_pos_std = 0.8 # Std dev for measurement noise in position
    sim_measurement_noise_yaw_std = np.radians(3.0) # Std dev for measurement noise in yaw

    for i in range(num_steps):
        # Apply true motion (with some random process noise)
        # Simulate a turn every 50 steps
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

        # Update true state based on current steer/speed
        yaw_prev = x_true[2]
        if abs(current_steer) < 1e-6:
            x_true[0] += current_speed * np.cos(yaw_prev) * dt
            x_true[1] += current_speed * np.sin(yaw_prev) * dt
        else:
            angular_vel_yaw = current_speed * np.tan(current_steer) / 2.0
            x_true[2] += angular_vel_yaw * dt
            x_true[0] += current_speed * np.cos(yaw_prev + angular_vel_yaw * dt / 2) * dt
            x_true[1] += current_speed * np.sin(yaw_prev + angular_vel_yaw * dt / 2) * dt
        
        # Add process noise to true state
        x_true[0] += np.random.normal(0, sim_process_noise_pos_std)
        x_true[1] += np.random.normal(0, sim_process_noise_pos_std)
        x_true[2] += np.random.normal(0, sim_process_noise_yaw_std)
        
        # Normalize yaw
        x_true[2] = np.arctan2(np.sin(x_true[2]), np.cos(x_true[2]))

        # Update steer_angle and speed in true state (they are inputs, but we track them)
        x_true[3] = current_steer
        x_true[4] = current_speed
        
        true_states.append(x_true.copy())

        # Simulate ICP measurement (noisy pose)
        if i % 10 == 0: # Measure every 10 steps
            meas_noise_pos = np.random.normal(0, sim_measurement_noise_pos_std, 2)
            meas_noise_yaw = np.random.normal(0, sim_measurement_noise_yaw_std)
            
            measurements.append(np.array([
                x_true[0] + meas_noise_pos[0],
                x_true[1] + meas_noise_pos[1],
                x_true[2] + meas_noise_yaw
            ]))
        else:
            measurements.append(None) # No measurement for this step

    return np.array(true_states), measurements

# --- Main EKF Loop ---
if __name__ == "__main__":
    dt = 0.1 # Time step
    num_steps = 200

    # 1. Initialize EKF
    # State: [x, y, yaw, steer_angle, speed] (dim_x = 5)
    initial_x = np.array([0.0, 0.0, 0.0, np.radians(0.0), 5.0])
    initial_P = np.diag([0.5**2, 0.5**2, np.radians(5.0)**2, np.radians(10.0)**2, 1.0**2])

    ekf = ExtendedKalmanFilter(dim_x=5, dim_z=3) # dim_x: state dim, dim_z: measurement dim (x,y,yaw)

    ekf.x = initial_x.copy()
    ekf.P = initial_P.copy()

    # Assign motion and measurement functions
    ekf.f = lambda x_current, u_dummy=None: f_x_vehicle_kinematics(x_current, dt)
    ekf.F_jac = lambda x_current, u_dummy=None: F_jacobian_vehicle_kinematics(x_current, dt)
    
    ekf.h = h_x_icp_measurement # h_x_icp_measurement needs x
    ekf.H_jac = H_jacobian_icp_measurement # H_jacobian_icp_measurement needs x

    # Assign noise covariances
    # Q: Process Noise Covariance (Uncertainty in our vehicle kinematic model)
    ekf.Q = np.diag([
        0.05**2, 0.05**2, # x,y noise (m^2)
        np.radians(0.5)**2, # yaw noise (rad^2)
        np.radians(1.0)**2, # steer_angle noise (rad^2) - how much steer_angle changes
        0.1**2              # speed noise (m/s)^2 - how much speed changes
    ])

    # R: Measurement Noise Covariance (from ICP/LiDAR)
    # This would be the 3x3 covariance from your ICP result for [x,y,yaw]
    ekf.R = np.diag([0.2**2, 0.2**2, np.radians(3.0)**2]) # 0.8m pos std dev, 3 deg yaw std dev

    # Simulate data
    true_states_full, measurements_full = simulate_vehicle_data(num_steps, dt)

    # Store results for plotting
    ekf_states_full = []
    
    print("\nStarting 5-DOF Vehicle EKF simulation...")
    for i in range(num_steps):
        # --- Prediction Step ---
        # Note: In this model, steer_angle and speed are part of the state,
        # so they are propagated directly by f_x_vehicle_kinematics.
        # If they were external control inputs, you'd pass them as u.
        ekf.predict() 

        # --- Update Step (if ICP measurement is available) ---
        if measurements_full[i] is not None:
            # In a real system, you'd get z_icp and R_icp from your icp_lane_registration function:
            # icp_results = icp_lane_registration(...)
            # z_icp = np.array([icp_results['t'][0], icp_results['t'][1], np.degrees(Rotation.from_matrix(icp_results['R']).as_euler('xyz'))[2]]) # Example, extract x,y,yaw
            # R_icp = icp_results['covariance'][np.array([3,4,5])[:,None], np.array([3,4,5])] # Extract 3x3 for x,y,yaw if covariance is (rx,ry,rz,tx,ty,tz)
            # R_icp = icp_results['covariance'] # If ICP directly outputs 3x3 for [x,y,yaw]

            # For this simulation, we use measurements_full[i] directly and ekf.R
            ekf.update(z=measurements_full[i], R=ekf.R, HJacobian=ekf.H_jac, Hx=ekf.h) 

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
    plt.title('Vehicle Pose Estimation (Simplified 5-DOF EKF - XY Plane Projection)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    print(f"\nFinal EKF Estimated Pose (m, rad):\n{ekf.x.round(3)}")
    print(f"Final EKF Covariance:\n{ekf.P.round(5)}")
    print(f"Final EKF Pose (degrees):\nRoll: {np.degrees(ekf.x[2]).round(3)} (yaw)") # Only yaw is in state