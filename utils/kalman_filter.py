import numpy as np
import matplotlib.pyplot as plt # For visualization

class EKF2DFusion:
    """
    A simplified 2D Extended Kalman Filter for fusing kinematic odometry
    and LiDAR odometry measurements.

    State: [x, y, yaw]
    """
    def __init__(self, initial_pose: np.ndarray, initial_pose_covariance: np.ndarray):
        """
        Initialize the EKF.

        Args:
            initial_pose (np.ndarray): Initial state [x, y, yaw]. Shape (3,).
            initial_pose_covariance (np.ndarray): Initial covariance matrix. Shape (3, 3).
        """
        self.x = initial_pose # Current state estimate
        self.P = initial_pose_covariance # Current covariance estimate

        # --- Process Noise Covariance (Q) ---
        # Represents uncertainty in our kinematic motion model.
        # Tunable parameter: Higher values mean less trust in kinematic prediction.
        # Example: [dx_noise, dy_noise, dyaw_noise] along diagonal.
        self.Q = np.diag([0.05**2, 0.05**2, np.radians(1.0)**2]) # Covariance for [x,y,yaw] noise per time step

        # --- Measurement Noise Covariance (R) ---
        # Represents uncertainty in the LiDAR odometry measurement.
        # This is the covariance output by your ICP/LiDAR odometry module.
        # Tunable parameter: Higher values mean less trust in LiDAR measurement.
        self.R = np.diag([0.1**2, 0.1**2, np.radians(2.0)**2]) # Covariance for [x,y,yaw] measurement noise

        print("EKF Initialized:")
        print(f"  Initial Pose: {self.x}")
        print(f"  Initial Covariance:\n{self.P}")

    def predict(self, linear_vel: float, angular_vel: float, dt: float):
        """
        Prediction step (using kinematic odometry).

        Args:
            linear_vel (float): Linear velocity (from wheel odometry).
            angular_vel (float): Angular velocity (from wheel odometry).
            dt (float): Time step.
        """
        # Non-linear motion model:
        # x_k = x_{k-1} + v * cos(yaw_{k-1}) * dt
        # y_k = y_{k-1} + v * sin(yaw_{k-1}) * dt
        # yaw_k = yaw_{k-1} + omega * dt

        yaw_prev = self.x[2]
        
        # Predicted state (x_bar)
        if abs(angular_vel) < 1e-6: # Avoid division by zero if turning radius is very large
            # Straight motion
            self.x[0] += linear_vel * np.cos(yaw_prev) * dt
            self.x[1] += linear_vel * np.sin(yaw_prev) * dt
            self.x[2] += angular_vel * dt
        else:
            # Arc motion
            radius = linear_vel / angular_vel
            self.x[0] += -radius * np.sin(yaw_prev) + radius * np.sin(yaw_prev + angular_vel * dt)
            self.x[1] += radius * np.cos(yaw_prev) - radius * np.cos(yaw_prev + angular_vel * dt)
            self.x[2] += angular_vel * dt

        # Jacobian of the motion model (F_k) - linearized around current state
        # F_k = df/dx_k-1
        F_k = np.array([
            [1.0, 0.0, -linear_vel * np.sin(yaw_prev) * dt],
            [0.0, 1.0,  linear_vel * np.cos(yaw_prev) * dt],
            [0.0, 0.0, 1.0]
        ])
        if abs(angular_vel) > 1e-6: # More complex Jacobian for arc motion
            F_k[0, 2] = radius * np.cos(yaw_prev + angular_vel * dt) - radius * np.cos(yaw_prev)
            F_k[1, 2] = radius * np.sin(yaw_prev + angular_vel * dt) - radius * np.sin(yaw_prev)
        
        # Predicted covariance (P_bar)
        self.P = F_k @ self.P @ F_k.T + self.Q

        # print(f"  Prediction: New Pose = {self.x.round(2)}, Covariance:\n{self.P.round(4)}")

    def update(self, lidar_pose_measurement: np.ndarray, lidar_pose_covariance: np.ndarray = None):
        """
        Update step (using LiDAR odometry measurement).

        Args:
            lidar_pose_measurement (np.ndarray): LiDAR's absolute pose measurement [x_m, y_m, yaw_m]. Shape (3,).
            lidar_pose_covariance (np.ndarray): Covariance matrix of the LiDAR measurement. Shape (3, 3).
                                                If None, uses the EKF's default R.
        """
        if lidar_pose_covariance is None:
            current_R_measurement = self.R
        else:
            current_R_measurement = lidar_pose_covariance # Use covariance from ICP output

        # Measurement model (H_k) - how state relates to measurement
        # Here, we assume LiDAR directly measures our state variables [x, y, yaw]
        # so H_k is simply the identity matrix.
        H_k = np.eye(3) 

        print("z : ", lidar_pose_measurement)
        print("self : ", self.x)
        
        # Measurement residual (y_k) - difference between actual measurement and expected measurement
        # Expected measurement is H_k @ self.x_bar (predicted state)
        y_k = lidar_pose_measurement - (H_k @ self.x) # This is z_k - h(x_bar_k)

        # Normalize yaw residual to be within [-pi, pi]
        y_k[2] = np.arctan2(np.sin(y_k[2]), np.cos(y_k[2]))


        # Innovation (S_k) - covariance of the residual
        S_k = H_k @ self.P @ H_k.T + current_R_measurement

        # Kalman Gain (K_k)
        K_k = self.P @ H_k.T @ np.linalg.inv(S_k)

        print('hx : ', K_k)
        
        # Updated state estimate (x_k)
        self.x = self.x + (K_k @ y_k)

        print("K_k : ", K_k)
        print("self2 : ", self.x)
        
        # Updated covariance estimate (P_k)
        self.P = (np.eye(self.x.shape[0]) - K_k @ H_k) @ self.P

        # Normalize yaw in state to be within [-pi, pi]
        self.x[2] = np.arctan2(np.sin(self.x[2]), np.cos(self.x[2]))


        # print(f"  Update: New Pose = {self.x.round(2)}, Covariance:\n{self.P.round(4)}")

# --- Simulation of Data ---
def generate_dummy_data(num_steps: int, dt: float):
    # True path (for visualization)
    true_poses = [np.array([0.0, 0.0, 0.0])] # x, y, yaw
    
    # Kinematic Odometry (noisy prediction)
    kinematic_linear_vels = np.full(num_steps, 1.0) # 1 m/s forward
    kinematic_angular_vels = np.zeros(num_steps)
    kinematic_angular_vels[num_steps // 4 : num_steps // 2] = np.radians(5) # Turn for a bit
    kinematic_angular_vels[num_steps // 2 : 3 * num_steps // 4] = np.radians(-5) # Turn back

    # LiDAR Odometry (less frequent, less noisy measurements)
    # Simulate a LiDAR measurement every 10 steps, with low noise
    lidar_measurements = []
    lidar_covariances = []
    
    # Simulating the generation of true values and noisy measurements
    current_true_x = np.array([0.0, 0.0, 0.0])
    for i in range(num_steps):
        v = kinematic_linear_vels[i]
        omega = kinematic_angular_vels[i]
        
        # Update true pose
        if abs(omega) < 1e-6:
            current_true_x[0] += v * np.cos(current_true_x[2]) * dt
            current_true_x[1] += v * np.sin(current_true_x[2]) * dt
        else:
            radius = v / omega
            current_true_x[0] += -radius * np.sin(current_true_x[2]) + radius * np.sin(current_true_x[2] + omega * dt)
            current_true_x[1] += radius * np.cos(current_true_x[2]) - radius * np.cos(current_true_x[2] + omega * dt)
        current_true_x[2] += omega * dt
        current_true_x[2] = np.arctan2(np.sin(current_true_x[2]), np.cos(current_true_x[2])) # Normalize yaw
        
        true_poses.append(current_true_x.copy())

        if (i + 1) % 10 == 0: # LiDAR measurement every 10 steps
            # Simulate LiDAR measurement with some noise
            lidar_noise = np.random.normal(0, [0.05, 0.05, np.radians(1.0)], 3) # Low noise
            lidar_measurements.append(current_true_x + lidar_noise)
            # Simulate ICP covariance for the measurement
            lidar_covariances.append(np.diag([0.05**2, 0.05**2, np.radians(1.5)**2])) # Lower uncertainty than default EKF R
        else:
            lidar_measurements.append(None) # No measurement for this step
            lidar_covariances.append(None)

    return np.array(true_poses), kinematic_linear_vels, kinematic_angular_vels, lidar_measurements, lidar_covariances

# --- Main Simulation Loop ---
if __name__ == "__main__":
    dt = 0.1 # Time step (seconds)
    num_simulation_steps = 200

    # Initial EKF state and uncertainty
    initial_pose = np.array([0.0, 0.0, 0.0]) # x, y, yaw
    initial_covariance = np.diag([0.1**2, 0.1**2, np.radians(5.0)**2]) # High initial uncertainty

    ekf_filter = EKF2DFusion(initial_pose, initial_covariance)

    # Generate dummy data
    true_poses_data, lin_vels, ang_vels, lidar_meas_data, lidar_cov_data = generate_dummy_data(num_simulation_steps, dt)

    # Store estimated poses for plotting
    estimated_poses = [ekf_filter.x.copy()]
    predicted_kinematic_only_poses = [initial_pose.copy()]

    for i in range(num_simulation_steps):
        # Kinematic-only prediction
        kin_x, kin_y, kin_yaw = predicted_kinematic_only_poses[-1]
        v_kin = lin_vels[i]
        omega_kin = ang_vels[i]
        
        if abs(omega_kin) < 1e-6:
            kin_x += v_kin * np.cos(kin_yaw) * dt
            kin_y += v_kin * np.sin(kin_yaw) * dt
        else:
            radius_kin = v_kin / omega_kin
            kin_x += -radius_kin * np.sin(kin_yaw) + radius_kin * np.sin(kin_yaw + omega_kin * dt)
            kin_y += radius_kin * np.cos(kin_yaw) - radius_kin * np.cos(kin_yaw + omega_kin * dt)
        kin_yaw += omega_kin * dt
        kin_yaw = np.arctan2(np.sin(kin_yaw), np.cos(kin_yaw))
        predicted_kinematic_only_poses.append(np.array([kin_x, kin_y, kin_yaw]))

        # EKF Prediction Step
        ekf_filter.predict(lin_vels[i], ang_vels[i], dt)
        
        # EKF Update Step (if LiDAR measurement is available)
        if lidar_meas_data[i] is not None:
            print(f"\nStep {i+1}: LiDAR Update triggered.")
            ekf_filter.update(lidar_meas_data[i], lidar_cov_data[i])

        estimated_poses.append(ekf_filter.x.copy())

    estimated_poses = np.array(estimated_poses)
    predicted_kinematic_only_poses = np.array(predicted_kinematic_only_poses)

    # --- Plotting Results ---
    plt.figure(figsize=(10, 8))
    plt.plot(true_poses_data[:, 0], true_poses_data[:, 1], 'g--', label='True Path')
    plt.plot(predicted_kinematic_only_poses[:, 0], predicted_kinematic_only_poses[:, 1], 'r-.', label='Kinematic Only Prediction')
    plt.plot(estimated_poses[:, 0], estimated_poses[:, 1], 'b-', label='EKF Fused Path')
    
    # Plot LiDAR measurements
    lidar_x_meas = [m[0] for m in lidar_meas_data if m is not None]
    lidar_y_meas = [m[1] for m in lidar_meas_data if m is not None]
    plt.scatter(lidar_x_meas, lidar_y_meas, color='purple', marker='o', s=50, label='LiDAR Measurements (used in update)')

    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.title('Vehicle Pose Estimation Fusion (Simplified 2D EKF)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # Equal scaling for x and y axes
    plt.show()