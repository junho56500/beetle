import numpy as np
import matplotlib.pyplot as plt # <--- Add this line
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise


# --- State Vector (dim_x = 12) ---
# [0] px: position x
# [1] py: position y
# [2] pz: position z
# [3] roll: orientation roll (rotation around x-axis)
# [4] pitch: orientation pitch (rotation around y-axis)
# [5] yaw: orientation yaw (rotation around z-axis)
# [6] vx: velocity in x-direction
# [7] vy: velocity in y-direction
# [8] vz: velocity in z-direction
# [9] omega_roll: angular velocity around x-axis (roll rate)
# [10] omega_pitch: angular velocity around y-axis (pitch rate)
# [11] omega_yaw: angular velocity around z-axis (yaw rate)

# --- 1. State Transition Function (f_x) ---
# x_k = f(x_{k-1}, dt)
def f_x_6dof_constant_velocity(x: np.ndarray, dt: float) -> np.ndarray:
    """
    State transition function for a 6-DOF constant velocity and angular velocity model.
    Assumes velocities and angular velocities are applied in the global frame.

    Args:
        x (np.ndarray): Current state vector (12 dimensions).
        dt (float): Time step.

    Returns:
        np.ndarray: Predicted state vector for the next time step.
    """
    x_k = x.copy() # Start with current state

    # Position update (simple integration of global velocities)
    x_k[0] += x_k[6] * dt  # px += vx * dt
    x_k[1] += x_k[7] * dt  # py += vy * dt
    x_k[2] += x_k[8] * dt  # pz += vz * dt

    # Orientation update (simple integration of angular velocities)
    x_k[3] += x_k[9] * dt   # roll += omega_roll * dt
    x_k[4] += x_k[10] * dt  # pitch += omega_pitch * dt
    x_k[5] += x_k[11] * dt  # yaw += omega_yaw * dt

    # Normalize angles to [-pi, pi]
    x_k[3] = np.arctan2(np.sin(x_k[3]), np.cos(x_k[3]))
    x_k[4] = np.arctan2(np.sin(x_k[4]), np.cos(x_k[4]))
    x_k[5] = np.arctan2(np.sin(x_k[5]), np.cos(x_k[5]))

    # Velocities and angular velocities remain constant (hence "constant velocity model")
    # x_k[6:12] = x[6:12] -- already handled by x_k = x.copy()

    return x_k

# --- 2. Jacobian of the State Transition Function (F_jacobian) ---
# F_k = df/dx_{k-1}
def F_jacobian_6dof_constant_velocity(x: np.ndarray, dt: float) -> np.ndarray:
    """
    Jacobian of the state transition function f_x_6dof_constant_velocity.
    For a constant velocity/angular velocity model with global rates, this is a sparse matrix.

    Args:
        x (np.ndarray): Current state vector (12 dimensions).
        dt (float): Time step.

    Returns:
        np.ndarray: The 12x12 Jacobian matrix.
    """
    F = np.eye(12) # Start with identity matrix (state depends on itself)

    # Derivatives of position w.r.t. velocity:
    # d(px_k)/d(vx_{k-1}) = dt
    # d(py_k)/d(vy_{k-1}) = dt
    # d(pz_k)/d(vz_{k-1}) = dt
    F[0, 6] = dt  # px depends on vx
    F[1, 7] = dt  # py depends on vy
    F[2, 8] = dt  # pz depends on vz

    # Derivatives of orientation w.r.t. angular velocity:
    # d(roll_k)/d(omega_roll_{k-1}) = dt
    # d(pitch_k)/d(omega_pitch_{k-1}) = dt
    # d(yaw_k)/d(omega_yaw_{k-1}) = dt
    F[3, 9] = dt   # roll depends on omega_roll
    F[4, 10] = dt  # pitch depends on omega_pitch
    F[5, 11] = dt  # yaw depends on omega_yaw

    # All other derivatives are 0 or 1 on the diagonal.
    # E.g., d(vx_k)/d(vx_{k-1}) = 1 (already in np.eye(12))
    # d(px_k)/d(px_{k-1}) = 1 (already in np.eye(12))

    return F

# --- Example Usage (Integration with filterpy) ---
if __name__ == "__main__":
    from filterpy.kalman import ExtendedKalmanFilter
    from filterpy.common import Q_discrete_white_noise

    dt_val = 0.1 # Time step
    num_simulation_steps = 500

    # 1. Initialize EKF for 6-DOF Pose + 6-DOF Velocity (12 states total)
    dim_x = 12
    dim_z = 6 # Assuming LiDAR measures full 6D pose [x,y,z,roll,pitch,yaw]

    # Initial state: [px, py, pz, roll, pitch, yaw, vx, vy, vz, omega_roll, omega_pitch, omega_yaw]
    initial_x_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Pose
                                1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Velocities (moving along X)
    
    # Initial covariance: High uncertainty initially
    initial_P_cov = np.diag([
        0.1**2, 0.1**2, 0.1**2,  # px,py,pz uncertainty (m^2)
        np.radians(5)**2, np.radians(5)**2, np.radians(5)**2, # roll,pitch,yaw uncertainty (rad^2)
        0.5**2, 0.5**2, 0.5**2,  # vx,vy,vz uncertainty (m/s)^2
        np.radians(10)**2, np.radians(10)**2, np.radians(10)**2 # omega_r,p,y uncertainty (rad/s)^2
    ])

    ekf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
    ekf.x = initial_x_state.copy()
    ekf.P = initial_P_cov.copy()

    # Assign motion model functions to EKF.
    # Use lambda to pass 'dt_val' as a fixed argument.
    ekf.f = lambda x_current, u_dummy=None: f_x_6dof_constant_velocity(x_current, dt_val)
    ekf.F_jac = lambda x_current, u_dummy=None: F_jacobian_6dof_constant_velocity(x_current, dt_val)

    # --- Measurement Model (for full 6D pose measurement) ---
    # h(x) takes full state [px,py,pz,roll,pitch,yaw,vx,vy,vz,omega_r,p,y]
    # and returns [px,py,pz,roll,pitch,yaw]
    def h_x_6dof_pose_measurement(x: np.ndarray) -> np.ndarray:
        return x[:6] # Returns the first 6 elements (pose)

    # H(x) is the Jacobian of h_x_6dof_pose_measurement w.r.t. state x
    # This will be dim_z x dim_x = 6 x 12
    def H_jacobian_6dof_pose_measurement(x: np.ndarray) -> np.ndarray:
        H_mat = np.zeros((6, 12))
        H_mat[0:6, 0:6] = np.eye(6) # Measures x,y,z,roll,pitch,yaw from state
        return H_mat

    ekf.h = h_x_6dof_pose_measurement
    ekf.H_jac = H_jacobian_6dof_pose_measurement

    # --- Noise Covariances ---
    # Q: Process Noise Covariance for 6-DOF (tuned based on how much velocities/angles drift)
    ekf.Q = np.diag([
        0.01**2, 0.01**2, 0.01**2, # px,py,pz noise (m/s)^2 (effectively noise in velocity integration)
        np.radians(0.1)**2, np.radians(0.1)**2, np.radians(0.1)**2, # roll,pitch,yaw noise (rad/s)^2 (effectively noise in angular rate integration)
        0.001**2, 0.001**2, 0.001**2, # vx,vy,vz noise (m/s/s)^2 (acceleration noise)
        np.radians(0.01)**2, np.radians(0.01)**2, np.radians(0.01)**2 # omega_r,p,y noise (rad/s/s)^2 (angular acceleration noise)
    ])

    # R: Measurement Noise Covariance for 6-DOF (from LiDAR/ICP)
    # Assume 0.1m position std dev and 1 deg angular std dev
    ekf.R = np.diag([
        0.1**2, 0.1**2, 0.1**2, # px,py,pz measurement noise (m^2)
        np.radians(1.0)**2, np.radians(1.0)**2, np.radians(1.0)**2 # roll,pitch,yaw measurement noise (rad^2)
    ])


    # --- Simulate Data (for 6-DOF) ---
    def simulate_6dof_data(num_steps_sim, dt_sim):
        true_states_sim = []
        measurements_sim = []

        x_true_sim = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # Initial pose
                               1.0, 0.0, 0.0, 0.0, 0.0, np.radians(0.5)]) # Initial velocities (move along X, rotate around Z)

        proc_noise_pos_std = 0.005 # per dt
        proc_noise_ang_std = np.radians(0.05) # per dt
        meas_noise_pos_std = 0.5
        meas_noise_ang_std = np.radians(2.0)

        for _ in range(num_steps_sim):
            # Apply true motion (with some random walk)
            x_true_sim[0] += x_true_sim[6] * dt_sim + np.random.normal(0, proc_noise_pos_std)
            x_true_sim[1] += x_true_sim[7] * dt_sim + np.random.normal(0, proc_noise_pos_std)
            x_true_sim[2] += x_true_sim[8] * dt_sim + np.random.normal(0, proc_noise_pos_std)
            x_true_sim[3] += x_true_sim[9] * dt_sim + np.random.normal(0, proc_noise_ang_std)
            x_true_sim[4] += x_true_sim[10] * dt_sim + np.random.normal(0, proc_noise_ang_std)
            x_true_sim[5] += x_true_sim[11] * dt_sim + np.random.normal(0, proc_noise_ang_std)
            
            # Normalize angles in true state (crucial for Euler)
            x_true_sim[3] = np.arctan2(np.sin(x_true_sim[3]), np.cos(x_true_sim[3]))
            x_true_sim[4] = np.arctan2(np.sin(x_true_sim[4]), np.cos(x_true_sim[4]))
            x_true_sim[5] = np.arctan2(np.sin(x_true_sim[5]), np.cos(x_true_sim[5]))

            true_states_sim.append(x_true_sim.copy())

            # Simulate measurement every 10 steps
            if (_ + 1) % 10 == 0:
                meas_pos_noise = np.random.normal(0, meas_noise_pos_std, 3)
                meas_ang_noise = np.random.normal(0, meas_noise_ang_std, 3)
                
                measurements_sim.append(np.concatenate((x_true_sim[:3] + meas_pos_noise,
                                                        x_true_sim[3:6] + meas_ang_noise)))
            else:
                measurements_sim.append(None)
        return np.array(true_states_sim), measurements_sim

    # Run Simulation
    true_states_full, measurements_full = simulate_6dof_data(num_simulation_steps, dt_val)

    # Store results
    ekf_states_full = []
    
    print("\nStarting 6-DOF EKF simulation...")
    for i in range(num_simulation_steps):
        # Predict step (no 'u' control input in this constant velocity model)
        ekf.predict()

        # Update step if measurement available
        if measurements_full[i] is not None:
            # print(f"Step {i+1}: Measurement available. Updating...")
            ekf.update(z=measurements_full[i], R=ekf.R, HJacobian=ekf.H_jac ,Hx=ekf.h) 

        ekf_states_full.append(ekf.x.copy())

    ekf_states_full = np.array(ekf_states_full)

    # --- Plotting Results (2D Projection of 3D path) ---
    plt.figure(figsize=(12, 8))
    plt.plot(true_states_full[:, 0], true_states_full[:, 1], 'g-', linewidth=2, label='True Path (XY)')
    plt.plot(ekf_states_full[:, 0], ekf_states_full[:, 1], 'b-', linewidth=2, alpha=0.8, label='EKF Estimated Path (XY)')
    
    # Plot measurements
    meas_x = [m[0] for m in measurements_full if m is not None]
    meas_y = [m[1] for m in measurements_full if m is not None]
    plt.scatter(meas_x, meas_y, color='purple', marker='o', s=50, alpha=0.6, label='Measurements (XY)')

    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.title('6-DOF Vehicle Pose Estimation (Simplified EKF - XY Plane Projection)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    print(f"\nFinal EKF Estimated Pose (m, rad):\n{ekf.x.round(3)}")
    print(f"Final EKF Covariance:\n{ekf.P.round(5)}")
    print(f"Final EKF Pose (degrees):\n{np.degrees(ekf.x[3:6]).round(3)}")