import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise # Helper for process noise

# --- 1. Define Motion Model and its Jacobian ---
# State: [x, y, vx, vy]

def f_x(x, dt):
    """
    Non-linear state transition function (constant velocity model).
    x_k = x_{k-1} + vx_{k-1} * dt
    y_k = y_{k-1} + vy_{k-1} * dt
    vx_k = vx_{k-1}
    vy_k = vy_{k-1}
    """
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    return F @ x

def F_jacobian(x, dt):
    """
    Jacobian of the state transition function f_x with respect to the state x.
    For a constant velocity model, this is simply the state transition matrix itself.
    """
    return np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

# --- 2. Define Measurement Model and its Jacobian ---
# Measurement: [x_m, y_m] (position only)

def h_x(x):
    """
    Non-linear measurement function.
    Measures position (x, y) from the state [x, y, vx, vy].
    """
    return np.array([x[0], x[1]])

def H_jacobian(x):
    """
    Jacobian of the measurement function h_x with respect to the state x.
    Since h_x is linear (just picks x and y), its Jacobian is constant.
    """
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])

# --- 3. Simulation Setup ---
def simulate_data(num_steps, dt):
    """Generates true path, noisy measurements, and noisy kinematic inputs."""
    true_states = []
    measurements = []
    kinematic_inputs = [] # [linear_vel, angular_vel] or [ax, ay]

    # Initial true state
    x_true = np.array([0.0, 0.0, 1.0, 0.5]) # [x, y, vx, vy]
    true_states.append(x_true.copy())

    # Noise parameters
    process_noise_std = 0.02 # Std dev for process noise
    measurement_noise_std = 0.5 # Std dev for measurement noise

    for i in range(num_steps):
        # Apply true motion (with some random process noise)
        x_true[0] += x_true[2] * dt + np.random.normal(0, process_noise_std)
        x_true[1] += x_true[3] * dt + np.random.normal(0, process_noise_std)
        # vx, vy are constant in this simple model, but could have noise too
        
        true_states.append(x_true.copy())

        # Simulate kinematic input (e.g., from wheel odometry)
        # For constant velocity model, kinematic input is just vx, vy
        # In a real system, this would be derived from wheel encoders/IMU
        kinematic_inputs.append(np.array([x_true[2], x_true[3]])) # Just pass current vx, vy as input

        # Simulate LiDAR measurement (noisy position)
        if i % 10 == 0: # Measure every 10 steps
            measurement_noise = np.random.normal(0, measurement_noise_std, 2)
            measurements.append(np.array([x_true[0], x_true[1]]) + measurement_noise)
        else:
            measurements.append(None) # No measurement for this step

    return np.array(true_states), measurements, np.array(kinematic_inputs)

# --- Main EKF Loop ---
if __name__ == "__main__":
    # Ensure filterpy is installed: pip install filterpy
    
    dt = 0.1 # Time step
    num_steps = 200

    # 1. Initialize EKF
    # State: [x, y, vx, vy]
    initial_x = np.array([0.0, 0.0, 1.0, 0.5]) # Initial guess for state
    initial_P = np.diag([0.1**2, 0.1**2, 0.5**2, 0.5**2]) # Initial uncertainty (high)

    ekf = ExtendedKalmanFilter(dim_x=4, dim_z=2) # dim_x: state dimension, dim_z: measurement dimension

    ekf.x = initial_x # Initial state estimate
    ekf.P = initial_P # Initial covariance

    # Assign motion and measurement functions
    # Use lambda to pass 'dt' as a fixed argument to f_x and F_jacobian
    # The functions assigned to ekf.f and ekf.F_jac should only take 'x' (state)
    # and optionally 'u' (control input).
    ekf.f = lambda x, u_dummy=None: f_x(x, dt) # f_x needs dt
    ekf.F_jac = lambda x, u_dummy=None: F_jacobian(x, dt) # F_jacobian needs dt

    # Assign measurement function and its Jacobian
    ekf.h = h_x 
    ekf.H_jac = H_jacobian 

    # Assign noise covariances
    # Q: Process noise (uncertainty in motion model)
    # For a constant velocity model, Q_discrete_white_noise is a good helper
    ekf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.25**2, block_size=2) # var is noise in acceleration

    # R: Measurement noise (uncertainty in sensor readings)
    # This would be the covariance from your LiDAR odometry (ICP)
    ekf.R = np.diag([4.0**2, 4.0**2]) # Example: 0.5m std dev for x, y measurements

    # Simulate data
    true_states, measurements, kinematic_inputs = simulate_data(num_steps, dt)

    
    # Store results for plotting
    ekf_states = []
    ekf_covariances = []
    
    for i in range(num_steps):
        # --- Prediction Step ---
        # Call predict *without* dt. dt is now 'baked into' ekf.f and ekf.F_jac via lambda.
        # If you had a control input 'u', you would pass it here: ekf.predict(u=u_input)
        ekf.predict() 

        # --- Update Step (if measurement is available) ---
        if measurements[i] is not None:
            # Pass the measurement (z) and its covariance (R)
            # In a real system, lidar_pose_covariance from ICP would go here instead of ekf.R
            ekf.update(z=measurements[i], R=ekf.R, HJacobian=ekf.H_jac, Hx=ekf.h)
            #ekf.update(z=measurements[i], R=ekf.R)

        ekf_states.append(ekf.y.copy())
        ekf_covariances.append(ekf.P.copy())

    ekf_states = np.array(ekf_states)

    # --- Plotting Results ---
    plt.figure(figsize=(12, 8))
    plt.plot(true_states[:, 0], true_states[:, 1], 'g-', label='True Path')
    plt.plot(ekf_states[:, 0], ekf_states[:, 1], 'b-', label='EKF Estimated Path')
    
    # Plot measurements
    meas_x = [m[0] for m in measurements if m is not None]
    meas_y = [m[1] for m in measurements if m is not None]
    plt.scatter(meas_x, meas_y, color='purple', marker='x', s=50, label='LiDAR Measurements')

    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.title('2D Vehicle Tracking with Simplified EKF')
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # Equal scaling for x and y axes
    plt.show()

    # You can also inspect the final covariance
    print(f"\nFinal EKF Estimated Pose: {ekf.x.round(3)}")
    print(f"Final EKF Covariance:\n{ekf.P.round(5)}")