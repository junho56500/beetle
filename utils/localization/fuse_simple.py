import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from typing import Tuple

def fuse_icp_measurement(
    x_predicted: np.ndarray,      # Predicted state vector (e.g., [x, y, z, roll, pitch, yaw])
    P_predicted: np.ndarray,      # Predicted covariance matrix
    z_icp: np.ndarray,            # ICP measurement vector (e.g., [x, y, z, roll, pitch, yaw]_measured)
    R_icp: np.ndarray,            # Covariance matrix of the ICP measurement
    measurement_function: callable, # h(x) function: maps state to expected measurement
    measurement_jacobian: callable  # H(x) function: Jacobian of h(x) w.r.t. x
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuses a predicted state and its covariance with an ICP measurement and its covariance
    using the update step of an Extended Kalman Filter.

    Args:
        x_predicted (np.ndarray): The state vector predicted by the motion model.
                                  e.g., [x, y, z, roll, pitch, yaw]
        P_predicted (np.ndarray): The covariance matrix of the predicted state.
        z_icp (np.ndarray): The measurement vector obtained from ICP.
                            e.g., [x, y, z, roll, pitch, yaw]_measured
        R_icp (np.ndarray): The covariance matrix of the ICP measurement.
        measurement_function (callable): A function h(x) that maps the state vector
                                         to the expected measurement vector.
        measurement_jacobian (callable): A function H(x) that returns the Jacobian
                                         of h(x) with respect to x.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - x_fused (np.ndarray): The fused (updated) state vector.
            - P_fused (np.ndarray): The fused (updated) covariance matrix.
    """
    
    # Create a temporary EKF instance to perform the update step.
    # We only need its internal update logic.
    # The dim_x and dim_z must match your state and measurement vector sizes.
    # For a 6D pose state and 6D pose measurement:
    dim_x = x_predicted.shape[0]
    dim_z = z_icp.shape[0]

    temp_ekf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)

    # Load the predicted state and covariance into the temporary EKF
    temp_ekf.x = x_predicted.copy()
    temp_ekf.P = P_predicted.copy()

    # Assign the measurement function and its Jacobian to the temporary EKF
    # These are needed by the update method internally.
    temp_ekf.h = measurement_function
    temp_ekf.H_jac = measurement_jacobian

    # Perform the update step
    # The 'z' is the actual measurement.
    # The 'R' is the covariance of the measurement.
    # 'hx' and 'Hx' are automatically used from temp_ekf.h and temp_ekf.H_jac
    temp_ekf.update(z=z_icp, R=R_icp, HJacobian=measurement_jacobian, Hx=measurement_function)

    # Return the updated state and covariance
    x_fused = temp_ekf.x
    P_fused = temp_ekf.P

    return x_fused, P_fused

# --- Example Usage ---
if __name__ == "__main__":
    # --- Define 3D Pose State and Measurement Models (Simplified) ---
    # State: [x, y, z, roll, pitch, yaw] (6 dimensions)
    # Measurement: [x, y, z, roll, pitch, yaw] (6 dimensions)

    def h_x_3d_pose(x: np.ndarray) -> np.ndarray:
        """Measures full 6D pose directly from state."""
        return x.copy() # Sensor directly measures the state

    def H_jacobian_3d_pose(x: np.ndarray) -> np.ndarray:
        """Jacobian of h_x_3d_pose is identity matrix."""
        return np.eye(6) # 6x6 Identity matrix

    # --- Simulate Predicted State and Covariance (from Prediction Step) ---
    # This would come from your EKF's predict() step or a motion model.
    predicted_x = np.array([1.0, 1.0, 0.5, np.radians(10), np.radians(5), np.radians(15)]) # Example predicted pose
    predicted_P = np.diag([0.2**2, 0.2**2, 0.1**2, np.radians(2)**2, np.radians(1)**2, np.radians(3)**2]) # Example predicted covariance

    print("--- Before Fusion ---")
    print(f"Predicted State:\n{predicted_x.round(3)}")
    print(f"Predicted Covariance:\n{predicted_P.round(4)}")

    # --- Simulate ICP Measurement and its Covariance ---
    # This would come directly from your ICP algorithm (e.g., Open3D's registration_result.covariance)
    # Let's assume ICP measured a slightly different pose with its own uncertainty
    icp_measurement_z = np.array([1.1, 1.05, 0.52, np.radians(11), np.radians(4), np.radians(16)]) # Noisy measurement
    icp_measurement_R = np.diag([0.05**2, 0.05**2, 0.03**2, np.radians(0.5)**2, np.radians(0.3)**2, np.radians(0.8)**2]) # ICP's covariance (lower than predicted P)

    print("\n--- ICP Measurement ---")
    print(f"Measurement (z): {icp_measurement_z.round(3)}")
    print(f"Measurement Covariance (R_icp):\n{icp_measurement_R.round(4)}")

    # --- Call the Fusion Function ---
    fused_x, fused_P = fuse_icp_measurement(
        x_predicted=predicted_x,
        P_predicted=predicted_P,
        z_icp=icp_measurement_z,
        R_icp=icp_measurement_R,
        measurement_function=h_x_3d_pose,
        measurement_jacobian=H_jacobian_3d_pose
    )

    print("\n--- After Fusion ---")
    print(f"Fused State:\n{fused_x.round(3)}")
    print(f"Fused Covariance:\n{fused_P.round(4)}")

    # Observe:
    # - fused_x should be a value between predicted_x and icp_measurement_z,
    #   closer to the one with lower uncertainty (lower variance).
    # - fused_P should be smaller (lower uncertainty) than both predicted_P and icp_measurement_R.