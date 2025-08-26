
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

class correction_pose:
    def correction(self, timestamps, dx, dy, dyaw):
        
        # Initialize position and yaw
        positions = [(0.0, 0.0)]  # Start at origin
        yaws = [0.0]              # Initial yaw (heading)
    
        # Accumulate position and yaw
        for i in range(1, len(timestamps)):
            # Get previous yaw
            yaw_prev = yaws[-1]

            # Rotate local dx, dy to global coordinates
            dx_global = dx[i] * np.cos(yaw_prev) - dy[i] * np.sin(yaw_prev)
            dy_global = dx[i] * np.sin(yaw_prev) + dy[i] * np.cos(yaw_prev)

            # Update position
            x_prev, y_prev = positions[-1]
            x_new = x_prev + dx_global
            y_new = y_prev + dy_global
            positions.append((x_new, y_new))

            # Update yaw
            yaws.append(yaw_prev + dyaw[i])

        # Convert to numpy arrays for plotting
        positions = np.array(positions)
        yaws = np.array(yaws)
        return positions, yaws


def main():
    # Example input data
    timestamps = np.array([0, 1, 2, 3, 4])  # time in seconds
    dx = np.array([1, 1, 1, 1, 1])          # delta x in local robot frame
    dy = np.array([0, 0.1, 0, -0.1, 0])     # delta y in local robot frame
    dyaw = np.radians([0, 5, -5, 10, 0])    # delta yaw in radians


    cp = correction_pose()
    positions, yaws = cp.correction(timestamps, dx, dy, dyaw)

    # Plot the trajectory
    plt.figure(figsize=(6, 6))
    plt.plot(positions[:, 0], positions[:, 1], marker='o')
    plt.quiver(positions[:, 0], positions[:, 1],
            np.cos(yaws), np.sin(yaws),
            angles='xy', scale_units='xy', scale=1, color='r', label='Yaw direction')
    plt.title("Simulated 2D Trajectory")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.show()

    

if __name__ == '__main__':
    main()