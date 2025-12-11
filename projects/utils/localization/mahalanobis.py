import numpy as np

def euclidean_distance(x, y):
    """
    Calculates the Euclidean distance between two points.
    
    Args:
        x (np.ndarray): The first point (as a NumPy array).
        y (np.ndarray): The second point (as a NumPy array).
        
    Returns:
        float: The Euclidean distance.
    """
    return np.linalg.norm(x - y)

def mahalanobis_distance(x, y, cov_matrix):
    """
    Calculates the Mahalanobis distance between two points.
    
    Args:
        x (np.ndarray): The first point.
        y (np.ndarray): The second point.
        cov_matrix (np.ndarray): The covariance matrix of the data.
        
    Returns:
        float: The Mahalanobis distance.
    """
    diff = x - y
    # Calculate the inverse of the covariance matrix.
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    # The formula is the square root of (diff^T * inv(cov) * diff).
    return np.sqrt(diff.T @ inv_cov_matrix @ diff)

def main():
    # --- Step 1: Create sample data with covariance ---
    # We will create data that is correlated and has different variances.
    # Imagine a scatter plot that forms an ellipse tilted at an angle.
    mean = np.array([0, 0])
    
    # Define a covariance matrix that creates a tilted ellipse shape.
    # The diagonal elements are the variances.
    # The off-diagonal elements are the covariances (correlation).
    cov_matrix = np.array([[2.0, 1.5],
                           [1.5, 2.0]])
    
    # Generate some random data points to visualize the distribution if needed.
    # data = np.random.multivariate_normal(mean, cov_matrix, 1000)
    
    print("Data's Covariance Matrix:")
    print(cov_matrix)
    print("---------------------------------")
    
    # --- Step 2: Define test points for comparison ---
    # These points will be measured from the origin (mean).
    
    # Point 1: Lies along the major axis of the data's ellipse.
    # The Mahalanobis distance should be smaller for this point.
    point1 = np.array([2.5, 2.5])
    
    # Point 2: Has the same Euclidean distance as Point 1, but lies off the axis.
    # The Mahalanobis distance should be larger for this point.
    point2 = np.array([3.5, 0.0])

    print("Test Points:")
    print(f"Point 1: {point1}")
    print(f"Point 2: {point2}")
    print("---------------------------------")

    # --- Step 3: Calculate and compare distances ---
    # The origin (0, 0) is the center of our distribution.
    origin = np.array([0, 0])

    # Euclidean Distance Calculation
    dist_euclidean1 = euclidean_distance(point1, origin)
    dist_euclidean2 = euclidean_distance(point2, origin)

    # Mahalanobis Distance Calculation
    dist_mahalanobis1 = mahalanobis_distance(point1, origin, cov_matrix)
    dist_mahalanobis2 = mahalanobis_distance(point2, origin, cov_matrix)

    print("Euclidean Distances:")
    print(f"Distance from origin to Point 1: {dist_euclidean1:.4f}")
    print(f"Distance from origin to Point 2: {dist_euclidean2:.4f}")
    print("\nObservation:")
    print("Euclidean distance is larger for Point 2, as it's geometrically farther.")
    print("---------------------------------")

    print("Mahalanobis Distances:")
    print(f"Distance from origin to Point 1: {dist_mahalanobis1:.4f}")
    print(f"Distance from origin to Point 2: {dist_mahalanobis2:.4f}")
    print("\nObservation:")
    print("Mahalanobis distance is larger for Point 2 because it's farther from the data's distribution.")
    print("Point 1 is closer in terms of the data's variance, so it has a smaller Mahalanobis distance.")
    
if __name__ == "__main__":
    main()
