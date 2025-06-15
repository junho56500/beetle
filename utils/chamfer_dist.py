import numpy as np
from scipy.spatial import cKDTree

class chamfer_distance:
    def op(self, pc1, pc2):
        """
        Compute symmetric Chamfer Distance between two point clouds.
        Args:
            pc1: (N, D) NumPy array (point cloud 1)
            pc2: (M, D) NumPy array (point cloud 2)
        Returns:
            Chamfer Distance (float)
        """
        tree1 = cKDTree(pc1)
        tree2 = cKDTree(pc2)

        # One direction: pc1 -> pc2
        dist1, _ = tree2.query(pc1)
        # Opposite direction: pc2 -> pc1
        dist2, _ = tree1.query(pc2)

        return np.mean(dist1**2) + np.mean(dist2**2)
    
    def op_linalg(self, pc1, pc2):
        """
        Chamfer Distance using numpy and linalg.norm.
        Args:
            pc1: (N, D) numpy array
            pc2: (M, D) numpy array
        Returns:
            float: Chamfer Distance
        """
        # Compute full pairwise distance matrix
        dist_matrix = np.linalg.norm(pc1[:, np.newaxis, :] - pc2[np.newaxis, :, :], axis=2)  # shape (N, M)
        pc1 = np.array([[1,1],[2,2],[3,3]])
        pc2= np.array([[1,1.5],[2,2.5],[3,3.5]])
        print(pc1[:, np.newaxis] - pc2[np.newaxis, :])
        
        # For each point in pc1, find the nearest point in pc2
        min_dist1 = np.min(dist_matrix, axis=1)

        # For each point in pc2, find the nearest point in pc1
        min_dist2 = np.min(dist_matrix, axis=0)

        # Symmetric Chamfer Distance (squared)
        return np.mean(min_dist1**2) + np.mean(min_dist2**2)


def main():
    # Example usage
    pc1 = np.random.rand(100, 3)  # 100 points in 3D
    pc2 = pc1 + np.random.normal(scale=0.01, size=pc1.shape)  # slightly perturbed

    cd = chamfer_distance()
    ret = cd.op(pc1, pc2)
    ret2 = cd.op_linalg(pc1, pc2)
    print(f"Chamfer Distance: {ret:.6f}")
    print(f"Chamfer Distance: {ret2:.6f}")
    

if __name__ == '__main__':
    main()