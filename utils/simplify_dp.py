
import numpy as np

class douglasPeuker:
    def __init__(self, epsilon = 0.01):
        self.epsilon = epsilon      #epsilon (float): The tolerance distance. Points closer than this distance
                                    #to the line segment formed by the endpoints will be discarded.


    def point_to_segment_distance(self, point, a, b):
        """
        Calculates the shortest distance between a point and a line segment.

        Args:
            point (numpy.ndarray): The point.
            a (numpy.ndarray): The first endpoint of the segment.
            b (numpy.ndarray): The second endpoint of the segment.

        Returns:
            float: The shortest distance.
        """
        if np.all(a == b):
            return np.linalg.norm(point - a)

        ab = b - a
        ap = point - a
        projection = np.dot(ap, ab) / np.dot(ab, ab)

        if projection <= 0:
            return np.linalg.norm(point - a)
        elif projection >= 1:
            return np.linalg.norm(point - b)
        else:
            closest_point = a + projection * ab
            return np.linalg.norm(point - closest_point)


    def simplify(self, points):
        """
        Reduces the number of points in a curve using the Douglas-Peucker algorithm (iterative).

        Args:
            points (list or numpy.ndarray): A list of 2D or 3D points (e.g., [[x1, y1], [x2, y2], ...]).

        Returns:
            list: A new list of points representing the simplified curve.
        """
        if len(points) <= 2:
            return points

        simplified = [points[0]]
        stack = [(0, len(points) - 1)]
        visited = [False] * len(points)
        visited[0] = True
        visited[-1] = True

        while stack:
            start_index, end_index = stack.pop()
            max_distance = 0
            farthest_index = -1

            start_point = np.array(points[start_index])
            end_point = np.array(points[end_index])

            for i in range(start_index + 1, end_index):
                if not visited[i]:
                    point = np.array(points[i])
                    distance = self.point_to_segment_distance(point, start_point, end_point)
                    if distance > max_distance:
                        max_distance = distance
                        farthest_index = i

            if max_distance > self.epsilon:
                visited[farthest_index] = True
                stack.append((start_index, farthest_index))
                stack.append((farthest_index, end_index))

        # Reconstruct the simplified path in the original order
        simplified_points = [points[i] for i, v in enumerate(visited) if v]
        return simplified_points


def main():
    # Example usage with 2D points
    points_2d = [[0, 0], [1, 1], [2, 0.5], [3, 1], [4, 0], [5, -0.5], [6, 1], [7, 0]]
    epsilon_2d = 0.8
    dp = douglasPeuker(epsilon_2d)
    simplified_points_2d = dp.simplify(points_2d)
    print("Original 2D points:", points_2d)
    print("Simplified 2D points (epsilon =", epsilon_2d, "):", simplified_points_2d)

    print("-" * 20)

    # Example usage with 3D points
    points_3d = [[0, 0, 0], [1, 1, 1], [2, 0.5, 0], [3, 1, 0.5], [4, 0, 1], [5, -0.5, 0], [6, 1, 1.5], [7, 0, 0]]
    epsilon_3d = 1.0
    dp2 = douglasPeuker(epsilon_3d)
    simplified_points_3d = dp2.simplify(points_3d)
    print("Original 3D points:", points_3d)
    print("Simplified 3D points (epsilon =", epsilon_3d, "):", simplified_points_3d)
    

if __name__ == '__main__':
    main()

