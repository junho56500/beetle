
import numpy as np

class IOU:
    def calculate(self, box1, box2):
        """
        Calculates the Intersection over Union (IoU) of two bounding boxes.

        Bounding boxes are expected in the format [x_min, y_min, x_max, y_max].

        Args:
            box1 (list or np.array): Coordinates of the first bounding box.
            box2 (list or np.array): Coordinates of the second bounding box.

        Returns:
            float: The IoU value, a float between 0 and 1.
        """

        # Ensure inputs are NumPy arrays for easier element-wise operations
        box1 = np.array(box1)
        box2 = np.array(box2)

        # 1. Determine the coordinates of the intersection rectangle
        # x_left: maximum of the x-coordinates of the left edges
        x_left = max(box1[0], box2[0])
        # y_top: maximum of the y-coordinates of the top edges
        y_top = max(box1[1], box2[1])
        # x_right: minimum of the x-coordinates of the right edges
        x_right = min(box1[2], box2[2])
        # y_bottom: minimum of the y-coordinates of the bottom edges
        y_bottom = min(box1[3], box2[3])

        # 2. Calculate the area of the intersection rectangle
        # If the boxes do not overlap, the intersection area is 0.
        # This happens if x_right <= x_left or y_bottom <= y_top.
        intersection_width = max(0, x_right - x_left)
        intersection_height = max(0, y_bottom - y_top)
        intersection_area = intersection_width * intersection_height

        # 3. Calculate the area of each bounding box
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # 4. Calculate the area of the union
        # Union Area = Area of Box1 + Area of Box2 - Area of Intersection
        union_area = box1_area + box2_area - intersection_area

        # 5. Compute IoU
        # Handle the case where union_area is zero to avoid division by zero.
        # This can happen if both boxes have zero area and no intersection.
        if union_area == 0:
            return 0.0
        else:
            iou = intersection_area / union_area
            return iou


def main():
    iou = IOU()
    
    # Example 1: Overlapping boxes
    box_a = [0, 0, 10, 10]  # x_min, y_min, x_max, y_max
    box_b = [5, 5, 15, 15]
    iou1 = iou.calculate(box_a, box_b)
    print(f"IoU between {box_a} and {box_b}: {iou1:.4f}") # Expected: ~0.1429 (25 / 175)

    # Example 2: Fully contained box
    box_c = [0, 0, 10, 10]
    box_d = [2, 2, 8, 8]
    iou2 = iou.calculate(box_c, box_d)
    print(f"IoU between {box_c} and {box_d}: {iou2:.4f}") # Expected: 0.3600 (36 / 100)

    # Example 3: Non-overlapping boxes
    box_e = [0, 0, 10, 10]
    box_f = [11, 11, 20, 20]
    iou3 = iou.calculate(box_e, box_f)
    print(f"IoU between {box_e} and {box_f}: {iou3:.4f}") # Expected: 0.0000

    # Example 4: Identical boxes
    box_g = [0, 0, 10, 10]
    box_h = [0, 0, 10, 10]
    iou4 = iou.calculate(box_g, box_h)
    print(f"IoU between {box_g} and {box_h}: {iou4:.4f}") # Expected: 1.0000

    # Example 5: Boxes with zero area (e.g., lines or points)
    box_i = [0, 0, 0, 0]
    box_j = [0, 0, 0, 0]
    iou5 = iou.calculate(box_i, box_j)
    print(f"IoU between {box_i} and {box_j}: {iou5:.4f}") # Expected: 0.0000

    box_k = [0, 0, 5, 5]
    box_l = [5, 5, 10, 10] # Touch at a single point
    iou6 = iou.calculate(box_k, box_l)
    print(f"IoU between {box_k} and {box_l}: {iou6:.4f}") # Expected: 0.0000

    box_m = [0, 0, 5, 5]
    box_n = [5, 0, 10, 5] # Touch along an edge
    iou7 = iou.calculate(box_m, box_n)
    print(f"IoU between {box_m} and {box_n}: {iou7:.4f}") # Expected: 0.0000 (intersection area is 0 because width is 0)    
    

if __name__ == '__main__':
    main()