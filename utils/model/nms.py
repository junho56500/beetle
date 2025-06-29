
import numpy as np

def calculate_midpoint(p1, p2):
    """Calculates the midpoint of a line segment."""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def calculate_slope(p1, p2):
    """Calculates the slope of a line segment."""
    if p2[0] - p1[0] == 0:  # Vertical line
        return float('inf') if p2[1] > p1[1] else float('-inf')
    return (p2[1] - p1[1]) / (p2[0] - p1[0])

def euclidean_distance(point1, point2):
    """Calculates the Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

class SegmentNMS:
    def __init__(self, midpoint_threshold=20, slope_threshold=0.1, iou_threshold=0.5):
        """
        Initializes the SegmentNMS with specified thresholds.

        Args:
            midpoint_threshold (float): Maximum Euclidean distance between midpoints
                                        for two segments to be considered overlapping.
            slope_threshold (float): Maximum absolute difference in slopes for two
                                     segments to be considered overlapping.
            iou_threshold (float): This parameter is currently a placeholder for a more
                                    complex overlap metric if needed. For simple line
                                    NMS, midpoint and slope thresholds are primary.
        """
        self.midpoint_threshold = midpoint_threshold
        self.slope_threshold = slope_threshold
        self.iou_threshold = iou_threshold

    def apply_nms(self, segment_predictions):
        """
        Applies Non-Maximum Suppression (NMS) to a list of line segment predictions
        using the thresholds configured during the class initialization.

        Args:
            segment_predictions (list): A list of dictionaries, where each dictionary
                                     represents a segment prediction and has the format:
                                     {
                                         'p1': (x1, y1),  # First endpoint of the line segment
                                         'p2': (x2, y2),  # Second endpoint of the line segment
                                         'score': float   # Confidence score of the segment
                                     }

        Returns:
            list: A list of the selected (non-suppressed and sufficiently long) segment predictions.
        """
        if not segment_predictions:
            return []

        # Sort predictions by score in descending order
        # We create a copy to avoid modifying the original list
        predictions = sorted(segment_predictions, key=lambda x: x['score'], reverse=True)
        selected_segments = []
        
        # Track which segments have been suppressed
        suppressed = [False] * len(predictions)

        for i in range(len(predictions)):
            if suppressed[i]:
                continue

            # This segment is selected (not suppressed)
            selected_segments.append(predictions[i])
            
            current_segment = predictions[i]
            current_midpoint = calculate_midpoint(current_segment['p1'], current_segment['p2'])
            current_slope = calculate_slope(current_segment['p1'], current_segment['p2'])

            for j in range(i + 1, len(predictions)):
                if suppressed[j]:
                    continue

                compare_segment = predictions[j]
                compare_midpoint = calculate_midpoint(compare_segment['p1'], compare_segment['p2'])
                compare_slope = calculate_slope(compare_segment['p1'], compare_segment['p2'])

                # Check for overlap based on midpoint proximity and slope similarity
                midpoint_dist = euclidean_distance(current_midpoint, compare_midpoint)
                slope_diff = abs(current_slope - compare_slope)

                if midpoint_dist < self.midpoint_threshold and slope_diff < self.slope_threshold:
                    # If they overlap, suppress the lower-scoring segment
                    suppressed[j] = True
                    
        return selected_segments



def main():
    # Define some example segment predictions
    # Each segment has two points (p1, p2) and a confidence score
    example_segments = [
        {'p1': (10, 100), 'p2': (200, 10), 'score': 0.95},  # Segment 1 (High score)
        {'p1': (15, 95), 'p2': (205, 12), 'score': 0.88},   # Segment 2 (Slightly shifted from 1)
        {'p1': (30, 80), 'p2': (220, 0), 'score': 0.70},    # Segment 3 (Similar to 1/2 but lower score)
        {'p1': (50, 150), 'p2': (250, 50), 'score': 0.92},  # Segment 4 (Distinct)
        {'p1': (55, 145), 'p2': (255, 52), 'score': 0.85},  # Segment 5 (Slightly shifted from 4)
        {'p1': (10, 200), 'p2': (150, 180), 'score': 0.60}, # Segment 6 (Distinct, low score)
    ]

    print("Original Segment Predictions:")
    for i, segment in enumerate(example_segments):
        print(f"  Segment {i+1}: P1={segment['p1']}, P2={segment['p2']}, Score={segment['score']:.2f}")

    print("\n--- Applying NMS ---")
    # You can adjust these thresholds based on your specific segment detection output
    nms = SegmentNMS(midpoint_threshold=30, slope_threshold=0.2)
    selected_segments = nms.apply_nms(
        example_segments
    )

    print("\nSelected Segment Predictions after NMS:")
    if not selected_segments:
        print("No segments selected.")
    for i, segment in enumerate(selected_segments):
        print(f"  Segment {i+1}: P1={segment['p1']}, P2={segment['p2']}, Score={segment['score']:.2f}")

    print("\n--- Testing with more distinct segments ---")
    distinct_segments = [
        {'p1': (10, 100), 'p2': (200, 10), 'score': 0.95},
        {'p1': (300, 100), 'p2': (500, 10), 'score': 0.90},
        {'p1': (10, 250), 'p2': (200, 160), 'score': 0.88},
    ]
    selected_distinct_segments = nms.apply_nms(distinct_segments)
    print("Selected Distinct Segments after NMS:")
    for i, segment in enumerate(selected_distinct_segments):
        print(f"  Segment {i+1}: P1={segment['p1']}, P2={segment['p2']}, Score={segment['score']:.2f}")

    print("\n--- Testing with all similar segments ---")
    similar_segments = [
        {'p1': (10, 100), 'p2': (200, 10), 'score': 0.95},
        {'p1': (15, 98), 'p2': (203, 12), 'score': 0.90},
        {'p1': (12, 102), 'p2': (198, 9), 'score': 0.85},
    ]
    selected_similar_segments = nms.apply_nms(similar_segments)
    print("Selected Similar Segments after NMS:")
    for i, segment in enumerate(selected_similar_segments):
        print(f"  Segment {i+1}: P1={segment['p1']}, P2={segment['p2']}, Score={segment['score']:.2f}")


if __name__ == '__main__':
    main()