import numpy as np

def calculate_confidence_scores(logits, bbox_pred=None, heatmap=None, iou_pred=None):
    """
    Demonstration of 5 confidence calculation methods.
    """
    
    # 1. Classification-Based (Softmax)
    # Applied to raw logits to get a probability distribution
    def get_softmax_score(logits):
        exp_logits = np.exp(logits - np.max(logits)) # Stability trick
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return np.max(probs) # Score of the winning class

    # 2. Objectness/Centerness (FCOS style)
    # Penalizes boxes that are off-center relative to the pixel/feature
    # left, right, top, bottom
    def get_centerness_aware_score(cls_score, l, t, r, b):
        # l,t,r,b are distances to the four sides of the bounding box
        term1 = min(l, r) / max(l, r)
        term2 = min(t, b) / max(t, b)
        centerness = np.sqrt(term1 * term2)
        return cls_score * centerness

    # 3. IoU-Aware Score (YOLOv8/PVRCNN style)
    # Multiplies classification certainty by the model's own predicted IoU
    def get_iou_aware_score(cls_score, predicted_iou):
        # predicted_iou is a scalar from a separate regression head
        return cls_score * predicted_iou

    # 4. Probabilistic/Uncertainty (Gaussian)
    # Lower variance (sigma) results in a higher final confidence
    def get_probabilistic_score(cls_score, sigma):
        # sigma represents spatial uncertainty (variance)
        # We use an exponential decay to turn high variance into a penalty
        uncertainty_penalty = np.exp(-sigma) 
        return cls_score * uncertainty_penalty

    # 5. Transformer Query-Based (DETR style)
    # Checks if the query is 'Background' or a valid 'Object'
    def get_query_confidence(query_logits, bg_class_idx=0):
        exp_logits = np.exp(query_logits)
        probs = exp_logits / np.sum(exp_logits)
        # Confidence is the sum of all non-background probabilities
        return 1.0 - probs[bg_class_idx]

    # --- Example Usage ---
    example_logits = np.array([0.1, 0.2, 4.5, 0.1]) # High score for index 2
    
    print(f"1. Softmax Score:      {get_softmax_score(example_logits):.4f}")
    print(f"2. Centerness Score:   {get_centerness_aware_score(0.9, 10, 12, 11, 13):.4f}")
    print(f"3. IoU-Aware Score:    {get_iou_aware_score(0.9, 0.85):.4f}")
    print(f"4. Probabilistic:      {get_probabilistic_score(0.9, 0.2):.4f}")
    print(f"5. Query Confidence:   {get_query_confidence(example_logits):.4f}")

calculate_confidence_scores(None)