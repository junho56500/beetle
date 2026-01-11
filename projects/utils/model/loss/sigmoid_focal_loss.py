import torch
import torch.nn as nn
import torch.nn.functional as F

def sigmoid_focal_loss(inputs: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    """
    Computes the Sigmoid Focal Loss.

    Args:
        inputs (torch.Tensor): Raw logits from the model, shape (N, C) or (N, C, H, W).
                               N = batch size, C = number of classes.
        targets (torch.Tensor): Ground truth labels, shape (N, C) or (N, C, H, W).
                                Should be float tensors with 0s and 1s.
        gamma (float): The focusing parameter. Higher values increase focus on hard examples.
        alpha (float): The weighting factor for classes (alpha_t in the formula).
                       Can be a scalar or a tensor of shape (C,).
                       If alpha is a scalar, it applies uniformly.
                       If alpha is a tensor, it applies per-class.

    Returns:
        torch.Tensor: Computed focal loss.
    """
    # Flatten inputs and targets for pixel-wise/instance-wise loss calculation
    # If inputs are (N, C, H, W), reshape to (N*H*W, C)
    if inputs.dim() > 2:
        inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, inputs.shape[1])
        targets = targets.permute(0, 2, 3, 1).contiguous().view(-1, targets.shape[1])
    
    # Ensure targets are float type for BCE_with_logits
    targets = targets.float()

    # Calculate Binary Cross-Entropy (BCE) loss per element
    # F.binary_cross_entropy_with_logits combines sigmoid and BCE
    # reduction='none' gives individual losses before applying the modulating factor
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

    # Calculate probabilities after sigmoid activation
    prob = torch.sigmoid(inputs)

    # Calculate p_t (probability of the true class)
    # p_t = p if y=1, else 1-p
    prob_t = targets * prob + (1 - targets) * (1 - prob)

    # Calculate the modulating factor (1 - p_t)^gamma
    modulating_factor = (1 - prob_t) ** gamma

    # Apply alpha weighting
    # alpha_factor = alpha if y=1, else 1-alpha
    if isinstance(alpha, (int, float)):
        # If alpha is a scalar, apply it universally
        alpha_factor = torch.ones_like(targets) * alpha
    else: # Assuming alpha is a tensor for class-specific weighting
        # Alpha should be a tensor of shape (C,) for C classes
        # For each element, select the alpha value corresponding to its class.
        # This requires `targets` to be one-hot encoded or similar for index-based selection.
        # A simpler approach for binary/multi-label is to apply alpha based on target value (0 or 1).
        # We assume alpha is defined for positive class and (1-alpha) for negative class, or is a per-class tensor.
        # For simplicity here, if alpha is a tensor, assume it matches input channels.
        if alpha.dim() == 1 and alpha.shape[0] == inputs.shape[1]: # Alpha per class
            alpha_factor = torch.where(targets == 1, alpha, 1 - alpha)
        else:
            raise ValueError("Alpha tensor must be a scalar or match the number of classes.")


    # Combine all parts
    focal_loss = alpha_factor * modulating_factor * bce_loss

    # Return the mean loss over all elements
    return focal_loss.mean()


if __name__ == '__main__':
    # --- Example Usage ---

    # Simulate model outputs (logits) for a batch of 2 examples, 3 classes (multi-label)
    # N=2, C=3
    # Each column represents a class, and values are independent logits.
    logits_multilabel = torch.tensor([
        [2.0, -1.0, 0.5],  # Example 1: High for class 0, Low for class 1, Medium for class 2
        [-0.5, 3.0, -2.0]  # Example 2: Low for class 0, High for class 1, Very Low for class 2
    ], dtype=torch.float32)

    # Simulate ground truth labels (binary 0 or 1 for each class independently)
    # For Example 1: Class 0 (true), Class 1 (false), Class 2 (true)
    # For Example 2: Class 0 (false), Class 1 (true), Class 2 (false)
    labels_multilabel = torch.tensor([
        [1, 0, 1],
        [0, 1, 0]
    ], dtype=torch.float32)

    print("--- Standard Binary Cross Entropy Loss (with Logits) ---")
    bce_loss_standard = F.binary_cross_entropy_with_logits(logits_multilabel, labels_multilabel)
    print(f"Standard BCE Loss: {bce_loss_standard.item():.4f}")

    print("\n--- Sigmoid Focal Loss (Gamma = 0) ---")
    # Gamma = 0 should be equivalent to standard BCE (ignoring alpha for now)
    focal_loss_gamma0 = sigmoid_focal_loss(logits_multilabel, labels_multilabel, gamma=0.0, alpha=0.5)
    print(f"Focal Loss (gamma=0, alpha=0.5): {focal_loss_gamma0.item():.4f}")
    # Note: A slight difference might occur due to mean reduction being applied differently or precision.

    print("\n--- Sigmoid Focal Loss (Gamma = 2.0) ---")
    focal_loss_gamma2 = sigmoid_focal_loss(logits_multilabel, labels_multilabel, gamma=2.0, alpha=0.25)
    print(f"Focal Loss (gamma=2.0, alpha=0.25): {focal_loss_gamma2.item():.4f}")

    # Example with specific alpha per class
    alpha_per_class = torch.tensor([0.7, 0.2, 0.5], dtype=torch.float32) # Alpha for class 0, 1, 2
    print("\n--- Sigmoid Focal Loss (Gamma = 2.0, Per-Class Alpha) ---")
    focal_loss_per_class_alpha = sigmoid_focal_loss(logits_multilabel, labels_multilabel, gamma=2.0, alpha=alpha_per_class)
    print(f"Focal Loss (gamma=2.0, alpha={alpha_per_class.tolist()}): {focal_loss_per_class_alpha.item():.4f}")

    print("\n--- Individual Prediction Analysis (Gamma = 2.0, Alpha = 0.25) ---")
    # Let's manually calculate for one hard example and one easy example
    # Example 1, Class 1: logit=-1.0, label=0 (Easy Negative: model confident it's negative, and it is)
    single_logit_easy_neg = logits_multilabel[0, 1].unsqueeze(0).unsqueeze(0)
    single_label_easy_neg = labels_multilabel[0, 1].unsqueeze(0).unsqueeze(0)

    bce_loss_easy_neg = F.binary_cross_entropy_with_logits(single_logit_easy_neg, single_label_easy_neg, reduction='none')
    prob_easy_neg = torch.sigmoid(single_logit_easy_neg)
    # p_t for negative class (label=0) is (1 - prob)
    prob_t_easy_neg = (1 - prob_easy_neg).squeeze()

    gamma_val = 2.0
    alpha_val = 0.25 # This alpha is for the positive class (target=1)

    # For negative target, we typically use (1-alpha) or a pre-defined alpha for negative class if `alpha` is a tensor
    # If alpha is a scalar (like 0.25 here), the standard focal loss formula uses `alpha_t` based on the true label.
    # So if target is 0, alpha_t would be (1 - alpha_val)
    alpha_factor_easy_neg = (1 - alpha_val) if single_label_easy_neg.item() == 0 else alpha_val

    modulating_factor_easy_neg = (1 - prob_t_easy_neg) ** gamma_val
    focal_loss_easy_neg = alpha_factor_easy_neg * modulating_factor_easy_neg * bce_loss_easy_neg
    print(f"  Easy Negative (Example 1, Class 1):")
    print(f"    Prob (sigmoid): {prob_easy_neg.item():.4f}, True Prob (p_t): {prob_t_easy_neg.item():.4f}")
    print(f"    BCE Loss: {bce_loss_easy_neg.item():.4f}")
    print(f"    Modulating Factor: {modulating_factor_easy_neg.item():.4f}")
    print(f"    Focal Loss: {focal_loss_easy_neg.item():.4f}")

    # Example 2, Class 0: logit=-0.5, label=0 (Slightly Hard Negative: model somewhat confident, but still negative)
    single_logit_hard_neg = logits_multilabel[1, 0].unsqueeze(0).unsqueeze(0)
    single_label_hard_neg = labels_multilabel[1, 0].unsqueeze(0).unsqueeze(0)

    bce_loss_hard_neg = F.binary_cross_entropy_with_logits(single_logit_hard_neg, single_label_hard_neg, reduction='none')
    prob_hard_neg = torch.sigmoid(single_logit_hard_neg)
    prob_t_hard_neg = (1 - prob_hard_neg).squeeze()

    alpha_factor_hard_neg = (1 - alpha_val) if single_label_hard_neg.item() == 0 else alpha_val

    modulating_factor_hard_neg = (1 - prob_t_hard_neg) ** gamma_val
    focal_loss_hard_neg = alpha_factor_hard_neg * modulating_factor_hard_neg * bce_loss_hard_neg
    print(f"\n  Slightly Hard Negative (Example 2, Class 0):")
    print(f"    Prob (sigmoid): {prob_hard_neg.item():.4f}, True Prob (p_t): {prob_t_hard_neg.item():.4f}")
    print(f"    BCE Loss: {bce_loss_hard_neg.item():.4f}")
    print(f"    Modulating Factor: {modulating_factor_hard_neg.item():.4f}")
    print(f"    Focal Loss: {focal_loss_hard_neg.item():.4f}")

    # Observe how the modulating factor is larger for the "harder" negative example,
    # leading to a proportionally higher focal loss compared to the easy negative.
