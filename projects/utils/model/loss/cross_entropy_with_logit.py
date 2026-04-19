import torch
import torch.nn as nn
import torch.nn.functional as F

def symmetric_cross_entropy(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 1.0, beta: float = 1.0) -> torch.Tensor:
    """
    Computes the Symmetric Cross Entropy Loss.

    Symmetric Cross Entropy combines standard Cross-Entropy (CE) and Reverse Cross-Entropy (RCE)
    to improve robustness against noisy labels.

    Args:
        inputs (torch.Tensor): Raw logits from the model, shape (N, C) or (N, C, H, W).
                               N = batch size, C = number of classes.
        targets (torch.Tensor): Ground truth class indices, shape (N) or (N, H, W).
                                Contains class indices (0 to C-1).
        alpha (float): Weighting factor for the standard Cross-Entropy term.
        beta (float): Weighting factor for the Reverse Cross-Entropy term.

    Returns:
        torch.Tensor: Computed Symmetric Cross Entropy loss.
    """
    # Flatten inputs and targets if they are multi-dimensional (e.g., from image segmentation)
    if inputs.dim() > 2:
        inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, inputs.shape[1])
        targets = targets.view(-1)

    # 1. Standard Cross-Entropy (CE) Term
    # This term pushes the model to fit the given labels.
    ce_loss = F.cross_entropy(inputs, targets, reduction='none')

    # 2. Reverse Cross-Entropy (RCE) Term
    # This term regularizes the learning to be more robust to noisy labels.
    # It's calculated as -sum(p_k * log(y_k)), where p_k are model's probabilities
    # and y_k are the one-hot encoded true labels.
    # A small epsilon is added to y_k to avoid log(0) and provide stability.
    num_classes = inputs.shape[1]
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
    
    epsilon = 1e-6 # Small value to prevent log(0)
    targets_smoothed = targets_one_hot.clamp(min=epsilon, max=1.0 - epsilon)

    log_targets = torch.log(targets_smoothed)
    prob_inputs = F.softmax(inputs, dim=1)

    rce_loss = -torch.sum(prob_inputs * log_targets, dim=1)

    # Combine CE and RCE terms with their respective weights
    total_loss = alpha * ce_loss.mean() + beta * rce_loss.mean()
    
    return total_loss

if __name__ == '__main__':
    # --- Example Usage ---

    # Simulate model outputs (logits) for a batch of 4 examples, 3 classes
    # Example 1, 2: Clean labels
    # Example 3, 4: Noisy labels (true class might be different from the given label)
    logits = torch.tensor([
        [2.0, 0.5, 0.1],  # Pred: Class 0 (high confidence)
        [0.3, 1.5, 0.7],  # Pred: Class 1 (high confidence)
        [0.8, 0.9, 2.5],  # Pred: Class 2 (high confidence)
        [1.8, 0.2, 0.3]   # Pred: Class 0 (high confidence)
    ], dtype=torch.float32)

    # Simulate ground truth labels (some of which are noisy for demonstration)
    labels = torch.tensor([0, 1, 2, 0], dtype=torch.long) # Assume labels 2 and 0 are noisy for ex 3 and 4

    print("--- Standard Cross Entropy Loss ---")
    ce_loss_standard = F.cross_entropy(logits, labels)
    print(f"Standard CE Loss: {ce_loss_standard.item():.4f}")

    print("\n--- Symmetric Cross Entropy Loss (alpha=1.0, beta=1.0) ---")
    # Equal weighting for CE and RCE
    sce_loss_default = symmetric_cross_entropy(logits, labels, alpha=1.0, beta=1.0)
    print(f"SCE Loss (alpha=1.0, beta=1.0): {sce_loss_default.item():.4f}")

    print("\n--- Symmetric Cross Entropy Loss (alpha=0.7, beta=0.3) ---")
    # Emphasize CE more by giving it a higher alpha weight
    sce_loss_weighted = symmetric_cross_entropy(logits, labels, alpha=0.7, beta=0.3)
    print(f"SCE Loss (alpha=0.7, beta=0.3): {sce_loss_weighted.item():.4f}")

    print("\n--- Individual Example Analysis (Noisy Label Case) ---")

    # Analyze Example 3: logit=[0.8, 0.9, 2.5], label=2 (assumed noisy, true might be 1)
    # The model is confident in predicting class 2 based on the logits.
    single_logit = logits[2].unsqueeze(0) # Select example 3's logits
    single_label = labels[2].unsqueeze(0) # Select example 3's label

    ce_single = F.cross_entropy(single_logit, single_label, reduction='none').item()
    print(f"\n  Example 3 (Noisy Label, True is 1, Model predicts 2):")
    print(f"    CE Loss: {ce_single:.4f}")

    # RCE calculation for this specific example
    num_classes = single_logit.shape[1]
    targets_one_hot = F.one_hot(single_label, num_classes=num_classes).float()
    epsilon = 1e-6
    targets_smoothed = targets_one_hot.clamp(min=epsilon, max=1.0 - epsilon)
    log_targets = torch.log(targets_smoothed)
    prob_inputs = F.softmax(single_logit, dim=1)
    rce_single = -torch.sum(prob_inputs * log_targets, dim=1).item()
    print(f"    RCE Loss: {rce_single:.4f}")

    # Total SCE loss for this example with alpha=1, beta=1
    total_sce_single = (1.0 * ce_single + 1.0 * rce_single) / 2 # Averaging since both weights are 1.0
    print(f"    SCE Loss (alpha=1, beta=1): {total_sce_single:.4f}")

    # This example shows how SCE provides robustness. Even if the label is noisy (e.g., true class is 1, labeled as 2),
    # the RCE term helps prevent the model from aggressively overfitting to this potentially incorrect label.
    # The `targets_smoothed` effectively softens the one-hot target for the RCE component, ensuring stability
    # and allowing gradients to be more balanced.
