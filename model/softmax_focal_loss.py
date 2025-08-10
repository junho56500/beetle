import torch
import torch.nn as nn
import torch.nn.functional as F

def softmax_focal_loss(inputs: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    """
    Computes the Softmax Focal Loss.

    Args:
        inputs (torch.Tensor): Raw logits from the model, shape (N, C) or (N, C, H, W).
                               N = batch size, C = number of classes.
        targets (torch.Tensor): Ground truth class indices, shape (N) or (N, H, W).
                                Contains class indices (0 to C-1).
        gamma (float): The focusing parameter. Higher values increase focus on hard examples.
        alpha (float): The weighting factor for classes (alpha_t in the formula).
                       Can be a scalar or a tensor of shape (C,).

    Returns:
        torch.Tensor: Computed focal loss.
    """
    # Apply softmax to logits to get probabilities
    # Permute dimensions if inputs are like (N, C, H, W) to (N*H*W, C) for F.cross_entropy
    if inputs.dim() > 2:
        inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, inputs.shape[1])
        targets = targets.view(-1)

    # Calculate standard Cross-Entropy Loss per element
    # F.cross_entropy combines log_softmax and nll_loss
    # We set reduction='none' to get individual losses before applying the modulating factor
    ce_loss = F.cross_entropy(inputs, targets, reduction='none')

    # Get the probabilities of the true class (p_t)
    # Use softmax and gather for p_t
    prob = F.softmax(inputs, dim=1)
    prob_t = prob.gather(1, targets.unsqueeze(1)).squeeze(1)

    # Calculate the modulating factor (1 - p_t)^gamma
    modulating_factor = (1 - prob_t) ** gamma

    # Apply alpha weighting (if alpha is a scalar, it applies to all, otherwise per-class)
    if isinstance(alpha, (int, float)):
        alpha_factor = alpha
    else: # Assuming alpha is a tensor for class-specific weighting
        alpha_factor = alpha.gather(0, targets)

    # Combine all parts
    focal_loss = alpha_factor * modulating_factor * ce_loss

    # Return the mean loss over the batch
    return focal_loss.mean()

if __name__ == '__main__':
    # --- Example Usage ---

    # Simulate model outputs (logits) for a batch of 4 examples, 3 classes
    # N=4, C=3
    logits = torch.tensor([
        [2.0, 0.5, 0.1],  # Example 1: Model confident in class 0
        [0.3, 1.5, 0.7],  # Example 2: Model confident in class 1
        [0.8, 0.9, 1.2],  # Example 3: Model less confident, slightly favoring class 2
        [0.1, 0.2, -0.5]  # Example 4: Model confident in class 1 (but it's wrong!)
    ], dtype=torch.float32)

    # Simulate ground truth labels
    # Corresponding to class 0, 1, 2, 0 respectively
    labels = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    print("--- Softmax Cross Entropy Loss ---")
    ce_loss_standard = F.cross_entropy(logits, labels)
    print(f"Standard CE Loss: {ce_loss_standard.item():.4f}")

    print("\n--- Softmax Focal Loss (Gamma = 0) ---")
    # Gamma = 0 should be equivalent to standard CE (ignoring alpha for now)
    focal_loss_gamma0 = softmax_focal_loss(logits, labels, gamma=0.0, alpha=1.0)
    print(f"Focal Loss (gamma=0, alpha=1.0): {focal_loss_gamma0.item():.4f}")
    # Note: A slight difference might occur due to mean vs sum reduction depending on implementation.
    # F.cross_entropy defaults to mean reduction.

    print("\n--- Softmax Focal Loss (Gamma = 2.0) ---")
    focal_loss_gamma2 = softmax_focal_loss(logits, labels, gamma=2.0, alpha=0.25)
    print(f"Focal Loss (gamma=2.0, alpha=0.25): {focal_loss_gamma2.item():.4f}")

    print("\n--- Individual Example Analysis (Gamma = 2.0, Alpha = 0.25) ---")
    # Let's manually calculate for one example to understand the modulating factor
    example_idx = 0 # First example: logit=[2.0, 0.5, 0.1], label=0
    single_logit = logits[example_idx].unsqueeze(0)
    single_label = labels[example_idx].unsqueeze(0)

    ce_loss_single = F.cross_entropy(single_logit, single_label, reduction='none')
    prob_single = F.softmax(single_logit, dim=1)
    prob_t_single = prob_single.gather(1, single_label.unsqueeze(1)).squeeze(1)

    print(f"  Example {example_idx}: True label probability (p_t): {prob_t_single.item():.4f}")
    print(f"  Example {example_idx}: Cross-Entropy Loss: {ce_loss_single.item():.4f}")

    gamma_val = 2.0
    alpha_val = 0.25
    modulating_factor_single = (1 - prob_t_single) ** gamma_val
    focal_loss_single = alpha_val * modulating_factor_single * ce_loss_single
    print(f"  Example {example_idx}: Modulating factor (1-p_t)^gamma: {modulating_factor_single.item():.4f}")
    print(f"  Example {example_idx}: Focal Loss: {focal_loss_single.item():.4f}")

    # Now consider Example 4, which is a hard example (wrongly predicted)
    example_idx = 3 # Fourth example: logit=[0.1, 0.2, -0.5], label=0
    single_logit = logits[example_idx].unsqueeze(0)
    single_label = labels[example_idx].unsqueeze(0)

    ce_loss_single_hard = F.cross_entropy(single_logit, single_label, reduction='none')
    prob_single_hard = F.softmax(single_logit, dim=1)
    prob_t_single_hard = prob_single_hard.gather(1, single_label.unsqueeze(1)).squeeze(1)

    print(f"\n  Example {example_idx} (Hard Case): True label probability (p_t): {prob_t_single_hard.item():.4f}")
    print(f"  Example {example_idx} (Hard Case): Cross-Entropy Loss: {ce_loss_single_hard.item():.4f}")

    modulating_factor_single_hard = (1 - prob_t_single_hard) ** gamma_val
    focal_loss_single_hard = alpha_val * modulating_factor_single_hard * ce_loss_single_hard
    print(f"  Example {example_idx} (Hard Case): Modulating factor (1-p_t)^gamma: {modulating_factor_single_hard.item():.4f}")
    print(f"  Example {example_idx} (Hard Case): Focal Loss: {focal_loss_single_hard.item():.4f}")

    # Notice that for Example 0 (easy), the focal loss is much lower than CE.
    # For Example 3 (hard), the focal loss is closer to CE (or still significant).
    # This demonstrates the down-weighting effect on easy examples.
