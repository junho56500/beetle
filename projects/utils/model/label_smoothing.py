import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    """
    Implements Label Smoothing Loss for classification tasks.

    Args:
        num_classes (int): The total number of classes.
        smoothing (float): The smoothing parameter (epsilon).
                           Typically a small value like 0.1.
        reduction (str): Specifies the reduction to apply to the output.
                         'none' | 'mean' | 'sum'. Default: 'mean'.
    """
    def __init__(self, num_classes: int, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction

        if not (0 <= smoothing < 1):
            raise ValueError("Smoothing parameter must be between 0 and 1 (exclusive of 1).")

        # The 'confidence' for the true class, which is 1 - smoothing
        self.confidence = 1.0 - smoothing
        # The 'smooth_value' to be distributed among other classes
        self.smooth_value = smoothing / (num_classes - 1) if num_classes > 1 else 0.0

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the label-smoothed loss.

        Args:
            logits (torch.Tensor): The model's raw output logits.
                                   Shape: (batch_size, num_classes).
            target (torch.Tensor): The ground truth class labels (integer indices).
                                   Shape: (batch_size,).
        Returns:
            torch.Tensor: The calculated smoothed loss.
        """
        # 1. Convert hard target labels to one-hot encoding
        # (batch_size, num_classes)
        one_hot_target = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1)

        # 2. Apply label smoothing to the one-hot target
        # For the true class (where one_hot_target is 1), replace with confidence.
        # For other classes (where one_hot_target is 0), replace with smooth_value.
        # This creates the soft_labels: (batch_size, num_classes)
        smooth_target = one_hot_target * self.confidence + self.smooth_value

        # 3. Apply log_softmax to model logits for numerical stability with KLDivLoss
        log_probs = F.log_softmax(logits, dim=-1)

        # 4. Calculate KL Divergence between log_probs and smooth_target
        # KLDivLoss(input, target) expects input to be log-probabilities and target to be probabilities.
        # The reduction in KLDivLoss is usually 'batchmean' or 'sum' depending on exact definition.
        # Here, we use 'none' to get element-wise loss, then apply our desired reduction.
        loss_unreduced = F.kl_div(log_probs, smooth_target, reduction='none')

        # Sum over classes, then apply desired reduction (mean across batch)
        # Summing loss over the class dimension
        loss_per_sample = loss_unreduced.sum(dim=-1)

        if self.reduction == 'mean':
            return loss_per_sample.mean()
        elif self.reduction == 'sum':
            return loss_per_sample.sum()
        elif self.reduction == 'none':
            return loss_per_sample
        else:
            raise ValueError("Unsupported reduction type.")


# --- Example Usage ---
if __name__ == "__main__":
    num_classes = 5
    smoothing_epsilon = 0.1
    batch_size = 4

    # Dummy model output (raw logits)
    # Higher logits mean higher confidence for that class
    dummy_logits = torch.randn(batch_size, num_classes)
    dummy_logits.requires_grad_(True) # Enable gradient tracking

    # Dummy ground truth hard labels (integer indices)
    # e.g., Batch 0 is class 2, Batch 1 is class 0, etc.
    dummy_targets_hard = torch.tensor([2, 0, 4, 1], dtype=torch.long)

    print(f"Original Hard Targets (integer indices): {dummy_targets_hard}")

    # Convert hard targets to one-hot for visualization
    one_hot_targets = torch.zeros(batch_size, num_classes).scatter_(1, dummy_targets_hard.unsqueeze(1), 1)
    print(f"\nOne-Hot Targets (example):\n{one_hot_targets[0]}") # Show one sample

    # --- Initialize Label Smoothing Loss ---
    label_smoothing_loss_fn = LabelSmoothingLoss(
        num_classes=num_classes,
        smoothing=smoothing_epsilon,
        reduction='mean'
    )

    # --- Calculate Loss with Label Smoothing ---
    smoothed_loss = label_smoothing_loss_fn(dummy_logits, dummy_targets_hard)

    print(f"\nLoss with Label Smoothing (epsilon={smoothing_epsilon}): {smoothed_loss.item():.4f}")

    # --- For comparison: Standard Cross-Entropy Loss ---
    standard_ce_loss = F.cross_entropy(dummy_logits, dummy_targets_hard, reduction='mean')
    print(f"Standard Cross-Entropy Loss: {standard_ce_loss.item():.4f}")


    # --- Visualize Smoothed Targets for a Single Example ---
    single_hard_target = torch.tensor([dummy_targets_hard[0]], dtype=torch.long)
    single_logits = dummy_logits[0].unsqueeze(0) # Unsqueeze to (1, num_classes)

    # Manually calculate smoothed target for one example
    one_hot_single = torch.zeros_like(single_logits).scatter_(1, single_hard_target.unsqueeze(1), 1)
    smooth_target_single = one_hot_single * (1.0 - smoothing_epsilon) + (smoothing_epsilon / (num_classes - 1))

    print(f"\n--- Visualization for a Single Sample (Hard Target: {single_hard_target.item()}) ---")
    print(f"Original One-Hot Target: {one_hot_single.squeeze().tolist()}")
    print(f"Smoothed Target (epsilon={smoothing_epsilon}): {smooth_target_single.squeeze().tolist()}")
    # Notice how the '1' becomes '1 - epsilon' and '0's become 'epsilon / (K-1)'