import torch
import torch.nn as nn
import torch.nn.functional as F

class SymmetricCrossEntropyLoss(nn.Module):
    """
    Implements Symmetric Cross-Entropy Loss for multi-class classification,
    designed to be robust to noisy labels.

    Combines standard Cross-Entropy (CE) and Reverse Cross-Entropy (RCE).

    Args:
        alpha (float): Weight for the standard Cross-Entropy term.
                       Often 1.0 (default) or tuned (e.g., 0.1).
        beta (float): Weight for the Reverse Cross-Entropy term.
                      Often 1.0 (default) or tuned (e.g., 1.0).
        num_classes (int): The total number of classes in the classification task.
        reduction (str): Specifies the reduction to apply to the output.
                         'none' | 'mean' | 'sum'. Default: 'mean'.
        epsilon (float): Small value added to log arguments for numerical stability.
    """
    def __init__(self,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 num_classes: int = None, # Mandatory for RCE, for one-hot conversion
                 reduction: str = 'mean',
                 epsilon: float = 1e-12): # Small constant for log stability
        super().__init__()
        
        if num_classes is None:
            raise ValueError("num_classes must be provided for SymmetricCrossEntropyLoss.")
            
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.reduction = reduction
        self.epsilon = epsilon # For log stability

    def forward(self, logits: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
        """
        Calculates Symmetric Cross-Entropy Loss.

        Args:
            logits (torch.Tensor): Model's raw output logits. Shape: (N, C)
            target_indices (torch.Tensor): Ground truth class labels (integer indices). Shape: (N,)

        Returns:
            torch.Tensor: The calculated symmetric cross-entropy loss.
        """
        # Ensure logits and target_indices are on the same device
        device = logits.device

        # 1. Predicted Probabilities (Q)
        # Apply softmax to logits to get probabilities
        pred_probs = F.softmax(logits, dim=-1) # Q(x)

        # 2. Ground Truth Probabilities (P) - One-hot encoded
        # Convert integer targets to one-hot vector
        target_one_hot = F.one_hot(target_indices, num_classes=self.num_classes).float().to(device) # P(x)

        # --- Calculate Standard Cross-Entropy (CE) ---
        # L_CE = - ∑ P(x) * log(Q(x))
        # F.cross_entropy does this directly with logits, but here we work with log_probs of Q
        log_pred_probs = torch.log(pred_probs + self.epsilon) # log Q(x)
        ce_loss_unreduced = - (target_one_hot * log_pred_probs).sum(dim=-1) # (N,) per sample

        # --- Calculate Reverse Cross-Entropy (RCE) ---
        # L_RCE = - ∑ Q(x) * log(P(x))
        # Add epsilon to target_one_hot before log to prevent log(0) for P(x) = 0
        log_target_probs = torch.log(target_one_hot + self.epsilon) # log P(x)
        rce_loss_unreduced = - (pred_probs * log_target_probs).sum(dim=-1) # (N,) per sample

        # --- Combine for Symmetric Loss ---
        total_loss_unreduced = self.alpha * ce_loss_unreduced + self.beta * rce_loss_unreduced

        # --- Apply Reduction ---
        if self.reduction == 'mean':
            return total_loss_unreduced.mean()
        elif self.reduction == 'sum':
            return total_loss_unreduced.sum()
        elif self.reduction == 'none':
            return total_loss_unreduced
        else:
            raise ValueError("Unsupported reduction type.")

# --- Example Usage ---
if __name__ == '__main__':
    num_classes = 3
    batch_size = 4

    # Dummy logits from a model
    dummy_logits = torch.randn(batch_size, num_classes)
    dummy_logits.requires_grad_(True) # Enable gradient tracking

    # Dummy ground truth labels (integer indices)
    dummy_targets = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    print(f"Dummy Logits:\n{dummy_logits}")
    print(f"\nDummy Targets: {dummy_targets.tolist()}")

    # --- Scenario 1: Balanced weights (alpha=1.0, beta=1.0) ---
    print("\n--- Scenario 1: Balanced Symmetric Cross-Entropy ---")
    sce_loss_fn_balanced = SymmetricCrossEntropyLoss(
        alpha=1.0, beta=1.0, num_classes=num_classes, reduction='mean'
    )
    loss_balanced = sce_loss_fn_balanced(dummy_logits, dummy_targets)
    print(f"Loss (alpha=1.0, beta=1.0): {loss_balanced.item():.4f}")

    # --- Scenario 2: More emphasis on RCE (e.g., alpha=0.1, beta=1.0 for more noise robustness) ---
    print("\n--- Scenario 2: Emphasis on Reverse CE for Noise Robustness ---")
    sce_loss_fn_robust = SymmetricCrossEntropyLoss(
        alpha=0.1, beta=1.0, num_classes=num_classes, reduction='mean'
    )
    loss_robust = sce_loss_fn_robust(dummy_logits, dummy_targets)
    print(f"Loss (alpha=0.1, beta=1.0): {loss_robust.item():.4f}")

    # --- Test with a noisy label to illustrate ---
    # Imagine target for sample 0 is actually 1, but we label it 0 (noisy)
    noisy_targets_dummy = torch.tensor([1, 1, 2, 0], dtype=torch.long) # Original target for sample 0 was 0, now it's 1

    # Let's make model predict confidently correct for sample 0 based on the noisy label
    # This scenario is hard to illustrate with random logits.
    # Let's consider a simple case where the model makes a very confident WRONG prediction
    # on sample 0, target is 0, but model predicts class 2 with high confidence.
    # Standard CE will be high. RCE will also be high.
    
    # Let's make logits predict wrong confidently for sample 0 (target is 0, model predicts 2)
    noisy_logits_scenario = torch.tensor([
        [-5.0, -5.0, 5.0],  # Sample 0 predicts class 2 confidently (wrong)
        [1.0, 0.5, 0.1],    # Sample 1 correct
        [-0.1, -0.2, 0.9],  # Sample 2 correct
        [0.9, 0.1, 0.0]     # Sample 3 correct
    ])
    noisy_targets_scenario = torch.tensor([0, 1, 2, 0], dtype=torch.long) # Sample 0 is correctly labeled as 0

    print("\n--- Scenario 3: Confident Wrong Prediction (Sample 0, target 0, pred 2) ---")
    # Standard CE for sample 0: Very high because P[0]=1, Q[0] very low.
    # RCE for sample 0: Very high because Q[2] is high, P[2]=0.
    
    # Calculate Standard CE for comparison
    standard_ce_loss_fn = nn.CrossEntropyLoss(reduction='mean')
    standard_loss = standard_ce_loss_fn(noisy_logits_scenario, noisy_targets_scenario)
    print(f"Standard CE Loss: {standard_loss.item():.4f}")

    # Calculate Symmetric CE
    sce_loss_fn_test = SymmetricCrossEntropyLoss(alpha=1.0, beta=1.0, num_classes=num_classes, reduction='mean')
    sce_loss_test = sce_loss_fn_test(noisy_logits_scenario, noisy_targets_scenario)
    print(f"Symmetric CE Loss (alpha=1.0, beta=1.0): {sce_loss_test.item():.4f}")
    
    sce_loss_fn_robust_test = SymmetricCrossEntropyLoss(alpha=0.1, beta=1.0, num_classes=num_classes, reduction='mean')
    sce_loss_robust_test = sce_loss_fn_robust_test(noisy_logits_scenario, noisy_targets_scenario)
    print(f"Symmetric CE Loss (alpha=0.1, beta=1.0): {sce_loss_robust_test.item():.4f}")

    # The effect of noise robustness is more apparent with a dataset of mixed clean/noisy labels
    # where the model struggles less to learn from clean labels due to softening the noisy ones.