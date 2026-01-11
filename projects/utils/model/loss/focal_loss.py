import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weighting factor for positive/negative classes
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction  # How to reduce the loss (mean, sum, none)

    def forward(self, inputs, targets):
        # inputs: raw logits from the model (before sigmoid/softmax)
        # targets: true labels (0 or 1 for binary, one-hot or class index for multi-class)

        # For binary classification, we typically use sigmoid
        # For multi-class, we use softmax.
        # This implementation assumes binary classification for simplicity.
        # For multi-class, you'd adapt by using F.log_softmax and F.nll_loss
        # or compute pt using softmax probabilities.

        # Ensure inputs are logits for numerical stability with F.binary_cross_entropy_with_logits
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Calculate pt: probability of the true class
        # For binary_cross_entropy_with_logits, it directly computes -log(pt)
        # So, pt = exp(-BCE_loss)
        # Use torch.sigmoid(inputs) to get probabilities
        p_t = torch.sigmoid(inputs)
        p_t = p_t * targets + (1 - p_t) * (1 - targets) # probability for the true class

        # Calculate modulating factor (1 - pt)^gamma
        modulating_factor = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        # alpha_t is alpha for positive class, 1-alpha for negative class
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Combine terms
        focal_loss = alpha_t * modulating_factor * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- Example Usage ---
if __name__ == "__main__":
    # Simulate some model outputs (logits) and true labels
    # Batch size = 5
    # Binary classification example

    # Case 1: Easy Positive Example (well-classified)
    # Logit for positive class = 4.0 -> P(positive) approx 0.98
    # True label = 1
    inputs1 = torch.tensor([4.0])
    targets1 = torch.tensor([1.0])

    # Case 2: Hard Positive Example (misclassified or low confidence)
    # Logit for positive class = -2.0 -> P(positive) approx 0.12
    # True label = 1
    inputs2 = torch.tensor([-2.0])
    targets2 = torch.tensor([1.0])

    # Case 3: Easy Negative Example (well-classified)
    # Logit for positive class = -5.0 -> P(positive) approx 0.006
    # True label = 0
    inputs3 = torch.tensor([-5.0])
    targets3 = torch.tensor([0.0])

    # Case 4: Hard Negative Example (misclassified or low confidence)
    # Logit for positive class = 1.0 -> P(positive) approx 0.73
    # True label = 0
    inputs4 = torch.tensor([1.0])
    targets4 = torch.tensor([0.0])

    # Combining into a batch
    inputs_batch = torch.cat([inputs1, inputs2, inputs3, inputs4]).unsqueeze(1)
    targets_batch = torch.cat([targets1, targets2, targets3, targets4]).unsqueeze(1)

    # Standard Binary Cross Entropy Loss
    bce_loss_fn = nn.BCEWithLogitsLoss()
    bce_loss = bce_loss_fn(inputs_batch, targets_batch)
    print(f"Standard BCE Loss: {bce_loss.item():.4f}\n")

    # Focal Loss (gamma=2, alpha=0.25)
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    focal_loss = focal_loss_fn(inputs_batch, targets_batch)
    print(f"Focal Loss (alpha=0.25, gamma=2.0): {focal_loss.item():.4f}\n")

    # Let's inspect individual contributions with reduction='none'
    focal_loss_individual_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction='none')
    individual_focal_losses = focal_loss_individual_fn(inputs_batch, targets_batch)
    
    bce_individual_fn = nn.BCEWithLogitsLoss(reduction='none')
    individual_bce_losses = bce_individual_fn(inputs_batch, targets_batch)

    print("-" * 30)
    print("Individual Loss Contributions:")
    print("-" * 30)
    
    prob_positive = torch.sigmoid(inputs_batch)

    for i in range(inputs_batch.shape[0]):
        input_val = inputs_batch[i].item()
        target_val = targets_batch[i].item()
        prob_pos_val = prob_positive[i].item()
        
        # Calculate pt for this example
        pt_val = prob_pos_val if target_val == 1 else (1 - prob_pos_val)
        
        # Calculate modulating factor (1-pt)^gamma
        mod_factor_val = (1 - pt_val) ** focal_loss_fn.gamma
        
        # Calculate alpha_t
        alpha_t_val = focal_loss_fn.alpha if target_val == 1 else (1 - focal_loss_fn.alpha)

        print(f"Example {i+1}:")
        print(f"  Logit: {input_val:.2f}, True Label: {int(target_val)}")
        print(f"  P(positive): {prob_pos_val:.4f}, P_t: {pt_val:.4f}")
        print(f"  Modulating Factor (1-p_t)^gamma: {mod_factor_val:.4f}")
        print(f"  Alpha_t: {alpha_t_val:.4f}")
        print(f"  Individual BCE Loss: {individual_bce_losses[i].item():.4f}")
        print(f"  Individual Focal Loss: {individual_focal_losses[i].item():.4f}")
        print("-" * 20)

    # Observe:
    # - For easy examples (1 and 3), Focal Loss is significantly lower than BCE Loss.
    # - For hard examples (2 and 4), Focal Loss is closer to BCE Loss,
    #   and even slightly higher if alpha favors the minority class.