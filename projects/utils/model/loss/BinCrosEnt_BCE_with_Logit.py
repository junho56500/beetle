import torch
import torch.nn.functional as F

# Assume your model outputs logits for a binary classification task
# Shape: (batch_size, 1) if single output neuron per sample
# Or (batch_size, H, W) for pixel-wise binary prediction like objectness
batch_size = 3
num_pixels = 4 # Example for 1D pixel-wise prediction

# Model's raw output (logits) for 3 samples, each making 4 pixel-wise binary predictions
# (N, C, H, W) -> for binary, C=1, so (N, 1, H, W)
model_logits = torch.tensor([
    [[[ 2.5, -1.0,  0.8, -2.0]]], # Sample 1 logits (1 channel)
    [[[-0.5,  3.0, -0.1,  1.5]]], # Sample 2 logits
    [[[ 1.0, -0.8,  0.3,  0.7]]]  # Sample 3 logits
]) # Shape: (3, 1, 1, 4) if thinking (N, C, H, W)

# BCE Target (Ground Truth Labels)
# Must match model_logits shape or be broadcastable, and values are 0.0 or 1.0
bce_target = torch.tensor([
    [[[ 1.0,  0.0,  1.0,  0.0]]], # Sample 1 labels (1.0 for positive, 0.0 for negative)
    [[[ 0.0,  1.0,  0.0,  1.0]]], # Sample 2 labels
    [[[ 1.0,  0.0,  0.0,  1.0]]]  # Sample 3 labels
]) # Shape: (3, 1, 1, 4)

# Ensure target is float type (important!)
bce_target = bce_target.float()

print(f"Model Logits shape: {model_logits.shape}")
print(f"BCE Target shape: {bce_target.shape}")
print(f"BCE Target data type: {bce_target.dtype}")

# Calculate BCEWithLogitsLoss
loss = F.binary_cross_entropy_with_logits(model_logits, bce_target, reduction='mean')

print(f"\nCalculated BCE Loss: {loss.item():.4f}")

# Example with different shapes (broadcastable)
# If target is (3,4) (no channel dim, no height/width if thinking pixel-wise, but flat list of predictions)
model_logits_flat = torch.tensor([
    [ 2.5, -1.0,  0.8, -2.0],
    [-0.5,  3.0, -0.1,  1.5],
    [ 1.0, -0.8,  0.3,  0.7]
]) # Shape: (3, 4)

bce_target_flat = torch.tensor([
    [ 1.0,  0.0,  1.0,  0.0],
    [ 0.0,  1.0,  0.0,  1.0],
    [ 1.0,  0.0,  0.0,  1.0]
]).float() # Shape: (3, 4)

loss_flat = F.binary_cross_entropy_with_logits(model_logits_flat, bce_target_flat, reduction='mean')
print(f"Calculated BCE Loss (flat): {loss_flat.item():.4f}")