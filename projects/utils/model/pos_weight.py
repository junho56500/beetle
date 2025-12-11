import torch

# Assuming your training labels are in a tensor (or list/array converted to tensor)
# Example: labels = torch.tensor([0., 1., 0., 0., 1., 0.])
# For pixel-wise labels in segmentation: labels = torch.tensor([[0.,1.],[0.,0.]]) etc.

# Flatten the labels tensor if it's multi-dimensional (e.g., from (N, H, W) to 1D)
# Ensure it's float type if it's not already, as BCE targets are float
flat_labels = your_training_labels_tensor.float().view(-1)

N_positive = flat_labels.sum().item() # Sum of 1s
N_negative = len(flat_labels) - N_positive # Total elements - sum of 1s

if N_positive > 0:
    calculated_pos_weight = N_negative / N_positive
    print(f"Number of positive samples: {N_positive}")
    print(f"Number of negative samples: {N_negative}")
    print(f"Calculated pos_weight: {calculated_pos_weight:.2f}")
    
    # Pass this to your loss function:
    # loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(calculated_pos_weight))
else:
    print("Warning: No positive samples found. pos_weight cannot be calculated based on frequency.")
    # Handle this case: pos_weight might remain 1.0 or you might raise an error.